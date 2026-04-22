"""
src/ar/server.py

Layer 4 — AR WebSocket Server
Streams map snapshots to connected AR clients over WebSocket.
Serves the Three.js WebXR frontend.
"""
import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn


app = FastAPI(title="Lowrance AR Map Server")


# ---------------------------------------------------------------------------
# Connection manager — broadcast to all connected AR clients
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()

# Injected by main.py
_mapping_worker = None

def set_mapping_worker(worker):
    global _mapping_worker
    _mapping_worker = worker


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/stats")
def stats():
    if _mapping_worker is None:
        return {"error": "mapping worker not initialised"}
    snap = _mapping_worker.get_snapshot()
    return snap or {"status": "no data yet"}


@app.get("/mesh")
def mesh():
    """Full mesh export — may be large. Use for offline inspection."""
    if _mapping_worker is None:
        return {"error": "mapping worker not initialised"}
    return _mapping_worker.get_full_mesh() or {"error": "mesh unavailable (install scikit-image)"}


@app.post("/api/speed")
def set_speed(value: float = 1.0):
    from utils import speed_control
    speed_control.set(value)
    return {"speed": speed_control.get()}


# ---------------------------------------------------------------------------
# WebSocket — streams live pointcloud updates
# ---------------------------------------------------------------------------

@app.websocket("/ws/map")
async def ws_map(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Client can send pose updates (AR headset position)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                pose = json.loads(data)
                # TODO: pass pose back to mapping engine for view-dependent LOD
            except asyncio.TimeoutError:
                pass

            snap = _mapping_worker.get_snapshot() if _mapping_worker else None
            if snap:
                await websocket.send_text(json.dumps({
                    "type": "map_update",
                    **snap
                }))
            await asyncio.sleep(0.1)   # 10 Hz update rate
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Serve the Three.js AR frontend
# ---------------------------------------------------------------------------

AR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lowrance AR Depth Map</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0f1a; color: #e0e8f0; font-family: monospace; overflow: hidden; }
  #hud {
    position: fixed; top: 16px; left: 16px; z-index: 10;
    background: rgba(10,20,40,0.7); border: 1px solid rgba(0,200,180,0.3);
    padding: 12px 16px; border-radius: 8px; font-size: 13px; line-height: 1.8;
  }
  #hud span { color: #00c8b4; }
  #canvas { position: fixed; inset: 0; }
  #xr-btn {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #00c8b4; color: #0a0f1a; border: none; border-radius: 8px;
    padding: 12px 28px; font-size: 15px; font-weight: 700; cursor: pointer;
    display: none;
  }
  #speed-ctrl {
    position: fixed; bottom: 24px; right: 24px; z-index: 10;
    background: rgba(10,20,40,0.75); border: 1px solid rgba(0,200,180,0.3);
    padding: 10px 14px; border-radius: 8px; font-family: monospace;
    font-size: 12px; color: #e0e8f0; display: flex; flex-direction: column; gap: 6px;
  }
  #speed-ctrl label { color: #00c8b4; }
  #speed-slider { width: 140px; accent-color: #00c8b4; }
</style>
</head>
<body>
<div id="hud">
  Depth Map AR v1.5<br>
  Points: <span id="h-pts">—</span><br>
  Coverage: <span id="h-cov">—</span><br>
  Mean depth: <span id="h-dep">—</span> m<br>
  Updates: <span id="h-upd">—</span>
</div>
<div id="speed-ctrl">
  <label>Speed: <span id="speed-val">1.0×</span></label>
  <input id="speed-slider" type="range" min="-2" max="2" step="0.5" value="0">
</div>
<canvas id="canvas"></canvas>
<button id="xr-btn">Enter AR</button>

<script type="importmap">
  { "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.162/build/three.module.js" } }
</script>
<script type="module">
import * as THREE from 'three';

// ── Three.js scene setup ──────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.xr.enabled = true;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, -30, 50);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0x334466, 1.5));
const dLight = new THREE.DirectionalLight(0x99ccff, 2);
dLight.position.set(50, -50, 100);
scene.add(dLight);

// ── 3D grid ───────────────────────────────────────────────────────────────
const GRID_SIZE  = 200;
const GRID_DIVS  = 40;
const GRID_DEPTH = 30;
const GRID_STEP  = 5;

for (let d = 0; d <= GRID_DEPTH; d += GRID_STEP) {
  const plane = new THREE.GridHelper(
    GRID_SIZE, GRID_DIVS,
    d === 0 ? 0x4a8ab8 : 0x2a5a82,
    d === 0 ? 0x2a5070 : 0x153048
  );
  plane.rotation.x = Math.PI / 2;
  plane.position.z = -d;
  plane.material.transparent = true;
  plane.material.opacity = d === 0 ? 0.70 : Math.max(0.12, 0.55 - d * 0.014);
  scene.add(plane);
}

(function() {
  const half = GRID_SIZE / 2, step = 20;
  const verts = [];
  for (let x = -half; x <= half + 0.1; x += step) {
    for (let y = -half; y <= half + 0.1; y += step) {
      verts.push(x, y, 0,  x, y, -GRID_DEPTH);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(verts), 3));
  scene.add(new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0x3a7aaa, transparent: true, opacity: 0.40 })));
})();

// ── Point cloud geometry ──────────────────────────────────────────────────
const MAX_PTS = 200_000;
const geom = new THREE.BufferGeometry();
const positions = new Float32Array(MAX_PTS * 3);
const colors    = new Float32Array(MAX_PTS * 3);
geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geom.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
geom.setDrawRange(0, 0);

const mat = new THREE.PointsMaterial({
  size: 0.4, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.85
});
const points = new THREE.Points(geom, mat);
scene.add(points);

// ── Boat position marker ──────────────────────────────────────────────────
const boatGeo = new THREE.ConeGeometry(1.2, 3.5, 4);
boatGeo.rotateX(Math.PI / 2);   // point forward along +Y
const boatMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc });
const boatMesh = new THREE.Mesh(boatGeo, boatMat);
boatMesh.visible = false;
scene.add(boatMesh);

// Heading ring around the marker
const ringGeo = new THREE.RingGeometry(2.0, 2.4, 32);
const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc, side: THREE.DoubleSide, transparent: true, opacity: 0.45 });
const boatRing = new THREE.Mesh(ringGeo, ringMat);
scene.add(boatRing);

// ── Colour mapping: depth + confidence → RGB ──────────────────────────────
function depthColor(depth, maxDepth, confidence, std) {
  // Hue: 240° (blue, shallow) → 0° (red, deep)
  const t = Math.min(depth / maxDepth, 1.0);
  const h = (1 - t) * 240 / 360;
  const s = 0.8;
  const l = 0.3 + confidence * 0.4;  // brighter = more confident
  return hsl2rgb(h, s, l);
}

function hsl2rgb(h, s, l) {
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue(p, q, h + 1/3);
    g = hue(p, q, h);
    b = hue(p, q, h - 1/3);
  }
  return [r, g, b];
}
function hue(p, q, t) {
  if (t < 0) t += 1; if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}

// ── WebSocket connection ──────────────────────────────────────────────────
let maxDepth = 20;
const ws = new WebSocket(`ws://${location.host}/ws/map`);

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type !== 'map_update') return;

  const pc = msg.pointcloud;
  if (!pc || !pc.x) return;

  const n = Math.min(pc.x.length, MAX_PTS);
  maxDepth = Math.max(maxDepth, ...pc.depth.slice(0, n));

  for (let i = 0; i < n; i++) {
    positions[i * 3]     = pc.x[i];
    positions[i * 3 + 1] = pc.y[i];
    positions[i * 3 + 2] = -pc.depth[i];   // depth is negative Z in ENU

    const [r, g, b] = depthColor(pc.depth[i], maxDepth, pc.confidence[i], pc.std[i]);
    colors[i * 3]     = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate    = true;
  geom.setDrawRange(0, n);

  // HUD
  document.getElementById('h-pts').textContent = n.toLocaleString();
  document.getElementById('h-cov').textContent = (msg.stats?.coverage_pct ?? '—') + '%';
  document.getElementById('h-dep').textContent = msg.stats?.mean_depth ?? '—';
  document.getElementById('h-upd').textContent = msg.stats?.updates ?? '—';

  // Boat marker
  const b = msg.boat;
  if (b && b.east !== undefined) {
    boatMesh.position.set(b.east, b.north, 0.5);
    boatMesh.rotation.z = -THREE.MathUtils.degToRad(b.heading);
    boatMesh.visible = true;
    boatRing.position.set(b.east, b.north, 0.2);
  }
};

ws.onclose = () => console.warn('[WS] Disconnected — will retry');

// ── Speed slider ──────────────────────────────────────────────────────────
document.getElementById('speed-slider').addEventListener('input', e => {
  const multiplier = Math.pow(2, parseFloat(e.target.value));
  document.getElementById('speed-val').textContent = multiplier.toFixed(2) + '×';
  fetch('/api/speed?value=' + multiplier, { method: 'POST' });
});

// ── Orbit controls — roll-free turntable (Z-up) ──────────────────────────
// Left/right drag: azimuth around world Z. Up/down drag: elevation.
// Elevation is clamped so the camera never rolls or flips.
(function () {
  const cam = camera;
  let radius = cam.position.length();             // ~58 m initial
  let az = Math.atan2(cam.position.y, cam.position.x);  // ≈ -π/2
  let el = Math.asin(Math.max(-1, Math.min(1, cam.position.z / radius))); // ≈ 59°

  function apply() {
    cam.position.set(
      radius * Math.cos(el) * Math.cos(az),
      radius * Math.cos(el) * Math.sin(az),
      radius * Math.sin(el)
    );
    cam.up.set(0, 0, 1);   // enforce Z-up so grid never tilts
    cam.lookAt(0, 0, 0);
  }

  let drag = false, px = 0, py = 0;
  canvas.addEventListener('mousedown',  e => { drag = true;  px = e.clientX; py = e.clientY; });
  canvas.addEventListener('mouseup',    () => drag = false);
  canvas.addEventListener('mouseleave', () => drag = false);
  canvas.addEventListener('mousemove',  e => {
    if (!drag) return;
    const dx = e.clientX - px, dy = e.clientY - py;
    px = e.clientX; py = e.clientY;
    az -= dx * 0.003;
    el  = Math.max(0.05, Math.min(Math.PI / 2 - 0.05, el - dy * 0.003));
    apply();
  });
  canvas.addEventListener('wheel', e => {
    radius = Math.max(5, Math.min(500, radius * (1 + e.deltaY * 0.001)));
    apply();
  });

  apply();   // sync camera to initial angles
})();

// ── WebXR AR button ───────────────────────────────────────────────────────
if ('xr' in navigator) {
  navigator.xr.isSessionSupported('immersive-ar').then(supported => {
    if (supported) {
      document.getElementById('xr-btn').style.display = 'block';
      document.getElementById('xr-btn').onclick = () => {
        navigator.xr.requestSession('immersive-ar', {
          requiredFeatures: ['local-floor'],
          optionalFeatures: ['bounded-floor', 'plane-detection']
        }).then(session => {
          renderer.xr.setSession(session);
        });
      };
    }
  });
}

// ── Render loop ───────────────────────────────────────────────────────────
renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);

  // Send pose to server for view-dependent LOD
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'pose',
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    }));
  }
});

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return AR_HTML


# ---------------------------------------------------------------------------
# Entry point (used when running standalone for development)
# ---------------------------------------------------------------------------

def start(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port, log_level="info")
