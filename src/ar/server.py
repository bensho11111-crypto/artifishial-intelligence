"""
src/ar/server.py

Layer 4 — AR WebSocket Server (mode-aware)
"""
import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="Lowrance AR Map Server")

# Injected by main.py via set_app_context()
_ctx            = None
_mapping_worker = None
_replay_ctrl    = None
_ground_truth   = None   # GroundTruthManifest | None
_gt_last_push   = 0.0


def set_app_context(ctx, worker, replay_ctrl=None, ground_truth=None):
    global _ctx, _mapping_worker, _replay_ctrl, _ground_truth
    _ctx            = ctx
    _mapping_worker = worker
    _replay_ctrl    = replay_ctrl
    _ground_truth   = ground_truth


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)


manager = ConnectionManager()


def _build_session_info() -> dict:
    from mode import AppMode
    has_gt = _ground_truth is not None
    duration = None
    if _replay_ctrl is not None:
        duration = _replay_ctrl.duration_s
    return {
        "type": "session_info",
        "mode": _ctx.mode.name if _ctx else "LIVE",
        "replay_duration_s": duration,
        "has_ground_truth": has_gt,
    }


def _build_map_update(snap: dict) -> dict:
    payload = {
        "type": "map_update",
        "ts": snap["ts"],
        "pointcloud": snap["pointcloud"],
        "stats": snap["stats"],
        "boat": snap["boat"],
    }
    if _replay_ctrl is not None:
        dur = _replay_ctrl.duration_s
        pos = _replay_ctrl.get_position_s()
        payload["replay"] = {
            "position_s": round(pos, 2),
            "duration_s": round(dur, 2),
            "fraction":   round(pos / dur, 4) if dur > 0 else 0.0,
            "paused":     _replay_ctrl.state.paused,
        }
    return payload


def _build_ground_truth_update() -> dict:
    from ground_truth.metrics import compute_metrics
    schools = _ground_truth.fish_schools if _ground_truth else []
    metrics = compute_metrics([], schools)
    return {
        "type": "ground_truth_update",
        "fish_schools": schools,
        "detection_metrics": {
            "fish_true_count":       metrics.fish_true_count,
            "fish_detected_count":   metrics.fish_detected_count,
            "precision":             metrics.precision,
            "recall":                metrics.recall,
            "mean_distance_error_m": metrics.mean_distance_error_m,
        },
    }


async def _dispatch(msg: dict):
    if _replay_ctrl is None:
        return
    t = msg.get("type")
    if t == "replay_seek":
        await _replay_ctrl.seek(float(msg.get("fraction", 0)))
    elif t == "replay_pause":
        _replay_ctrl.pause()
    elif t == "replay_play":
        _replay_ctrl.play()
    elif t == "replay_speed":
        _replay_ctrl.set_speed(float(msg.get("value", 1.0)))


@app.websocket("/ws/map")
async def ws_map(websocket: WebSocket):
    global _gt_last_push
    await manager.connect(websocket)
    await websocket.send_text(json.dumps(_build_session_info()))
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                await _dispatch(json.loads(data))
            except asyncio.TimeoutError:
                pass

            snap = _mapping_worker.get_snapshot() if _mapping_worker else None
            if snap:
                await websocket.send_text(json.dumps(_build_map_update(snap)))

            # Ground truth push at 1 Hz in SYNTHETIC_REPLAY mode
            from mode import AppMode
            if (_ctx and _ctx.mode == AppMode.SYNTHETIC_REPLAY
                    and _ground_truth is not None
                    and time.time() - _gt_last_push >= 1.0):
                await websocket.send_text(json.dumps(_build_ground_truth_update()))
                _gt_last_push = time.time()

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/stats")
def stats():
    if _mapping_worker is None:
        return {"error": "mapping worker not initialised"}
    return _mapping_worker.get_snapshot() or {"status": "no data yet"}


@app.get("/mesh")
def mesh():
    if _mapping_worker is None:
        return {"error": "mapping worker not initialised"}
    return _mapping_worker.get_full_mesh() or {"error": "mesh unavailable"}


@app.get("/api/floor-grid")
def floor_grid_endpoint():
    if _ground_truth is None or _ground_truth.floor_grid is None:
        return {"error": "no floor grid available in this mode"}
    return _ground_truth.floor_grid


# Serve static frontend files
_frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_frontend_dir / "index.html"))
