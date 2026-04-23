// synthetic_overlay.js — fish school ground truth 3D overlay
// Loaded dynamically only in SYNTHETIC_REPLAY mode.
import * as THREE from 'three';

const SPECIES_COLORS = {
  'largemouth bass': 0xff6b35,
  'rainbow trout':   0x7ecef4,
  'common carp':     0xc8a96e,
  'bluegill bream':  0xa8e063,
};

let _scene = null;
const _markers = [];   // {group, school} pairs

export function init(scene) {
  _scene = scene;
  console.log('[SyntheticOverlay] Initialized');
}

export function updateSchools(fishSchools) {
  // Remove old markers
  for (const { group } of _markers) {
    _scene.remove(group);
    group.traverse(o => { if (o.geometry) o.geometry.dispose(); });
  }
  _markers.length = 0;

  if (!_scene || !fishSchools) return;

  for (const school of fishSchools) {
    const color = SPECIES_COLORS[school.species] ?? 0xff9f1c;
    const group = new THREE.Group();

    // Flat cylinder (disc)
    const cylGeo = new THREE.CylinderGeometry(school.radius_m, school.radius_m, school.radius_m * 0.4, 32);
    const cylMat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false,
    });
    const cyl = new THREE.Mesh(cylGeo, cylMat);
    // Three.js CylinderGeometry is Y-up; scene is Z-up with depth as -Z
    // Rotate 90° around X so the flat face is horizontal (in XY plane)
    cyl.rotation.x = Math.PI / 2;
    group.add(cyl);

    // Wireframe ring
    const ringGeo = new THREE.RingGeometry(school.radius_m - 0.3, school.radius_m, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color, side: THREE.DoubleSide, transparent: true, opacity: 0.75,
    });
    group.add(new THREE.Mesh(ringGeo, ringMat));

    // Vertical pole from surface (z=0) down to school depth
    const poleGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -school.depth_m),
    ]);
    const poleMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 });
    group.add(new THREE.Line(poleGeo, poleMat));

    group.position.set(school.east_m, school.north_m, -school.depth_m);
    _scene.add(group);
    _markers.push({ group, school });
  }
}

export function setVisible(visible) {
  for (const { group } of _markers) {
    group.visible = visible;
  }
}
