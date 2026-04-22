# Lowrance Elite 7 → Probabilistic 3D AR Map

Agentic pipeline that ingests sonar, GPS, and motion data from a Lowrance Elite 7 transducer, builds a probabilistic 3D terrain map, and renders it in augmented reality.

## Architecture

```
Layer 1 — Data Ingestion
  NMEA 2000 / serial  →  StructureScan sonar  →  GPS + heading  →  IMU

Layer 2 — Agent Orchestrator
  Sensor Fusion Agent  →  Motion Compensation Agent  →  Validation Agent

Layer 3 — Probabilistic Mapping Engine
  Occupancy Grid  →  Bayesian Update  →  Mesh Reconstruction  →  Confidence Map

Layer 4 — AR Display
  WebXR / ARKit  →  3D Overlay Renderer  →  Confidence Viz  →  Target Device

Layer 5 — Storage & State
  Voxel Store  →  Track Log  →  Session Archive  →  Map Merge
```

## Stack

| Concern | Technology |
|---|---|
| Data ingestion | Python, `pyserial`, `python-can` |
| NMEA parsing | `pynmea2`, custom NMEA 2000 decoder |
| Agent orchestration | Python async, `asyncio` queues |
| Sensor fusion | Kalman filter (`filterpy`) |
| Probabilistic mapping | NumPy voxel grid + Gaussian Process (`GPy`) |
| Mesh reconstruction | `scikit-image` marching cubes |
| AR rendering | Three.js + WebXR (browser) or ARKit (iOS native) |
| Storage | SQLite (voxels), GeoJSON (track logs) |
| API | FastAPI WebSocket server |

## Quick start

```bash
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# Edit config.yaml: set serial port, GPS offset, transducer draft
python src/main.py
# Open http://localhost:8000 on an AR-capable device
```

## Directory layout

```
src/
  ingestion/      # Serial readers, NMEA parsers, sensor drivers
  agents/         # Orchestrator, fusion, motion comp, validation
  mapping/        # Voxel grid, Bayesian updater, mesh builder
  ar/             # WebXR server, Three.js renderer, WebSocket bridge
  storage/        # Voxel DB, track log writer, session manager
  utils/          # Coordinate transforms, logging, config loader
config/
tests/
docs/
```
