"""
src/main.py

Entry point — wires all five layers together and starts the event loop.

Usage:
  python src/main.py                    # simulated data (no hardware needed)
  python src/main.py --port /dev/ttyUSB0   # Lowrance NMEA serial
  python src/main.py --sl2 /path/to/file.sl2  # replay an SL2 recording
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ingestion.nmea_reader import SimulatedSource, NMEA0183Reader, StructureScanReader
from agents.orchestrator import Orchestrator
from mapping.voxel_map import MappingWorker, MapConfig
from ar.server import app, set_mapping_worker
import uvicorn


def parse_args():
    p = argparse.ArgumentParser(description="Lowrance AR Depth Map")
    p.add_argument("--port",    default=None,       help="Serial port for NMEA 0183 (e.g. /dev/ttyUSB0)")
    p.add_argument("--sl2",     default=None,       help="Path to .sl2 file for replay")
    p.add_argument("--host",    default="0.0.0.0",  help="WebSocket server host")
    p.add_argument("--http-port", default=8000, type=int, help="HTTP/WS port")
    p.add_argument("--voxel-size", default=0.5, type=float, help="Voxel resolution in metres")
    p.add_argument("--sim-hz",    default=10, type=int,   help="Simulated source Hz")
    return p.parse_args()


async def main():
    args = parse_args()

    # ── Layer 2: Agent orchestrator (creates the raw_q and obs_q) ──────────
    orch = Orchestrator()

    # ── Layer 1: Data source ──────────────────────────────────────────────
    if args.sl2:
        print(f"[main] Replaying SL2 file: {args.sl2}")
        source = StructureScanReader(args.sl2, orch.raw_q, is_file=True)
    elif args.port:
        print(f"[main] Reading NMEA from serial port: {args.port}")
        source = NMEA0183Reader(args.port, orch.raw_q)
    else:
        print("[main] No hardware specified — using simulated source")
        source = SimulatedSource(orch.raw_q, hz=args.sim_hz)

    # ── Layer 3: Mapping engine ───────────────────────────────────────────
    cfg = MapConfig(voxel_size_m=args.voxel_size)
    worker = MappingWorker(orch.obs_q, cfg)

    # ── Layer 4: AR server ────────────────────────────────────────────────
    set_mapping_worker(worker)
    config = uvicorn.Config(
        app, host=args.host, port=args.http_port,
        log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(config)

    print(f"\n{'='*50}")
    print(f"  Lowrance AR Map — starting up")
    print(f"  AR viewer: http://localhost:{args.http_port}")
    print(f"  WebSocket: ws://localhost:{args.http_port}/ws/map")
    print(f"  Map stats: http://localhost:{args.http_port}/stats")
    print(f"  Voxel res: {args.voxel_size} m")
    print(f"{'='*50}\n")

    # ── Run all layers concurrently ───────────────────────────────────────
    await asyncio.gather(
        source.run(),
        orch.run(),
        worker.run(),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
