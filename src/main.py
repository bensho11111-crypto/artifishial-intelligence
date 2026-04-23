"""
src/main.py

Entry point — resolves AppMode from CLI args, wires all layers, starts the event loop.

Usage:
  python src/main.py                                           # simulated synthetic (live physics)
  python src/main.py --synthetic                              # same as above
  python src/main.py --port COM3                              # LIVE: real Lowrance hardware
  python src/main.py --sl2 data/synthetic/sample.sl2         # TRUE_REPLAY
  python src/main.py --sl2 data/synthetic/sample.sl2 \
    --ground-truth data/synthetic/ground_truth.json           # SYNTHETIC_REPLAY (file-based)
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mode import AppMode, AppContext
from agents.orchestrator import Orchestrator
from mapping.voxel_map import MappingWorker, MapConfig
from ar.server import app, set_app_context
import uvicorn


def parse_args():
    p = argparse.ArgumentParser(description="Lowrance AR Depth Map")
    p.add_argument("--port",         default=None,  help="Serial port for NMEA 0183 (LIVE mode)")
    p.add_argument("--sl2",          default=None,  help="Path to .sl2 file for replay")
    p.add_argument("--ground-truth", default=None,  dest="ground_truth",
                   help="Path to ground_truth.json (enables SYNTHETIC_REPLAY with fish overlay)")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use physics-based synthetic live source (SYNTHETIC_REPLAY, in-memory GT)")
    p.add_argument("--host",         default="0.0.0.0")
    p.add_argument("--http-port",    default=8000, type=int)
    p.add_argument("--voxel-size",   default=0.5, type=float)
    p.add_argument("--sim-hz",       default=10, type=int)
    return p.parse_args()


def build_context(args) -> AppContext:
    if args.port:
        return AppContext(mode=AppMode.LIVE, serial_port=args.port,
                         host=args.host, http_port=args.http_port,
                         voxel_size_m=args.voxel_size, sim_hz=args.sim_hz)
    if args.sl2 and args.ground_truth:
        return AppContext(mode=AppMode.SYNTHETIC_REPLAY, sl2_path=args.sl2,
                         ground_truth_path=args.ground_truth,
                         host=args.host, http_port=args.http_port,
                         voxel_size_m=args.voxel_size, sim_hz=args.sim_hz)
    if args.sl2:
        return AppContext(mode=AppMode.TRUE_REPLAY, sl2_path=args.sl2,
                         host=args.host, http_port=args.http_port,
                         voxel_size_m=args.voxel_size, sim_hz=args.sim_hz)
    # Default / --synthetic: live physics simulation with in-memory ground truth
    return AppContext(mode=AppMode.SYNTHETIC_REPLAY,
                     host=args.host, http_port=args.http_port,
                     voxel_size_m=args.voxel_size, sim_hz=args.sim_hz)


async def main():
    args = parse_args()
    ctx = build_context(args)

    print(f"\n{'='*54}")
    print(f"  Lowrance AR Map — mode: {ctx.mode.name}")
    print(f"  AR viewer: http://localhost:{ctx.http_port}")
    print(f"  WebSocket: ws://localhost:{ctx.http_port}/ws/map")
    print(f"  Voxel res: {ctx.voxel_size_m} m")
    print(f"{'='*54}\n")

    orch   = Orchestrator()
    cfg    = MapConfig(voxel_size_m=ctx.voxel_size_m)
    worker = MappingWorker(orch.obs_q, cfg)

    replay_ctrl   = None
    ground_truth  = None
    source_task   = None

    if ctx.mode == AppMode.LIVE:
        from ingestion.nmea_reader import NMEA0183Reader
        source = NMEA0183Reader(ctx.serial_port, orch.raw_q)
        source_task = source.run()

    elif ctx.mode == AppMode.TRUE_REPLAY:
        from ingestion.sl2_replay import make_sl2_replay_controller
        replay_ctrl = make_sl2_replay_controller(
            ctx.sl2_path, orch.raw_q,
            on_seek=orch.reset,
        )
        source_task = replay_ctrl.run()

    elif ctx.mode == AppMode.SYNTHETIC_REPLAY:
        if ctx.sl2_path and ctx.ground_truth_path:
            # File-based synthetic replay
            from ingestion.sl2_replay import make_sl2_replay_controller
            from ground_truth.manifest import GroundTruthManifest
            replay_ctrl  = make_sl2_replay_controller(
                ctx.sl2_path, orch.raw_q,
                on_seek=orch.reset,
            )
            ground_truth = GroundTruthManifest.from_json_file(ctx.ground_truth_path)
            source_task  = replay_ctrl.run()
        else:
            # Live physics simulation with in-memory ground truth
            from ingestion.synthetic_live import SyntheticLiveSource
            from ground_truth.manifest import GroundTruthManifest
            source       = SyntheticLiveSource(orch.raw_q, hz=ctx.sim_hz)
            ground_truth = GroundTruthManifest.from_synthetic_live_source(source)
            source_task  = source.run()

    set_app_context(ctx, worker, replay_ctrl, ground_truth)

    uvicorn_config = uvicorn.Config(
        app, host=ctx.host, port=ctx.http_port,
        log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(uvicorn_config)

    await asyncio.gather(
        source_task,
        orch.run(),
        worker.run(),
        server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
