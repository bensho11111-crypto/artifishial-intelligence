"""
src/mapping/voxel_map.py

Layer 3 — Probabilistic 3D Mapping Engine
Maintains a live 3D occupancy grid updated with each FusedObservation.
Exports meshes and uncertainty fields for the AR renderer.
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Optional dependency — used only for mesh export
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MapConfig:
    voxel_size_m: float = 0.5          # spatial resolution (50 cm)
    x_range_m: float   = 500.0        # east extent from origin
    y_range_m: float   = 500.0        # north extent
    z_range_m: float   = 60.0         # max depth
    prior_prob: float  = 0.5          # initial occupancy probability
    l_occupied: float  = 0.7          # log-odds update for hit
    l_free: float      = 0.3          # log-odds update for miss
    min_log_odds: float = -5.0
    max_log_odds: float =  5.0
    gp_length_scale: float = 2.0      # GP interpolation length scale (m)
    gp_noise_var: float    = 0.1


# ---------------------------------------------------------------------------
# Occupancy voxel grid
# ---------------------------------------------------------------------------

class VoxelMap:
    """
    3D log-odds occupancy grid in ENU local coordinates.

    Each voxel stores:
      log_odds   — occupancy probability (log-odds form)
      n_hits     — number of sonar returns
      sum_depth  — running depth sum for mean estimation
      sum_sq     — running depth^2 for variance estimation
      confidence — exponential moving average of observation confidence
    """

    def __init__(self, cfg: MapConfig = MapConfig()):
        self.cfg = cfg
        s = cfg.voxel_size_m

        self.nx = int(cfg.x_range_m / s)
        self.ny = int(cfg.y_range_m / s)
        self.nz = int(cfg.z_range_m / s)

        # Allocate grids (float32 saves memory)
        lo0 = np.log(cfg.prior_prob / (1 - cfg.prior_prob))
        self.log_odds  = np.full((self.nx, self.ny, self.nz), lo0,  dtype=np.float32)
        self.n_hits    = np.zeros((self.nx, self.ny),               dtype=np.int32)
        self.sum_depth = np.zeros((self.nx, self.ny),               dtype=np.float64)
        self.sum_sq    = np.zeros((self.nx, self.ny),               dtype=np.float64)
        self.confidence= np.zeros((self.nx, self.ny),               dtype=np.float32)
        self.last_hit  = np.zeros((self.nx, self.ny),               dtype=np.float64)

        self._origin_offset = np.array([cfg.x_range_m / 2, cfg.y_range_m / 2])
        self._update_count = 0
        self._created_at = time.time()

        print(f"[VoxelMap] Allocated {self.nx}×{self.ny}×{self.nz} voxels "
              f"({self.nx * self.ny * self.nz * 4 / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obs) -> bool:
        """
        Ingest one FusedObservation.
        Returns True if the observation fell within the map bounds.
        """
        ix, iy = self._world_to_grid(obs.easting_m, obs.northing_m)
        if not self._in_bounds_xy(ix, iy):
            return False

        iz = self._depth_to_iz(obs.depth_m)
        if iz < 0 or iz >= self.nz:
            return False

        cfg = self.cfg

        # Log-odds update: free cells above the return, occupied at return
        for z in range(min(iz, self.nz - 1)):
            self.log_odds[ix, iy, z] = np.clip(
                self.log_odds[ix, iy, z] + np.log((1 - cfg.l_free) / cfg.l_free),
                cfg.min_log_odds, cfg.max_log_odds
            )
        self.log_odds[ix, iy, iz] = np.clip(
            self.log_odds[ix, iy, iz] + np.log(cfg.l_occupied / (1 - cfg.l_occupied)),
            cfg.min_log_odds, cfg.max_log_odds
        )

        # Running statistics for depth mean / variance
        self.n_hits[ix, iy]    += 1
        self.sum_depth[ix, iy] += obs.depth_m
        self.sum_sq[ix, iy]    += obs.depth_m ** 2
        self.last_hit[ix, iy]   = obs.ts

        # EMA confidence
        alpha = 0.1
        self.confidence[ix, iy] = (1 - alpha) * self.confidence[ix, iy] + alpha * obs.confidence

        self._update_count += 1
        return True

    def depth_mean(self) -> np.ndarray:
        """Return (nx, ny) array of estimated mean depths."""
        n = np.maximum(self.n_hits, 1)
        return (self.sum_depth / n).astype(np.float32)

    def depth_variance(self) -> np.ndarray:
        """Return (nx, ny) array of depth variance (uncertainty)."""
        n = np.maximum(self.n_hits, 1)
        mean = self.sum_depth / n
        var  = self.sum_sq / n - mean ** 2
        return np.maximum(var, 0.0).astype(np.float32)

    def depth_std(self) -> np.ndarray:
        return np.sqrt(self.depth_variance())

    def occupancy_prob(self) -> np.ndarray:
        """Convert log-odds to probability [0, 1]. Shape: (nx, ny, nz)."""
        return (1.0 / (1.0 + np.exp(-self.log_odds))).astype(np.float32)

    def surface_slice(self, threshold: float = 0.6) -> np.ndarray:
        """
        2D depth map: for each (x, y) column, find the deepest voxel
        with occupancy probability > threshold.
        Returns (nx, ny) array in metres (NaN where unobserved).
        """
        prob = self.occupancy_prob()
        mask = prob > threshold
        out  = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        for iz in range(self.nz):
            hit_xy = mask[:, :, iz]
            depth_val = (iz + 0.5) * self.cfg.voxel_size_m
            out[hit_xy] = depth_val
        return out

    def export_mesh(self, threshold: float = 0.5) -> Optional[dict]:
        """
        Run marching cubes on the occupancy grid to produce a triangle mesh.
        Returns dict with 'verts', 'faces', 'normals' arrays, or None.
        """
        if not HAS_SKIMAGE:
            print("[VoxelMap] scikit-image not installed — mesh export unavailable")
            return None

        prob = self.occupancy_prob()
        verts, faces, normals, _ = marching_cubes(prob, level=threshold)

        # Scale from voxel units to metres and shift to ENU
        s = self.cfg.voxel_size_m
        verts_world = verts * s
        verts_world[:, 0] -= self._origin_offset[0]
        verts_world[:, 1] -= self._origin_offset[1]

        return {
            "verts":   verts_world.tolist(),
            "faces":   faces.tolist(),
            "normals": normals.tolist(),
        }

    def export_pointcloud(self, min_hits: int = 2) -> dict:
        """
        Export observed cells as a point cloud for lightweight AR streaming.
        Returns dict with arrays: x, y, depth, confidence, std.
        """
        mask = self.n_hits >= min_hits
        ixs, iys = np.where(mask)
        if len(ixs) == 0:
            return {"x": [], "y": [], "depth": [], "confidence": [], "std": []}

        s  = self.cfg.voxel_size_m
        ox, oy = self._origin_offset

        x = (ixs * s - ox).tolist()
        y = (iys * s - oy).tolist()
        d = (self.sum_depth[ixs, iys] / self.n_hits[ixs, iys]).tolist()
        c = self.confidence[ixs, iys].tolist()
        v = np.sqrt(np.maximum(
            self.sum_sq[ixs, iys] / self.n_hits[ixs, iys]
            - (self.sum_depth[ixs, iys] / self.n_hits[ixs, iys]) ** 2,
            0.0
        )).tolist()

        return {"x": x, "y": y, "depth": d, "confidence": c, "std": v}

    def stats(self) -> dict:
        observed = int(np.sum(self.n_hits > 0))
        return {
            "updates":    self._update_count,
            "observed_cells": observed,
            "coverage_pct": round(100 * observed / (self.nx * self.ny), 2),
            "mean_depth": round(float(np.nanmean(self.depth_mean()[self.n_hits > 0])), 2)
                          if observed > 0 else 0.0,
            "uptime_s":   round(time.time() - self._created_at, 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, east_m: float, north_m: float) -> Tuple[int, int]:
        ox, oy = self._origin_offset
        ix = int((east_m  + ox) / self.cfg.voxel_size_m)
        iy = int((north_m + oy) / self.cfg.voxel_size_m)
        return ix, iy

    def _depth_to_iz(self, depth_m: float) -> int:
        return int(depth_m / self.cfg.voxel_size_m)

    def _in_bounds_xy(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny


# ---------------------------------------------------------------------------
# Mapping worker — consumes FusedObservations and serves the AR layer
# ---------------------------------------------------------------------------

class MappingWorker:
    """
    Wraps VoxelMap and exposes an async interface.
    Runs as a background task; callers snapshot the map via get_snapshot().
    """

    SNAPSHOT_INTERVAL = 1.0   # seconds between pointcloud snapshots

    def __init__(self, obs_q, cfg: MapConfig = MapConfig()):
        self.obs_q = obs_q
        self.vmap  = VoxelMap(cfg)
        self._snapshot: Optional[dict] = None
        self._snapshot_ts = 0.0

    async def run(self):
        import asyncio
        print("[MappingWorker] Running")
        while True:
            obs = await self.obs_q.get()
            self.vmap.update(obs)
            now = time.time()
            if now - self._snapshot_ts > self.SNAPSHOT_INTERVAL:
                self._snapshot = {
                    "ts": now,
                    "stats": self.vmap.stats(),
                    "pointcloud": self.vmap.export_pointcloud(),
                }
                self._snapshot_ts = now

    def get_snapshot(self) -> Optional[dict]:
        return self._snapshot

    def get_full_mesh(self) -> Optional[dict]:
        return self.vmap.export_mesh()
