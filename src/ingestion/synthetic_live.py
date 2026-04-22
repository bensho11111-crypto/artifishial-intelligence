"""
src/ingestion/synthetic_live.py

Physics-based synthetic sonar source for offline development and testing.

Drop-in replacement for SimulatedSource that models:
  - Fractal lake bathymetry (bowl + multi-octave ridges)
  - Fish schools with realistic sonar physics (arch formation, slant range)
  - Trolling boat trajectory with back-and-forth passes
  - 200 kHz conical beam physics
  - SL2-style 512-byte echo arrays with proper bottom + fish returns

The hidden TrueWorld state is NEVER placed directly on the pipeline queue.
The queue receives standard SensorFrame objects indistinguishable from
real hardware — fish only appear in the raw echo array, not in depth_m.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from utils import speed_control

# Re-use the data models already defined in nmea_reader so the pipeline
# receives the exact same types it expects from real hardware.
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from nmea_reader import (
    GPSFrame, SonarReturn, MotionFrame, SensorFrame,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lake grid
GRID_CELLS   = 401          # 401 × 401 cells
GRID_STEP_M  = 0.5          # metres per cell  →  200 m × 200 m lake
GRID_HALF    = (GRID_CELLS - 1) / 2 * GRID_STEP_M  # 100 m

# Transducer beam (200 kHz, 20° full angle)
BEAM_HALF_ANGLE_DEG = 10.0
TAN_BEAM_HALF       = math.tan(math.radians(BEAM_HALF_ANGLE_DEG))  # ≈ 0.1763

# GPS origin (Puddingstone Reservoir, San Dimas CA)
ORIGIN_LAT = 33.9003
ORIGIN_LON = -117.5012

ECHO_SIZE   = 512           # sonar echo array length (bytes)
ECHO_RANGE  = 60.0          # metres full-scale

RNG_SEED    = 7             # fixed seed for reproducible lake


# ---------------------------------------------------------------------------
# Fractal lake bottom
# ---------------------------------------------------------------------------

def _build_lake_bottom(rng: np.random.Generator) -> np.ndarray:
    """
    Returns a (GRID_CELLS × GRID_CELLS) float32 array of water depth in metres.

    Construction:
      1.  Bowl shape: deepest at centre (~12 m), shallow at edges (~2 m).
      2.  Octave 1: large ridges, wavelength ~100 m, amplitude ±3 m.
      3.  Octave 2: medium features, wavelength ~30 m, amplitude ±1 m.
      4.  Octave 3: fine texture, wavelength ~8 m, amplitude ±0.3 m.
    """
    xs = np.linspace(-GRID_HALF, GRID_HALF, GRID_CELLS)   # east axis
    ys = np.linspace(-GRID_HALF, GRID_HALF, GRID_CELLS)   # north axis
    east_grid, north_grid = np.meshgrid(xs, ys)            # shape (401, 401)

    # ── bowl shape ──────────────────────────────────────────────────────────
    r2 = (east_grid / GRID_HALF) ** 2 + (north_grid / GRID_HALF) ** 2
    # r2=0 at centre → depth 12 m;  r2=1 at edge → depth 2 m
    bowl = 2.0 + 10.0 * np.clip(1.0 - r2, 0.0, 1.0)

    # ── octave helper: sum of N random sin-wave planes ───────────────────────
    def _octave(n_waves: int, wavelength_m: float, amplitude: float) -> np.ndarray:
        acc = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
        for _ in range(n_waves):
            angle  = rng.uniform(0, 2 * math.pi)
            phase  = rng.uniform(0, 2 * math.pi)
            kx     = math.cos(angle) * (2 * math.pi / wavelength_m)
            ky     = math.sin(angle) * (2 * math.pi / wavelength_m)
            acc   += np.sin(kx * east_grid + ky * north_grid + phase).astype(np.float32)
        return acc * (amplitude / n_waves)

    oct1 = _octave(4,  100.0, 3.0)
    oct2 = _octave(6,   30.0, 1.0)
    oct3 = _octave(8,    8.0, 0.3)

    depth = (bowl + oct1 + oct2 + oct3).astype(np.float32)
    depth = np.clip(depth, 0.5, 20.0)   # physical limits
    return depth


# ---------------------------------------------------------------------------
# Fish school
# ---------------------------------------------------------------------------

SPECIES_PARAMS = {
    #              target-strength factor,  preferred_depth_m
    "bass":        (0.85, 3.5),
    "trout":       (0.70, 5.0),
    "carp":        (0.60, 6.5),
    "bream":       (0.50, 4.5),
}


@dataclass
class FishSchool:
    east_m:   float
    north_m:  float
    depth_m:  float          # depth in water column
    radius_m: float          # horizontal extent
    density:  float          # 0–1, affects echo amplitude
    vx:       float          # drift velocity east (m/s)
    vy:       float          # drift velocity north (m/s)
    species:  str

    # Internal: oscillation phase for depth variation
    _depth_phase: float = field(default=0.0, repr=False)
    _preferred_depth: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._preferred_depth = self.depth_m

    @classmethod
    def random(cls, rng: np.random.Generator) -> "FishSchool":
        species = rng.choice(list(SPECIES_PARAMS.keys()))
        _, pref_depth = SPECIES_PARAMS[species]
        depth   = float(rng.uniform(pref_depth - 1.0, pref_depth + 1.0))
        depth   = max(1.0, min(depth, 9.0))
        obj = cls(
            east_m   = float(rng.uniform(-80, 80)),
            north_m  = float(rng.uniform(-80, 80)),
            depth_m  = depth,
            radius_m = float(rng.uniform(3.0, 15.0)),
            density  = float(rng.uniform(0.3, 1.0)),
            vx       = float(rng.uniform(0.05, 0.20) * rng.choice([-1, 1])),
            vy       = float(rng.uniform(0.05, 0.20) * rng.choice([-1, 1])),
            species  = species,
            _depth_phase = float(rng.uniform(0, 2 * math.pi)),
        )
        obj._preferred_depth = depth
        return obj

    def step(self, dt: float, rng: np.random.Generator):
        """Update school position and depth (called every ping)."""
        self._depth_phase += dt * 0.08   # slow oscillation

        # Position drift
        self.east_m  += self.vx * dt
        self.north_m += self.vy * dt

        # Gentle boundary reflection
        if abs(self.east_m) > 95:
            self.vx *= -1
            self.east_m = float(np.clip(self.east_m, -95, 95))
        if abs(self.north_m) > 95:
            self.vy *= -1
            self.north_m = float(np.clip(self.north_m, -95, 95))

        # Random walk on velocity (very slow)
        self.vx += float(rng.normal(0, 0.005))
        self.vy += float(rng.normal(0, 0.005))
        # Clamp speed
        spd = math.sqrt(self.vx**2 + self.vy**2)
        if spd > 0.25:
            self.vx *= 0.25 / spd
            self.vy *= 0.25 / spd

        # Depth oscillation ±0.5 m
        self.depth_m = self._preferred_depth + 0.5 * math.sin(self._depth_phase)
        self.depth_m = max(0.5, self.depth_m)


# ---------------------------------------------------------------------------
# Boat (trolling route)
# ---------------------------------------------------------------------------

@dataclass
class Boat:
    east_m:      float = -80.0
    north_m:     float = -80.0
    heading_deg: float = 45.0    # NE to start
    speed_ms:    float = 2.5
    _t:          float = field(default=0.0, repr=False)
    _heading_noise: float = field(default=0.0, repr=False)

    # Turning state
    _turning: bool = field(default=False, repr=False)
    _turn_remaining: float = field(default=0.0, repr=False)

    def step(self, dt: float, rng: np.random.Generator):
        self._t += dt

        # Speed: typical trolling variation
        self.speed_ms = 2.5 + 0.4 * math.sin(self._t * 0.1)

        # Heading noise
        self._heading_noise += float(rng.normal(0, 1.0)) * dt
        self._heading_noise *= 0.98    # damping

        if self._turning:
            step_turn = min(abs(self._turn_remaining), 180.0 * dt)
            sign = 1 if self._turn_remaining > 0 else -1
            self.heading_deg = (self.heading_deg + sign * step_turn) % 360
            self._turn_remaining -= sign * step_turn
            if abs(self._turn_remaining) < 0.5:
                self._turning = False
        else:
            h = self.heading_deg + self._heading_noise
            rad = math.radians(h)
            self.east_m  += math.sin(rad) * self.speed_ms * dt
            self.north_m += math.cos(rad) * self.speed_ms * dt

            # Check lake boundary → turn 180°
            if abs(self.east_m) > 95 or abs(self.north_m) > 95:
                # Clamp position
                self.east_m  = float(np.clip(self.east_m,  -95, 95))
                self.north_m = float(np.clip(self.north_m, -95, 95))
                # Schedule 180° turn
                self._turning = True
                self._turn_remaining = 180.0

    @property
    def heading_rad(self) -> float:
        return math.radians(self.heading_deg)


# ---------------------------------------------------------------------------
# True world (hidden ground truth)
# ---------------------------------------------------------------------------

@dataclass
class TrueWorld:
    lake_bottom:    np.ndarray           # (401, 401) depth in metres
    fish_schools:   List[FishSchool]
    boat:           Boat
    temperature_C:  float = 18.3
    thermocline_m:  float = 4.5
    sound_velocity: float = 1480.0

    def _sample_depth(self, east_m: float, north_m: float) -> float:
        """Bilinear interpolation into lake_bottom grid."""
        # Convert ENU metres to grid indices
        col = (east_m  + GRID_HALF) / GRID_STEP_M   # east axis → column
        row = (north_m + GRID_HALF) / GRID_STEP_M   # north axis → row
        col = float(np.clip(col, 0, GRID_CELLS - 1.001))
        row = float(np.clip(row, 0, GRID_CELLS - 1.001))

        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, GRID_CELLS - 1), min(r0 + 1, GRID_CELLS - 1)
        fc, fr = col - c0, row - r0

        d = (self.lake_bottom[r0, c0] * (1 - fc) * (1 - fr)
             + self.lake_bottom[r0, c1] * fc       * (1 - fr)
             + self.lake_bottom[r1, c0] * (1 - fc) * fr
             + self.lake_bottom[r1, c1] * fc       * fr)
        return float(d)

    def ping(self, rng: np.random.Generator) -> dict:
        """
        Simulate one sonar ping.  Returns a dict with everything the
        SyntheticLiveSource needs to build a SensorFrame.
        """
        bx, by = self.boat.east_m, self.boat.north_m

        # ── 1. Bottom return ────────────────────────────────────────────────
        true_depth = self._sample_depth(bx, by)
        meas_depth = true_depth + float(rng.normal(0, 0.04))
        meas_depth = max(0.3, meas_depth)

        speed_penalty = min(1.0, self.boat.speed_ms / 4.0)
        amplitude     = 0.85 * (1.0 - speed_penalty)   # base_hardness = 0.85
        signal_db     = 60.0 + 20.0 * amplitude + float(rng.normal(0, 3.0))
        hardness      = min(1.0, signal_db / 100.0)

        # ── 2. Fish echoes ───────────────────────────────────────────────────
        fish_returns = []   # list of (slant_range_m, echo_amplitude)
        for school in self.fish_schools:
            horiz_offset = math.sqrt(
                (school.east_m - bx) ** 2 + (school.north_m - by) ** 2
            )
            beam_radius = school.depth_m * TAN_BEAM_HALF
            if horiz_offset > beam_radius + school.radius_m:
                continue   # fish not in beam

            overlap = max(0.0,
                (beam_radius + school.radius_m - horiz_offset) / school.radius_m)
            overlap = min(1.0, overlap)

            slant_range = math.sqrt(horiz_offset ** 2 + school.depth_m ** 2)
            ts_factor, _ = SPECIES_PARAMS[school.species]
            echo_amp = school.density * overlap * 0.4 * ts_factor
            fish_returns.append((slant_range, echo_amp))

        # ── 3. Build 512-byte echo array ────────────────────────────────────
        echo = _make_echo(meas_depth, fish_returns, rng)

        return {
            "depth_m":    meas_depth,
            "true_depth": true_depth,
            "signal_db":  signal_db,
            "hardness":   hardness,
            "fish_returns": fish_returns,
            "echo":       echo,
        }


def _make_echo(
    bottom_depth_m: float,
    fish_returns: list,   # list of (slant_range_m, amplitude)
    rng: np.random.Generator,
) -> bytes:
    """
    Build a 512-byte SL2-style sonar echo array.

    Index layout: index i corresponds to range = i / ECHO_SIZE * ECHO_RANGE metres.
    """
    echo = np.array([int(rng.integers(2, 13)) for _ in range(ECHO_SIZE)],
                    dtype=np.int16)

    def _gaussian(centre_idx: int, amp: int, sigma: float):
        lo = max(0, int(centre_idx - 4 * sigma))
        hi = min(ECHO_SIZE, int(centre_idx + 4 * sigma) + 1)
        idxs = np.arange(lo, hi)
        vals = (amp * np.exp(-0.5 * ((idxs - centre_idx) / sigma) ** 2)).astype(np.int16)
        echo[lo:hi] += vals

    # Bottom return
    bot_idx = int(bottom_depth_m / ECHO_RANGE * ECHO_SIZE)
    bot_idx = max(0, min(ECHO_SIZE - 1, bot_idx))
    sigma_b = max(3.0, ECHO_SIZE * 0.008)
    _gaussian(bot_idx, 200, sigma_b)

    # Second-bounce echo
    sec_idx = bot_idx * 2
    if 0 < sec_idx < ECHO_SIZE:
        _gaussian(sec_idx, 70, sigma_b + 2)

    # Fish echoes
    for slant_range, fish_amp in fish_returns:
        fish_idx = int(slant_range / ECHO_RANGE * ECHO_SIZE)
        fish_idx = max(0, min(ECHO_SIZE - 1, fish_idx))
        amp_counts = int(fish_amp * 180)
        if amp_counts > 0:
            _gaussian(fish_idx, amp_counts, 3.0)

    echo = np.clip(echo, 0, 255).astype(np.uint8)
    return bytes(echo)


# ---------------------------------------------------------------------------
# GPS helpers — ENU metres → decimal degrees
# ---------------------------------------------------------------------------

def _enu_to_latlon(east_m: float, north_m: float,
                   origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Flat-earth inverse of the Orchestrator's latlon_to_enu."""
    R = 6_371_000.0
    lat = origin_lat + math.degrees(north_m / R)
    lon = origin_lon + math.degrees(east_m / (R * math.cos(math.radians(origin_lat))))
    return lat, lon


# ---------------------------------------------------------------------------
# SyntheticLiveSource  (public API)
# ---------------------------------------------------------------------------

class SyntheticLiveSource:
    """
    Physics-based drop-in replacement for SimulatedSource.

    Puts SensorFrame objects onto `queue` at `hz` Hz.
    The hidden TrueWorld is never exposed to the pipeline;
    use get_true_state_snapshot() for debug visualisation.
    """

    def __init__(self, queue: asyncio.Queue, hz: int = 10):
        self.queue = queue
        self.hz    = hz
        self._dt   = 1.0 / hz
        self._seed = RNG_SEED
        self._elapsed = 0.0

        rng = np.random.default_rng(self._seed)

        # ── Build hidden ground truth ────────────────────────────────────────
        lake = _build_lake_bottom(rng)

        n_schools = int(rng.integers(3, 7))   # 3–6
        schools = [FishSchool.random(rng) for _ in range(n_schools)]

        self._world = TrueWorld(
            lake_bottom  = lake,
            fish_schools = schools,
            boat         = Boat(),
        )

        # Separate live RNG so snapshot doesn't perturb it
        self._rng = np.random.default_rng(self._seed + 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        print(f"[SyntheticLive] Physics-based source running at {self.hz} Hz  "
              f"| {len(self._world.fish_schools)} fish schools")
        depth_range = (float(self._world.lake_bottom.min()),
                       float(self._world.lake_bottom.max()))
        print(f"[SyntheticLive] Lake depth range: "
              f"{depth_range[0]:.2f} m – {depth_range[1]:.2f} m")

        while True:
            t_start = asyncio.get_event_loop().time()

            # Advance physics
            self._elapsed += self._dt
            self._world.boat.step(self._dt, self._rng)
            for school in self._world.fish_schools:
                school.step(self._dt, self._rng)

            # Simulate ping
            ping = self._world.ping(self._rng)

            # Build SensorFrame
            frame = self._build_frame(ping)
            await self.queue.put(frame)

            # Pace to hz
            elapsed = asyncio.get_event_loop().time() - t_start
            delay   = max(0.0, self._dt / speed_control.get() - elapsed)
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _build_frame(self, ping: dict) -> SensorFrame:
        ts   = time.time()
        boat = self._world.boat

        # GPS with realistic position noise (≈1.5 m sigma)
        east_noisy  = boat.east_m  + float(self._rng.normal(0, 1.5))
        north_noisy = boat.north_m + float(self._rng.normal(0, 1.5))
        lat, lon = _enu_to_latlon(east_noisy, north_noisy,
                                   ORIGIN_LAT, ORIGIN_LON)
        speed_kts = boat.speed_ms / 0.5144

        gps = GPSFrame(
            ts         = ts,
            lat        = lat,
            lon        = lon,
            speed_kts  = speed_kts,
            course_deg = boat.heading_deg,
            hdop       = 1.2 + 0.3 * abs(math.sin(ts * 0.05)),
        )

        sonar = SonarReturn(
            ts                = ts,
            depth_m           = ping["depth_m"],
            frequency_hz      = 200_000,
            signal_strength_db= ping["signal_db"],
            bottom_hardness   = ping["hardness"],
        )

        motion = MotionFrame(
            ts          = ts,
            speed_ms    = boat.speed_ms,
            heading_deg = boat.heading_deg,
        )

        return SensorFrame(
            ts     = ts,
            gps    = gps,
            sonar  = sonar,
            motion = motion,
            raw    = "",   # raw echo bytes not carried on SensorFrame (pipeline doesn't use it)
        )

    # ------------------------------------------------------------------
    # Debug / visualisation  (pipeline never calls this)
    # ------------------------------------------------------------------

    def get_true_state_snapshot(self) -> dict:
        """
        Returns the hidden ground truth for debugging or visualisation.
        Never called by the pipeline.
        fish_schools list includes at minimum: east, north, depth_m, radius_m, species.
        """
        boat = self._world.boat
        schools_info = [
            {
                "east":     s.east_m,
                "north":    s.north_m,
                "depth_m":  s.depth_m,
                "radius_m": s.radius_m,
                "depth":    s.depth_m,   # legacy alias
                "radius":   s.radius_m,  # legacy alias
                "density":  s.density,
                "species":  s.species,
                "vx":       s.vx,
                "vy":       s.vy,
            }
            for s in self._world.fish_schools
        ]

        # Sample a grid of bottom depths for quick visualisation
        sample_rows = np.linspace(0, GRID_CELLS - 1, 20, dtype=int)
        sample_cols = np.linspace(0, GRID_CELLS - 1, 20, dtype=int)
        sample_depths = self._world.lake_bottom[
            np.ix_(sample_rows, sample_cols)
        ].tolist()

        return {
            "fish_schools": schools_info,
            "boat": {
                "east":    boat.east_m,
                "north":   boat.north_m,
                "heading": boat.heading_deg,
                "speed":   boat.speed_ms,
            },
            "lake_bottom": {
                "min_depth": float(self._world.lake_bottom.min()),
                "max_depth": float(self._world.lake_bottom.max()),
                "shape":     list(self._world.lake_bottom.shape),
                "sample_20x20": sample_depths,
            },
            "uptime_s": round(self._elapsed, 1),
        }

    def reset(self):
        """Re-create the world from the same seed and reset elapsed time."""
        self._elapsed = 0.0
        rng = np.random.default_rng(self._seed)
        lake = _build_lake_bottom(rng)
        n_schools = int(rng.integers(3, 7))
        schools = [FishSchool.random(rng) for _ in range(n_schools)]
        self._world = TrueWorld(
            lake_bottom  = lake,
            fish_schools = schools,
            boat         = Boat(),
        )
        self._rng = np.random.default_rng(self._seed + 1)

    # ------------------------------------------------------------------
    # Utility: compute slant-range profile for one school (debug helper)
    # ------------------------------------------------------------------

    def compute_arch_slant_ranges(self, school_idx: int,
                                  n_steps: int = 50) -> list[float]:
        """
        Simulate the boat passing directly over fish_schools[school_idx]
        along the east axis and return the slant ranges at each step.
        Useful for verifying the arch shape.
        """
        school = self._world.fish_schools[school_idx]
        east_positions = np.linspace(
            school.east_m - school.radius_m * 4,
            school.east_m + school.radius_m * 4,
            n_steps,
        )
        ranges = []
        for bx in east_positions:
            horiz = abs(bx - school.east_m)
            slant = math.sqrt(horiz ** 2 + school.depth_m ** 2)
            ranges.append(round(slant, 3))
        return ranges
