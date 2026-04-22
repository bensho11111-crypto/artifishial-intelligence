"""
src/agents/orchestrator.py

Layer 2 — Agent Orchestrator
Three cooperating async agents in a pipeline:
  1. SensorFusionAgent    — time-aligns streams, applies Kalman filter
  2. MotionCompAgent      — corrects sonar position for boat movement
  3. ValidationAgent      — rejects outliers, emits clean FusedObservation
"""
import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Deque
from collections import deque

import numpy as np

# Relative import from ingestion layer
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ingestion.nmea_reader import SensorFrame, GPSFrame, SonarReturn, MotionFrame


# ---------------------------------------------------------------------------
# Output of the full agent pipeline
# ---------------------------------------------------------------------------

@dataclass
class FusedObservation:
    """
    A single georeferenced depth observation, ready for the mapping engine.
    Position is in local ENU (East-North-Up) metres relative to session origin.
    """
    ts: float
    easting_m: float          # E relative to session origin
    northing_m: float         # N relative to session origin
    depth_m: float            # water depth (positive downward)
    confidence: float         # 0.0–1.0  (higher = more certain)
    speed_ms: float           # boat speed at time of observation
    heading_deg: float
    signal_db: float
    bottom_hardness: float


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def latlon_to_enu(lat: float, lon: float, origin_lat: float, origin_lon: float):
    """Simple flat-earth ENU conversion (accurate to ~500 m radius)."""
    R = 6_371_000.0
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    north = R * dlat
    east  = R * dlon * math.cos(math.radians(origin_lat))
    return east, north


# ---------------------------------------------------------------------------
# Agent 1: Sensor Fusion (Kalman filter on position + velocity)
# ---------------------------------------------------------------------------

class SensorFusionAgent:
    """
    Subscribes to raw SensorFrames.
    Runs a simple 4-state Kalman filter [x, y, vx, vy] in local ENU.
    Emits time-aligned FusedFrames to the next stage queue.
    """

    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q = in_q
        self.out_q = out_q
        self._origin: Optional[tuple[float, float]] = None

        # Kalman state: [east, north, ve, vn]
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._Q = np.diag([0.1, 0.1, 0.5, 0.5])   # process noise
        self._R_gps = np.diag([4.0, 4.0])           # GPS noise (2m sigma)

        self._last_ts = time.time()
        self._pending_sonar: Optional[SonarReturn] = None
        self._pending_motion: Optional[MotionFrame] = None

    async def run(self):
        print("[FusionAgent] Running")
        while True:
            frame: SensorFrame = await self.in_q.get()
            fused = self._process(frame)
            if fused:
                await self.out_q.put(fused)

    def _process(self, frame: SensorFrame) -> Optional[dict]:
        dt = frame.ts - self._last_ts
        self._last_ts = frame.ts
        dt = max(dt, 0.001)

        # Store sub-frames
        if frame.sonar:
            self._pending_sonar = frame.sonar
        if frame.motion:
            self._pending_motion = frame.motion

        if frame.gps is None:
            return None  # only emit on GPS ticks

        gps = frame.gps

        # Set origin on first fix
        if self._origin is None:
            self._origin = (gps.lat, gps.lon)
            print(f"[FusionAgent] Session origin set: {gps.lat:.6f}, {gps.lon:.6f}")

        east, north = latlon_to_enu(gps.lat, gps.lon, *self._origin)

        # Kalman predict
        F = np.array([[1, 0, dt, 0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

        # Kalman update (GPS measurement)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = np.array([east, north])
        y = z - H @ self._x
        S = H @ self._P @ H.T + self._R_gps
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ H) @ self._P

        return {
            "ts": frame.ts,
            "east": self._x[0],
            "north": self._x[1],
            "ve": self._x[2],
            "vn": self._x[3],
            "gps": gps,
            "sonar": self._pending_sonar,
            "motion": self._pending_motion,
        }


# ---------------------------------------------------------------------------
# Agent 2: Motion Compensation
# ---------------------------------------------------------------------------

class MotionCompAgent:
    """
    Corrects the sonar measurement point for:
      - Transducer offset from GPS antenna
      - Boat pitch/roll (heave correction)
      - Sound velocity profile (basic)
    """

    # Transducer offset relative to GPS antenna (metres, boat frame)
    TRANSD_FWD   = -0.5   # 0.5 m aft of GPS
    TRANSD_STBD  =  0.0   # centred
    TRANSD_DOWN  =  0.3   # 30 cm below waterline (draft)

    SOUND_VELOCITY = 1500.0  # m/s in fresh water (~1480–1530)

    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q = in_q
        self.out_q = out_q

    async def run(self):
        print("[MotionCompAgent] Running")
        while True:
            fused: dict = await self.in_q.get()
            corrected = self._compensate(fused)
            if corrected:
                await self.out_q.put(corrected)

    def _compensate(self, fused: dict) -> Optional[dict]:
        sonar = fused.get("sonar")
        if sonar is None:
            return None  # no depth measurement — skip

        heading_rad = math.radians(fused.get("motion", None).heading_deg
                                   if fused.get("motion") else fused["gps"].course_deg)
        pitch = roll = 0.0
        if fused.get("motion"):
            pitch = math.radians(fused["motion"].pitch_deg)
            roll  = math.radians(fused["motion"].roll_deg)

        # Rotate transducer offset into world ENU
        cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
        east_offset  =  self.TRANSD_FWD * sin_h + self.TRANSD_STBD * cos_h
        north_offset =  self.TRANSD_FWD * cos_h - self.TRANSD_STBD * sin_h

        # Heave and tilt correction
        depth_corrected = sonar.depth_m * math.cos(pitch) * math.cos(roll)
        depth_corrected -= self.TRANSD_DOWN  # subtract draft

        fused["east"]  += east_offset
        fused["north"] += north_offset
        fused["depth_m"] = max(0.01, depth_corrected)
        return fused


# ---------------------------------------------------------------------------
# Agent 3: Data Validation
# ---------------------------------------------------------------------------

class ValidationAgent:
    """
    Rejects bad observations and emits clean FusedObservation objects.

    Rejection rules:
      - Depth outlier (> 3σ from recent window)
      - Speed > 15 kts (sonar unreliable at high speed)
      - GPS HDOP > 5
      - Duplicate timestamp
    """

    WINDOW = 30          # samples for rolling stats
    MAX_SPEED_KTS = 15.0
    MAX_HDOP = 5.0

    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q = in_q
        self.out_q = out_q
        self._depth_window: Deque[float] = deque(maxlen=self.WINDOW)
        self._last_ts = 0.0
        self._rejected = 0
        self._accepted = 0

    async def run(self):
        print("[ValidationAgent] Running")
        while True:
            fused: dict = await self.in_q.get()
            obs = self._validate(fused)
            if obs:
                self._accepted += 1
                await self.out_q.put(obs)
            else:
                self._rejected += 1

    def _validate(self, fused: dict) -> Optional[FusedObservation]:
        depth = fused.get("depth_m")
        gps: GPSFrame = fused["gps"]
        ts: float = fused["ts"]

        if depth is None:
            return None

        # Duplicate timestamp
        if ts <= self._last_ts:
            return None
        self._last_ts = ts

        # Speed gate
        if gps.speed_kts > self.MAX_SPEED_KTS:
            return None

        # HDOP gate
        if gps.hdop > self.MAX_HDOP:
            return None

        # Depth outlier gate
        if len(self._depth_window) >= 5:
            mu = float(np.mean(self._depth_window))
            sigma = float(np.std(self._depth_window)) + 0.01
            if abs(depth - mu) > 3 * sigma:
                return None

        self._depth_window.append(depth)

        # Build confidence score
        speed_penalty = min(1.0, gps.speed_kts / self.MAX_SPEED_KTS)
        hdop_penalty  = min(1.0, (gps.hdop - 1.0) / (self.MAX_HDOP - 1.0))
        sonar: Optional[SonarReturn] = fused.get("sonar")
        signal_conf = (sonar.signal_strength_db / 100.0) if sonar else 0.5
        confidence = signal_conf * (1 - 0.3 * speed_penalty) * (1 - 0.2 * hdop_penalty)

        motion: Optional[MotionFrame] = fused.get("motion")
        speed_ms  = math.sqrt(fused.get("ve", 0)**2 + fused.get("vn", 0)**2)
        heading   = motion.heading_deg if motion else gps.course_deg

        return FusedObservation(
            ts=ts,
            easting_m=fused["east"],
            northing_m=fused["north"],
            depth_m=depth,
            confidence=round(confidence, 3),
            speed_ms=speed_ms,
            heading_deg=heading,
            signal_db=sonar.signal_strength_db if sonar else 0.0,
            bottom_hardness=sonar.bottom_hardness if sonar else 0.0,
        )


# ---------------------------------------------------------------------------
# Orchestrator — wires the pipeline together
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Connects all agents with asyncio queues.

    raw_q  → FusionAgent → fused_q → MotionCompAgent → comp_q → ValidationAgent → obs_q
    """

    def __init__(self):
        self.raw_q   = asyncio.Queue(maxsize=500)
        self._fused_q = asyncio.Queue(maxsize=200)
        self._comp_q  = asyncio.Queue(maxsize=200)
        self.obs_q   = asyncio.Queue(maxsize=500)   # consumers read from here

        self._fusion  = SensorFusionAgent(self.raw_q, self._fused_q)
        self._motion  = MotionCompAgent(self._fused_q, self._comp_q)
        self._valid   = ValidationAgent(self._comp_q, self.obs_q)

    async def run(self):
        await asyncio.gather(
            self._fusion.run(),
            self._motion.run(),
            self._valid.run(),
        )
