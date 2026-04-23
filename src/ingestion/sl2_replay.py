"""
src/ingestion/sl2_replay.py

Loads an SL2 file into memory as a list of raw packet byte-strings,
and provides a decode function for use with ReplayController.

When a matching track.geojson exists alongside the SL2, GPS coordinates
are injected into each packet so the fusion agent can place observations
on the map. The SL2 and geojson are 1:1 synchronized (same boat model).
"""
import struct
import time
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ingestion.nmea_reader import SensorFrame, SonarReturn, GPSFrame, MotionFrame
from replay.controller import ReplayController


def load_sl2_packets(path: str) -> List[bytes]:
    """Read all raw packet bytes from an SL2 file into a list."""
    with open(path, "rb") as f:
        data = f.read()
    packets = []
    offset = 8  # skip file header
    while offset + 2 < len(data):
        block_size = struct.unpack_from("<H", data, offset)[0]
        if block_size == 0:
            break
        end = offset + block_size
        if end > len(data):
            break
        packets.append(data[offset:end])
        offset = end
    return packets


def _load_geojson_gps(geojson_path: str) -> List[dict]:
    """Load per-fix GPS properties from track.geojson Point features."""
    geo = json.loads(Path(geojson_path).read_text())
    points = []
    for f in geo.get("features", []):
        if f.get("geometry", {}).get("type") == "Point":
            coord = f["geometry"]["coordinates"]  # [lon, lat]
            props = f.get("properties", {})
            points.append({
                "lat":         coord[1],
                "lon":         coord[0],
                "speed_kts":   props.get("speed_kts", 0.0),
                "heading_deg": props.get("heading_deg", 0.0),
                "hdop":        props.get("hdop", 1.0),
            })
    return points


def _pair_packets_with_gps(
    sl2_path: str,
) -> List[Tuple[bytes, Optional[dict]]]:
    """
    Pair each SL2 packet with a GPS fix from the adjacent track.geojson.
    Falls back to sonar-only tuples if no geojson found.
    """
    packets = load_sl2_packets(sl2_path)
    geojson_path = Path(sl2_path).with_name("track.geojson")
    gps_fixes = _load_geojson_gps(str(geojson_path)) if geojson_path.exists() else []
    paired = []
    for i, pkt in enumerate(packets):
        gps = gps_fixes[i] if i < len(gps_fixes) else None
        paired.append((pkt, gps))
    return paired


def decode_sl2_packet(block: bytes) -> Optional[SensorFrame]:
    """Sonar-only decode (no GPS). Used when no geojson is available."""
    if len(block) < 40:
        return None
    try:
        depth_ft = struct.unpack_from("<f", block, 24)[0]
        depth_m  = depth_ft * 0.3048
        strength = float(block[36]) / 255.0 * 100.0
        ts = time.time()
        return SensorFrame(
            ts=ts,
            sonar=SonarReturn(
                ts=ts,
                depth_m=depth_m,
                frequency_hz=200_000,
                signal_strength_db=strength,
                bottom_hardness=min(1.0, strength / 80.0),
            ),
        )
    except (struct.error, IndexError):
        return None


def decode_sl2_paired(item: Tuple[bytes, Optional[dict]]) -> Optional[SensorFrame]:
    """Decode an SL2 packet and inject GPS from the paired geojson fix."""
    block, gps_fix = item
    if len(block) < 40:
        return None
    try:
        depth_ft = struct.unpack_from("<f", block, 24)[0]
        depth_m  = depth_ft * 0.3048
        strength = float(block[36]) / 255.0 * 100.0
        ts = time.time()
        gps = None
        motion = None
        if gps_fix:
            gps = GPSFrame(
                ts=ts,
                lat=gps_fix["lat"],
                lon=gps_fix["lon"],
                speed_kts=gps_fix["speed_kts"],
                course_deg=gps_fix["heading_deg"],
                hdop=gps_fix["hdop"],
            )
            motion = MotionFrame(
                ts=ts,
                speed_ms=gps_fix["speed_kts"] * 0.5144,
                heading_deg=gps_fix["heading_deg"],
            )
        return SensorFrame(
            ts=ts,
            gps=gps,
            sonar=SonarReturn(
                ts=ts,
                depth_m=depth_m,
                frequency_hz=200_000,
                signal_strength_db=strength,
                bottom_hardness=min(1.0, strength / 80.0),
            ),
            motion=motion,
        )
    except (struct.error, IndexError):
        return None


def make_sl2_replay_controller(
    path: str,
    queue: asyncio.Queue,
    on_seek=None,
    packet_interval_s: float = 1.0,
) -> ReplayController:
    """
    Load an SL2 file (+ adjacent track.geojson if present) and return a
    configured ReplayController.

    packet_interval_s defaults to 1.0 s because the synthetic SL2 has
    1 packet per GPS second (120 packets = 120 s of data).
    """
    paired = _pair_packets_with_gps(path)
    has_gps = any(gps is not None for _, gps in paired)
    decode_fn = decode_sl2_paired if has_gps else decode_sl2_packet
    packets = paired if has_gps else [p for p, _ in paired]
    return ReplayController(
        packets=packets,
        packet_interval_s=packet_interval_s,
        queue=queue,
        decode_fn=decode_fn,
        on_seek=on_seek,
    )
