#!/usr/bin/env python3
"""
generate.py — synthetic Lowrance Elite 7 data generator.

Produces three files that represent 2 minutes of a fishing boat
trolling across Puddingstone Reservoir, CA:

  nmea_0183.nmea  — NMEA 0183 serial stream (1 Hz GPS / 5 Hz sonar)
  track.geojson   — GPS track with per-point depth/heading properties
  sample.sl2      — Minimal binary SL2 file (primary 200 kHz channel)

Run:  python generate.py
"""
import math
import struct
import json
import random
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
random.seed(42)

# ── simulation parameters ─────────────────────────────────────────────────
DURATION_S        = 120        # 2 minutes of data
GPS_HZ            = 1          # Lowrance default GPS output rate
SONAR_HZ          = 5          # typical downward-sonar ping rate
START_LAT         = 33.9003    # Puddingstone Reservoir, San Dimas CA
START_LON         = -117.5012
START_HEADING_DEG = 52.0       # NE along the main channel
SPEED_KTS         = 3.5        # typical slow-trolling speed
WATER_TEMP_C      = 18.3       # spring surface temp
BASE_DEPTH_M      = 6.0        # mean channel depth
ORIGIN_UTC_S      = 8*3600 + 32*60   # session start 08:32:00 UTC


# ── NMEA helpers ──────────────────────────────────────────────────────────

def _checksum(body: str) -> str:
    """XOR of every byte between $ and * (exclusive)."""
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"

def _dec_to_nmea(deg: float, is_lat: bool) -> tuple:
    """Decimal degrees → NMEA ddmm.mmmmm / dddmm.mmmmm + hemisphere."""
    hemi = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    d = abs(deg)
    dd = int(d)
    mm = (d - dd) * 60.0
    return (f"{dd:02d}{mm:09.6f}", hemi) if is_lat else (f"{dd:03d}{mm:09.6f}", hemi)

def _s(body: str) -> str:
    """Wrap body in $…*checksum\r\n."""
    return f"${body}*{_checksum(body)}\r\n"


# ── boat physics (Euler integration at SONAR_HZ steps) ───────────────────

class Boat:
    """
    Simple forward-Euler kinematic model.

    The heading follows a slow sinusoidal S-curve (like trolling back and
    forth across a fishing hole), and the depth model combines three
    sinusoidal components to mimic a realistic lake-floor profile:
      - main channel gradient
      - underwater ridge / shelf
      - fine-scale sonar texture + quantisation noise
    """

    def __init__(self):
        self.t          = 0.0
        self.lat        = START_LAT
        self.lon        = START_LON
        self.heading    = START_HEADING_DEG
        self.speed_kts  = SPEED_KTS
        self.depth_m    = BASE_DEPTH_M
        self._phase     = random.uniform(0, 2 * math.pi)

    def step(self, dt: float):
        self.t += dt
        t = self.t

        # Heading: slow S-curve turn (realistic trolling pattern)
        turn_rate = 9.0 * math.sin(t * 0.040) + 2.5 * math.sin(t * 0.14)
        self.heading = (self.heading + turn_rate * dt) % 360.0

        # Speed variation (engine trim, wind)
        self.speed_kts = SPEED_KTS + 0.55 * math.sin(t * 0.11) + 0.15 * random.gauss(0, 1)
        self.speed_kts = max(0.5, self.speed_kts)
        speed_ms = self.speed_kts * 0.5144

        # Euler position integration
        h = math.radians(self.heading)
        self.lat += speed_ms * dt * math.cos(h) / 111_320.0
        self.lon += speed_ms * dt * math.sin(h) / (111_320.0 * math.cos(math.radians(self.lat)))

        # Depth: three-component floor model + noise
        self.depth_m = (
            BASE_DEPTH_M
            + 3.8 * math.sin(t * 0.052)            # main channel profile
            + 1.3 * math.sin(t * 0.21 + 1.1)       # secondary ridge
            + 0.40 * math.sin(t * 0.85 + self._phase)  # fine sonar texture
            + 0.06 * random.gauss(0, 1)             # quantisation / noise floor
        )
        self.depth_m = max(0.6, self.depth_m)

    def hdop(self) -> float:
        # Mild HDOP variation; occasional spike near canyon walls
        base = 0.8 + 0.35 * abs(math.sin(self.t * 0.031))
        return round(base + (0.4 if 55 < self.t < 62 else 0.0), 1)

    def signal_strength(self) -> int:
        # 140–220 counts; drops slightly at high speed / shallow
        base = 185 - int(self.speed_kts * 8) + random.randint(-12, 12)
        return max(100, min(255, base))


# ── NMEA sentence builders ────────────────────────────────────────────────

def _time_fields(t_offset: float) -> str:
    ts = ORIGIN_UTC_S + t_offset
    hh = int(ts // 3600) % 24
    mm = int(ts // 60) % 60
    ss = ts % 60
    return f"{hh:02d}{mm:02d}{ss:06.3f}"

def gpgga(b: Boat) -> str:
    tf = _time_fields(b.t)
    ls, lh = _dec_to_nmea(b.lat, True)
    ws, wh = _dec_to_nmea(b.lon, False)
    body = f"GPGGA,{tf},{ls},{lh},{ws},{wh},1,09,{b.hdop():.1f},245.0,M,-26.1,M,,"
    return _s(body)

def gprmc(b: Boat) -> str:
    tf = _time_fields(b.t)
    ls, lh = _dec_to_nmea(b.lat, True)
    ws, wh = _dec_to_nmea(b.lon, False)
    body = f"GPRMC,{tf},A,{ls},{lh},{ws},{wh},{b.speed_kts:.2f},{b.heading:.1f},220426,,"
    return _s(body)

def gpvtg(b: Boat) -> str:
    body = f"GPVTG,{b.heading:.1f},T,,M,{b.speed_kts:.2f},N,{b.speed_kts*1.852:.2f},K,A"
    return _s(body)

def sddbt(depth_m: float) -> str:
    body = f"SDDBT,{depth_m/0.3048:.1f},f,{depth_m:.2f},M,{depth_m/1.8288:.1f},F"
    return _s(body)

def sddpt(depth_m: float) -> str:
    # Second field is transducer offset below waterline (0.30 m keel depth)
    body = f"SDDPT,{depth_m:.2f},0.30"
    return _s(body)

def hehdt(heading: float) -> str:
    body = f"HEHDT,{heading:.1f},T"
    return _s(body)

def sdmtw(temp_c: float) -> str:
    body = f"SDMTW,{temp_c:.1f},C"
    return _s(body)


# ── NMEA stream generation ────────────────────────────────────────────────

def generate_nmea() -> tuple:
    """
    Returns (lines, coords, props).

    Interleaving follows the real Lowrance Elite 7 pattern:
      - Depth + heading sentences at SONAR_HZ
      - GPS sentences (GGA + RMC + VTG) at GPS_HZ
      - Water temperature every ~10 s
    """
    boat = Boat()
    lines = []
    coords = []
    props  = []

    sonar_dt = 1.0 / SONAR_HZ   # 0.20 s
    gps_every = SONAR_HZ // GPS_HZ  # every Nth sonar tick

    n_steps = int(DURATION_S * SONAR_HZ)
    for step in range(n_steps):
        boat.step(sonar_dt)

        # ── sonar + heading (every tick) ──
        lines.append(sddbt(boat.depth_m))
        lines.append(sddpt(boat.depth_m))
        lines.append(hehdt(boat.heading))

        # ── GPS (1 Hz = every 5th sonar tick) ──
        if step % gps_every == 0:
            lines.append(gpgga(boat))
            lines.append(gprmc(boat))
            lines.append(gpvtg(boat))
            if step % (10 * SONAR_HZ) == 0:   # temperature every 10 s
                lines.append(sdmtw(WATER_TEMP_C + 0.05 * math.sin(boat.t * 0.04)))

            coords.append([round(boat.lon, 7), round(boat.lat, 7)])
            props.append({
                "t_s":        round(boat.t, 1),
                "depth_m":    round(boat.depth_m, 2),
                "heading_deg": round(boat.heading, 1),
                "speed_kts":  round(boat.speed_kts, 2),
                "hdop":       boat.hdop(),
            })

    return lines, coords, props


# ── GeoJSON track ─────────────────────────────────────────────────────────

def generate_geojson(coords: list, props: list) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "device": "Lowrance Elite 7",
                    "session_date": "2026-04-22",
                    "start_utc": "08:32:00",
                    "water_body": "Puddingstone Reservoir, San Dimas CA",
                    "note": "Synthetic — generated by generate.py"
                }
            },
            *[
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coords[i]},
                    "properties": p
                }
                for i, p in enumerate(props)
            ]
        ]
    }


# ── SL2 binary ────────────────────────────────────────────────────────────
#
# Block layout decoded by StructureScanReader._decode_block:
#
#   offset  0  uint16  block_size        (total bytes this block)
#   offset  2  uint16  last_block_size
#   offset  4  uint16  channel_id        (0 = primary 200 kHz downward)
#   offset  6  uint16  packet_size       (sonar echo bytes that follow header)
#   offset  8  uint32  frame_index
#   offset 12  float32 upper_limit_ft
#   offset 16  float32 lower_limit_ft
#   offset 20  float32 frequency_hz
#   offset 24  float32 water_depth_ft    ← app reads this
#   offset 28  float32 water_temp_c      ← app reads this
#   offset 32  float32 speed_gps_cms
#   offset 36  uint8   signal_strength   ← app reads this
#   offset 37  3 bytes padding
#   offset 40  N bytes sonar echo data   (N = packet_size)
#
# Real SL2 packets typically have 512–3000 echo bytes; we use 512.

ECHO_SIZE  = 512
HDR_SIZE   = 40
BLOCK_SIZE = HDR_SIZE + ECHO_SIZE   # 552 bytes

_HDR_FMT = "<HHHHIffffffB3s"  # 40 bytes total

# ── Beam physics constant (200 kHz, 20° full angle) ───────────────────────
_TAN_BEAM_HALF = math.tan(math.radians(10.0))   # ≈ 0.1763


# ── TrueWorldSnapshot for SL2 generation ─────────────────────────────────

class _FishSchoolSL2:
    """Lightweight fish school for SL2 generation (no async needed)."""
    def __init__(self, east_m, north_m, depth_m, radius_m, density, species):
        self.east_m   = east_m
        self.north_m  = north_m
        self.depth_m  = depth_m
        self.radius_m = radius_m
        self.density  = density
        self.species  = species


_SL2_SPECIES_TS = {"bass": 0.85, "trout": 0.70, "carp": 0.60, "bream": 0.50}


def _build_sl2_fish_schools(boat_route_positions: list) -> list:
    """
    Place 3–4 fish schools at fixed positions relative to the trolling route
    so that several arches are visible in the SL2 output.

    At least one school is placed directly on the trolling path to guarantee
    a complete arch.
    """
    rng = random.Random(2024)   # deterministic for reproducibility

    # Estimate centre of the trolling route
    if boat_route_positions:
        mid = len(boat_route_positions) // 2
        cx = sum(p[0] for p in boat_route_positions[max(0, mid-5):mid+5]) / 10
        cy = sum(p[1] for p in boat_route_positions[max(0, mid-5):mid+5]) / 10
    else:
        cx, cy = 0.0, 0.0

    schools = []

    # School 0: directly on trolling path → guaranteed complete arch
    schools.append(_FishSchoolSL2(
        east_m   = cx,
        north_m  = cy,
        depth_m  = 3.5,
        radius_m = 8.0,
        density  = 0.9,
        species  = "bass",
    ))

    # School 1: slightly off-path (partial arch)
    schools.append(_FishSchoolSL2(
        east_m   = cx + 15.0,
        north_m  = cy + 5.0,
        depth_m  = 5.0,
        radius_m = 6.0,
        density  = 0.7,
        species  = "trout",
    ))

    # School 2: further off-path (edge echo)
    schools.append(_FishSchoolSL2(
        east_m   = cx - 10.0,
        north_m  = cy - 8.0,
        depth_m  = 6.5,
        radius_m = 10.0,
        density  = 0.55,
        species  = "carp",
    ))

    # School 3: near route start
    if boat_route_positions:
        sx, sy = boat_route_positions[len(boat_route_positions) // 5]
    else:
        sx, sy = cx + 20, cy - 20
    schools.append(_FishSchoolSL2(
        east_m   = sx,
        north_m  = sy,
        depth_m  = 4.0,
        radius_m = 5.0,
        density  = 0.8,
        species  = "bream",
    ))

    return schools


def _lat_lon_to_enu_simple(lat, lon, origin_lat, origin_lon):
    """Flat-earth approximation: returns (east_m, north_m)."""
    R = 6_371_000.0
    north = R * math.radians(lat - origin_lat)
    east  = R * math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat))
    return east, north


def _sl2_header() -> bytes:
    # 8-byte file header: format=2 (SL2), device=Elite 7 (0x0100)
    return struct.pack("<HHHH", 2, 0x0100, 0, 0)

def _sl2_block(frame_idx: int, depth_m: float, temp_c: float,
               speed_kts: float, strength: int, last_size: int,
               fish_schools: list, boat_east: float, boat_north: float) -> bytes:
    depth_ft  = depth_m / 0.3048
    speed_cms = speed_kts * 51.44
    hdr = struct.pack(
        _HDR_FMT,
        BLOCK_SIZE,            # block_size
        last_size,             # last_block_size
        0,                     # channel_id = primary 200 kHz
        ECHO_SIZE,             # packet_size
        frame_idx,             # frame_index
        0.0,                   # upper_limit_ft
        60.0 / 0.3048,         # lower_limit_ft (60 m max range)
        200_000.0,             # frequency_hz
        depth_ft,              # water_depth_ft  (offset 24)
        temp_c,                # water_temp_c    (offset 28)
        speed_cms,             # speed_gps_cms   (offset 32)
        strength & 0xFF,       # signal_strength (offset 36)
        b'\x00\x00\x00',       # padding         (offset 37)
    )
    echo = _make_echo(depth_m, 60.0, ECHO_SIZE, fish_schools, boat_east, boat_north)
    return hdr + echo

def _make_echo(depth_m: float, max_range_m: float, n: int,
               fish_schools: list = None, boat_east: float = 0.0,
               boat_north: float = 0.0) -> bytes:
    """
    Simulate a realistic sonar A-scope return:
      - noise floor (0–15 counts)
      - first-bottom return: narrow Gaussian at depth_m
      - second bottom echo: weaker return at 2× depth (typical multi-path)
      - physics-based fish arch echoes using slant-range geometry

    When fish_schools is provided (list of _FishSchoolSL2), proper arch-shaped
    echoes are computed from the beam physics model.  Each school contributes
    a return at the slant range rather than the true depth, creating the
    characteristic arch as the boat passes overhead.
    """
    echo = bytearray(n)
    first_idx  = int(depth_m / max_range_m * n)
    first_idx  = max(0, min(n - 1, first_idx))
    sigma      = max(3, int(n * 0.012))

    # Noise floor
    for i in range(n):
        echo[i] = random.randint(2, 15)

    def _gaussian(centre, amp, width):
        for i in range(max(0, centre - width*4), min(n, centre + width*4 + 1)):
            v = int(amp * math.exp(-0.5 * ((i - centre) / width) ** 2))
            echo[i] = min(255, echo[i] + v)

    # Primary bottom return
    _gaussian(first_idx, 215, sigma)

    # Second bottom echo (multi-path)
    if first_idx * 2 < n:
        _gaussian(first_idx * 2, 80, sigma + 2)

    # Physics-based fish arch echoes
    if fish_schools:
        for school in fish_schools:
            horiz_offset = math.sqrt(
                (school.east_m - boat_east) ** 2 +
                (school.north_m - boat_north) ** 2
            )
            beam_radius = school.depth_m * _TAN_BEAM_HALF
            if horiz_offset > beam_radius + school.radius_m:
                continue   # school not in beam

            overlap = max(0.0,
                (beam_radius + school.radius_m - horiz_offset) / school.radius_m)
            overlap = min(1.0, overlap)

            slant_range = math.sqrt(horiz_offset ** 2 + school.depth_m ** 2)
            ts_factor = _SL2_SPECIES_TS.get(school.species, 0.65)
            echo_amp   = school.density * overlap * 0.4 * ts_factor
            amp_counts = int(echo_amp * 180)

            if amp_counts < 1:
                continue

            fish_idx = int(slant_range / max_range_m * n)
            fish_idx = max(0, min(n - 1, fish_idx))

            # Only render if above the bottom return (prevents overlap confusion)
            if fish_idx < first_idx - sigma * 2:
                _gaussian(fish_idx, amp_counts, 3)

    return bytes(echo)

def generate_sl2(coords: list, props: list) -> bytes:
    # Build ENU positions for the boat route so we can compute fish offsets
    origin_lat = START_LAT
    origin_lon = START_LON
    route_enu = []
    for coord, p in zip(coords, props):
        lon_c, lat_c = coord
        east, north = _lat_lon_to_enu_simple(lat_c, lon_c, origin_lat, origin_lon)
        route_enu.append((east, north))

    fish_schools = _build_sl2_fish_schools(route_enu)
    print(f"  SL2 fish schools: {len(fish_schools)} "
          f"({', '.join(s.species for s in fish_schools)})")

    data = _sl2_header()
    last = 0
    for i, (p, (boat_e, boat_n)) in enumerate(zip(props, route_enu)):
        block = _sl2_block(
            i, p["depth_m"], WATER_TEMP_C + 0.05 * math.sin(p["t_s"] * 0.04),
            p["speed_kts"], random.randint(140, 225), last,
            fish_schools, boat_e, boat_n,
        )
        data += block
        last = BLOCK_SIZE
    data += struct.pack("<H", 0)   # EOF sentinel
    return data


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic Lowrance Elite 7 data …")

    nmea_lines, coords, props = generate_nmea()

    nmea_path = OUT_DIR / "nmea_0183.nmea"
    nmea_path.write_text("".join(nmea_lines), encoding="ascii")
    print(f"  {nmea_path.name}: {len(nmea_lines)} sentences  ({nmea_path.stat().st_size:,} bytes)")

    geo_path = OUT_DIR / "track.geojson"
    geo_path.write_text(json.dumps(generate_geojson(coords, props), indent=2), encoding="utf-8")
    print(f"  {geo_path.name}: {len(coords)} track points  ({geo_path.stat().st_size:,} bytes)")

    sl2_data = generate_sl2(coords, props)
    sl2_path = OUT_DIR / "sample.sl2"
    sl2_path.write_bytes(sl2_data)
    print(f"  {sl2_path.name}: {len(props)} packets  ({sl2_path.stat().st_size:,} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
