"""
src/ingestion/nmea_reader.py

Layer 1 — Data Ingestion
Reads NMEA 0183 and NMEA 2000 sentences from the Lowrance Elite 7.
Emits typed SensorFrame objects onto an asyncio queue.
"""
import asyncio
import time
import math
import struct
from dataclasses import dataclass, field
from typing import Optional
import serial_asyncio  # pip install pyserial-asyncio
import pynmea2         # pip install pynmea2
from utils import speed_control


# ---------------------------------------------------------------------------
# Data models — one per sensor type
# ---------------------------------------------------------------------------

@dataclass
class GPSFrame:
    ts: float                  # Unix timestamp (UTC)
    lat: float                 # decimal degrees
    lon: float                 # decimal degrees
    speed_kts: float           # knots
    course_deg: float          # true heading, degrees
    hdop: float = 1.0


@dataclass
class SonarReturn:
    ts: float
    depth_m: float             # water depth below transducer
    frequency_hz: int          # 83 kHz (wide) or 200 kHz (narrow)
    signal_strength_db: float  # 0–100 dB
    bottom_hardness: float     # 0.0–1.0 (derived from return strength)


@dataclass
class MotionFrame:
    ts: float
    speed_ms: float            # m/s over ground
    heading_deg: float         # true heading
    pitch_deg: float = 0.0
    roll_deg: float  = 0.0
    heave_m: float   = 0.0     # vertical displacement


@dataclass
class SensorFrame:
    """Unified frame emitted to the agent queue."""
    ts: float
    gps: Optional[GPSFrame]      = None
    sonar: Optional[SonarReturn] = None
    motion: Optional[MotionFrame]= None
    raw: str                     = ""


# ---------------------------------------------------------------------------
# NMEA 0183 reader (Lowrance outputs standard sentences on USB serial)
# ---------------------------------------------------------------------------

class NMEA0183Reader:
    """
    Connects to the Lowrance Elite 7 via USB-to-serial (typically /dev/ttyUSB0
    or COM3 on Windows) at 4800 / 38400 baud.

    Parses:
      - $GPGGA / $GPRMC  → GPSFrame
      - $SDDBT / $SDDPT  → SonarReturn (depth)
      - $HEHDT / $TIROT  → heading / rate-of-turn
    """

    BAUD = 38400

    def __init__(self, port: str, queue: asyncio.Queue):
        self.port = port
        self.queue = queue
        self._last_gps: Optional[GPSFrame] = None

    async def run(self):
        reader, _ = await serial_asyncio.open_serial_connection(
            url=self.port, baudrate=self.BAUD
        )
        print(f"[NMEA0183] Connected on {self.port} @ {self.BAUD} baud")
        while True:
            try:
                raw = await reader.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                frame = self._parse(line)
                if frame:
                    await self.queue.put(frame)
            except Exception as e:
                print(f"[NMEA0183] Parse error: {e}")

    def _parse(self, line: str) -> Optional[SensorFrame]:
        ts = time.time()
        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return None

        # GPS position + speed
        if isinstance(msg, (pynmea2.GGA, pynmea2.RMC)):
            if msg.latitude and msg.longitude:
                gps = GPSFrame(
                    ts=ts,
                    lat=msg.latitude,
                    lon=msg.longitude,
                    speed_kts=getattr(msg, "spd_over_grnd", 0.0) or 0.0,
                    course_deg=getattr(msg, "true_course", 0.0) or 0.0,
                )
                self._last_gps = gps
                return SensorFrame(ts=ts, gps=gps, raw=line)

        # Depth below transducer (DBT) or depth below surface (DPT)
        if msg.sentence_type in ("DBT", "DPT"):
            depth_m = float(getattr(msg, "depth_meters", 0) or 0)
            return SensorFrame(
                ts=ts,
                sonar=SonarReturn(
                    ts=ts,
                    depth_m=depth_m,
                    frequency_hz=200_000,
                    signal_strength_db=50.0,   # not available in NMEA
                    bottom_hardness=0.5,
                ),
                raw=line,
            )

        # Heading (HDT = true)
        if msg.sentence_type == "HDT":
            hdg = float(msg.heading or 0)
            spd = self._last_gps.speed_kts * 0.5144 if self._last_gps else 0.0
            return SensorFrame(
                ts=ts,
                motion=MotionFrame(ts=ts, speed_ms=spd, heading_deg=hdg),
                raw=line,
            )

        return None


# ---------------------------------------------------------------------------
# StructureScan binary reader (Lowrance SL2/SL3 proprietary format)
# ---------------------------------------------------------------------------

SL2_HEADER = struct.Struct("<HHHHHIIIIIffff")  # simplified — real SL2 has more fields

class StructureScanReader:
    """
    Reads a live SL2 stream or replays a recorded .sl2 file.
    Each packet contains depth, water temperature, and raw sonar amplitude.
    """

    PACKET_SIZE = 1970  # bytes per SL2 block (varies by channel)

    def __init__(self, source: str, queue: asyncio.Queue, is_file: bool = False):
        self.source = source   # serial port or file path
        self.queue = queue
        self.is_file = is_file

    async def run(self):
        if self.is_file:
            await self._replay_file()
        else:
            await self._stream_live()

    async def _replay_file(self):
        with open(self.source, "rb") as f:
            data = f.read()
        # Walk packets (real parser would check magic bytes / channel ID)
        offset = 8  # skip file header
        while offset + 4 < len(data):
            block_size = struct.unpack_from("<H", data, offset)[0]
            if block_size == 0:
                break
            block = data[offset: offset + block_size]
            frame = self._decode_block(block)
            if frame:
                await self.queue.put(frame)
            await asyncio.sleep(0.05 / speed_control.get())
            offset += block_size

    async def _stream_live(self):
        # Live SL2 streaming is not natively exposed by Lowrance Elite 7.
        # Use a GoFree WiFi adapter + UDP broadcast, or record SD card files.
        print("[StructureScan] Live SL2 streaming: connect via GoFree WiFi at 192.168.0.1:2000")
        reader, _ = await serial_asyncio.open_serial_connection(
            url=self.source, baudrate=115200
        )
        buf = b""
        while True:
            chunk = await reader.read(4096)
            buf += chunk
            while len(buf) >= self.PACKET_SIZE:
                frame = self._decode_block(buf[:self.PACKET_SIZE])
                if frame:
                    await self.queue.put(frame)
                buf = buf[self.PACKET_SIZE:]

    def _decode_block(self, block: bytes) -> Optional[SensorFrame]:
        if len(block) < 40:
            return None
        try:
            # Offsets for SL2 format (channel 0 = primary sonar)
            depth_ft = struct.unpack_from("<f", block, 24)[0]
            depth_m = depth_ft * 0.3048
            temp_c  = struct.unpack_from("<f", block, 28)[0]
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
        except struct.error:
            return None


# ---------------------------------------------------------------------------
# Mock / simulation source  (for development without hardware)
# ---------------------------------------------------------------------------

class SimulatedSource:
    """Generates synthetic sensor data for offline development and testing."""

    def __init__(self, queue: asyncio.Queue, hz: int = 10):
        self.queue = queue
        self.hz = hz
        self._t = 0.0

    async def run(self):
        print("[Sim] Simulated sensor source running")
        while True:
            self._t += 1.0 / self.hz
            ts = time.time()
            # Simple sinusoidal lake bed
            depth = 5.0 + 3.0 * math.sin(self._t * 0.1) + 0.3 * (hash(int(self._t * 10)) % 10) / 10
            lat = 33.900 + math.sin(self._t * 0.02) * 0.001
            lon = -117.500 + math.cos(self._t * 0.02) * 0.001
            speed_ms = 1.5 + 0.2 * math.sin(self._t)
            heading = (self._t * 5.0) % 360.0

            await self.queue.put(SensorFrame(
                ts=ts,
                gps=GPSFrame(ts=ts, lat=lat, lon=lon,
                             speed_kts=speed_ms / 0.5144, course_deg=heading),
                sonar=SonarReturn(ts=ts, depth_m=depth,
                                  frequency_hz=200_000, signal_strength_db=65.0,
                                  bottom_hardness=0.6),
                motion=MotionFrame(ts=ts, speed_ms=speed_ms, heading_deg=heading),
            ))
            await asyncio.sleep(1.0 / (self.hz * speed_control.get()))
