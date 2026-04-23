from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class AppMode(Enum):
    LIVE             = auto()   # real serial hardware; no scrub, no ground truth
    TRUE_REPLAY      = auto()   # recorded SL2/NMEA file; has slider; no ground truth
    SYNTHETIC_REPLAY = auto()   # synthetic SL2 + ground_truth.json; has slider + metrics

@dataclass
class AppContext:
    mode: AppMode
    serial_port: Optional[str] = None        # LIVE only
    sl2_path: Optional[str] = None           # TRUE_REPLAY and SYNTHETIC_REPLAY
    nmea_path: Optional[str] = None          # optional NMEA file for replay
    ground_truth_path: Optional[str] = None  # SYNTHETIC_REPLAY only
    host: str = "0.0.0.0"
    http_port: int = 8000
    voxel_size_m: float = 0.5
    sim_hz: int = 10
