from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MapSnapshot:
    """Delivered to AR clients in ALL modes."""
    ts: float
    pointcloud: dict          # {x, y, depth, confidence, std}
    stats: dict               # {updates, coverage_pct, mean_depth, uptime_s}
    boat: dict                # {east, north, heading}
    replay_position_s: Optional[float] = None   # None in LIVE
    replay_duration_s: Optional[float] = None   # None in LIVE
    replay_paused: Optional[bool] = None        # None in LIVE


@dataclass
class FishSchoolGroundTruth:
    east_m: float
    north_m: float
    depth_m: float
    radius_m: float
    density: float
    species: str


@dataclass
class SyntheticGroundTruth:
    """
    Only constructed when mode == SYNTHETIC_REPLAY.
    Must never appear in LIVE or TRUE_REPLAY code paths.
    """
    fish_schools: List[FishSchoolGroundTruth] = field(default_factory=list)
    lake_bottom_sample: Optional[list] = None
    current_boat_true: Optional[dict] = None


@dataclass
class SyntheticMapSnapshot(MapSnapshot):
    """
    Extends MapSnapshot with ground-truth fields.
    Only ever instantiated in SYNTHETIC_REPLAY mode.
    The AR server checks isinstance(snapshot, SyntheticMapSnapshot) before
    including fish overlay fields in the WebSocket payload.
    """
    fish_ground_truth: Optional[SyntheticGroundTruth] = None
    detection_metrics: Optional[dict] = None
