from dataclasses import dataclass, field
from typing import List, Optional
import json
import pathlib


@dataclass
class GroundTruthManifest:
    """
    Loaded from ground_truth.json (written by generate.py).
    In live-synthetic mode, produced directly from SyntheticLiveSource.
    """
    origin_lat: float
    origin_lon: float
    fish_schools: List[dict]
    lake_bottom_sample_20x20: Optional[list] = None
    floor_grid: Optional[dict] = None
    generated_by: str = "generate.py"

    @classmethod
    def from_json_file(cls, path: str) -> "GroundTruthManifest":
        data = json.loads(pathlib.Path(path).read_text())
        # Forward-compatible: ignore unknown keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_synthetic_live_source(cls, source) -> "GroundTruthManifest":
        """Extract live ground truth from a SyntheticLiveSource instance."""
        snap = source.get_true_state_snapshot()
        return cls(
            origin_lat=snap.get("origin_lat", 33.9003),
            origin_lon=snap.get("origin_lon", -117.5012),
            fish_schools=snap.get("fish_schools", []),
            lake_bottom_sample_20x20=snap.get("lake_bottom", {}).get("sample_20x20"),
            generated_by="SyntheticLiveSource",
        )
