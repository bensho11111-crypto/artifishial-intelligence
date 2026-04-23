from dataclasses import dataclass
from typing import List
import math


@dataclass
class DetectionMetrics:
    fish_true_count: int
    fish_detected_count: int
    precision: float
    recall: float
    mean_distance_error_m: float


def compute_metrics(
    detected_positions: List[dict],
    true_schools: List[dict],
    match_radius_m: float = 10.0,
) -> DetectionMetrics:
    """
    Greedy matching: a detection is a TP if within match_radius_m of any true school.
    detected_positions: list of {east_m, north_m, depth_m}
    true_schools: list of {east_m, north_m, depth_m, radius_m, ...}
    """
    if not true_schools:
        return DetectionMetrics(0, len(detected_positions), 0.0, 0.0, 0.0)

    matched_true = set()
    matched_det  = set()
    distances    = []

    for di, det in enumerate(detected_positions):
        best_dist = float("inf")
        best_ti   = None
        for ti, true in enumerate(true_schools):
            if ti in matched_true:
                continue
            d = math.sqrt(
                (det.get("east_m", 0) - true.get("east_m", 0)) ** 2 +
                (det.get("north_m", 0) - true.get("north_m", 0)) ** 2 +
                (det.get("depth_m", 0) - true.get("depth_m", 0)) ** 2
            )
            if d < best_dist:
                best_dist = d
                best_ti   = ti
        if best_ti is not None and best_dist <= match_radius_m:
            matched_true.add(best_ti)
            matched_det.add(di)
            distances.append(best_dist)

    tp = len(matched_true)
    precision = tp / len(detected_positions) if detected_positions else 0.0
    recall    = tp / len(true_schools) if true_schools else 0.0
    mean_err  = sum(distances) / len(distances) if distances else 0.0

    return DetectionMetrics(
        fish_true_count=len(true_schools),
        fish_detected_count=len(detected_positions),
        precision=round(precision, 3),
        recall=round(recall, 3),
        mean_distance_error_m=round(mean_err, 2),
    )
