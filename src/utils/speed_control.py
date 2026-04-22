# src/utils/speed_control.py
_multiplier: float = 1.0

def get() -> float:
    return _multiplier

def set(value: float) -> None:
    global _multiplier
    _multiplier = max(0.1, min(10.0, float(value)))
