"""
Tool: Traffic Prediction (VTZ-aware)

Computes N-step-ahead load forecast directly from the VTZ traffic-profile
schedule so the Planner can anticipate surges and proactively wake cells
before QoS degrades — rather than reacting after throughput has already fallen.

Key transitions detected as surges (ue_ratio ≥ 1.5 × current):
    Night  (0.1×) → Early     (1.5×)  at 08:00
    Afternoon (0.5×) → Evening_1 (2.0×) at 18:00
"""
from __future__ import annotations

import pandas as pd

_BASE_UES = 200   # 50 indoor + 150 outdoor at 1.0×, matching common.py

_TRAFFIC_PROFILES = [
    {"name": "Night",      "start_hour":  0, "duration_hours": 8, "ue_ratio": 0.1},
    {"name": "Early",      "start_hour":  8, "duration_hours": 1, "ue_ratio": 1.5},
    {"name": "Morning",    "start_hour":  9, "duration_hours": 4, "ue_ratio": 1.0},
    {"name": "Afternoon",  "start_hour": 13, "duration_hours": 5, "ue_ratio": 0.5},
    {"name": "Evening_1",  "start_hour": 18, "duration_hours": 2, "ue_ratio": 2.0},
    {"name": "Evening_2",  "start_hour": 20, "duration_hours": 2, "ue_ratio": 1.0},
    {"name": "Late Night", "start_hour": 22, "duration_hours": 2, "ue_ratio": 0.1},
]


def _get_profile(ts: pd.Timestamp) -> dict:
    hour = ts.hour
    for p in _TRAFFIC_PROFILES:
        if p["start_hour"] <= hour < p["start_hour"] + p["duration_hours"]:
            return p
    return _TRAFFIC_PROFILES[0]


def query(
    timestamp: pd.Timestamp | str,
    cell_id: int | None = None,   # unused; network-wide forecast — kept for interface compat
    horizon: int = 3,
    step_minutes: int = 15,
) -> dict:
    """
    Return load forecast for the next `horizon` steps (each `step_minutes` apart).

    Returns a network-wide dict:
        {
            "current":        {"profile": str, "ue_ratio": float, "ues": int},
            "forecast":       [
                {"step": 1, "profile": str, "ue_ratio": float, "ues": int, "surge": bool},
                ...
            ],
            "surge_imminent": bool,
            "surge_step":     int | None,   # first step where a surge occurs, or None
        }

    A step is flagged as a surge when its ue_ratio >= 1.5 × current ue_ratio,
    giving the Planner clear advance warning to wake sleeping cells before the
    load transition hits.
    """
    ts              = pd.Timestamp(timestamp)
    current_profile = _get_profile(ts)
    current_ratio   = current_profile["ue_ratio"]

    forecast_steps: list[dict] = []
    surge_step: int | None = None

    for step in range(1, horizon + 1):
        future_ts      = ts + pd.Timedelta(minutes=step_minutes * step)
        future_profile = _get_profile(future_ts)
        is_surge       = future_profile["ue_ratio"] >= current_ratio * 1.5
        if is_surge and surge_step is None:
            surge_step = step
        forecast_steps.append({
            "step":     step,
            "profile":  future_profile["name"],
            "ue_ratio": future_profile["ue_ratio"],
            "ues":      int(_BASE_UES * future_profile["ue_ratio"]),
            "surge":    is_surge,
        })

    return {
        "current": {
            "profile":  current_profile["name"],
            "ue_ratio": current_ratio,
            "ues":      int(_BASE_UES * current_ratio),
        },
        "forecast":       forecast_steps,
        "surge_imminent": surge_step is not None,
        "surge_step":     surge_step,
    }
