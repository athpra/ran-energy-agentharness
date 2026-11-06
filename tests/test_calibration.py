"""
Tests for the VIAVI-calibrated mock RSG (mock_rsg/calibrate.py, mock_rsg/scenario.py).
No endpoint or VIAVI needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_calibration.py)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["USE_MOCK_RSG"] = "1"

from agent.policy import DEFAULT_POLICY  # noqa: E402
from experiments.common import connect_scenario  # noqa: E402
from experiments.tune_policy import evaluate  # noqa: E402
from mock_rsg.calibrate import _disagreement, _fit_line  # noqa: E402
from mock_rsg.scenario import calibrated_throughput  # noqa: E402

PROFILE = {"tp_by_sleeping_mbps": {"0": 6.0, "10": 5.0}, "tp_drop_per_sleeping": 0.2}


def test_throughput_interpolates_inside_and_extends_outside_the_observed_counts():
    assert calibrated_throughput(PROFILE, 0) == 6.0
    assert abs(calibrated_throughput(PROFILE, 5) - 5.5) < 1e-9
    assert abs(calibrated_throughput(PROFILE, 12) - 4.6) < 1e-9


def test_throughput_uses_the_logged_value_for_a_logged_set_of_sleeping_cells():
    profile = {**PROFILE, "tp_by_sleeping_set_mbps": {"1,3": 5.9}}
    assert calibrated_throughput(profile, 2, "1,3") == 5.9            # logged set: exact
    assert abs(calibrated_throughput(profile, 2, "0,1") - 5.8) < 1e-9  # same count, not logged
    assert abs(calibrated_throughput(PROFILE, 2, "1,3") - 5.8) < 1e-9  # older file without sets


def test_fit_line_reports_a_non_negative_drop():
    tp0, drop = _fit_line([(0, 7.0), (10, 6.0), (20, 5.0)])
    assert abs(tp0 - 7.0) < 1e-9 and abs(drop - 0.1) < 1e-9
    assert _fit_line([(4, 5.0), (4, 6.0)]) == (5.5, 0.0)


def test_runs_from_another_viavi_setup_are_left_out():
    ref = ({(1.0, "0,2"): 5.20}, {(1.0, 1): 20.0})
    assert _disagreement(({(1.0, "0,2"): 5.23}, {(1.0, 1): 22.0}), ref) is None   # same setup
    assert "throughput" in _disagreement(({(1.0, "0,2"): 5.40}, {(1.0, 1): 20.0}), ref)
    assert "PRB" in _disagreement(({(1.0, "4"): 6.0}, {(1.0, 1): 40.0}), ref)
    assert _disagreement(({(0.1, ""): 7.7}, {(0.1, 1): 3.0}), ref) is None        # nothing shared


def test_calibration_can_be_switched_off():
    os.environ["MOCK_RSG_CALIBRATION"] = "none"
    try:
        assert connect_scenario().calibration is None
    finally:
        del os.environ["MOCK_RSG_CALIBRATION"]


def test_calibrated_mock_reproduces_the_viavi_default_run():
    """The 5 Mbps default-policy full harness on VIAVI (three identical runs):
    reward 0.6448, 65.5% of sleepable cells asleep, 100 switches."""
    if connect_scenario().calibration is None:
        return  # no calibration file
    m = evaluate(DEFAULT_POLICY, [5.0], 96, "energy", 5.0, 0.01)
    assert (m["reward"], m["mean_sleep_fraction"], m["switches"]) == (0.6448, 0.6553, 100)
    assert m["avoidable_violation_rate"] == 0.0 and m["qos_violation_rate"] == 0.125


if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
