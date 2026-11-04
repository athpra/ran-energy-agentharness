"""
Tests for the harness policy, the episode reward and the policy tuner.
Uses the mock RSG and stand-in models; no endpoint or VIAVI needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_policy.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["USE_MOCK_RSG"] = "1"

import pandas as pd  # noqa: E402

from agent.context import ALL_TOOLS, assemble  # noqa: E402
from agent.policy import DEFAULT_POLICY, Policy, load_policy, save_policy  # noqa: E402
from experiments.reward import episode_metrics  # noqa: E402
from experiments.tune_policy import FallbackPlanner, RuleValidator, evaluate  # noqa: E402
from tools.traffic_prediction import query as forecast_query  # noqa: E402

TS = pd.Timestamp("2024-01-01 03:00")          # off-peak, no surge, no faults expected to matter
UTIL = {c: 10.0 + c for c in range(21)}         # cells 0–1 below 12%, more below 25%


def _candidates(policy, ts=TS, tp=10.0):
    ctx, _ = assemble(ts, list(range(21)), UTIL, ALL_TOOLS, sleeping_cell_ids=[],
                      current_throughput_mbps=tp, intent_mbps=5.0, policy=policy)
    return ctx["sleep_candidates"]


def test_default_policy_matches_the_old_constants():
    assert (DEFAULT_POLICY.prb_sleep_threshold, DEFAULT_POLICY.prb_sleep_threshold_peak,
            DEFAULT_POLICY.throughput_headroom, DEFAULT_POLICY.surge_ratio,
            DEFAULT_POLICY.overload_threshold_pct) == (12.0, 25.0, 1.1, 1.5, 85.0)
    assert load_policy(None) is DEFAULT_POLICY
    assert _candidates(None) == _candidates(DEFAULT_POLICY)


def test_policy_thresholds_change_the_candidates():
    loose = replace(DEFAULT_POLICY, prb_sleep_threshold=20.0)
    assert set(_candidates(DEFAULT_POLICY)) < set(_candidates(loose))
    strict = replace(DEFAULT_POLICY, throughput_headroom=1.5)
    assert _candidates(strict, tp=7.0) == [] and _candidates(DEFAULT_POLICY, tp=7.0)


def test_surge_ratio_is_tunable():
    afternoon = pd.Timestamp("2024-01-01 17:30")   # Afternoon 0.5× → Evening_1 2.0× (4×)
    assert forecast_query(afternoon, surge_ratio=1.5)["surge_imminent"]
    assert not forecast_query(afternoon, surge_ratio=5.0)["surge_imminent"]


def test_load_policy_round_trip_and_bounds():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mine.yaml"
        save_policy(replace(DEFAULT_POLICY, prb_sleep_threshold=15.0), path, meta={"note": "x"})
        p = load_policy(path)
        assert p.prb_sleep_threshold == 15.0 and p.name == "default"
        path.write_text("throughput_headroom: 0.9\n")
        for bad in ("throughput_headroom: 0.9\n", "prb_sleep_threshold: 20\nprb_sleep_threshold_peak: 10\n",
                    "fault_blocking: false\n"):
            path.write_text(bad)
            try:
                load_policy(path)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad!r}")


def _it(ts, sleeping, tp, actions=0):
    state = {"sleeping_cells": sleeping, "avg_throughput_mbps": tp, "per_site": {i: {} for i in range(21)}}
    return {"timestamp": ts, "pre_kpis": state, "post_kpis": state, "approved_actions": [{}] * actions}


def test_reward_penalises_only_avoidable_violations():
    base = [_it("2024-01-01 03:00", 21, 6.0, 1), _it("2024-01-01 12:00", 0, 4.0)]
    m = episode_metrics(base, 5.0)
    assert m["mean_sleep_fraction"] == 0.5 and m["qos_violation_rate"] == 0.5
    assert m["avoidable_violation_rate"] == 0.0 and m["reward"] == round(0.5 - 0.01 * 0.5, 4)
    worse = episode_metrics([base[0], _it("2024-01-01 12:00", 3, 4.0)], 5.0)
    assert worse["avoidable_violation_rate"] == 0.5 and worse["reward"] < m["reward"]
    cost = episode_metrics(base, 5.0, objective="cost")
    assert 0 < cost["cost_weighted_saving"] < m["mean_sleep_fraction"]


def test_rule_validator_rejects_sleeps_only_below_intent():
    def msg(tp):
        acts = [{"action": "sleep", "cell_id": 1}, {"action": "wake", "cell_id": 2}]
        return [{"role": "user", "content": f"### Proposed Actions\n```json\n{json.dumps(acts)}\n```\n\n"
                 f"### Simulation Results\navg_throughput={tp} Mbps  sleeping=3/42\n\n### Operator Intent\n5 Mbps\n"}]
    assert json.loads(RuleValidator().invoke(msg(4.2)).content)["rejected"] == [
        {"action": "sleep", "cell_id": 1, "reason": "Sim 1 throughput below intent"}]
    assert json.loads(RuleValidator().invoke(msg(6.0)).content)["rejected"] == []
    assert FallbackPlanner().invoke([]).content == "[]"


def test_mock_episode_scores_and_is_deterministic():
    a = evaluate(DEFAULT_POLICY, [5.0], 12, "energy", 5.0, 0.01)
    b = evaluate(DEFAULT_POLICY, [5.0], 12, "energy", 5.0, 0.01)
    assert a == b and 0 <= a["mean_sleep_fraction"] <= 1
    loose = evaluate(replace(DEFAULT_POLICY, prb_sleep_threshold=30.0, prb_sleep_threshold_peak=40.0),
                     [5.0], 12, "energy", 5.0, 0.01)
    assert loose["mean_sleep_fraction"] >= a["mean_sleep_fraction"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
