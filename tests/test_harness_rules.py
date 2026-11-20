"""
Tests for the harness's guidance rule (agent/harness.py): with the
pre-computed guidance in the tool context, the Planner may sleep only the
harness's sleep candidates, and the wakes the harness asks for are added if
the Planner's answer leaves them out. No LLM or RSG needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_harness_rules.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

import agent.harness as harness_mod  # noqa: E402
from agent.harness import AgentHarness  # noqa: E402


class _Reply:
    def __init__(self, content):
        self.content = content


class _LLM:
    model_name = "stub"

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return _Reply(self.reply)


def _run(planner_actions, ctx, sleeping=(), wake_cands=None):
    """One iteration with assemble() replaced by a fixed tool context."""
    applied = []
    sim = lambda actions: ("avg_throughput=9.00 Mbps  sleeping=0/42", {})  # noqa: E731

    def apply(actions):
        applied.extend(actions)
        return sim(actions)

    h = AgentHarness(llm=_LLM(json.dumps(planner_actions)), sim_test_fn=sim, sim_apply_fn=apply,
                     operator_intent="5 Mbps", validator_llm=_LLM('{"approved": [], "rejected": []}'))
    with mock.patch.object(harness_mod, "assemble", return_value=(ctx, "### Pre-computed Action Guidance\n")):
        r = h.run_iteration(1, pd.Timestamp("2026-11-20 08:00"), "kpis", list(range(6)),
                            {c: 5.0 for c in range(6)}, sleeping_cell_ids=list(sleeping),
                            static_wake_candidates=wake_cands, current_throughput_mbps=9.0)
    return r, sorted((a["action"], a["cell_id"]) for a in applied)


def test_planner_sleeps_outside_the_candidates_are_rejected_and_its_subset_kept():
    ctx = {"sleep_candidates": [0, 1, 2], "wake_candidates": []}
    r, applied = _run([{"action": "sleep", "cell_id": 0}, {"action": "sleep", "cell_id": 4}], ctx)
    assert applied == [("sleep", 0)]                      # its own subset, not all candidates
    assert r.rejected_actions == [{"action": "sleep", "cell_id": 4,
                                   "reason": "guidance rule: not in the harness's sleep candidates"}]


def test_surge_wakes_the_planner_left_out_are_added():
    ctx = {"sleep_candidates": [], "wake_candidates": [2, 3]}
    _, applied = _run([{"action": "wake", "cell_id": 2}], ctx, sleeping=[2, 3])
    assert applied == [("wake", 2), ("wake", 3)]


def test_qos_wakes_are_added_only_when_the_planner_sleeps_nothing():
    ctx = {"sleep_candidates": [0], "wake_candidates": []}
    _, applied = _run([{"action": "wake", "cell_id": 5}], ctx, sleeping=[3, 5], wake_cands=[3, 5])
    assert applied == [("wake", 3), ("wake", 5)]
    _, applied = _run([{"action": "sleep", "cell_id": 0}], ctx, sleeping=[3], wake_cands=[3])
    assert applied == [("sleep", 0)]


def test_an_empty_or_fully_rejected_answer_falls_back_as_before():
    ctx = {"sleep_candidates": [0, 1], "wake_candidates": []}
    for answer in ([], [{"action": "sleep", "cell_id": 5}]):
        r, applied = _run(answer, ctx)
        assert applied == [("sleep", 0), ("sleep", 1)] and r.planner_raw.startswith("[fallback]")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
