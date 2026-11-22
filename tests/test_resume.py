"""
Tests for resuming a run that stopped early (experiments/common.load_resume,
make_sim_fns(replay_actions=...), and run(..., resume=...)). Uses the mock RSG
and the policy tuner's LLM stand-ins: no VIAVI or LLM endpoint needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_resume.py)
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import PROJECT_ROOT, load_resume, replay_sleep_state  # noqa: E402

CONFIG = {"condition": "full_harness_vtzcycle__test", "intent": "5 Mbps", "iterations": 12}


def _write_run(d: Path, iterations: list[dict], config: dict = CONFIG) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_config.json").write_text(json.dumps(config))
    (d / "iterations.jsonl").write_text("".join(json.dumps(i) + "\n" for i in iterations))
    return d


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_replay_applies_actions_in_order():
    state = {}
    replay_sleep_state(state, {0: "A/N1/1", 1: "B/N1/1"}, [
        {"action": "sleep", "cell_id": 0}, {"action": "sleep", "cell_id": 1},
        {"action": "wake", "cell_id": 0}])
    assert state == {"A/N1/1": False, "B/N1/1": True}


def test_resume_refuses_a_run_it_cannot_continue():
    its = [{"iteration": 1, "timestamp": "2026-11-21 00:00:00", "approved_actions": []},
           {"iteration": 2, "timestamp": "2026-11-21 00:15:00", "approved_actions": []}]
    with tempfile.TemporaryDirectory() as d:
        for run_its, expected, why in [
            (its, {"intent": "4 Mbps"}, "intent is '5 Mbps'"),
            (its[1:], {}, "not 1..1 in order"),
            (its, {"iterations": 12}, None),
        ]:
            run_dir = _write_run(Path(d) / "run", run_its)
            try:
                r = _quiet(load_resume, run_dir, expected)
            except SystemExit as e:
                assert why and why in str(e), (why, e)
            else:
                assert why is None and r.done == 2 and str(r.base_ts) == "2026-11-21 00:00:00"
        done = _write_run(Path(d) / "done", its, {**CONFIG, "iterations": 2})
        try:
            _quiet(load_resume, done, {})
        except SystemExit as e:
            assert "already complete" in str(e)
        else:
            raise AssertionError("resumed a complete run")


def test_a_resumed_run_continues_exactly_where_the_full_run_went():
    """Run 12 iterations; cut a copy to its first 6 and resume it: iterations
    7-12 match the uninterrupted run's, sleep state included."""
    import os
    os.environ["USE_MOCK_RSG"] = "1"
    from mock_rsg.scenario import Scenario as MockScenario
    from experiments.full_harness_vtzcycle import run
    from experiments.tune_policy import FallbackPlanner, RuleValidator

    conf = str(PROJECT_ROOT / "ai_rsg_config" / "config.conf")

    def go(run_dir, resume=None):
        _quiet(run, MockScenario(conf, "mock://"), "5 Mbps", 12, run_dir=run_dir,
               llm=FallbackPlanner(), validator_llm=RuleValidator(), verbose=False, resume=resume)
        return [json.loads(line) for line in open(run_dir / "iterations.jsonl")]

    def key(it):
        return (it["iteration"], it["timestamp"], it["pre_kpis"]["sleeping_cells"],
                sorted((a["action"], a["cell_id"]) for a in it["approved_actions"]),
                it["post_kpis"].get("sleeping_cells"), it["post_kpis"].get("avg_throughput_mbps"))

    with tempfile.TemporaryDirectory() as d:
        full = go(_write_run(Path(d) / "full", []))
        assert any(it["approved_actions"] for it in full[:6]), "nothing to replay"
        cut = _write_run(Path(d) / "cut", full[:6])
        resumed = go(cut, _quiet(load_resume, cut, {"intent": "5 Mbps", "iterations": 12}))
        config = json.loads((cut / "run_config.json").read_text())
    assert [key(it) for it in resumed] == [key(it) for it in full]
    assert config["resumed"][0]["at_iteration"] == 7


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
