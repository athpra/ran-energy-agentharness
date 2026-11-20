"""
Tests for role readers (agent/roles.py), role bindings in the catalog and
discovery, and the harness's certified-blocker rule. No LLM or RSG needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_roles.py)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from agent.catalog import DATA_DIR, load_catalog, profile_file  # noqa: E402
from agent.discovery import resolve  # noqa: E402
from agent.harness import AgentHarness  # noqa: E402
from agent.roles import BindingError, cell_blocker  # noqa: E402
from tools.alarm_fault import query as alarm_fault_query  # noqa: E402

WINDOWS = pd.DataFrame({
    "ticket": ["T1", "T2"], "cell": [3, 5], "work": ["drive_test", "upgrade"],
    "from": ["2024-01-01 01:00:00", "2024-01-01 02:30:00"],
    "to":   ["2024-01-01 02:00:00", "2024-01-01 03:00:00"],
})
WINDOW_MAPPING = {"cell": "cell", "start": "from", "end": "to", "reason": "work"}


def _windows_file(d: Path) -> Path:
    path = d / "windows.csv"
    WINDOWS.to_csv(path, index=False)
    return path


def test_windows_block_from_start_until_end_on_any_date():
    with tempfile.TemporaryDirectory() as d:
        b = {"role": "cell_blocker", "source": str(_windows_file(Path(d))), "mapping": WINDOW_MAPPING}
        assert cell_blocker("2026-11-18 00:45", b) == {}
        assert cell_blocker("2026-11-18 01:00", b) == {3: "drive_test"}
        assert cell_blocker("2026-11-18 01:45", b) == {3: "drive_test"}
        assert cell_blocker("2026-11-18 02:00", b) == {}            # end is exclusive
        assert cell_blocker("2026-11-18 02:30", b) == {5: "upgrade"}


def test_event_shape_reads_the_fault_feed_like_the_fault_tool():
    """The fault feed is already cell_blocker data: bound by a mapping, the
    generic reader finds the same faulted cells as tools/alarm_fault.py."""
    b = {"role": "cell_blocker", "source": str(DATA_DIR / "faults.csv"),
         "mapping": {"cell": "cell_id", "time": "timestamp", "active": "active", "reason": "fault_type"}}
    for hhmm in ("06:45", "12:30", "18:00", "23:45"):
        tool = {c for c, v in alarm_fault_query(f"2026-11-18 {hhmm}").items() if v["active"]}
        assert set(cell_blocker(f"2026-11-18 {hhmm}", b)) == tool


def _catalog_with(d: Path, policy: dict) -> dict:
    path = _windows_file(d)
    entry = {"description": "work windows", "files": [path.name], "tool": None,
             "columns": profile_file(path)["columns"]}
    (d / "catalog.yaml").write_text(yaml.safe_dump({"work": entry}))
    (d / "policy.yaml").write_text(yaml.safe_dump(policy))
    return load_catalog(d / "catalog.yaml", d / "policy.yaml")


def test_only_a_certified_entry_is_bound_and_its_source_sits_next_to_the_catalog():
    with tempfile.TemporaryDirectory() as d:
        rule = {"role": "cell_blocker", "mapping": WINDOW_MAPPING}
        assert _catalog_with(Path(d), {"work": {"certified": False, **rule}})["work"]["binding"] is None
        bound = _catalog_with(Path(d), {"work": {"certified": True, **rule}})["work"]["binding"]
        assert bound["role"] == "cell_blocker" and bound["source"] == str(Path(d) / "windows.csv")


def test_a_binding_that_cannot_be_read_is_refused():
    with tempfile.TemporaryDirectory() as d:
        for rule, why in [
            ({"role": "cell_sleeper", "mapping": WINDOW_MAPPING}, "unknown role"),
            ({"role": "cell_blocker", "mapping": {"cell": "cell", "start": "from"}}, "fits no"),
            ({"role": "cell_blocker", "mapping": {**WINDOW_MAPPING, "end": "until"}}, "not in the data"),
        ]:
            try:
                _catalog_with(Path(d), {"work": {"certified": True, **rule}})
            except BindingError as e:
                assert why in str(e)
            else:
                raise AssertionError(f"accepted {rule}")


def test_resolve_carries_the_bindings_of_chosen_datasets_only():
    binding = {"role": "cell_blocker", "source": "x.csv", "mapping": WINDOW_MAPPING}
    catalog = {"faults": {"tool": "alarm_fault", "safety_critical": True},
               "work":   {"tool": None, "binding": binding},
               "other":  {"tool": None, "binding": None}}
    assert resolve([{"dataset": "work", "reason": ""}], catalog).bindings == {"work": binding}
    assert resolve([{"dataset": "other", "reason": ""}], catalog).bindings == {}


class _Reply:
    def __init__(self, content):
        self.content = content


class _Planner:
    """Proposes to sleep every awake cell."""
    model_name = "stub-planner"

    def __init__(self, cells):
        self.cells = cells

    def invoke(self, messages):
        return _Reply(json.dumps([{"action": "sleep", "cell_id": c, "reason": "low load"} for c in self.cells]))


class _ApproveAll:
    model_name = "stub-validator"

    def invoke(self, messages):
        return _Reply('{"approved": [], "rejected": []}')


def test_harness_rule_blocks_sleep_and_wakes_cells_in_a_window_whatever_the_llms_say():
    applied = []
    sim = lambda actions: ("avg_throughput=9.00 Mbps  sleeping=0/42", {})  # noqa: E731

    def apply(actions):
        applied.extend(actions)
        return sim(actions)

    with tempfile.TemporaryDirectory() as d:
        b = {"role": "cell_blocker", "source": str(_windows_file(Path(d))), "mapping": WINDOW_MAPPING}
        harness = AgentHarness(llm=_Planner([1, 3]), sim_test_fn=sim, sim_apply_fn=apply,
                               operator_intent="5 Mbps", enabled_tools=frozenset(),
                               validator_llm=_ApproveAll(), bindings={"work": b})
        r = harness.run_iteration(1, pd.Timestamp("2026-11-18 02:30"), "kpis", list(range(6)),
                                  {c: 5.0 for c in range(6)}, sleeping_cell_ids=[5],
                                  current_throughput_mbps=9.0)
    # cell 3's window is over at 02:30, cell 5's is open: 5 is woken, 1 and 3 may sleep
    assert sorted((a["action"], a["cell_id"]) for a in applied) == [("sleep", 1), ("sleep", 3), ("wake", 5)]
    assert r.tool_context["tool_blocks"]["role_blocked"] == {"work": [5]}

    with tempfile.TemporaryDirectory() as d:
        b = {"role": "cell_blocker", "source": str(_windows_file(Path(d))), "mapping": WINDOW_MAPPING}
        harness = AgentHarness(llm=_Planner([1, 3]), sim_test_fn=sim, sim_apply_fn=sim,
                               operator_intent="5 Mbps", enabled_tools=frozenset(),
                               validator_llm=_ApproveAll(), bindings={"work": b})
        r = harness.run_iteration(1, pd.Timestamp("2026-11-18 01:30"), "kpis", list(range(6)),
                                  {c: 5.0 for c in range(6)}, sleeping_cell_ids=[],
                                  current_throughput_mbps=9.0)
    assert [a["cell_id"] for a in r.approved_actions] == [1]
    # cell 3 is out of the harness's sleep candidates, so the guidance rule rejects it first
    assert r.rejected_actions[0]["cell_id"] == 3
    assert r.rejected_actions[0]["reason"].startswith(("guidance rule", "certified rule"))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
