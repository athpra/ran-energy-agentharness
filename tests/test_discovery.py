"""
Tests for the data catalog and the discovery step. No LLM or RSG needed:
a stub model returns canned selections.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_discovery.py)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from agent.catalog import DATA_DIR, POLICY_PATH, catalog_prompt, load_catalog  # noqa: E402
from agent.discovery import _parse_selection, discover, resolve  # noqa: E402
from generators import generate_catalog, profile_catalog  # noqa: E402

SOURCES = ["historical_kpi.csv", "traffic_forecast.csv", "faults.csv",
           "neighbor_graph.json", "load_shift_matrix.csv", "energy_pricing.csv"]


class StubLLM:
    model_name = "stub"

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, messages):
        return type("Reply", (), {"content": self.reply})()


def _generated_catalog(tmp: Path) -> Path:
    for f in SOURCES:
        shutil.copy(DATA_DIR / f, tmp / f)
    entries = generate_catalog.generate(tmp, seed=42)
    path = tmp / "catalog.yaml"
    generate_catalog.write_catalog(entries, path, header="test")
    return path


def test_generated_catalog_maps_every_tool_and_adds_distractors():
    with tempfile.TemporaryDirectory() as d:
        catalog = load_catalog(_generated_catalog(Path(d)))
    tools = {e["tool"] for e in catalog.values()} - {None}
    assert tools == {"historical_kpi", "traffic_forecast", "alarm_fault", "interference", "energy_pricing"}
    assert [n for n, e in catalog.items() if e["tool"] is None] == [
        "subscriber_billing", "site_inventory", "legacy_alarm_archive"]
    severity = catalog["faults"]["columns"]["severity"]
    assert severity["type"] == "text" and severity["meaning"] == "fault severity" and severity["values"]
    assert catalog["faults"]["columns"]["cell_id"] == {"meaning": "cell index", "type": "number", "min": 0, "max": 20}


def test_generation_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pa, pb = _generated_catalog(Path(a)), _generated_catalog(Path(b))
        assert pa.read_text() == pb.read_text()
        assert (Path(a) / "legacy_alarm_archive.csv").read_text() == (Path(b) / "legacy_alarm_archive.csv").read_text()


def test_policy_certifies_and_only_certified_can_be_safety_critical():
    with tempfile.TemporaryDirectory() as d:
        path = _generated_catalog(Path(d))
        policy = Path(d) / "policy.yaml"
        policy.write_text(yaml.safe_dump({
            "faults": {"certified": True, "safety_critical": True, "decision_use": "never sleep a faulted cell"},
            "site_inventory": {"certified": False, "safety_critical": True},
        }))
        catalog = load_catalog(path, policy)
    assert catalog["faults"]["safety_critical"] and catalog["faults"]["certified"]
    assert not catalog["site_inventory"]["safety_critical"]
    assert not catalog["energy_pricing"]["certified"]


def test_prompt_hides_tool_and_safety_flag_and_uncertified_decision_use():
    catalog = {
        "faults": {"description": "fault feed", "tool": "alarm_fault", "certified": True,
                   "safety_critical": True, "decision_use": "never sleep a faulted cell"},
        "drafty": {"description": "draft", "tool": None, "certified": False, "decision_use": "guess"},
    }
    text = catalog_prompt(catalog)
    assert "alarm_fault" not in text and "safety" not in text.lower()
    assert "never sleep a faulted cell" in text and "guess" not in text


def test_parse_selection_tolerates_reasoning_and_bare_lists():
    raw = '<think>maybe {"selected": []}</think>\n{"selected": [{"dataset": "faults", "reason": "safety"}]}'
    assert _parse_selection(raw) == [{"dataset": "faults", "reason": "safety"}]
    assert _parse_selection('["faults", "energy_pricing"]')[1]["dataset"] == "energy_pricing"
    assert _parse_selection("no json here") == []


def test_resolve_maps_tools_adds_missed_safety_and_ignores_unknown():
    catalog = {
        "faults":         {"tool": "alarm_fault", "safety_critical": True},
        "energy_pricing": {"tool": "energy_pricing", "safety_critical": False},
        "site_inventory": {"tool": None, "safety_critical": False},
    }
    res = resolve([{"dataset": "energy_pricing", "reason": ""}, {"dataset": "site_inventory", "reason": ""},
                   {"dataset": "weather", "reason": ""}], catalog)
    assert res.enabled_tools == frozenset({"energy_pricing", "alarm_fault"})
    assert res.safety_added == ["faults"] and res.unknown == ["weather"]
    assert resolve([], catalog, enforce_safety=False).enabled_tools == frozenset()


def test_discover_with_real_policy_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        catalog = load_catalog(_generated_catalog(Path(d)), POLICY_PATH)
    llm = StubLLM('{"selected": [{"dataset": "historical_kpi", "reason": "load"}, '
                  '{"dataset": "legacy_alarm_archive", "reason": "alarms"}]}')
    res = discover(llm, "5 Mbps", catalog)
    assert res.selected == ["historical_kpi", "legacy_alarm_archive"]
    assert res.safety_added == ["faults"]
    assert res.enabled_tools == frozenset({"historical_kpi", "alarm_fault"})
    assert res.to_dict()["enabled_tools"] == ["alarm_fault", "historical_kpi"]


def test_profile_catalog_without_llm():
    with tempfile.TemporaryDirectory() as d:
        _generated_catalog(Path(d))
        draft = profile_catalog.draft_catalog(Path(d))
    assert "catalog" not in draft
    assert draft["faults"]["tool"] == "alarm_fault" and draft["subscriber_billing"]["tool"] is None
    assert draft["neighbor_graph"]["structure"] == {"neighbor_graph.json": "object with 21 keys"}


def test_profile_catalog_uses_llm_draft():
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(DATA_DIR / "faults.csv", Path(d) / "faults.csv")
        llm = StubLLM('{"description": "Fault events per cell.", "concepts": ["fault"], '
                      '"columns": {"severity": "how bad the fault is"}}')
        entry = profile_catalog.draft_catalog(Path(d), llm)["faults"]
    assert entry["description"] == "Fault events per cell." and entry["drafted_by"] == "stub"
    assert entry["columns"]["severity"]["meaning"] == "how bad the fault is"
    assert entry["columns"]["cell_id"]["meaning"] == ""


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
