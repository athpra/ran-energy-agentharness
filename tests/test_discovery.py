"""
Tests for the data catalog and the discovery step. No LLM or RSG needed:
a stub model returns canned selections.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_discovery.py)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from agent.catalog import DATA_DIR, POLICY_PATH, catalog_prompt, load_catalog  # noqa: E402
from agent.discovery import _parse_selection, discover, resolve  # noqa: E402
from generators import generate_catalog, generate_maintenance, profile_catalog  # noqa: E402

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
    generate_maintenance.generate(tmp / "historical_kpi.csv").to_csv(tmp / "maintenance_windows.csv", index=False)
    entries = generate_catalog.generate(tmp, seed=42)
    path = tmp / "catalog.yaml"
    generate_catalog.write_catalog(entries, path, header="test")
    return path


def test_generated_catalog_maps_every_tool_and_adds_distractors():
    with tempfile.TemporaryDirectory() as d:
        catalog = load_catalog(_generated_catalog(Path(d)))
    tools = {e["tool"] for e in catalog.values()} - {None}
    assert tools == {"historical_kpi", "traffic_forecast", "alarm_fault", "interference", "energy_pricing"}
    assert [n for n, e in catalog.items() if e["tool"] is None and not e["binding"]] == [
        "subscriber_billing", "site_inventory", "legacy_alarm_archive"]
    assert catalog["maintenance_windows"]["binding"]["role"] == "cell_blocker"
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


def test_hard_catalog_adds_unusable_near_misses_next_to_their_twins():
    from eval.eval_discovery import relevant_datasets
    with tempfile.TemporaryDirectory() as d:
        path = _generated_catalog(Path(d))
        entries = yaml.safe_load(path.read_text())
        hard = generate_catalog.generate_hard(entries, Path(d), seed=42)
        generate_catalog.write_catalog(hard, Path(d) / "catalog_hard.yaml", header="test")
        easy, hard = load_catalog(path), load_catalog(Path(d) / "catalog_hard.yaml")
    names = list(hard)
    for name, spec in generate_catalog.HARD_DISTRACTORS.items():
        assert names.index(name) == names.index(spec["after"]) + 1
        assert hard[name]["tool"] is None and hard[name]["binding"] is None
    assert relevant_datasets(hard) == relevant_datasets(easy)
    assert set(hard["fault_feed_lab"]["columns"]) == set(hard["faults"]["columns"])


def test_eval_keeps_raw_responses_and_counts_unparsed_ones_apart_from_empty_choices():
    from eval.eval_discovery import run_discovery_eval
    catalog = {"faults": {"tool": "alarm_fault", "safety_critical": True},
               "energy_pricing": {"tool": "energy_pricing"},
               "site_inventory": {"tool": None}}

    class Replies(StubLLM):
        def __init__(self, replies):
            self.replies = iter(replies)

        def invoke(self, messages):
            return type("Reply", (), {"content": next(self.replies)})()

    llm = Replies(['<think>long</think>{"selected": [{"dataset": "faults", "reason": "safety"}, '
                   '{"dataset": "site_inventory", "reason": "cells"}]}',
                   '<think>ran out of tokens', '{"selected": []}'])
    out = run_discovery_eval(catalog, {"faults"}, llm, "5 Mbps", 3, verbose=False)
    assert [r["parse_failed"] for r in out["runs"]] == [False, True, False]
    assert out["runs"][1]["raw"] == "<think>ran out of tokens"
    assert out["runs"][0]["precision"] == 0.5 and out["runs"][0]["recall"] == 0.5
    assert out["parse_failure_rate"] == 0.333 and out["distractor_rate"] == 0.333
    assert out["safety_critical_miss_rate"] == 0.667


def test_compare_models_prints_a_discovery_row_per_model_and_mode():
    import contextlib
    import io
    from eval.compare_models import compare
    run = {"missed": ["historical_kpi"], "distractors_chosen": ["fault_feed_lab"]}
    files = []
    with tempfile.TemporaryDirectory() as d:
        for model, policy in [("org/model-b", True), ("org/model-a", False), ("org/model-a", True)]:
            path = Path(d) / f"{model.split('/')[-1]}_{policy}.json"
            path.write_text(json.dumps({"model_id": model, "catalog_name": "catalog_hard", "policy": policy,
                                        "runs": [run, run], "mean_precision": 0.8, "mean_recall": 0.9,
                                        "safety_critical_miss_rate": 0.0, "distractor_rate": 1.0,
                                        "parse_failure_rate": 0.0}))
            files.append(path)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare(files)
    rows = [l for l in out.getvalue().splitlines() if l.startswith("model-")]
    assert [r.split()[:3] for r in rows] == [["model-a", "catalog_hard", "on"], ["model-a", "catalog_hard", "off"],
                                            ["model-b", "catalog_hard", "on"]]
    assert "historical_kpi×2" in rows[0] and "fault_feed_lab×2" in rows[0]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
