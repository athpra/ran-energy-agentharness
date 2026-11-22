"""
Tests for the multi-model benchmark's models file and discovery modes
(eval/benchmark.py). No LLM needed.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_benchmark.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.catalog import CATALOG_PATH  # noqa: E402
from eval.benchmark import discovery_modes, load_models  # noqa: E402
from eval.eval_discovery import HARD_CATALOG_PATH  # noqa: E402


def test_models_file_takes_ids_with_or_without_an_endpoint():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "models.yaml"
        path.write_text("models:\n  - id: org/a\n  - id: org/b\n    api_base: https://b/v1\n"
                        "    timeout: 600\n  - org/c\n")
        assert load_models(path) == [{"id": "org/a", "api_base": None, "timeout": None},
                                     {"id": "org/b", "api_base": "https://b/v1", "timeout": 600.0},
                                     {"id": "org/c", "api_base": None, "timeout": None}]
        for bad in ("models: []\n", "models:\n  - api_base: https://x/v1\n"):
            path.write_text(bad)
            try:
                load_models(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {bad!r}")


def test_discovery_modes_go_from_easiest_to_the_drafted_catalog():
    draft = Path("catalog_draft.yaml")
    assert discovery_modes(None) == [(CATALOG_PATH, False), (CATALOG_PATH, True),
                                     (HARD_CATALOG_PATH, False), (HARD_CATALOG_PATH, True)]
    assert discovery_modes(draft)[-1] == (draft, True)


def _run(model, planner_f1s, validator=(1.0, 1.0)):
    """A planner/validator result with one scenario per F1 value."""
    per = [{"scenario_id": f"S{i:02d}", "f1": f} for i, f in enumerate(planner_f1s, 1)]
    return {"model": {"model_id": model},
            "planner": {"f1_mean": sum(planner_f1s) / len(planner_f1s), "per_scenario": per},
            "validator": {"risky_rejection_rate_mean": validator[0],
                          "safe_approval_rate_mean": validator[1],
                          "per_scenario": [{"scenario_id": "V01", "risky_rejection_rate": validator[0],
                                            "safe_approval_rate": validator[1]}]}}


def test_across_runs_counts_perfect_runs_and_where_each_model_missed():
    import contextlib
    import io
    from eval.compare_models import across_runs, compare_repeats
    results = [_run("org/a", [1, 1]), _run("org/a", [1, 0]), _run("org/a", [1, 1]),
               _run("org/b", [1, 1], validator=(0.5, 1.0))]
    a, b = across_runs(results)
    assert (a["runs"], a["all_correct_runs"], a["scenarios_correct"]) == (3, "2 of 3", "2-3/3")
    assert a["planner_f1_min"] == 0.5 and a["misses"] == {"S02": 1}
    assert (b["all_correct_runs"], b["scenarios_correct"], b["misses"]) == ("0 of 1", "2/3", {"V01": 1})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        compare_repeats(results)
    assert "Across runs" in out.getvalue() and "S02 1/3" in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        compare_repeats(results[:1] + results[3:])    # one run per model: no table
    assert out.getvalue() == ""


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
