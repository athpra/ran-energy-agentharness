"""
Model comparison tool.

Loads two or more eval result JSON files and prints a side-by-side
comparison table across models.

Usage:
    python -m eval.compare_models eval/results/eval_*.json

    # or compare specific files:
    python -m eval.compare_models \
        eval/results/eval_20260920_120000_Qwen2.5-7B-Instruct.json \
        eval/results/eval_20260920_130000_Nemotron-70B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {p}")
    return json.loads(p.read_text())


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _print_table(rows: list[dict], headers: list[str]) -> None:
    widths = [max(len(h), max(len(_fmt(r.get(h, "—"))) for r in rows)) for h in headers]
    sep    = "  ".join("-" * w for w in widths)
    header = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(_fmt(row.get(h, "—")).ljust(w) for h, w in zip(headers, widths)))


def compare(paths: list[str | Path], show_per_scenario: bool = False) -> None:
    results = [load_result(p) for p in paths]

    print("\n═══ Model Metadata ═══")
    meta_headers = ["model_id", "api_base", "temperature", "eval_run_id", "timestamp"]
    meta_rows = []
    for r in results:
        m = r.get("model", {})
        meta_rows.append({
            "model_id":    m.get("model_id", "?"),
            "api_base":    m.get("api_base", "?")[:50],
            "temperature": m.get("temperature"),
            "eval_run_id": r.get("eval_run_id", "?"),
            "timestamp":   r.get("timestamp", "?")[:19],
        })
    _print_table(meta_rows, meta_headers)

    # ── Planner aggregate ─────────────────────────────────────────────────────
    planner_keys = [
        "precision_mean", "recall_mean", "f1_mean",
        "format_valid_rate", "no_hallucinations_rate",
    ]
    planner_rows = []
    for r in results:
        m = r.get("model", {})
        p = r.get("planner", {})
        row = {"model": m.get("model_id", "?")[:30]}
        for k in planner_keys:
            row[k] = p.get(k)
        planner_rows.append(row)

    if any(r.get("planner") for r in results):
        print("\n═══ Planner Scores ═══")
        _print_table(planner_rows, ["model"] + planner_keys)

    # ── Validator aggregate ───────────────────────────────────────────────────
    validator_keys = ["risky_rejection_rate_mean", "safe_approval_rate_mean", "format_valid_rate"]
    validator_rows = []
    for r in results:
        m = r.get("model", {})
        v = r.get("validator", {})
        row = {"model": m.get("model_id", "?")[:30]}
        for k in validator_keys:
            row[k] = v.get(k)
        validator_rows.append(row)

    if any(r.get("validator") for r in results):
        print("\n═══ Validator Scores ═══")
        _print_table(validator_rows, ["model"] + validator_keys)

    # ── Per-scenario breakdown ────────────────────────────────────────────────
    if show_per_scenario:
        print("\n═══ Per-Scenario Planner F1 ═══")
        # Gather all scenario IDs
        all_ids = []
        for r in results:
            for s in r.get("planner", {}).get("per_scenario", []):
                if s["scenario_id"] not in all_ids:
                    all_ids.append(s["scenario_id"])

        model_names = [r["model"]["model_id"][:20] for r in results]
        headers     = ["scenario_id"] + model_names
        rows = []
        for sid in all_ids:
            row = {"scenario_id": sid}
            for r in results:
                m = r["model"]["model_id"][:20]
                per = {s["scenario_id"]: s for s in r.get("planner", {}).get("per_scenario", [])}
                row[m] = per[sid]["f1"] if sid in per else None
            rows.append(row)
        _print_table(rows, headers)

        print("\n═══ Per-Scenario Validator risky_rejection_rate ═══")
        all_val_ids = []
        for r in results:
            for s in r.get("validator", {}).get("per_scenario", []):
                if s["scenario_id"] not in all_val_ids:
                    all_val_ids.append(s["scenario_id"])

        rows = []
        for sid in all_val_ids:
            row = {"scenario_id": sid}
            for r in results:
                m = r["model"]["model_id"][:20]
                per = {s["scenario_id"]: s for s in r.get("validator", {}).get("per_scenario", [])}
                row[m] = per[sid]["risky_rejection_rate"] if sid in per else None
            rows.append(row)
        _print_table(rows, ["scenario_id"] + model_names)

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare eval results across models")
    parser.add_argument("files", nargs="+", help="Paths to eval result JSON files")
    parser.add_argument("--per-scenario", action="store_true",
                        help="Show per-scenario breakdown in addition to aggregates")
    args = parser.parse_args()

    if len(args.files) < 1:
        parser.error("Provide at least one result JSON file.")

    compare(args.files, show_per_scenario=args.per_scenario)


if __name__ == "__main__":
    main()
