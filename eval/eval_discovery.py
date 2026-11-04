"""
Discovery eval.

Runs the discovery step several times against the data catalog and scores the
choices: a dataset is relevant when a harness tool reads it, and a distractor
when none does. Needs only the LLM endpoint, not the VIAVI RSG.

Usage:
    python -m eval.eval_discovery                  # certified catalog, 10 runs
    python -m eval.eval_discovery --no-policy      # descriptions only, nothing certified
    python -m eval.eval_discovery --catalog data/synthetic/catalog_draft.yaml

Writes eval/results/discovery_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv

from agent.catalog import CATALOG_PATH, POLICY_PATH, load_catalog
from agent.discovery import discover


def score(selected: list[str], catalog: dict[str, dict], critical: set[str]) -> dict:
    """critical comes from the policy even when discovery ran without it, so a
    missed fault feed is counted in every mode."""
    relevant = {n for n, e in catalog.items() if e.get("tool")}
    chosen   = set(selected)
    hits     = chosen & relevant
    return {
        "precision":             round(len(hits) / len(chosen), 3) if chosen else 0.0,
        "recall":                round(len(hits) / len(relevant), 3) if relevant else 0.0,
        "missed":                sorted(relevant - chosen),
        "distractors_chosen":    sorted(chosen - relevant),
        "missed_safety_critical": sorted((critical & set(catalog)) - chosen),
    }


def run_discovery_eval(catalog: dict[str, dict], critical: set[str], llm, intent: str, runs: int,
                       verbose: bool = True) -> dict:
    results = []
    for i in range(runs):
        res = discover(llm, intent, catalog)
        s = score(res.selected, catalog, critical)
        results.append({**s, "selected": res.selected, "reasons": res.reasons,
                        "unknown": res.unknown, "elapsed_s": round(res.elapsed_s, 2)})
        if verbose:
            print(f"  run {i + 1:>2}: precision={s['precision']:.2f}  recall={s['recall']:.2f}  "
                  f"missed={s['missed']}  distractors={s['distractors_chosen']}")
    n = len(results) or 1
    return {
        "runs":                    results,
        "mean_precision":          round(sum(r["precision"] for r in results) / n, 3),
        "mean_recall":             round(sum(r["recall"] for r in results) / n, 3),
        "safety_critical_miss_rate": round(sum(bool(r["missed_safety_critical"]) for r in results) / n, 3),
        "distractor_rate":         round(sum(bool(r["distractors_chosen"]) for r in results) / n, 3),
    }


def main():
    load_dotenv(find_dotenv(), override=True)
    parser = argparse.ArgumentParser(description="Score data discovery against the catalog")
    parser.add_argument("--catalog",   type=Path, default=CATALOG_PATH)
    parser.add_argument("--no-policy", action="store_true",
                        help="Ignore data/catalog_policy.yaml: nothing certified, no decision_use shown")
    parser.add_argument("--runs",      type=int, default=10)
    parser.add_argument("--intent",    default="5 Mbps")
    parser.add_argument("--model",     default=None)
    parser.add_argument("--output",    type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    from experiments.common import get_llm
    llm = get_llm(args.model)
    catalog = load_catalog(args.catalog, None if args.no_policy else POLICY_PATH)
    critical = {n for n, e in load_catalog(args.catalog).items() if e["safety_critical"]}

    print(f"Discovery eval: {args.runs} runs, catalog={args.catalog}, "
          f"policy={'off' if args.no_policy else 'on'}, model={llm.model_name}")
    summary = run_discovery_eval(catalog, critical, llm, args.intent, args.runs)
    print(f"\nmean precision={summary['mean_precision']:.2f}  mean recall={summary['mean_recall']:.2f}  "
          f"safety-critical miss rate={summary['safety_critical_miss_rate']:.0%}  "
          f"runs choosing a distractor={summary['distractor_rate']:.0%}")

    now = datetime.now(timezone.utc)
    out = args.output / f"discovery_{now.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": now.isoformat(),
        "model_id":  llm.model_name,
        "catalog":   str(args.catalog),
        "policy":    not args.no_policy,
        "intent":    args.intent,
        **summary,
    }, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
