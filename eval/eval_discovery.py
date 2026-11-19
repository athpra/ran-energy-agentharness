"""
Discovery eval.

Runs the discovery step several times against the data catalog and scores the
choices: a dataset is relevant when a harness tool or a certified role reader
(agent/roles.py) reads it, and a distractor when neither does. Recall counts
capabilities (tools and role readers) covered, so two files that feed one tool
count once. Needs only the
LLM endpoint, not the VIAVI RSG.

To benchmark models, run each one in all four modes and compare the files:
    python -m eval.eval_discovery --model <id> --api-base <url>    # certified catalog
    python -m eval.eval_discovery --model <id> --no-policy         # descriptions only
    python -m eval.eval_discovery --model <id> --hard              # + near-miss distractors
    python -m eval.eval_discovery --model <id> --hard --no-policy  # hardest
    python -m eval.compare_models eval/results/discovery_*.json

--no-policy hides the certified "Decision use" lines, so the model judges from
descriptions and columns alone. --hard uses data/synthetic/catalog_hard.yaml
(generators/generate_catalog.py). Every run keeps the raw response, so a
response that could not be parsed (parse_failed) is not mistaken for a wrong
choice.

Writes eval/results/discovery_<timestamp>_<model>_<catalog>_<policy>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv

from agent.catalog import CATALOG_PATH, DATA_DIR, POLICY_PATH, load_catalog
from agent.discovery import _parse_selection, discover

HARD_CATALOG_PATH = DATA_DIR / "catalog_hard.yaml"


def relevant_datasets(catalog: dict[str, dict]) -> dict[str, str]:
    """{dataset: the capability it provides}: its tool, or its role reader.
    Several datasets can provide one capability (neighbor_graph and
    load_shift_matrix both feed the interference tool in a drafted catalog)."""
    return {n: e["tool"] if e.get("tool") else f"{e['binding']['role']}:{n}"
            for n, e in catalog.items() if e.get("tool") or e.get("binding")}


def score(selected: list[str], catalog: dict[str, dict], critical: set[str],
          relevant: dict[str, str] | None = None) -> dict:
    """Recall counts capabilities, not files: a capability is covered when any
    dataset providing it is chosen, so choosing one of two interference files
    loses nothing. Precision counts chosen datasets. critical and relevant
    come from the policy even when discovery ran without it, so a missed fault
    feed or maintenance schedule is counted in every mode."""
    relevant = relevant_datasets(catalog) if relevant is None else relevant
    relevant = {n: c for n, c in relevant.items() if n in catalog}
    capabilities = set(relevant.values())
    chosen   = set(selected)
    covered  = {relevant[n] for n in chosen if n in relevant}
    return {
        "precision":             round(len(chosen & set(relevant)) / len(chosen), 3) if chosen else 0.0,
        "recall":                round(len(covered) / len(capabilities), 3) if capabilities else 0.0,
        "missed":                sorted(n for n, c in relevant.items() if c not in covered),
        "distractors_chosen":    sorted(chosen - set(relevant)),
        "missed_safety_critical": sorted((critical & set(catalog)) - chosen),
    }


def run_discovery_eval(catalog: dict[str, dict], critical: set[str], llm, intent: str, runs: int,
                       verbose: bool = True, relevant: dict[str, str] | None = None) -> dict:
    results = []
    for i in range(runs):
        res = discover(llm, intent, catalog)
        s = score(res.selected, catalog, critical, relevant)
        # an explicit empty selection is a choice, not a parse failure
        parse_failed = (not _parse_selection(res.raw)
                        and not re.search(r'"selected"\s*:\s*\[\s*\]', res.raw or ""))
        results.append({**s, "selected": res.selected, "reasons": res.reasons,
                        "unknown": res.unknown, "parse_failed": parse_failed,
                        "elapsed_s": round(res.elapsed_s, 2), "raw": res.raw})
        if verbose:
            print(f"  run {i + 1:>2}: precision={s['precision']:.2f}  recall={s['recall']:.2f}  "
                  f"missed={s['missed']}  distractors={s['distractors_chosen']}"
                  + ("  ⚠ response not parsed" if parse_failed else ""))
    n = len(results) or 1
    return {
        "runs":                    results,
        "mean_precision":          round(sum(r["precision"] for r in results) / n, 3),
        "mean_recall":             round(sum(r["recall"] for r in results) / n, 3),
        "safety_critical_miss_rate": round(sum(bool(r["missed_safety_critical"]) for r in results) / n, 3),
        "distractor_rate":         round(sum(bool(r["distractors_chosen"]) for r in results) / n, 3),
        "parse_failure_rate":      round(sum(r["parse_failed"] for r in results) / n, 3),
    }


def main():
    load_dotenv(find_dotenv(), override=True)
    parser = argparse.ArgumentParser(description="Score data discovery against the catalog")
    parser.add_argument("--catalog",   type=Path, default=None,
                        help=f"Catalog file (default: {CATALOG_PATH.name}, or {HARD_CATALOG_PATH.name} with --hard)")
    parser.add_argument("--hard",      action="store_true",
                        help="Use the catalog with near-miss distractors (data/synthetic/catalog_hard.yaml)")
    parser.add_argument("--no-policy", action="store_true",
                        help="Ignore data/catalog_policy.yaml: nothing certified, no decision_use shown")
    parser.add_argument("--runs",      type=int, default=10)
    parser.add_argument("--intent",    default="5 Mbps")
    parser.add_argument("--model",     default=None, help="Model ID (default: LLM_MODEL in .env)")
    parser.add_argument("--api-base",  default=None,
                        help="Endpoint URL serving --model (default: CDSW_API_URL in .env). On Cloudera "
                             "AI Inference each model usually has its own endpoint URL")
    parser.add_argument("--output",    type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    if args.catalog is None:
        args.catalog = HARD_CATALOG_PATH if args.hard else CATALOG_PATH
    if not args.catalog.exists():
        sys.exit(f"No catalog at {args.catalog}. Run python generators/generate_catalog.py first.")

    from experiments.common import get_llm
    evaluate(get_llm(args.model, args.api_base), args.catalog, args.no_policy, args.runs,
             args.intent, args.output)


def evaluate(llm, catalog_path: Path, no_policy: bool, runs: int, intent: str = "5 Mbps",
             output: Path = Path("eval/results"), verbose: bool = True) -> Path:
    """One model on one catalog and policy mode; returns the results file."""
    catalog_path = Path(catalog_path)
    catalog = load_catalog(catalog_path, None if no_policy else POLICY_PATH)
    certified = load_catalog(catalog_path)
    critical = {n for n, e in certified.items() if e["safety_critical"]}
    relevant = relevant_datasets(certified)

    print(f"Discovery eval: {runs} runs, catalog={catalog_path}, "
          f"policy={'off' if no_policy else 'on'}, model={llm.model_name}")
    summary = run_discovery_eval(catalog, critical, llm, intent, runs, verbose=verbose, relevant=relevant)
    print(f"\nmean precision={summary['mean_precision']:.2f}  mean recall={summary['mean_recall']:.2f}  "
          f"safety-critical miss rate={summary['safety_critical_miss_rate']:.0%}  "
          f"runs choosing a distractor={summary['distractor_rate']:.0%}  "
          f"unparsed responses={summary['parse_failure_rate']:.0%}")

    now = datetime.now(timezone.utc)
    model_slug = llm.model_name.split("/")[-1] or "llm"
    out = Path(output) / (f"discovery_{now.strftime('%Y%m%d_%H%M%S')}_{model_slug}_"
                          f"{catalog_path.stem}_{'nopolicy' if no_policy else 'policy'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": now.isoformat(),
        "model_id":  llm.model_name,
        "api_base":  str(getattr(llm, "openai_api_base", "") or ""),
        "catalog":   str(catalog_path),
        "catalog_name": catalog_path.stem,
        "runs_requested": runs,
        "temperature": getattr(llm, "temperature", None),
        "policy":    not no_policy,
        "intent":    intent,
        **summary,
    }, indent=2))
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    main()
