"""
Benchmark several models on every eval.

For each model in a models file, runs:
  planner, validator   eval/run_evals.py: the scenario catalogue, one call each
  discovery            eval/eval_discovery.py in up to five modes:
                         catalog        policy on, policy off
                         catalog_hard   policy on, policy off   (near-miss distractors)
                         catalog_draft  policy off               (LLM-drafted from the data;
                                                                  only if the file exists)

All results go to one folder per benchmark, so a table never mixes runs from
different catalogs or code versions, and end in one comparison table
(eval/compare_models.py), also saved as summary.txt. A step that fails is
recorded in benchmark.json and the rest go on; if a model's first step fails
(its endpoint is down, say), its remaining steps are skipped.

The drafted catalog is written once, by one model, before benchmarking, so
every model is scored on the same text:
    python generators/profile_catalog.py      # → data/synthetic/catalog_draft.yaml

Run:
    cp eval/models.example.yaml eval/models.yaml     # list your models
    python -m eval.benchmark --models eval/models.yaml
    python -m eval.benchmark --models eval/models.yaml --suites discovery --runs 5

Output: eval/results/benchmark_<timestamp>/ (result files, summary.txt, benchmark.json)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402
from dotenv import find_dotenv, load_dotenv  # noqa: E402

from agent.catalog import CATALOG_PATH, DATA_DIR  # noqa: E402
from eval.compare_models import compare  # noqa: E402
from eval.eval_discovery import HARD_CATALOG_PATH, evaluate  # noqa: E402

DRAFT_CATALOG_PATH = DATA_DIR / "catalog_draft.yaml"
SUITES = ("planner", "validator", "discovery")


def load_models(path: Path) -> list[dict]:
    """[{"id": ..., "api_base": ... or None}] from the models file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    models = []
    for m in data.get("models") or []:
        m = {"id": m} if isinstance(m, str) else dict(m)
        if not m.get("id"):
            raise ValueError(f"{path}: every model needs an id")
        models.append({"id": str(m["id"]), "api_base": m.get("api_base") or None})
    if not models:
        raise ValueError(f"{path}: no models listed")
    return models


def discovery_modes(drafted: Path | None) -> list[tuple[Path, bool]]:
    """(catalog, no_policy) pairs, easiest first."""
    modes = [(CATALOG_PATH, False), (CATALOG_PATH, True),
             (HARD_CATALOG_PATH, False), (HARD_CATALOG_PATH, True)]
    if drafted is not None:
        modes.append((drafted, True))
    return modes


def main():
    load_dotenv(find_dotenv(), override=True)
    parser = argparse.ArgumentParser(description="Run every eval on several models")
    parser.add_argument("--models", type=Path, required=True, help="YAML file listing the models")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--runs",   type=int, default=10, help="Discovery runs per mode")
    parser.add_argument("--intent", default="5 Mbps")
    parser.add_argument("--drafted", type=Path, default=DRAFT_CATALOG_PATH,
                        help="LLM-drafted catalog for the hardest discovery mode (skipped if missing)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Results folder (default: eval/results/benchmark_<timestamp>)")
    args = parser.parse_args()

    models = load_models(args.models)
    if "discovery" in args.suites:
        missing = [p for p in (CATALOG_PATH, HARD_CATALOG_PATH) if not p.exists()]
        if missing:
            sys.exit(f"Missing {[str(p) for p in missing]}. Run python generators/generate_maintenance.py "
                     f"&& python generators/generate_catalog.py")
    drafted = args.drafted if args.drafted.exists() else None
    if "discovery" in args.suites and drafted is None:
        print(f"Note: no drafted catalog at {args.drafted}; skipping that mode "
              f"(python generators/profile_catalog.py writes it)")

    now = datetime.now(timezone.utc)
    out = args.output or Path("eval/results") / f"benchmark_{now.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"started": now.isoformat(), "models": models, "suites": args.suites,
                "discovery_runs": args.runs, "intent": args.intent,
                "drafted_catalog": str(drafted) if drafted else None, "failures": []}

    from experiments.common import get_llm
    from eval.run_evals import run_evals

    for i, m in enumerate(models, 1):
        print(f"\n{'═' * 70}\n[{i}/{len(models)}] {m['id']}  ({m['api_base'] or 'CDSW_API_URL'})\n{'═' * 70}")
        llm = get_llm(m["id"], m["api_base"])
        steps = []
        if "planner" in args.suites or "validator" in args.suites:
            steps.append(("planner/validator", lambda: run_evals(
                output_dir=out, operator_intent=args.intent, llm=llm, verbose=False,
                planner_only="validator" not in args.suites,
                validator_only="planner" not in args.suites)))
        if "discovery" in args.suites:
            for catalog, no_policy in discovery_modes(drafted):
                name = f"discovery {catalog.stem} policy={'off' if no_policy else 'on'}"
                steps.append((name, lambda c=catalog, n=no_policy: evaluate(
                    llm, c, n, args.runs, args.intent, out, verbose=False)))
        for k, (name, step) in enumerate(steps):
            print(f"\n── {name} ──")
            try:
                step()
            except Exception as e:   # keep going: one bad endpoint should not end the benchmark
                print(f"  ✗ failed: {e}")
                manifest["failures"].append({"model": m["id"], "step": name, "error": repr(e),
                                             "traceback": traceback.format_exc()})
                if k == 0:
                    skipped = [n for n, _ in steps[1:]]
                    print(f"  skipping this model's other {len(skipped)} step(s)")
                    manifest["failures"].append({"model": m["id"], "step": "skipped", "error":
                                                 "first step failed", "skipped": skipped})
                    break

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    (out / "benchmark.json").write_text(json.dumps(manifest, indent=2))

    table = io.StringIO()
    results = sorted(p for p in out.glob("*.json") if p.name != "benchmark.json")
    with contextlib.redirect_stdout(table):
        compare(results)
    (out / "summary.txt").write_text(table.getvalue())
    print(table.getvalue())
    failed = sorted({f["model"] for f in manifest["failures"]})
    if failed:
        print(f"✗ Failures for {failed}; see {out / 'benchmark.json'}")
    print(f"Saved → {out}/ (summary.txt, benchmark.json, {len(results)} result files)")


if __name__ == "__main__":
    main()
