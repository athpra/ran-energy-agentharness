"""
Main eval runner.

Evaluates both the Planner and Validator against the scenario catalogue
and saves a timestamped JSON result file that includes full model metadata.

Usage:
    python -m eval.run_evals [--output eval/results/]

The saved JSON can be compared across models using eval/compare_models.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv, find_dotenv

from experiments.common import get_llm
from eval.fixtures.scenarios import SCENARIOS
from eval.eval_planner import run_planner_eval
from eval.eval_validator import run_validator_eval
from eval.scoring import aggregate_planner_scores, aggregate_validator_scores


def _model_metadata(llm) -> dict:
    """Extract model metadata from a ChatOpenAI instance."""
    return {
        "model_id":   getattr(llm, "model_name",      "unknown"),
        "api_base":   getattr(llm, "openai_api_base",  "unknown"),
        "temperature": getattr(llm, "temperature",     None),
        "max_tokens": getattr(llm, "max_tokens",       None),
    }


def run_evals(
    output_dir: str | Path = "eval/results",
    operator_intent: str   = "5 Mbps",
    planner_only: bool     = False,
    validator_only: bool   = False,
    verbose: bool          = True,
) -> dict:
    """
    Run the full eval suite and return the result dict.
    Also writes a JSON file to output_dir.
    """
    load_dotenv(find_dotenv(), override=True)
    llm = get_llm()

    now    = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    result: dict = {
        "eval_run_id":      run_id,
        "timestamp":        now.isoformat(),
        "model":            _model_metadata(llm),
        "operator_intent":  operator_intent,
    }

    if not validator_only:
        print(f"\n── Planner eval ({sum(1 for s in SCENARIOS if not s.is_validator_scenario)} scenarios) ──")
        planner_per = run_planner_eval(llm, SCENARIOS, operator_intent=operator_intent, verbose=verbose)
        planner_agg = aggregate_planner_scores(planner_per)
        result["planner"] = {**planner_agg, "per_scenario": planner_per}
        print(
            f"\n  Aggregate: P={planner_agg.get('precision_mean',0):.3f}  "
            f"R={planner_agg.get('recall_mean',0):.3f}  "
            f"F1={planner_agg.get('f1_mean',0):.3f}  "
            f"format={planner_agg.get('format_valid_rate',0):.3f}  "
            f"no_halluc={planner_agg.get('no_hallucinations_rate',0):.3f}"
        )

    if not planner_only:
        print(f"\n── Validator eval ({sum(1 for s in SCENARIOS if s.is_validator_scenario)} scenarios) ──")
        validator_per = run_validator_eval(llm, SCENARIOS, operator_intent=operator_intent, verbose=verbose)
        validator_agg = aggregate_validator_scores(validator_per)
        result["validator"] = {**validator_agg, "per_scenario": validator_per}
        print(
            f"\n  Aggregate: risky_rej={validator_agg.get('risky_rejection_rate_mean',0):.3f}  "
            f"safe_appr={validator_agg.get('safe_approval_rate_mean',0):.3f}  "
            f"format={validator_agg.get('format_valid_rate',0):.3f}"
        )

    # Save to file
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_tag = (
        result["model"]["model_id"]
        .replace("/", "_")
        .replace(":", "-")
        .replace(" ", "_")
    )
    filename = out_path / f"eval_{run_id}_{model_tag}.json"
    filename.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved: {filename}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run the RAN Energy eval suite")
    parser.add_argument("--output",   default="eval/results", help="Directory to save result JSON")
    parser.add_argument("--intent",   default="5 Mbps",       help="Operator QoS intent string")
    parser.add_argument("--planner-only",   action="store_true")
    parser.add_argument("--validator-only", action="store_true")
    parser.add_argument("--quiet",    action="store_true",    help="Suppress per-scenario output")
    args = parser.parse_args()

    run_evals(
        output_dir=args.output,
        operator_intent=args.intent,
        planner_only=args.planner_only,
        validator_only=args.validator_only,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
