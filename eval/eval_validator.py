"""
Validator evaluator.

Runs the Validator LLM against each validator EvalScenario and
returns scored results per scenario.

Usage:
    from eval.eval_validator import run_validator_eval
    results = run_validator_eval(llm, scenarios)
"""
from __future__ import annotations

import time
from typing import Any

from langchain_openai import ChatOpenAI

from agent.validator import validate
from eval.fixtures.scenarios import EvalScenario
from eval.scoring import score_validator


def run_validator_eval(
    llm: ChatOpenAI,
    scenarios: list[EvalScenario],
    operator_intent: str = "5 Mbps",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate the validator on all validator scenarios.

    Returns a list of per-scenario result dicts.
    """
    validator_scenarios = [s for s in scenarios if s.is_validator_scenario]
    results = []

    for scenario in validator_scenarios:
        if verbose:
            print(f"  Validator eval: {scenario.id} ({scenario.description[:60]}...)")

        assert scenario.validator_proposals   is not None
        assert scenario.sim_result_summary    is not None
        assert scenario.risky_proposal_ids    is not None
        assert scenario.safe_proposal_ids     is not None

        # Determine tool context: V03 scenario injects fault info
        tool_context = "### Tool Context\n(none — eval harness)\n"
        if hasattr(scenario, "_tool_ctx_override") and scenario._tool_ctx_override:
            tool_context = scenario._tool_ctx_override

        t0 = time.time()
        approved, rejected, raw, elapsed = validate(
            proposed_actions=scenario.validator_proposals,
            sim_result_summary=scenario.sim_result_summary,
            operator_intent=operator_intent,
            tool_context_block=tool_context,
            llm=llm,
        )
        wall_s = round(time.time() - t0, 3)

        scores = score_validator(
            approved=approved,
            rejected=rejected,
            raw_response=raw,
            risky_proposal_ids=scenario.risky_proposal_ids,
            safe_proposal_ids=scenario.safe_proposal_ids,
        )

        result = {
            "scenario_id":          scenario.id,
            "tags":                 scenario.tags,
            "risky_proposal_ids":   scenario.risky_proposal_ids,
            "safe_proposal_ids":    scenario.safe_proposal_ids,
            "approved_ids":         [a["cell_id"] for a in approved],
            "rejected_ids":         [r["cell_id"] for r in rejected],
            "validator_raw":        raw,
            "elapsed_s":            elapsed,
            "wall_s":               wall_s,
            **scores,
        }

        if verbose:
            status = "PASS" if (
                scores["risky_rejection_rate"] >= 0.8
                and scores["safe_approval_rate"] >= 0.8
                and scores["format_valid"]
            ) else "FAIL"
            print(
                f"    [{status}] risky_rej={scores['risky_rejection_rate']:.2f}  "
                f"safe_appr={scores['safe_approval_rate']:.2f}  "
                f"format={scores['format_valid']}"
            )

        results.append(result)

    return results
