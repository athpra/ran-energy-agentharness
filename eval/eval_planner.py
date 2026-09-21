"""
Planner evaluator.

Runs the Planner LLM against each non-validator EvalScenario and
returns scored results per scenario.

Usage:
    from eval.eval_planner import run_planner_eval
    results = run_planner_eval(llm, scenarios)
"""
from __future__ import annotations

import time
from typing import Any

from langchain_openai import ChatOpenAI

from agent.planner import plan, SYSTEM_PROMPT
from eval.fixtures.scenarios import EvalScenario
from eval.scoring import score_planner


def run_planner_eval(
    llm: ChatOpenAI,
    scenarios: list[EvalScenario],
    system_prompt: str = SYSTEM_PROMPT,
    operator_intent: str = "5 Mbps",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate the planner on all non-validator scenarios.

    Returns a list of per-scenario result dicts.
    """
    planner_scenarios = [s for s in scenarios if not s.is_validator_scenario]
    results = []

    for scenario in planner_scenarios:
        if verbose:
            print(f"  Planner eval: {scenario.id} ({scenario.description[:60]}...)")

        t0 = time.time()
        proposed, raw, elapsed = plan(
            kpi_summary=scenario.kpi_summary,
            operator_intent=operator_intent,
            tool_context_block="### Tool Context\n(none — eval harness)\n",
            llm=llm,
            system_prompt=system_prompt,
        )
        wall_s = round(time.time() - t0, 3)

        all_ids = [c.cell_id for c in scenario.cells]
        scores  = score_planner(
            proposed=proposed,
            raw_response=raw,
            ground_truth_sleep=scenario.ground_truth_sleep,
            ground_truth_wake=scenario.ground_truth_wake,
            all_cell_ids=all_ids,
        )

        result = {
            "scenario_id":        scenario.id,
            "tags":               scenario.tags,
            "ground_truth_sleep": scenario.ground_truth_sleep,
            "ground_truth_wake":  scenario.ground_truth_wake,
            "proposed_sleep":     [a["cell_id"] for a in proposed if a.get("action") == "sleep"],
            "proposed_wake":      [a["cell_id"] for a in proposed if a.get("action") == "wake"],
            "proposed_raw":       raw,
            "elapsed_s":          elapsed,
            "wall_s":             wall_s,
            **scores,
        }

        if verbose:
            status = "PASS" if scores["f1"] >= 0.8 and scores["format_valid"] else "FAIL"
            print(
                f"    [{status}] P={scores['precision']:.2f}  R={scores['recall']:.2f}  "
                f"F1={scores['f1']:.2f}  format={scores['format_valid']}  "
                f"halluc={not scores['no_hallucinations']}"
            )

        results.append(result)

    return results
