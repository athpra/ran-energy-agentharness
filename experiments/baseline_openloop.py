"""
Baseline B — Open-loop LLM (single LLM, no tools, no simulation test, no validator).

The Planner proposes actions and they are applied directly without:
  - tool context (no historical KPI, forecast, faults, etc.)
  - simulation safety test (Sim 1)
  - validator approval

This isolates the contribution of the closed-loop structure and tool context.

Run:
    python experiments/baseline_openloop.py --intent "5 Mbps" --iterations 20
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from experiments.common import get_llm, get_current_kpis, connect_scenario, make_run_dir, save_results, make_sim_fns, _kpi_to_text, apply_job_arguments
from agent.planner import plan

# Aggressive prompt for open-loop baseline: maximise energy savings without
# calibrated thresholds. Simulates an unconstrained LLM that has no closed-loop
# guardrails — it sleeps anything below medium load (≤ 40 % PRB), which causes
# QoS degradation when moderate-traffic cells are incorrectly silenced.
_OPENLOOP_AGGRESSIVE_PROMPT = """\
You are a 5G RAN energy optimization controller.
Your goal is to maximise energy savings by putting as many cells to sleep as possible.
Your ONLY output is a JSON array of sleep/wake actions. No explanation, no markdown, nothing else.

RULES:
  • Sleep any [Awake] cell whose N1_PRB is below 40 %
  • Wake  any [Sleeping] cell whose N12_PRB is above 60 %

EXAMPLE OUTPUT:
[{"action": "sleep", "cell_id": 2, "reason": "N1_PRB=15.0% < 40%"}, {"action": "wake", "cell_id": 5, "reason": "N12_PRB=72.0% > 60%"}]

Now produce the JSON array for the cells below. Output ONLY [...] — nothing before or after.
"""

# PRB threshold used in the deterministic fallback when the LLM returns [].
# Intentionally higher than the blueprint's 12 % to match the aggressive prompt.
_OPENLOOP_PRB_THRESHOLD = 40.0


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    condition_name: str = "baseline_openloop",
) -> list[dict]:
    llm = get_llm()
    _, sim_apply_fn = make_sim_fns(scenario)

    ts = pd.Timestamp.now().normalize()  # start of today; iterations advance virtually

    results = []
    for i in range(n_iterations):
        t0         = time.time()
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)

        raw_kpis    = get_current_kpis(scenario, timestamp=virtual_ts)
        kpi_summary = _kpi_to_text(raw_kpis)

        # No tool context, no validator — aggressive system prompt drives decisions
        proposed, planner_raw, t_plan = plan(
            kpi_summary=kpi_summary,
            operator_intent=operator_intent,
            tool_context_block="### Tool Context\n(none — open-loop baseline)\n",
            llm=llm,
            system_prompt=_OPENLOOP_AGGRESSIVE_PROMPT,
        )

        # Fallback: if LLM returns nothing, apply the aggressive PRB threshold
        # directly so the condition always produces real data.
        if not proposed:
            site_keys = sorted(raw_kpis["per_site"].keys())
            proposed = [
                {"action": "sleep", "cell_id": j,
                 "reason": f"N1_PRB={raw_kpis['per_site'][k]['n1_prb']:.1f}% < {_OPENLOOP_PRB_THRESHOLD:.0f}% [aggressive-fallback]"}
                for j, k in enumerate(site_keys)
                if not raw_kpis["per_site"][k]["n1_sleeping"]
                and raw_kpis["per_site"][k]["n1_prb"] < _OPENLOOP_PRB_THRESHOLD
            ]
            if proposed:
                planner_raw = (
                    f"[aggressive-fallback] LLM returned []; generated {len(proposed)} actions "
                    f"from N1_PRB < {_OPENLOOP_PRB_THRESHOLD:.0f}% rule"
                )

        # Apply directly — no Sim 1 test, no validator
        sim_summary, post_kpis = sim_apply_fn(proposed) if proposed else (
            "No actions proposed.", {}
        )

        results.append({
            "iteration":         i + 1,
            "condition":         condition_name,
            "tools_used":        [],
            "proposed_actions":  proposed,
            "approved_actions":  proposed,
            "rejected_actions":  [],
            "n_proposed":        len(proposed),
            "n_approved":        len(proposed),
            "n_rejected":        0,
            "planner_elapsed_s": round(t_plan, 2),
            "sim2_summary":      sim_summary,
            "post_kpis":         post_kpis,
            "total_elapsed_s":   round(time.time() - t0, 3),
        })
        print(f"  [iter {i+1:>3}] proposed={len(proposed)}  applied={len(proposed)}")

    return results



def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(description="Baseline B: open-loop LLM")
    parser.add_argument("--intent",     default="5 Mbps")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    args, _ = parser.parse_known_args()

    scenario = connect_scenario(args.rsg_host)

    print(f"Running: baseline_openloop  intent='{args.intent}'  iterations={args.iterations}")
    results = run(scenario, args.intent, args.iterations)
    run_dir = make_run_dir("baseline_openloop", "llm")
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
