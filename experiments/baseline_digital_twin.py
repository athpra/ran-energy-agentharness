"""
Baseline C — Digital twin only (equivalent to the existing PoC).

Dual-LLM closed loop (Planner + Validator) with VIAVI AI RSG as the
only tool. No additional tool context (no historical KPI, forecasts,
faults, interference, energy pricing).

Run:
    python experiments/baseline_digital_twin.py --intent "5 Mbps" --iterations 20
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from agent.harness   import AgentHarness
from agent.context   import ALL_TOOLS
from agent.planner   import BLUEPRINT_SYSTEM_PROMPT
from experiments.common import get_llm, get_current_kpis, connect_scenario, make_run_dir, save_results, make_sim_fns, _kpi_to_text, apply_job_arguments


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    condition_name: str = "baseline_digital_twin",
) -> list[dict]:
    llm                       = get_llm()
    sim_test_fn, sim_apply_fn, _ = make_sim_fns(scenario)

    harness = AgentHarness(
        llm=llm,
        sim_test_fn=sim_test_fn,
        sim_apply_fn=sim_apply_fn,
        operator_intent=operator_intent,
        enabled_tools=frozenset(),          # no additional tools
        planner_system_prompt=BLUEPRINT_SYSTEM_PROMPT,
    )

    ts = pd.Timestamp.now().normalize()  # start of today; iterations advance virtually
    for i in range(n_iterations):
        virtual_ts  = ts + pd.Timedelta(minutes=15 * i)
        kpis        = get_current_kpis(scenario, timestamp=virtual_ts)
        site_keys   = sorted(kpis["per_site"].keys())
        cell_ids    = list(range(len(site_keys)))
        utilization       = {j: kpis["per_site"][k]["n1_prb"] for j, k in enumerate(site_keys)}
        sleeping_cell_ids = [j for j, k in enumerate(site_keys) if kpis["per_site"][k]["n1_sleeping"]]
        kpi_summary       = _kpi_to_text(kpis)

        # Blueprint PRB rule: N1_PRB < 12 → sleep candidate (mirrors blueprint SQL).
        # Passed as static fallback in case the LLM planner returns [].
        sleep_cands = [
            j for j, k in enumerate(site_keys)
            if not kpis["per_site"][k]["n1_sleeping"] and kpis["per_site"][k]["n1_prb"] < 12.0
        ]

        result = harness.run_iteration(
            iteration=i + 1,
            timestamp=virtual_ts,
            kpi_summary=kpi_summary,
            cell_ids=cell_ids,
            current_utilization=utilization,
            static_sleep_candidates=sleep_cands or None,
            sleeping_cell_ids=sleeping_cell_ids,
        )
        print(
            f"  [iter {i+1:>3}] proposed={len(result.proposed_actions)}  "
            f"approved={len(result.approved_actions)}  "
            f"rejected={len(result.rejected_actions)}"
        )

    summary = harness.summary()
    for r in summary:
        r["condition"] = condition_name
    return summary


def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(description="Baseline C: digital twin only")
    parser.add_argument("--intent",     default="5 Mbps")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    args, _ = parser.parse_known_args()

    scenario = connect_scenario(args.rsg_host)

    print(f"Running: baseline_digital_twin  intent='{args.intent}'  iterations={args.iterations}")
    results = run(scenario, args.intent, args.iterations)
    run_dir = make_run_dir("baseline_digital_twin", "llm")
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
