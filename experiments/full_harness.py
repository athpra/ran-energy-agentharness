"""
Proposed system — Full agent harness.

Dual-LLM closed loop (Planner + Validator) with VIAVI AI RSG +
all five additional tools active:
  - historical_kpi
  - traffic_forecast
  - alarm_fault
  - interference
  - energy_pricing

Pass --tools to run a subset for ablation experiments:
    python experiments/full_harness.py --tools historical_kpi traffic_forecast

Run:
    python experiments/full_harness.py --intent "5 Mbps" --iterations 20
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from agent.harness   import AgentHarness
from agent.context   import ALL_TOOLS
from experiments.common import (
    get_llm, get_current_kpis, connect_scenario, make_run_dir, save_results, append_result, make_sim_fns, _kpi_to_text,
    apply_job_arguments,
)


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    enabled_tools: frozenset[str] = ALL_TOOLS,
    condition_name: str = "full_harness",
    run_dir=None,
) -> list[dict]:
    llm                       = get_llm()
    sim_test_fn, sim_apply_fn, _ = make_sim_fns(scenario)

    harness = AgentHarness(
        llm=llm,
        sim_test_fn=sim_test_fn,
        sim_apply_fn=sim_apply_fn,
        operator_intent=operator_intent,
        enabled_tools=enabled_tools,
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

        result = harness.run_iteration(
            iteration=i + 1,
            timestamp=virtual_ts,
            kpi_summary=kpi_summary,
            cell_ids=cell_ids,
            current_utilization=utilization,
            sleeping_cell_ids=sleeping_cell_ids,
        )
        r_dict = result.to_dict()
        r_dict["condition"] = condition_name
        print(
            f"  [iter {i+1:>3}] proposed={len(result.proposed_actions)}  "
            f"approved={len(result.approved_actions)}  "
            f"rejected={len(result.rejected_actions)}  "
            f"tools={sorted(result.tools_used)}"
        )
        if run_dir is not None:
            append_result(run_dir, r_dict)

    summary = harness.summary()
    for r in summary:
        r["condition"] = condition_name
    return summary


def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(description="Full agent harness with all tools")
    parser.add_argument("--intent",     default="5 Mbps")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITER", 96)))
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    parser.add_argument(
        "--tools", nargs="*", default=list(ALL_TOOLS),
        choices=list(ALL_TOOLS),
        help="Subset of tools to enable (default: all). Use for ablation runs.",
    )
    args, _ = parser.parse_known_args()

    enabled = frozenset(args.tools)
    label   = "full_harness" if enabled == ALL_TOOLS else "ablation_" + "_".join(sorted(enabled))

    scenario = connect_scenario(args.rsg_host)

    run_dir = make_run_dir(label, "llm")
    print(f"Running: {label}  tools={sorted(enabled)}  intent='{args.intent}'  iterations={args.iterations}")
    print(f"Saving incrementally to: {run_dir}")
    results = run(scenario, args.intent, args.iterations, enabled, label, run_dir=run_dir)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
