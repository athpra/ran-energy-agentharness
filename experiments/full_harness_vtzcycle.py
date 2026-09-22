"""
Full agent harness — VTZ-cycle variant.

Each iteration runs with a virtual timestamp advanced by 15 minutes, covering a
full 24-hour traffic cycle across 96 iterations.  The traffic profile (UE count)
for every simulation is selected from the virtual clock — Night / Morning /
Evening etc. — so KPIs vary meaningfully across iterations rather than staying
fixed at whatever the wall-clock hour happens to be.

Dual-LLM closed loop (Planner + Validator) with VIAVI AI RSG +
all five additional tools active.

Run:
    python experiments/full_harness_vtzcycle.py --intent "3 Mbps" --iterations 96
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from agent.harness import AgentHarness
from agent.context import ALL_TOOLS
from experiments.common import (
    apply_job_arguments,
    connect_scenario,
    get_current_kpis,
    get_llm,
    make_run_dir,
    make_sim_fns,
    save_results,
    append_result,
    _kpi_to_text_viavi,
)


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    enabled_tools: frozenset[str] = ALL_TOOLS,
    condition_name: str = "full_harness_vtzcycle",
    run_dir=None,
) -> list[dict]:
    llm = get_llm()
    sim_test_fn, sim_apply_fn, set_virtual_ts = make_sim_fns(scenario)

    harness = AgentHarness(
        llm=llm,
        sim_test_fn=sim_test_fn,
        sim_apply_fn=sim_apply_fn,
        operator_intent=operator_intent,
        enabled_tools=enabled_tools,
    )

    ts = pd.Timestamp.now().normalize()  # midnight today; iterations advance virtually

    for i in range(n_iterations):
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)

        # Set virtual timestamp so sim_test_fn / sim_apply_fn use the right traffic profile
        set_virtual_ts(virtual_ts)

        # Get current network state at this virtual time
        kpis = get_current_kpis(scenario, timestamp=virtual_ts)

        site_keys   = sorted(kpis["per_site"].keys())
        cell_ids    = list(range(len(site_keys)))
        utilization = {j: kpis["per_site"][k]["n1_prb"] for j, k in enumerate(site_keys)}
        kpi_summary = _kpi_to_text_viavi(kpis)

        result = harness.run_iteration(
            iteration=i + 1,
            timestamp=virtual_ts,
            kpi_summary=kpi_summary,
            cell_ids=cell_ids,
            current_utilization=utilization,
        )
        r_dict = result.to_dict()
        r_dict["condition"]   = condition_name
        r_dict["kpi_summary"] = kpi_summary

        print(
            f"  [iter {i+1:>3}] proposed={len(result.proposed_actions)}  "
            f"approved={len(result.approved_actions)}  "
            f"rejected={len(result.rejected_actions)}  "
            f"tp={kpis.get('avg_throughput_mbps', 0):.2f} Mbps  "
            f"sleeping={kpis.get('sleeping_cells', 0)}/{kpis.get('total_cells', 0)}"
        )

        if run_dir is not None:
            append_result(run_dir, r_dict)

    summary = harness.summary()
    for r in summary:
        r["condition"] = condition_name
    return summary


def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(
        description="Full agent harness — VTZ-cycle variant"
    )
    parser.add_argument("--intent",     default="3 Mbps")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITER", 96)))
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    parser.add_argument(
        "--tools", nargs="*", default=list(ALL_TOOLS), choices=list(ALL_TOOLS),
        help="Subset of tools to enable (default: all).  Use for ablation runs.",
    )
    args, _ = parser.parse_known_args()

    enabled = frozenset(args.tools)
    label   = (
        "full_harness_vtzcycle"
        if enabled == ALL_TOOLS
        else "ablation_vtzcycle_" + "_".join(sorted(enabled))
    )

    scenario = connect_scenario(args.rsg_host)
    run_dir  = make_run_dir(label, "llm")

    print(f"Running: {label}")
    print(f"  tools={sorted(enabled)}  intent='{args.intent}'  iterations={args.iterations}")
    print(f"  Saving incrementally to: {run_dir}")

    results = run(scenario, args.intent, args.iterations, enabled, label, run_dir=run_dir)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
