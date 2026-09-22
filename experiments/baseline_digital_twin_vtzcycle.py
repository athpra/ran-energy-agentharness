"""
Baseline C — Digital twin only, VTZ-cycle variant.

Dual-LLM closed loop (Planner + Validator) with VIAVI AI RSG but NO
additional tool context (no historical KPI, forecasts, faults, etc.).
Uses the blueprint's original PRB-12% system prompt.

The virtual timestamp advances 15 minutes per iteration so traffic profiles
(UE counts) cycle through Night / Morning / Evening across 96 iterations,
giving the Validator varying network conditions to reason about.

Run:
    python experiments/baseline_digital_twin_vtzcycle.py --intent "3 Mbps" --iterations 96
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from agent.harness  import AgentHarness
from agent.planner  import BLUEPRINT_SYSTEM_PROMPT
from experiments.common import (
    apply_job_arguments,
    connect_scenario,
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
    condition_name: str = "baseline_digital_twin_vtzcycle",
    run_dir=None,
) -> list[dict]:
    llm = get_llm()
    sim_test_fn, sim_apply_fn, set_virtual_ts = make_sim_fns(scenario)

    harness = AgentHarness(
        llm=llm,
        sim_test_fn=sim_test_fn,
        sim_apply_fn=sim_apply_fn,
        operator_intent=operator_intent,
        enabled_tools=frozenset(),              # no additional tools — digital twin only
        planner_system_prompt=BLUEPRINT_SYSTEM_PROMPT,
    )

    ts = pd.Timestamp.now().normalize()  # midnight today; used for tool context timestamps

    for i in range(n_iterations):
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)

        # Set virtual timestamp so all sim calls use the right traffic profile
        set_virtual_ts(virtual_ts)

        # Get current network state including accumulated sleep history
        _, kpis = sim_test_fn([])

        site_keys   = sorted(kpis["per_site"].keys())
        cell_ids    = list(range(len(site_keys)))
        utilization = {j: kpis["per_site"][k]["n1_prb"] for j, k in enumerate(site_keys)}
        kpi_summary = _kpi_to_text_viavi(kpis)

        # Blueprint PRB rule: N1_PRB < 12 % → sleep candidate (mirrors blueprint SQL).
        # Sort by PRB ascending so the LLM sees most-underloaded cells first.
        sleep_cands = sorted(
            (j for j, k in enumerate(site_keys)
             if not kpis["per_site"][k]["n1_sleeping"]
             and kpis["per_site"][k]["n1_prb"] < 12.0),
            key=lambda j: kpis["per_site"][site_keys[j]]["n1_prb"],
        )

        # Wake candidates: sleeping cells when QoS is below the operator intent.
        # Sort by N12_PRB descending (most overloaded neighbor first).
        intent_mbps = float(operator_intent.split()[0])
        wake_cands = sorted(
            (j for j, k in enumerate(site_keys) if kpis["per_site"][k]["n1_sleeping"]),
            key=lambda j: kpis["per_site"][site_keys[j]]["n12_prb"],
            reverse=True,
        ) if kpis.get("avg_throughput_mbps", 0) < intent_mbps else []

        result = harness.run_iteration(
            iteration=i + 1,
            timestamp=virtual_ts,
            kpi_summary=kpi_summary,
            cell_ids=cell_ids,
            current_utilization=utilization,
            static_sleep_candidates=sleep_cands or None,
            static_wake_candidates=wake_cands or None,
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
        description="Baseline C: digital twin only — VTZ-cycle variant"
    )
    parser.add_argument("--intent",     default="3 Mbps")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITER", 96)))
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    args, _ = parser.parse_known_args()

    scenario = connect_scenario(args.rsg_host)
    run_dir  = make_run_dir("baseline_digital_twin_vtzcycle", "llm")

    print(
        f"Running: baseline_digital_twin_vtzcycle  "
        f"intent='{args.intent}'  iterations={args.iterations}"
    )
    print(f"  Saving incrementally to: {run_dir}")

    results = run(scenario, args.intent, args.iterations, run_dir=run_dir)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
