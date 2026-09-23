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

from dotenv import find_dotenv, load_dotenv

import pandas as pd

from agent.harness import AgentHarness
from agent.context import ALL_TOOLS
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
    enabled_tools: frozenset[str] = ALL_TOOLS,
    condition_name: str = "full_harness_vtzcycle",
    model: str | None = None,
    run_dir=None,
) -> list[dict]:
    llm = get_llm(model)
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

        # Set virtual timestamp so all sim calls use the right traffic profile
        set_virtual_ts(virtual_ts)

        # Get current network state including accumulated sleep history.
        # sim_test_fn([]) replays cell_sleep_state so the Planner sees which
        # cells are sleeping and can propose wake actions when load increases.
        _, kpis = sim_test_fn([])

        site_keys      = sorted(kpis["per_site"].keys())
        cell_ids       = list(range(len(site_keys)))
        utilization    = {j: kpis["per_site"][k]["n1_prb"] for j, k in enumerate(site_keys)}
        kpi_summary    = _kpi_to_text_viavi(kpis)
        sleeping_ids   = [j for j, k in enumerate(site_keys) if kpis["per_site"][k]["n1_sleeping"]]

        # Reactive wake candidates: sleeping cells when QoS is already below intent.
        # The forecast tool also adds proactive wake guidance when a surge is imminent.
        intent_mbps = float(operator_intent.split()[0])
        wake_cands = sorted(
            sleeping_ids,
            key=lambda j: kpis["per_site"][site_keys[j]]["n12_prb"],
            reverse=True,
        ) if kpis.get("avg_throughput_mbps", 0) < intent_mbps else []

        result = harness.run_iteration(
            iteration=i + 1,
            timestamp=virtual_ts,
            kpi_summary=kpi_summary,
            cell_ids=cell_ids,
            current_utilization=utilization,
            static_wake_candidates=wake_cands or None,
            sleeping_cell_ids=sleeping_ids or None,
        )
        r_dict = result.to_dict()
        r_dict["condition"]   = condition_name
        r_dict["kpi_summary"] = kpi_summary

        post_tp  = (r_dict["post_kpis"].get("avg_throughput_mbps")
                    or kpis.get("avg_throughput_mbps", 0))
        sleeping = (r_dict["post_kpis"].get("sleeping_cells",
                    kpis.get("sleeping_cells", 0)))
        total    = r_dict["post_kpis"].get("total_cells", kpis.get("total_cells", 42))
        violated = post_tp < intent_mbps
        r_dict["qos_violated"] = violated

        qos_tag = "✗ QoS VIOLATION" if violated else "✓"
        print(
            f"  [iter {i+1:>3}] proposed={len(result.proposed_actions)}  "
            f"approved={len(result.approved_actions)}  "
            f"rejected={len(result.rejected_actions)}  "
            f"tp={post_tp:.2f} Mbps  sleeping={sleeping}/{total}  {qos_tag}"
        )

        if run_dir is not None:
            append_result(run_dir, r_dict)

    summary = harness.summary()
    for r in summary:
        r["condition"] = condition_name
    return summary


def main():
    load_dotenv(find_dotenv(), override=True)
    apply_job_arguments()
    parser = argparse.ArgumentParser(
        description="Full agent harness — VTZ-cycle variant"
    )
    parser.add_argument("--intent",     default="3 Mbps")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITER", 96)))
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    parser.add_argument("--model",      default=None,
                        help="LLM model ID override (e.g. nvidia/nemotron-3-super-120b-a12b). "
                             "Defaults to LLM_MODEL env var.")
    parser.add_argument(
        "--tools", nargs="*", default=list(ALL_TOOLS), choices=list(ALL_TOOLS),
        help="Subset of tools to enable (default: all).  Use for ablation runs.",
    )
    args, _ = parser.parse_known_args()

    enabled    = frozenset(args.tools)
    model_slug = args.model.split("/")[-1] if args.model else os.getenv("LLM_MODEL", "llm").split("/")[-1]
    label      = (
        "full_harness_vtzcycle"
        if enabled == ALL_TOOLS
        else "ablation_vtzcycle_" + "_".join(sorted(enabled))
    )
    condition  = f"{label}__{model_slug}"

    scenario = connect_scenario(args.rsg_host)
    run_dir  = make_run_dir(label, model_slug)

    print(f"Running: {condition}")
    print(f"  model={model_slug}  tools={sorted(enabled)}  intent='{args.intent}'  iterations={args.iterations}")
    print(f"  Saving incrementally to: {run_dir}")

    results = run(scenario, args.intent, args.iterations, enabled, condition, args.model, run_dir=run_dir)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
