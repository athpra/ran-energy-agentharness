"""
Baseline B — Open-loop LLM, VTZ-cycle variant.

Same aggressive planner as baseline_openloop.py but the virtual timestamp
advances 15 minutes per iteration so traffic profiles (UE counts) cycle through
Night / Morning / Evening across 96 iterations.

Run:
    python experiments/baseline_openloop_vtzcycle.py --intent "3 Mbps" --iterations 96
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from agent.planner import plan
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

_OPENLOOP_PRB_THRESHOLD = 40.0


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    condition_name: str = "baseline_openloop_vtzcycle",
    run_dir=None,
) -> list[dict]:
    llm = get_llm()
    _, sim_apply_fn, set_virtual_ts = make_sim_fns(scenario)

    ts = pd.Timestamp.now().normalize()

    results = []
    for i in range(n_iterations):
        t0         = time.time()
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)

        # Set virtual timestamp so sim_apply_fn uses the right traffic profile
        set_virtual_ts(virtual_ts)

        kpis        = get_current_kpis(scenario, timestamp=virtual_ts)
        kpi_summary = _kpi_to_text_viavi(kpis)

        proposed, planner_raw, t_plan = plan(
            kpi_summary=kpi_summary,
            operator_intent=operator_intent,
            tool_context_block="### Tool Context\n(none — open-loop baseline)\n",
            llm=llm,
            system_prompt=_OPENLOOP_AGGRESSIVE_PROMPT,
        )

        if not proposed:
            site_keys = sorted(kpis["per_site"].keys())
            proposed = [
                {
                    "action": "sleep", "cell_id": j,
                    "reason": (
                        f"N1_PRB={kpis['per_site'][k]['n1_prb']:.1f}% "
                        f"< {_OPENLOOP_PRB_THRESHOLD:.0f}% [aggressive-fallback]"
                    ),
                }
                for j, k in enumerate(site_keys)
                if not kpis["per_site"][k]["n1_sleeping"]
                and kpis["per_site"][k]["n1_prb"] < _OPENLOOP_PRB_THRESHOLD
            ]
            if proposed:
                planner_raw = (
                    f"[aggressive-fallback] LLM returned []; generated {len(proposed)} "
                    f"actions from N1_PRB < {_OPENLOOP_PRB_THRESHOLD:.0f}% rule"
                )

        sim_summary, post_kpis = (
            sim_apply_fn(proposed) if proposed else ("No actions proposed.", {})
        )

        r = {
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
            "planner_raw":       planner_raw,
            "kpi_summary":       kpi_summary,
            "sim2_summary":      sim_summary,
            "post_kpis":         post_kpis,
            "total_elapsed_s":   round(time.time() - t0, 3),
        }
        print(
            f"  [iter {i+1:>3}] proposed={len(proposed)}  "
            f"tp={kpis.get('avg_throughput_mbps', 0):.2f} Mbps  "
            f"sleeping={kpis.get('sleeping_cells', 0)}/{kpis.get('total_cells', 0)}"
        )
        results.append(r)
        if run_dir is not None:
            append_result(run_dir, r)

    return results


def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(
        description="Baseline B: open-loop LLM — VTZ-cycle variant"
    )
    parser.add_argument("--intent",     default="3 Mbps")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITER", 96)))
    parser.add_argument("--rsg-host",   default=os.getenv("RSG_HOST", ""))
    args, _ = parser.parse_known_args()

    scenario = connect_scenario(args.rsg_host)
    run_dir  = make_run_dir("baseline_openloop_vtzcycle", "llm")

    print(
        f"Running: baseline_openloop_vtzcycle  "
        f"intent='{args.intent}'  iterations={args.iterations}"
    )
    print(f"  Saving incrementally to: {run_dir}")

    results = run(scenario, args.intent, args.iterations, run_dir=run_dir)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
