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
    get_llm,
    make_run_dir,
    make_sim_fns,
    save_results,
    save_run_config,
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
    model_id: str = "",
) -> list[dict]:
    llm = get_llm()
    sim_test_fn, sim_apply_fn, set_virtual_ts = make_sim_fns(scenario)

    ts = pd.Timestamp.now().normalize()

    results = []
    for i in range(n_iterations):
        t0         = time.time()
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)

        # Set virtual timestamp so all sim calls use the right traffic profile
        set_virtual_ts(virtual_ts)

        # Get current network state including accumulated sleep history
        _, kpis = sim_test_fn([])
        kpi_summary = _kpi_to_text_viavi(kpis)

        proposed, planner_raw, t_plan = plan(
            kpi_summary=kpi_summary,
            operator_intent=operator_intent,
            tool_context_block="### Tool Context\n(none — open-loop baseline)\n",
            llm=llm,
            system_prompt=_OPENLOOP_AGGRESSIVE_PROMPT,
        )

        # Filter out invalid LLM actions: LLM sometimes proposes sleep on
        # already-sleeping cells (sees PRB=0% < threshold, ignores [Sleeping] tag).
        if proposed:
            _site_keys = sorted(kpis["per_site"].keys())
            _site_by_id = {j: k for j, k in enumerate(_site_keys)}
            proposed = [
                a for a in proposed
                if a.get("cell_id") in _site_by_id and (
                    (a["action"] == "sleep" and not kpis["per_site"][_site_by_id[a["cell_id"]]]["n1_sleeping"])
                    or (a["action"] == "wake"  and     kpis["per_site"][_site_by_id[a["cell_id"]]]["n1_sleeping"])
                )
            ]

        if not proposed:
            site_keys = sorted(kpis["per_site"].keys())
            # Sleep rule: awake cells below PRB threshold
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
            else:
                # Wake rule: sleeping cells with overloaded N12 neighbor (> 60%)
                proposed = [
                    {
                        "action": "wake", "cell_id": j,
                        "reason": (
                            f"N12_PRB={kpis['per_site'][k]['n12_prb']:.1f}% "
                            f"> 60% [aggressive-fallback]"
                        ),
                    }
                    for j, k in enumerate(site_keys)
                    if kpis["per_site"][k]["n1_sleeping"]
                    and kpis["per_site"][k]["n12_prb"] > 60.0
                ]
                if proposed:
                    planner_raw = (
                        f"[aggressive-fallback] generated {len(proposed)} "
                        f"wake actions from N12_PRB > 60% rule"
                    )

        sim_summary, post_kpis = (
            sim_apply_fn(proposed) if proposed else ("No actions proposed.", {})
        )

        intent_mbps = float(operator_intent.split()[0])
        post_tp  = post_kpis.get("avg_throughput_mbps") or kpis.get("avg_throughput_mbps", 0)
        sleeping = post_kpis.get("sleeping_cells", kpis.get("sleeping_cells", 0))
        total    = post_kpis.get("total_cells",    kpis.get("total_cells",    42))
        violated = post_tp < intent_mbps

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
            "planner_model":     model_id,
            "validator_model":   "",
            "pre_kpis":          kpis,
            "post_kpis":         post_kpis,
            "qos_violated":      violated,
            "total_elapsed_s":   round(time.time() - t0, 3),
        }
        qos_tag = "✗ QoS VIOLATION" if violated else "✓"
        print(
            f"  [iter {i+1:>3}] proposed={len(proposed)}  "
            f"tp={post_tp:.2f} Mbps  sleeping={sleeping}/{total}  {qos_tag}"
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

    model_id   = os.getenv("LLM_MODEL", "")
    model_slug = model_id.split("/")[-1] if model_id else "llm"

    scenario = connect_scenario(args.rsg_host)
    run_dir  = make_run_dir("baseline_openloop_vtzcycle", model_slug)

    save_run_config(run_dir, {
        "condition":       "baseline_openloop_vtzcycle",
        "planner_model":   model_id,
        "validator_model": "",
        "intent":          args.intent,
        "iterations":      args.iterations,
        "tools":           [],
    })

    print(
        f"Running: baseline_openloop_vtzcycle  "
        f"intent='{args.intent}'  iterations={args.iterations}"
    )
    print(f"  Saving incrementally to: {run_dir}")

    results = run(scenario, args.intent, args.iterations, run_dir=run_dir, model_id=model_id)
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
