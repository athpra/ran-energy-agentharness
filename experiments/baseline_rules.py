"""
Baseline A — Rule-based (no LLM).

Deterministic policy: sleep any cell whose current utilization is below
`sleep_threshold_pct`. Wake any sleeping cell whose utilization rises
above `wake_threshold_pct`. No tool context. No simulation test.

Run:
    python experiments/baseline_rules.py --intent "5 Mbps" --iterations 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from experiments.common import make_run_dir, save_results, apply_job_arguments


def rule_policy(
    cell_kpis: dict[int, dict],
    sleep_threshold_pct: float = 15.0,
    wake_threshold_pct:  float = 30.0,
) -> list[dict]:
    actions = []
    for cid, k in cell_kpis.items():
        util  = k.get("utilization_pct", 0)
        sleep = k.get("sleep_state", 0)
        if sleep == 0 and util < sleep_threshold_pct:
            actions.append({"action": "sleep", "cell_id": cid, "reason": f"util={util:.1f}% < threshold"})
        elif sleep == 1 and util > wake_threshold_pct:
            actions.append({"action": "wake",  "cell_id": cid, "reason": f"util={util:.1f}% > threshold"})
    return actions


def run(
    scenario,
    operator_intent: str,
    n_iterations: int,
    sleep_threshold_pct: float = 15.0,
    wake_threshold_pct:  float = 30.0,
    condition_name: str  = "baseline_rules",
) -> list[dict]:
    from experiments.common import make_sim_fns, _kpi_to_text_viavi
    sim_test_fn, sim_apply_fn, set_virtual_ts = make_sim_fns(scenario)

    intent_mbps = float(operator_intent.split()[0])
    ts = pd.Timestamp.now().normalize()

    results = []
    for i in range(n_iterations):
        t0         = time.time()
        virtual_ts = ts + pd.Timedelta(minutes=15 * i)
        set_virtual_ts(virtual_ts)

        # Read current state including accumulated sleep history
        _, kpis = sim_test_fn([])
        sk = sorted(kpis["per_site"].keys())
        cell_kpis = {
            j: {
                "utilization_pct": kpis["per_site"][k]["n1_prb"],
                "sleep_state":     1 if kpis["per_site"][k]["n1_sleeping"] else 0,
            }
            for j, k in enumerate(sk)
        }
        actions = rule_policy(cell_kpis, sleep_threshold_pct, wake_threshold_pct)

        sim_summary, post_kpis = sim_apply_fn(actions) if actions else (
            "No actions — nothing applied.", {}
        )

        post_tp  = post_kpis.get("avg_throughput_mbps") or kpis.get("avg_throughput_mbps", 0)
        sleeping = post_kpis.get("sleeping_cells", kpis.get("sleeping_cells", 0))
        total    = post_kpis.get("total_cells",    kpis.get("total_cells",    42))
        violated = post_tp < intent_mbps

        results.append({
            "iteration":         i + 1,
            "condition":         condition_name,
            "tools_used":        [],
            "proposed_actions":  actions,
            "approved_actions":  actions,
            "rejected_actions":  [],
            "n_proposed":        len(actions),
            "n_approved":        len(actions),
            "n_rejected":        0,
            "sim2_summary":      sim_summary,
            "post_kpis":         post_kpis,
            "qos_violated":      violated,
            "total_elapsed_s":   round(time.time() - t0, 3),
        })
        qos_tag = "✗ QoS VIOLATION" if violated else "✓"
        print(
            f"  [iter {i+1:>3}] actions={len(actions)}  "
            f"tp={post_tp:.2f} Mbps  sleeping={sleeping}/{total}  {qos_tag}"
        )

    return results


def main():
    apply_job_arguments()
    parser = argparse.ArgumentParser(description="Baseline A: rule-based policy")
    parser.add_argument("--intent",       default="5 Mbps")
    parser.add_argument("--iterations",   type=int,   default=20)
    parser.add_argument("--sleep-thresh", type=float, default=15.0)
    parser.add_argument("--wake-thresh",  type=float, default=30.0)
    parser.add_argument("--rsg-host",     default=os.getenv("RSG_HOST", ""))
    args, _ = parser.parse_known_args()

    from experiments.common import connect_scenario
    scenario = connect_scenario(args.rsg_host)

    print(f"Running: baseline_rules  intent='{args.intent}'  iterations={args.iterations}")
    results  = run(scenario, args.intent, args.iterations, args.sleep_thresh, args.wake_thresh)
    run_dir  = make_run_dir("baseline_rules", "rules")
    save_results(run_dir, results)


if __name__ == "__main__":
    main()
