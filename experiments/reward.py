"""
Episode reward.

One number per run, computed from the per-iteration results the harness
already logs (pre_kpis / post_kpis / approved_actions), so it works on a
results list in memory or on any output/run_*/iterations.jsonl.

    reward = saving − qos_weight × avoidable_violation_rate − switch_weight × switches_per_iteration

  saving                   mean share of sleepable N1 cells asleep ("energy"),
                           or the same weighted by the electricity price ("cost")
  avoidable_violation_rate share of iterations below intent while a cell was
                           asleep; violations with every cell awake are a
                           capacity limit the policy cannot fix, so they are
                           reported but not penalised
  switches_per_iteration   sleep + wake actions applied, per iteration
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.energy_pricing import query as energy_pricing_query

QOS_WEIGHT    = 5.0
SWITCH_WEIGHT = 0.01


def load_iterations(run_dir: str | Path) -> list[dict]:
    with open(Path(run_dir) / "iterations.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_metrics(
    iterations: list[dict],
    intent_mbps: float,
    objective: str = "energy",
    qos_weight: float = QOS_WEIGHT,
    switch_weight: float = SWITCH_WEIGHT,
) -> dict:
    if objective not in ("energy", "cost"):
        raise ValueError("objective must be 'energy' or 'cost'")
    n = len(iterations)
    if n == 0:
        raise ValueError("no iterations to score")

    saving = cost = violations = avoidable = switches = 0.0
    peak_price = max(energy_pricing_query(pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h))["current_price_per_kwh"]
                     for h in range(24))
    for it in iterations:
        state = it.get("post_kpis") or it.get("pre_kpis") or {}
        sleepable = len(state.get("per_site", {})) or 21
        frac = state.get("sleeping_cells", 0) / sleepable
        price = energy_pricing_query(it["timestamp"])["current_price_per_kwh"]
        saving += frac
        cost += frac * price / peak_price
        tp = state.get("avg_throughput_mbps", 0.0)
        if tp < intent_mbps:
            violations += 1
            if state.get("sleeping_cells", 0) > 0:
                avoidable += 1
        switches += len(it.get("approved_actions", []))

    gain = (saving if objective == "energy" else cost) / n
    reward = gain - qos_weight * avoidable / n - switch_weight * switches / n
    return {
        "reward":                   round(reward, 4),
        "objective":                objective,
        "mean_sleep_fraction":      round(saving / n, 4),
        "cost_weighted_saving":     round(cost / n, 4),
        "qos_violation_rate":       round(violations / n, 4),
        "avoidable_violation_rate": round(avoidable / n, 4),
        "switches":                 int(switches),
        "iterations":               n,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a finished run")
    parser.add_argument("run_dir")
    parser.add_argument("--intent", type=float, required=True, help="Intent in Mbps, e.g. 5")
    parser.add_argument("--objective", choices=["energy", "cost"], default="energy")
    args = parser.parse_args()
    print(json.dumps(episode_metrics(load_iterations(args.run_dir), args.intent, args.objective), indent=2))
