"""
Calibrate the mock RSG against VIAVI runs.

Reads the per-iteration KPIs that VIAVI runs already log, and fits, for each
traffic level (the profile's UE ratio):
  - throughput for each set of sleeping N1 cells VIAVI was run with. VIAVI is
    deterministic, so the mock reproduces any logged state exactly;
  - throughput by number of sleeping N1 cells, for sets never logged: the mean
    of the logged points at each count, with a least-squares line for counts
    never observed. Different sets with the same count differ in throughput,
    so this is an estimate;
  - each N1 cell's PRB utilisation while awake. On VIAVI it does not change
    when neighbours sleep, so one value per cell and traffic level is enough.

The mock (mock_rsg/scenario.py) uses the result when it exists, so the policy
tuner sees VIAVI's throughput levels instead of the analytic radio model.

Run (VIAVI runs only; mock runs are skipped):
    python -m mock_rsg.calibrate output/run_*
    python -m mock_rsg.calibrate output/run_2026092* --out /tmp/calibration.json

Output: mock_rsg/viavi_calibration.json (default)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import _get_profile  # noqa: E402

CALIBRATION_PATH = Path(__file__).with_name("viavi_calibration.json")


def _is_mock_run(iterations: list[dict]) -> bool:
    for it in iterations:
        for k in ("pre_kpis", "post_kpis"):
            sites = (it.get(k) or {}).get("per_site") or {}
            if sites:
                return any(s.startswith("gNB") for s in sites)   # mock cell names
    return False


def _sleeping_key(state: dict) -> str:
    """The sleeping N1 cells of a state, as sorted cell indices: "0,2,4".
    Cell index j is the j-th site in sorted order, the same as the mock's."""
    return ",".join(str(j) for j, key in enumerate(sorted(state["per_site"]))
                    if state["per_site"][key].get("n1_sleeping"))


def _states(run_dir: Path):
    """(ue_ratio, kpi state) for every logged pre/post state of one run."""
    path = run_dir / "iterations.jsonl"
    if not path.exists():
        return None
    iterations = [json.loads(line) for line in open(path) if line.strip()]
    if not iterations or _is_mock_run(iterations):
        return None
    out = []
    for it in iterations:
        # Older logs have no timestamp; iteration n is 15 × (n − 1) minutes after midnight
        ts = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * (it["iteration"] - 1))
        ratio = _get_profile(ts)["ue_ratio"]
        for k in ("pre_kpis", "post_kpis"):
            state = it.get(k) or {}
            if "avg_throughput_mbps" in state and state.get("per_site"):
                out.append((ratio, state))
    return out


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Intercept and (non-negative) drop per sleeping cell."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    if len(set(xs)) < 2:
        return statistics.mean(ys), 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    drop = max(0.0, -slope)
    return my + drop * mx, drop


def calibrate(run_dirs: list[Path]) -> dict:
    tp_points: dict[float, list[tuple[int, float]]] = defaultdict(list)
    tp_sets: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prb: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    used = []
    for d in run_dirs:
        states = _states(Path(d))
        if not states:
            continue
        used.append(Path(d).name)
        for ratio, state in states:
            tp_points[ratio].append((state["sleeping_cells"], state["avg_throughput_mbps"]))
            tp_sets[ratio][_sleeping_key(state)].append(state["avg_throughput_mbps"])
            for j, key in enumerate(sorted(state["per_site"])):
                site = state["per_site"][key]
                if not site.get("n1_sleeping"):
                    prb[ratio][j].append(site["n1_prb"])
    if not used:
        raise ValueError("no VIAVI runs found")

    n_cells = 1 + max(j for cells in prb.values() for j in cells)
    profiles = {}
    for ratio in sorted(tp_points):
        pts = tp_points[ratio]
        tp0, drop = _fit_line(pts)
        resid = [abs(y - (tp0 - drop * x)) for x, y in pts]
        by_n: dict[int, list[float]] = defaultdict(list)
        for x, y in pts:
            by_n[x].append(y)
        sets = tp_sets[ratio]
        profiles[f"{ratio:g}"] = {
            "tp_by_sleeping_set_mbps": {k: round(statistics.mean(v), 3) for k, v in sorted(sets.items())},
            "max_set_spread_mbps":   round(max(max(v) - min(v) for v in sets.values()), 3),
            "tp_by_sleeping_mbps":   {str(n): round(statistics.mean(v), 3) for n, v in sorted(by_n.items())},
            "tp_all_awake_mbps":     round(tp0, 3),
            "tp_drop_per_sleeping":  round(drop, 4),
            "max_abs_error_mbps":    round(max(resid), 3),
            "n_points":              len(pts),
            "max_sleeping_observed": max(x for x, _ in pts),
            "n1_prb": [round(statistics.median(prb[ratio][j]), 1) if prb[ratio].get(j) else None
                       for j in range(n_cells)],
        }
    return {"source_runs": used, "n1_cells": n_cells, "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description="Calibrate the mock RSG against VIAVI runs")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()

    cal = calibrate(args.run_dirs)
    args.out.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"✓ Calibration from {len(cal['source_runs'])} VIAVI runs → {args.out}")
    for ratio, p in cal["profiles"].items():
        obs = p["tp_by_sleeping_mbps"]
        print(f"  load {ratio:>4}×  {p['n_points']:>3} points, {len(p['tp_by_sleeping_set_mbps']):>3} sleeping sets "
              f"(spread within a set {p['max_set_spread_mbps']:.3f} Mbps)  tp with 0 asleep "
              f"{obs.get('0', p['tp_all_awake_mbps']):.2f} Mbps, sleeping counts seen {list(obs)}")


if __name__ == "__main__":
    main()
