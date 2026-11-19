"""
Maintenance scenario: a new kind of data, used with no new code.

Tonight's planned-maintenance schedule (data/synthetic/maintenance_windows.csv)
lists cells that must stay on air. No harness tool reads it. Two harnesses run
the same 24 h cycle on the mock RSG:

  fixed      the hand-wired full harness: the five tools, nothing else
  discovery  datasets chosen from the data catalog; maintenance_windows is read
             by the cell_blocker role reader through its certified catalog
             policy (agent/roles.py, data/catalog_policy.yaml)

and are scored on the reward (experiments/reward.py) and on maintenance
conflicts: 15-minute steps in which a cell was asleep inside its maintenance
window.

Discovery uses the LLM when an endpoint is configured (--discover llm, the
default when LLM_MODEL is set); --discover oracle selects every certified
relevant dataset, so the scenario runs with no endpoint. The Planner and
Validator are the tuner's stand-ins (experiments/tune_policy.py) on the mock;
confirm on VIAVI with the real models:
    python experiments/full_harness_vtzcycle.py --intent "5 Mbps" --discover \\
        --policy policies/tuned_calibrated.yaml

Run (after python generators/generate_all.py, or generate_maintenance.py and
generate_catalog.py):
    python experiments/maintenance_scenario.py
    python experiments/maintenance_scenario.py --discover oracle --intent 4

Output: output/maintenance_<timestamp>/report.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

from agent.catalog import CATALOG_PATH, load_catalog  # noqa: E402
from agent.context import ALL_TOOLS  # noqa: E402
from agent.discovery import DiscoveryResult, discover, resolve  # noqa: E402
from agent.policy import load_policy, tunable_params  # noqa: E402
from agent.roles import cell_blocker  # noqa: E402
from experiments.common import OUTPUT_DIR, connect_scenario  # noqa: E402
from experiments.reward import QOS_WEIGHT, SWITCH_WEIGHT, episode_metrics  # noqa: E402

DATASET = "maintenance_windows"


def asleep_cells(result: dict) -> set[int]:
    """N1 cells asleep after the iteration's actions."""
    kpis = result.get("post_kpis") or result.get("pre_kpis") or {}
    sites = kpis.get("per_site") or {}
    return {j for j, k in enumerate(sorted(sites)) if sites[k].get("n1_sleeping")}


def conflicts(results: list[dict], binding: dict) -> list[dict]:
    """Every (step, cell) where a cell was asleep inside its maintenance window."""
    out = []
    for r in results:
        windows = cell_blocker(r["timestamp"], binding)
        for cid in sorted(asleep_cells(r) & set(windows)):
            out.append({"iteration": r["iteration"], "time": str(r["timestamp"])[11:16],
                        "cell_id": cid, "work_type": windows[cid]})
    return out


def run_condition(name: str, intent: float, iterations: int, policy, tools, bindings) -> list[dict]:
    from experiments.full_harness_vtzcycle import run
    from experiments.tune_policy import FallbackPlanner, RuleValidator
    with contextlib.redirect_stdout(io.StringIO()):
        return run(connect_scenario(), f"{intent:g} Mbps", iterations, enabled_tools=tools,
                   condition_name=name, policy=policy, llm=FallbackPlanner(),
                   validator_llm=RuleValidator(), verbose=False, bindings=bindings)


def oracle(catalog: dict[str, dict]) -> DiscoveryResult:
    """What a perfect discovery step selects: every dataset a tool or a
    certified role reader can use."""
    chosen = [n for n, e in catalog.items() if e.get("tool") or e.get("binding")]
    return resolve([{"dataset": n, "reason": "oracle"} for n in chosen], catalog)


def main():
    load_dotenv(find_dotenv(), override=True)
    parser = argparse.ArgumentParser(description="Fixed vs discovery harness with a maintenance schedule")
    parser.add_argument("--intent",     type=float, default=5.0)
    parser.add_argument("--iterations", type=int,   default=96)
    parser.add_argument("--policy",     default="policies/tuned_calibrated.yaml")
    parser.add_argument("--catalog",    type=Path,  default=CATALOG_PATH)
    parser.add_argument("--discover",   choices=["llm", "oracle"],
                        default="llm" if os.getenv("LLM_MODEL") else "oracle")
    parser.add_argument("--model",      default=None, help="Discovery model (default: LLM_MODEL)")
    args = parser.parse_args()

    os.environ["USE_MOCK_RSG"] = "1"
    logging.disable(logging.INFO)
    policy  = load_policy(args.policy)
    catalog = load_catalog(args.catalog)
    if not (catalog.get(DATASET) or {}).get("binding"):
        sys.exit(f"{DATASET} is missing from the catalog or has no certified role. Run "
                 f"python generators/generate_maintenance.py && python generators/generate_catalog.py")
    truth = catalog[DATASET]["binding"]   # scores both harnesses, whatever discovery chose

    if args.discover == "llm":
        from experiments.common import get_llm
        llm = get_llm(args.model)
        found = discover(llm, f"{args.intent:g} Mbps", catalog)
        discovery_by = llm.model_name
    else:
        found, discovery_by = oracle(catalog), "oracle"
    print(found.summary())

    conditions = {
        "fixed":     (ALL_TOOLS, None),
        "discovery": (found.enabled_tools, found.bindings),
    }
    report = {
        "intent_mbps": args.intent, "iterations": args.iterations,
        "policy": {"name": policy.name, **tunable_params(policy)},
        "simulator": "mock_rsg", "planner": "fallback-planner", "validator": "rule-validator",
        "discovery": {"by": discovery_by, **found.to_dict()},
        "maintenance_windows": truth, "conditions": {},
    }
    print(f"\n{'':<10} {'reward':>7} {'asleep':>7} {'avoid. viol.':>12} {'switches':>8} "
          f"{'maint. conflicts':>16} {'cells':>6}")
    for name, (tools, bindings) in conditions.items():
        results = run_condition(name, args.intent, args.iterations, policy, tools, bindings)
        m = episode_metrics(results, args.intent, "energy", QOS_WEIGHT, SWITCH_WEIGHT)
        c = conflicts(results, truth)
        report["conditions"][name] = {
            "tools": sorted(tools), "bound_datasets": sorted(bindings or {}),
            "metrics": m, "maintenance_conflicts": len(c),
            "cells_in_conflict": sorted({x["cell_id"] for x in c}), "conflict_steps": c,
            "rule_rejections": sum(1 for r in results for x in r["rejected_actions"]
                                   if str(x.get("reason", "")).startswith("certified rule")),
        }
        print(f"{name:<10} {m['reward']:>7.4f} {m['mean_sleep_fraction']:>7.3f} "
              f"{m['avoidable_violation_rate']:>12.3f} {m['switches']:>8} {len(c):>16} "
              f"{len({x['cell_id'] for x in c}):>6}")

    out = OUTPUT_DIR / f"maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved → {out / 'report.json'}")


if __name__ == "__main__":
    main()
