"""
Tune the harness policy (agent/policy.py) offline.

Random search over the policy thresholds, scored by experiments/reward.py on
full 96-step traffic cycles. A candidate is promoted to policies/tuned.yaml
only if it beats the default policy on HELD-OUT intents, with no more
avoidable QoS violations there than the default has.

Each trial runs on the mock RSG with two stand-ins, so a trial takes seconds
and costs no inference. The mock uses the VIAVI calibration in
mock_rsg/viavi_calibration.json when it exists (python -m mock_rsg.calibrate
output/run_*); without it, the analytic radio model, whose throughput levels
differ from VIAVI's. The stand-ins:
  - Planner: returns [], so the harness uses its tool-derived candidates.
    That is what the LLM Planner does in most acting iterations today.
  - Validator: the rule in the Validator prompt's primary check: reject every
    sleep when the Sim 1 throughput is below intent.
Confirm a promoted policy with the real models on VIAVI before reporting it:
    python experiments/full_harness_vtzcycle.py --intent "5 Mbps" --policy policies/tuned.yaml

Run:
    python experiments/tune_policy.py                     # 40 trials
    python experiments/tune_policy.py --trials 100 --objective cost
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import random
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.policy import DEFAULT_POLICY, POLICIES_DIR, SEARCH_SPACE, Policy, check, save_policy, tunable_params  # noqa: E402
from experiments.common import OUTPUT_DIR, connect_scenario  # noqa: E402
from experiments.reward import QOS_WEIGHT, SWITCH_WEIGHT, episode_metrics  # noqa: E402


class _Reply:
    def __init__(self, content: str):
        self.content = content


class FallbackPlanner:
    """Planner stand-in: proposes nothing, so the harness acts on its
    tool-derived sleep and wake candidates."""
    model_name = "fallback-planner"

    def invoke(self, messages):
        return _Reply("[]")


class RuleValidator:
    """Validator stand-in: the prompt's primary check as code. Rejects every
    sleep action when the Sim 1 throughput is below the intent."""
    model_name = "rule-validator"

    def invoke(self, messages):
        text = messages[-1]["content"]
        tp = float(re.search(r"### Simulation Results\s*\navg_throughput=([\d.]+)", text).group(1))
        intent = float(re.search(r"### Operator Intent\s*\n([\d.]+)", text).group(1))
        actions = json.loads(re.search(r"```json\n(.*?)\n```", text, re.DOTALL).group(1))
        rejected = [
            {"action": a["action"], "cell_id": a["cell_id"], "reason": "Sim 1 throughput below intent"}
            for a in actions if a["action"] == "sleep"
        ] if tp < intent else []
        return _Reply(json.dumps({"approved": [], "rejected": rejected}))


def evaluate(policy: Policy, intents: list[float], iterations: int, objective: str,
             qos_weight: float, switch_weight: float) -> dict:
    """Mean metrics over one full episode per intent."""
    from experiments.full_harness_vtzcycle import run

    per_intent = {}
    for intent in intents:
        with contextlib.redirect_stdout(io.StringIO()):   # the per-run setup chatter
            results = run(connect_scenario(), f"{intent:g} Mbps", iterations, policy=policy,
                          llm=FallbackPlanner(), validator_llm=RuleValidator(), verbose=False)
        per_intent[f"{intent:g}"] = episode_metrics(results, intent, objective, qos_weight, switch_weight)
    keys = ("reward", "mean_sleep_fraction", "cost_weighted_saving", "qos_violation_rate",
            "avoidable_violation_rate", "switches")
    mean = {k: round(sum(m[k] for m in per_intent.values()) / len(per_intent), 4) for k in keys}
    return {**mean, "per_intent": per_intent}


def _simulator_name() -> str:
    if os.environ.get("USE_MOCK_RSG") != "1":
        return "viavi"
    from mock_rsg.scenario import load_calibration
    cal = load_calibration()
    return f"mock_rsg (VIAVI-calibrated, {len(cal['source_runs'])} runs)" if cal else "mock_rsg (analytic)"


def sample(rng: random.Random, trial: int) -> Policy:
    values = {k: round(rng.uniform(lo, hi), 2) for k, (lo, hi) in SEARCH_SPACE.items()}
    values["prb_sleep_threshold_peak"] = max(values["prb_sleep_threshold_peak"], values["prb_sleep_threshold"])
    return check(replace(DEFAULT_POLICY, name=f"trial_{trial:03d}", **values))


def main():
    parser = argparse.ArgumentParser(description="Tune the harness policy on the mock RSG")
    parser.add_argument("--trials",          type=int,   default=40)
    parser.add_argument("--train-intents",   type=float, nargs="+", default=[4.0, 5.0])
    parser.add_argument("--holdout-intents", type=float, nargs="+", default=[3.0, 6.0])
    parser.add_argument("--iterations",      type=int,   default=96, help="Steps per episode (96 = 24 h)")
    parser.add_argument("--objective",       choices=["energy", "cost"], default="energy")
    parser.add_argument("--qos-weight",      type=float, default=QOS_WEIGHT)
    parser.add_argument("--switch-weight",   type=float, default=SWITCH_WEIGHT)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--out",             type=Path,  default=POLICIES_DIR / "tuned.yaml")
    args = parser.parse_args()

    if set(args.train_intents) & set(args.holdout_intents):
        sys.exit("Train and held-out intents must differ.")
    os.environ.setdefault("USE_MOCK_RSG", "1")
    logging.disable(logging.INFO)
    score = lambda p, intents: evaluate(p, intents, args.iterations, args.objective,  # noqa: E731
                                        args.qos_weight, args.switch_weight)

    run_dir = OUTPUT_DIR / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_log = open(run_dir / "trials.jsonl", "w")

    def log(policy: Policy, train: dict) -> None:
        trials_log.write(json.dumps({"policy": policy.name, **tunable_params(policy), "train": train}) + "\n")
        trials_log.flush()

    print(f"Tuning on {_simulator_name()}: {args.trials} trials, train intents {args.train_intents}, "
          f"held-out {args.holdout_intents}, objective={args.objective}")
    base_train = score(DEFAULT_POLICY, args.train_intents)
    log(DEFAULT_POLICY, base_train)
    print(f"  default   train reward={base_train['reward']:.4f}  sleep={base_train['mean_sleep_fraction']:.3f}  "
          f"avoidable violations={base_train['avoidable_violation_rate']:.3f}")

    rng = random.Random(args.seed)
    best, best_train = DEFAULT_POLICY, base_train
    for t in range(1, args.trials + 1):
        cand = sample(rng, t)
        train = score(cand, args.train_intents)
        log(cand, train)
        if train["reward"] > best_train["reward"]:
            best, best_train = cand, train
            print(f"  trial {t:>3} train reward={train['reward']:.4f}  sleep={train['mean_sleep_fraction']:.3f}  "
                  f"avoidable violations={train['avoidable_violation_rate']:.3f}  ← best so far")
    trials_log.close()

    # Promotion gate: judged only on intents the search never saw
    base_hold = score(DEFAULT_POLICY, args.holdout_intents)
    best_hold = score(best, args.holdout_intents)
    promoted = (
        best != DEFAULT_POLICY
        and best_hold["reward"] > base_hold["reward"]
        and best_hold["avoidable_violation_rate"] <= base_hold["avoidable_violation_rate"]
    )
    report = {
        "promoted":        promoted,
        "objective":       args.objective,
        "weights":         {"qos": args.qos_weight, "switch": args.switch_weight},
        "train_intents":   args.train_intents,
        "holdout_intents": args.holdout_intents,
        "trials":          args.trials,
        "default":         {"params": tunable_params(DEFAULT_POLICY), "train": base_train, "holdout": base_hold},
        "best":            {"params": tunable_params(best), "train": best_train, "holdout": best_hold},
        "simulator":       _simulator_name(),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\nHeld-out  default reward={base_hold['reward']:.4f}  best reward={best_hold['reward']:.4f}  "
          f"(sleep {base_hold['mean_sleep_fraction']:.3f} → {best_hold['mean_sleep_fraction']:.3f}, "
          f"avoidable violations {base_hold['avoidable_violation_rate']:.3f} → "
          f"{best_hold['avoidable_violation_rate']:.3f}, switches {base_hold['switches']:g} → {best_hold['switches']:g})")
    if promoted:
        tuned = replace(best, name=args.out.stem)
        save_policy(tuned, args.out, meta={
            "tuned_from": best.name, "tuning_run": run_dir.name, "objective": args.objective,
            "train_reward": best_train["reward"], "holdout_reward": best_hold["reward"],
            "default_holdout_reward": base_hold["reward"], "simulator": report["simulator"],
        })
        print(f"✓ Promoted → {args.out}")
    else:
        print("✗ Not promoted: the best candidate did not beat the default on held-out intents.")
    print(f"Trials and report → {run_dir}")


if __name__ == "__main__":
    main()
