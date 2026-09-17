"""
Generate synthetic alarm and fault events.

Each cell independently draws a fault at each interval from a Bernoulli
process with configurable probability. Faults have a duration drawn from
a geometric distribution (mean ~3 intervals). Overlapping faults on the
same cell are merged.

Output: data/synthetic/faults.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HISTORICAL_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "historical_kpi.csv"
OUT_PATH        = Path(__file__).parent.parent / "data" / "synthetic" / "faults.csv"

FAULT_TYPES = [
    "hardware_degradation",
    "high_interference",
    "handover_failure",
    "power_anomaly",
]

FAULT_SEVERITY = {
    "hardware_degradation": "high",
    "high_interference":    "medium",
    "handover_failure":     "medium",
    "power_anomaly":        "high",
}


def generate(
    historical_path: Path = HISTORICAL_PATH,
    fault_probability: float = 0.05,
    mean_duration_intervals: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(historical_path, parse_dates=["timestamp"])

    rows = []
    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df   = cell_df.sort_values("timestamp").reset_index(drop=True)
        n         = len(cell_df)
        active    = False
        remaining = 0

        for i in range(n):
            ts = cell_df.loc[i, "timestamp"]

            if active:
                remaining -= 1
                if remaining <= 0:
                    active = False
            elif rng.random() < fault_probability:
                active    = True
                remaining = int(rng.geometric(p=1.0 / mean_duration_intervals))
                fault_type = rng.choice(FAULT_TYPES)
                rows.append(
                    {
                        "timestamp": ts,
                        "cell_id":   int(cell_id),
                        "fault_type": fault_type,
                        "severity":  FAULT_SEVERITY[fault_type],
                        "active":    True,
                    }
                )
                continue

            if active:
                # Carry forward the most recent fault type for this cell
                last = next(
                    (r for r in reversed(rows) if r["cell_id"] == int(cell_id)), None
                )
                fault_type = last["fault_type"] if last else rng.choice(FAULT_TYPES)
                rows.append(
                    {
                        "timestamp": ts,
                        "cell_id":   int(cell_id),
                        "fault_type": fault_type,
                        "severity":  FAULT_SEVERITY[fault_type],
                        "active":    True,
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "cell_id", "fault_type", "severity", "active"]
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fault/alarm events")
    parser.add_argument("--historical",      type=Path,  default=HISTORICAL_PATH)
    parser.add_argument("--fault-prob",      type=float, default=0.05)
    parser.add_argument("--mean-duration",   type=int,   default=3,
                        help="Mean fault duration in intervals")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--out",             type=Path,  default=OUT_PATH)
    args = parser.parse_args()

    df = generate(
        historical_path=args.historical,
        fault_probability=args.fault_prob,
        mean_duration_intervals=args.mean_duration,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✓ Fault feed       → {args.out}  ({len(df):,} fault-interval records)")


if __name__ == "__main__":
    main()
