"""
Generate tonight's planned-maintenance schedule.

Planned work that needs the cell on air (post-upgrade verification, drive
tests, antenna alignment) in the usual overnight window, 00:00–05:00. That is
also when the agent puts most capacity cells to sleep, so a harness that does
not read this schedule sleeps cells that engineers are working on.

No harness tool reads this file. The agent uses it only through the data
catalog: discovery chooses it, and its certified catalog policy binds it to
the cell_blocker role reader (agent/roles.py). It tests whether a new kind of
data can be used without new code.

Output: data/synthetic/maintenance_windows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR        = Path(__file__).parent.parent / "data" / "synthetic"
HISTORICAL_PATH = DATA_DIR / "historical_kpi.csv"
OUT_PATH        = DATA_DIR / "maintenance_windows.csv"

WORK_TYPES = ["sw_upgrade_verification", "drive_test", "antenna_alignment"]


def generate(historical_path: Path = HISTORICAL_PATH, n_windows: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hist = pd.read_csv(historical_path, usecols=["timestamp", "cell_id"])
    day = pd.Timestamp(hist["timestamp"].min()).normalize()     # the data's reference day
    cells = rng.choice(sorted(hist["cell_id"].unique()), size=n_windows, replace=False)

    rows = []
    for i, cell in enumerate(sorted(int(c) for c in cells)):
        start = int(rng.integers(0, 12)) * 15                    # 00:00–02:45
        length = int(rng.integers(4, 9)) * 15                    # 1–2 h
        rows.append({
            "ticket_id":    f"CHG-{4100 + i}",
            "cell_id":      cell,
            "window_start": str(day + pd.Timedelta(minutes=start)),
            "window_end":   str(day + pd.Timedelta(minutes=start + length)),
            "work_type":    str(rng.choice(WORK_TYPES)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate the planned-maintenance schedule")
    parser.add_argument("--historical", type=Path, default=HISTORICAL_PATH)
    parser.add_argument("--out",        type=Path, default=OUT_PATH)
    parser.add_argument("--n-windows",  type=int,  default=5)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    df = generate(args.historical, args.n_windows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"✓ Maintenance windows → {args.out}  ({len(df)} windows on cells {df.cell_id.tolist()})")


if __name__ == "__main__":
    main()
