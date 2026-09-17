"""
Generate synthetic historical KPI data for N cells over T intervals.

Each cell has a diurnal traffic pattern (sinusoidal with a business-hours peak),
Gaussian noise, and an independent random sleep state history.

Output: data/synthetic/historical_kpi.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "historical_kpi.csv"

# Per-cell profile offsets so cells aren't identical
_CELL_PHASE_OFFSETS = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1]
_CELL_PEAK_SCALES   = [1.0, 0.8,  1.2, 0.9,  1.1, 0.7]


def diurnal_load(hour: float, peak_hour: float = 13.0, width: float = 6.0) -> float:
    """Return a [0, 1] load factor for a given hour using a Gaussian envelope."""
    return float(np.exp(-0.5 * ((hour - peak_hour) / width) ** 2))


def generate(
    n_cells: int = 6,
    n_days: int = 7,
    interval_minutes: int = 15,
    peak_throughput_mbps: float = 100.0,
    noise_std: float = 5.0,
    fault_probability: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intervals_per_day = (24 * 60) // interval_minutes
    total_intervals = n_days * intervals_per_day

    timestamps = pd.date_range(
        start="2024-01-01", periods=total_intervals, freq=f"{interval_minutes}min"
    )

    rows = []
    for cell_id in range(n_cells):
        phase  = _CELL_PHASE_OFFSETS[cell_id % len(_CELL_PHASE_OFFSETS)]
        scale  = _CELL_PEAK_SCALES[cell_id % len(_CELL_PEAK_SCALES)]

        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0 + phase
            load_factor = diurnal_load(hour) * scale
            throughput  = load_factor * peak_throughput_mbps
            throughput += rng.normal(0, noise_std)
            throughput  = float(np.clip(throughput, 0, peak_throughput_mbps * 1.2))

            utilization = throughput / peak_throughput_mbps
            latency_ms  = 5.0 + (1 - load_factor) * 10 + rng.normal(0, 0.5)
            latency_ms  = float(np.clip(latency_ms, 1, 50))

            # Cell sleeps when load is very low and no recent activity
            sleep_state = int(utilization < 0.10 and rng.random() > 0.3)

            rows.append(
                {
                    "timestamp":        ts,
                    "cell_id":          cell_id,
                    "throughput_mbps":  round(throughput, 2),
                    "utilization_pct":  round(utilization * 100, 2),
                    "latency_ms":       round(latency_ms, 2),
                    "sleep_state":      sleep_state,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic historical KPI data")
    parser.add_argument("--n-cells",    type=int,   default=6)
    parser.add_argument("--n-days",     type=int,   default=7)
    parser.add_argument("--interval",   type=int,   default=15,    help="Minutes per interval")
    parser.add_argument("--peak-mbps",  type=float, default=100.0)
    parser.add_argument("--noise-std",  type=float, default=5.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        type=Path,  default=OUT_PATH)
    args = parser.parse_args()

    df = generate(
        n_cells=args.n_cells,
        n_days=args.n_days,
        interval_minutes=args.interval,
        peak_throughput_mbps=args.peak_mbps,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✓ Historical KPI  → {args.out}  ({len(df):,} rows, {args.n_cells} cells, {args.n_days} days)")


if __name__ == "__main__":
    main()
