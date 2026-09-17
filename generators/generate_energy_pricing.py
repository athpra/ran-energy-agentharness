"""
Generate synthetic energy pricing data.

Uses a three-tier time-of-use (ToU) tariff structure calibrated to
published utility profiles (e.g., UK Octopus Agile, Singapore SP Group):
  - Off-peak : 22:00 – 07:00  → low rate
  - Shoulder  : 07:00 – 09:00, 21:00 – 22:00
  - Peak      : 09:00 – 21:00  → high rate

A small random noise term is added to simulate real-time pricing variation.

Output: data/synthetic/energy_pricing.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "energy_pricing.csv"

# Tariff rates in USD/kWh (representative mid-range values)
TARIFF = {
    "off_peak": 0.08,
    "shoulder": 0.14,
    "peak":     0.22,
}


def _tier(hour: int) -> str:
    if 22 <= hour or hour < 7:
        return "off_peak"
    if (7 <= hour < 9) or (21 <= hour < 22):
        return "shoulder"
    return "peak"


def generate(
    n_days: int = 7,
    interval_minutes: int = 15,
    noise_std: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2024-01-01", periods=n_days * (24 * 60 // interval_minutes),
        freq=f"{interval_minutes}min"
    )

    rows = []
    for ts in timestamps:
        tier       = _tier(ts.hour)
        base_price = TARIFF[tier]
        price      = float(np.clip(base_price + rng.normal(0, noise_std), 0.01, None))
        rows.append(
            {
                "timestamp":       ts,
                "tier":            tier,
                "price_per_kwh":   round(price, 4),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic energy pricing data")
    parser.add_argument("--n-days",    type=int,   default=7)
    parser.add_argument("--interval",  type=int,   default=15, help="Minutes per interval")
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       type=Path,  default=OUT_PATH)
    args = parser.parse_args()

    df = generate(
        n_days=args.n_days,
        interval_minutes=args.interval,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✓ Energy pricing   → {args.out}  ({len(df):,} intervals, {args.n_days} days)")
    print(f"  Tiers: off_peak=${TARIFF['off_peak']}/kWh  "
          f"shoulder=${TARIFF['shoulder']}/kWh  "
          f"peak=${TARIFF['peak']}/kWh  (±noise)")


if __name__ == "__main__":
    main()
