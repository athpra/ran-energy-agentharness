"""
Generate synthetic traffic forecasts from historical KPI data.

Uses a rolling mean over the historical throughput series to produce
an N-step-ahead forecast per cell. Adds calibrated noise so forecasts
are plausible but not perfect — this models real predictor uncertainty.

Output: data/synthetic/traffic_forecast.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HISTORICAL_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "historical_kpi.csv"
OUT_PATH        = Path(__file__).parent.parent / "data" / "synthetic" / "traffic_forecast.csv"


def generate(
    historical_path: Path = HISTORICAL_PATH,
    horizon: int = 3,
    window: int = 8,
    forecast_noise_std: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each (timestamp, cell_id) in the historical data, emit a horizon-step
    forecast using a rolling window mean over the past `window` observations.
    """
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(historical_path, parse_dates=["timestamp"])

    rows = []
    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("timestamp").reset_index(drop=True)
        throughput = cell_df["throughput_mbps"].values

        for i in range(window, len(cell_df)):
            base_forecast = throughput[max(0, i - window) : i].mean()
            ts = cell_df.loc[i, "timestamp"]

            for step in range(1, horizon + 1):
                noise     = rng.normal(0, forecast_noise_std)
                predicted = float(np.clip(base_forecast + noise, 0, None))
                rows.append(
                    {
                        "timestamp":             ts,
                        "cell_id":               int(cell_id),
                        "forecast_step":         step,
                        "predicted_mbps":        round(predicted, 2),
                        "forecast_window_mean":  round(base_forecast, 2),
                    }
                )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic forecasts")
    parser.add_argument("--historical", type=Path,  default=HISTORICAL_PATH)
    parser.add_argument("--horizon",    type=int,   default=3,  help="Steps ahead to forecast")
    parser.add_argument("--window",     type=int,   default=8,  help="Rolling mean window size")
    parser.add_argument("--noise-std",  type=float, default=3.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        type=Path,  default=OUT_PATH)
    args = parser.parse_args()

    df = generate(
        historical_path=args.historical,
        horizon=args.horizon,
        window=args.window,
        forecast_noise_std=args.noise_std,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✓ Traffic forecast → {args.out}  ({len(df):,} rows, horizon={args.horizon})")


if __name__ == "__main__":
    main()
