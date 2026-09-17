"""
Tool: Energy Pricing

Returns the current electricity price and tier, plus a short-horizon
forecast. The agent can use this to modulate aggressiveness: sleep more
cells during peak pricing, be conservative during off-peak.
"""
import pandas as pd
from pathlib import Path
from functools import lru_cache

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "energy_pricing.csv"


@lru_cache(maxsize=1)
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


def query(
    timestamp: pd.Timestamp | str,
    horizon: int = 3,
    data_path: Path = _DEFAULT_PATH,
) -> dict:
    """
    Return current price and a horizon-step price forecast.

    Returns:
      - current_price_per_kwh (float)
      - current_tier          (str: off_peak | shoulder | peak)
      - forecast              list of {step, price_per_kwh, tier}
    """
    df = _load(str(data_path))
    ts = pd.Timestamp(timestamp)

    # Current: nearest record at or before ts
    past    = df[df["timestamp"] <= ts]
    current = past.iloc[-1] if not past.empty else df.iloc[0]

    # Forecast: next `horizon` records after ts
    future  = df[df["timestamp"] > ts].head(horizon)
    forecast = [
        {
            "step":             i + 1,
            "price_per_kwh":    round(row.price_per_kwh, 4),
            "tier":             row.tier,
        }
        for i, row in enumerate(future.itertuples())
    ]

    return {
        "current_price_per_kwh": round(float(current["price_per_kwh"]), 4),
        "current_tier":          str(current["tier"]),
        "forecast":              forecast,
    }
