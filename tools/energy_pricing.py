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

    Lookup is time-of-day based so virtual timestamps (VTZ cycle) return the
    correct tier regardless of what calendar date the virtual clock shows.

    Returns:
      - current_price_per_kwh (float)
      - current_tier          (str: off_peak | shoulder | peak)
      - forecast              list of {step, price_per_kwh, tier}
    """
    df = _load(str(data_path))
    ts = pd.Timestamp(timestamp)

    # Normalise to the CSV's reference date so time-of-day lookup works
    # regardless of the virtual timestamp's calendar date.
    ref_date = df["timestamp"].iloc[0].normalize()
    ts_norm  = ref_date + pd.Timedelta(hours=ts.hour, minutes=ts.minute)

    past    = df[df["timestamp"] <= ts_norm]
    current = past.iloc[-1] if not past.empty else df.iloc[0]

    future   = df[df["timestamp"] > ts_norm].head(horizon)
    forecast = [
        {
            "step":          i + 1,
            "price_per_kwh": round(row.price_per_kwh, 4),
            "tier":          row.tier,
        }
        for i, row in enumerate(future.itertuples())
    ]

    return {
        "current_price_per_kwh": round(float(current["price_per_kwh"]), 4),
        "current_tier":          str(current["tier"]),
        "forecast":              forecast,
    }
