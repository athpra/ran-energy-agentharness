"""
Tool: Historical KPI Store

Answers questions about past cell behaviour — load patterns, typical
sleep rates, baseline throughput — over a configurable lookback window.
"""
import pandas as pd
from pathlib import Path
from functools import lru_cache

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "historical_kpi.csv"


@lru_cache(maxsize=1)
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["cell_id", "timestamp"], inplace=True)
    return df


def query(
    cell_id: int | None = None,
    lookback_intervals: int = 96,   # ~24 hours at 15-min intervals
    data_path: Path = _DEFAULT_PATH,
) -> dict:
    """
    Return summary statistics for the requested cell(s) over the most
    recent `lookback_intervals` intervals in the historical record.

    Returns a dict keyed by cell_id with:
      - mean_throughput_mbps
      - mean_utilization_pct
      - sleep_rate  (fraction of intervals where the cell was sleeping)
      - p10/p90_throughput  (traffic envelope)
    """
    df = _load(str(data_path))

    if cell_id is not None:
        df = df[df["cell_id"] == cell_id]

    result = {}
    for cid, g in df.groupby("cell_id"):
        recent = g.tail(lookback_intervals)
        result[int(cid)] = {
            "mean_throughput_mbps": round(recent["throughput_mbps"].mean(), 2),
            "mean_utilization_pct": round(recent["utilization_pct"].mean(), 2),
            "sleep_rate":           round(recent["sleep_state"].mean(), 3),
            "p10_throughput_mbps":  round(recent["throughput_mbps"].quantile(0.10), 2),
            "p90_throughput_mbps":  round(recent["throughput_mbps"].quantile(0.90), 2),
        }

    return result
