"""
Tool: Traffic Prediction

Returns the N-step-ahead throughput forecast for each cell at the
current simulation timestamp (or the closest available entry).
"""
import pandas as pd
from pathlib import Path
from functools import lru_cache

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "traffic_forecast.csv"


@lru_cache(maxsize=1)
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["cell_id", "timestamp", "forecast_step"], inplace=True)
    return df


def query(
    timestamp: pd.Timestamp | str,
    cell_id: int | None = None,
    horizon: int = 3,
    data_path: Path = _DEFAULT_PATH,
) -> dict:
    """
    Return the traffic forecast for `horizon` steps ahead of `timestamp`.

    Returns a dict keyed by cell_id, each containing a list of
    {step, predicted_mbps} for steps 1..horizon.
    """
    df = _load(str(data_path))
    ts = pd.Timestamp(timestamp)

    # Find the nearest available forecast timestamp
    available = df["timestamp"].drop_duplicates().sort_values()
    nearest   = available.iloc[(available - ts).abs().argmin()]
    df = df[df["timestamp"] == nearest]

    if cell_id is not None:
        df = df[df["cell_id"] == cell_id]

    df = df[df["forecast_step"] <= horizon]

    result = {}
    for cid, g in df.groupby("cell_id"):
        result[int(cid)] = [
            {"step": int(row.forecast_step), "predicted_mbps": round(row.predicted_mbps, 2)}
            for row in g.itertuples()
        ]

    return result
