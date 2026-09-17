"""
Tool: Alarm and Fault Feed

Returns cells with active faults at the given simulation timestamp.
The agent should avoid proposing sleep actions on faulted cells.
"""
import pandas as pd
from pathlib import Path
from functools import lru_cache

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "faults.csv"


@lru_cache(maxsize=1)
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values(["cell_id", "timestamp"], inplace=True)
    return df


def query(
    timestamp: pd.Timestamp | str,
    cell_id: int | None = None,
    data_path: Path = _DEFAULT_PATH,
) -> dict:
    """
    Return active fault status at `timestamp`.

    Returns a dict keyed by cell_id:
      - active (bool)
      - fault_type (str or None)
      - severity (str or None)

    Cells with no record are returned as active=False.
    """
    df = _load(str(data_path))
    ts = pd.Timestamp(timestamp)

    # Use the most recent fault record at or before the requested timestamp
    df = df[df["timestamp"] <= ts]

    if cell_id is not None:
        df = df[df["cell_id"] == cell_id]

    result: dict[int, dict] = {}
    for cid, g in df.groupby("cell_id"):
        latest = g.sort_values("timestamp").iloc[-1]
        result[int(cid)] = {
            "active":     bool(latest["active"]),
            "fault_type": str(latest["fault_type"]) if latest["active"] else None,
            "severity":   str(latest["severity"])   if latest["active"] else None,
        }

    return result
