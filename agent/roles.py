"""
Role readers.

A role reader is one generic function per way data can affect the sleep
decision. It is written once; a dataset is bound to it by a certified catalog
entry (data/catalog_policy.yaml) that names the role and maps the dataset's
columns onto the reader's fields. A new dataset of an existing role then needs
a catalog entry and a review, not a new tool.

Roles so far:
  cell_blocker  the cell must be on air while the record is active: it may not
                be put to sleep, and a sleeping cell is woken. The harness
                enforces this in code (agent/harness.py), not through the LLMs.

A binding is the dict a reader takes:
  {"role": "cell_blocker", "source": "<file>", "mapping": {<field>: <column>}}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROLES = frozenset({"cell_blocker"})

# The fields each shape of cell_blocker data maps onto. A dataset uses one shape.
CELL_BLOCKER_SHAPES = {
    "windows": {"cell", "start", "end"},     # one row per window: blocked while start <= t < end
    "events":  {"cell", "time", "active"},   # state changes: blocked while the latest event is active
}


class BindingError(ValueError):
    pass


def check_binding(name: str, binding: dict) -> None:
    """Raise BindingError if a binding cannot be read: unknown role, or a
    mapping that matches no shape of the role."""
    role, mapping = binding.get("role"), binding.get("mapping") or {}
    if role not in ROLES:
        raise BindingError(f"{name}: unknown role {role!r} (known: {sorted(ROLES)})")
    if not any(fields <= set(mapping) for fields in CELL_BLOCKER_SHAPES.values()):
        shapes = "; ".join(f"{k}: {sorted(v)}" for k, v in CELL_BLOCKER_SHAPES.items())
        raise BindingError(f"{name}: mapping {sorted(mapping)} fits no cell_blocker shape ({shapes})")


@lru_cache(maxsize=16)
def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _time_of_day(ts: pd.Timestamp, ref_date: pd.Timestamp) -> pd.Timestamp:
    """The synthetic data covers one reference day; like tools/alarm_fault.py,
    read it by time of day so any virtual date works."""
    return ref_date + pd.Timedelta(hours=ts.hour, minutes=ts.minute)


def cell_blocker(timestamp: pd.Timestamp | str, binding: dict) -> dict[int, str]:
    """Cells blocked at `timestamp`, each with a reason: {cell_id: reason}."""
    m  = binding["mapping"]
    df = _load(str(binding["source"]))
    ts = pd.Timestamp(timestamp)

    def reason(row) -> str:
        return str(row[m["reason"]]) if m.get("reason") else "blocked"

    out: dict[int, str] = {}
    if CELL_BLOCKER_SHAPES["windows"] <= set(m):
        start = pd.to_datetime(df[m["start"]])
        end   = pd.to_datetime(df[m["end"]])
        t     = _time_of_day(ts, start.min().normalize())
        for i in df.index[(start <= t) & (t < end)]:
            out[int(df.at[i, m["cell"]])] = reason(df.loc[i])
        return out

    times = pd.to_datetime(df[m["time"]])
    t     = _time_of_day(ts, times.min().normalize())
    past  = df[times <= t].assign(_t=times[times <= t]).sort_values("_t")
    for cid, g in past.groupby(m["cell"]):
        last = g.iloc[-1]
        if str(last[m["active"]]).lower() in ("true", "1"):
            out[int(cid)] = reason(last)
    return out


def blocked_cells(timestamp: pd.Timestamp | str, bindings: dict[str, dict]) -> dict[str, dict[int, str]]:
    """{dataset name: {cell_id: reason}} for every cell_blocker binding."""
    return {name: cell_blocker(timestamp, b) for name, b in (bindings or {}).items()
            if b.get("role") == "cell_blocker"}
