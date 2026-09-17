"""
Tool: Inter-Cell Interference Model

Given a proposed set of cells to sleep, returns the estimated load
increase on each neighbor cell. The agent should reject proposals that
would push any neighbor above a configurable overload threshold.
"""
import json
import pandas as pd
from pathlib import Path
from functools import lru_cache

_GRAPH_PATH  = Path(__file__).parent.parent / "data" / "synthetic" / "neighbor_graph.json"
_MATRIX_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "load_shift_matrix.csv"


@lru_cache(maxsize=1)
def _load_graph(path: str) -> dict:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


@lru_cache(maxsize=1)
def _load_matrix(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def query(
    sleeping_cells: list[int],
    current_utilization: dict[int, float],   # cell_id -> current utilization % (0–100)
    overload_threshold_pct: float = 85.0,
    graph_path: Path  = _GRAPH_PATH,
    matrix_path: Path = _MATRIX_PATH,
) -> dict:
    """
    Estimate neighbor load after sleeping the given cells.

    Returns:
      - neighbor_load_pct: {cell_id: estimated_utilization_after_sleeping}
      - overloaded_cells:  list of cells that would exceed overload_threshold_pct
      - safe:              True if no neighbor would be overloaded
    """
    matrix = _load_matrix(str(matrix_path))
    rows   = matrix[matrix["sleeping_cell"].isin(sleeping_cells)]

    neighbor_delta: dict[int, float] = {}
    for _, row in rows.iterrows():
        absorbing = int(row["absorbing_cell"])
        sleeping  = int(row["sleeping_cell"])
        shift_pct = row["load_shift_fraction"] * current_utilization.get(sleeping, 0)
        neighbor_delta[absorbing] = neighbor_delta.get(absorbing, 0) + shift_pct

    neighbor_load: dict[int, float] = {}
    for cell_id, delta in neighbor_delta.items():
        base  = current_utilization.get(cell_id, 0)
        neighbor_load[cell_id] = round(base + delta, 2)

    overloaded = [c for c, load in neighbor_load.items() if load > overload_threshold_pct]

    return {
        "neighbor_load_pct": neighbor_load,
        "overloaded_cells":  overloaded,
        "safe":              len(overloaded) == 0,
    }
