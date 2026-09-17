"""
Generate a synthetic inter-cell interference model.

Produces two artefacts:
  1. neighbor_graph.json  — adjacency list: which cells are neighbors of each cell
  2. load_shift_matrix.csv — if cell A sleeps, what fraction of its load shifts to each neighbor

The load-shift fractions are derived from a synthetic distance-based signal
strength model: closer neighbors absorb more of the sleeping cell's traffic.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"


def _cell_positions(n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Place cells on a 10×10 km grid with minimum separation."""
    positions = []
    while len(positions) < n_cells:
        candidate = rng.uniform(0, 10, size=2)
        if all(np.linalg.norm(candidate - p) >= 1.5 for p in positions):
            positions.append(candidate)
    return np.array(positions)


def generate(
    n_cells: int = 6,
    neighbor_radius_km: float = 4.0,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    rng       = np.random.default_rng(seed)
    positions = _cell_positions(n_cells, rng)

    # Build neighbor graph: cells within neighbor_radius_km
    neighbor_graph: dict[str, list[int]] = {}
    for i in range(n_cells):
        neighbors = []
        for j in range(n_cells):
            if i != j:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist <= neighbor_radius_km:
                    neighbors.append(j)
        neighbor_graph[str(i)] = neighbors

    # Load-shift matrix: fraction of cell i's load absorbed by cell j when i sleeps
    # Uses inverse-distance weighting; rows sum to 1 (all load redistributes)
    rows = []
    for i in range(n_cells):
        neighbors = neighbor_graph[str(i)]
        if not neighbors:
            continue
        dists   = np.array([np.linalg.norm(positions[i] - positions[j]) for j in neighbors])
        weights = 1.0 / dists
        weights = weights / weights.sum()
        for j, w in zip(neighbors, weights):
            rows.append(
                {
                    "sleeping_cell":     i,
                    "absorbing_cell":    j,
                    "load_shift_fraction": round(float(w), 4),
                    "distance_km":       round(float(np.linalg.norm(positions[i] - positions[j])), 3),
                }
            )

    return neighbor_graph, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate inter-cell interference model")
    parser.add_argument("--n-cells",         type=int,   default=6)
    parser.add_argument("--neighbor-radius", type=float, default=4.0, help="km")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--out-dir",         type=Path,  default=OUT_DIR)
    args = parser.parse_args()

    graph, matrix_df = generate(
        n_cells=args.n_cells,
        neighbor_radius_km=args.neighbor_radius,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    graph_path  = args.out_dir / "neighbor_graph.json"
    matrix_path = args.out_dir / "load_shift_matrix.csv"

    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)
    matrix_df.to_csv(matrix_path, index=False)

    print(f"✓ Neighbor graph   → {graph_path}")
    print(f"✓ Load-shift matrix→ {matrix_path}  ({len(matrix_df)} cell-pair entries)")


if __name__ == "__main__":
    main()
