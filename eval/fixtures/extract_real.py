"""
Extract KPI fixtures from real iterations.jsonl runs.

Reads an existing JSONL output file and converts selected iterations
into EvalScenario-compatible fixtures that can be used in the eval suite.

Usage:
    python -m eval.fixtures.extract_real \
        --jsonl output/full_harness_*/iterations.jsonl \
        --out   eval/fixtures/real_scenarios.py \
        --max   5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _iter_to_cell_states(iteration: dict) -> list[dict] | None:
    """
    Convert a single iteration record to a list of CellState-like dicts.

    Returns None if the iteration lacks the required KPI fields.
    """
    post = iteration.get("post_kpis") or {}
    pre  = iteration.get("kpi_summary", "")

    if not pre:
        return None

    # Parse kpi_summary lines:
    # "  cell_id=N  site: N1_PRB=X%  N12_PRB=Y%  QoS=Z Mbps  [State]"
    cells = []
    for line in pre.splitlines():
        line = line.strip()
        if not line.startswith("cell_id="):
            continue
        try:
            parts   = {}
            for token in line.replace("  ", " ").split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    parts[k] = v.rstrip("%,")
            cell_id    = int(parts["cell_id"])
            n1_prb     = float(parts.get("N1_PRB", 0))
            n12_prb    = float(parts.get("N12_PRB", 0))
            avg_qos    = float(parts.get("QoS", 0))
            sleeping   = "[Sleeping]" in line
            site_match = line.split("cell_id=")[1].split(":")[0].strip().split()
            site       = site_match[1] if len(site_match) > 1 else "unknown"
            cells.append({
                "cell_id":    cell_id,
                "site":       site,
                "n1_prb":     n1_prb,
                "n12_prb":    n12_prb,
                "avg_qos":    avg_qos,
                "n1_sleeping": sleeping,
            })
        except Exception:
            continue

    return cells if cells else None


def extract_scenarios(
    jsonl_path: str | Path,
    max_scenarios: int = 5,
    condition: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract up to max_scenarios iterations from a JSONL file.

    Picks iterations that have non-trivial proposed actions (n_proposed > 0)
    and at least one sleeping/wake candidate for diversity.

    Returns a list of dicts, each matching the EvalScenario constructor args.
    """
    records = _load_jsonl(jsonl_path)
    if condition:
        records = [r for r in records if r.get("condition") == condition]

    chosen  = []
    seen_patterns: set[tuple] = set()

    for rec in records:
        if len(chosen) >= max_scenarios:
            break

        cells = _iter_to_cell_states(rec)
        if not cells:
            continue

        # Deduplicate by (n_sleeping, n_proposed) to get diverse patterns
        n_sleeping = sum(1 for c in cells if c["n1_sleeping"])
        n_proposed = rec.get("n_proposed", 0)
        key        = (n_sleeping, min(n_proposed, 5))
        if key in seen_patterns:
            continue
        seen_patterns.add(key)

        proposed   = rec.get("proposed_actions", [])
        approved   = rec.get("approved_actions", [])
        rejected   = rec.get("rejected_actions", [])
        iteration  = rec.get("iteration", 0)
        cond       = rec.get("condition", "unknown")

        chosen.append({
            "id":          f"REAL_{cond}_iter{iteration:03d}",
            "description": (
                f"Real data from {cond} iter {iteration}: "
                f"{n_sleeping}/{len(cells)} sleeping, {n_proposed} proposed"
            ),
            "cells":       cells,
            "proposed":    proposed,
            "approved":    approved,
            "rejected":    rejected,
            "tags":        ["real", cond],
        })

    return chosen


def write_python_fixtures(
    scenarios: list[dict],
    out_path: str | Path,
) -> None:
    """Write extracted scenarios as Python source so they can be imported."""
    lines = [
        '"""Auto-generated real-data fixtures. Do not edit by hand."""',
        "from eval.fixtures.scenarios import CellState, EvalScenario",
        "",
        "REAL_SCENARIOS: list[EvalScenario] = [",
    ]
    for s in scenarios:
        cells_code = ",\n        ".join(
            f"CellState({c['cell_id']!r}, {c['site']!r}, "
            f"n1_prb={c['n1_prb']}, n12_prb={c['n12_prb']}, "
            f"avg_qos={c['avg_qos']}, n1_sleeping={c['n1_sleeping']})"
            for c in s["cells"]
        )
        lines.append(
            f"    EvalScenario(\n"
            f"        id={s['id']!r},\n"
            f"        description={s['description']!r},\n"
            f"        tags={s['tags']!r},\n"
            f"        cells=[\n        {cells_code},\n        ],\n"
            f"    ),"
        )
    lines += ["]", ""]
    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {len(scenarios)} real fixtures to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract real KPI fixtures from iterations.jsonl")
    parser.add_argument("--jsonl",     required=True,       help="Path to iterations.jsonl")
    parser.add_argument("--out",       default="eval/fixtures/real_scenarios.py")
    parser.add_argument("--max",       type=int, default=5, help="Maximum scenarios to extract")
    parser.add_argument("--condition", default=None,        help="Filter by condition name")
    args = parser.parse_args()

    scenarios = extract_scenarios(args.jsonl, args.max, args.condition)
    print(f"Extracted {len(scenarios)} scenarios from {args.jsonl}")
    write_python_fixtures(scenarios, args.out)


if __name__ == "__main__":
    main()
