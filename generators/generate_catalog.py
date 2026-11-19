"""
Generate the data catalog for the synthetic datasets.

The other generators create every synthetic dataset, so the meanings here are
written by the code that made the data, not guessed. Column structure (types,
ranges, values) is profiled from the files themselves. Policy fields
(certified, safety_critical, decision_use) are not set here: they live in
data/catalog_policy.yaml and are merged in by agent/catalog.py.

maintenance_windows has no tool either: the harness can use it only through
its certified role (cell_blocker, agent/roles.py).

Also writes three distractor datasets that no tool reads, so discovery has
something to leave out:
  subscriber_billing    unrelated to cell sleep
  site_inventory        related to the network, but static
  legacy_alarm_archive  looks like the fault feed, but is a frozen 2023 export

Also writes a harder catalog for benchmarking discovery across models
(eval/eval_discovery.py --hard): the same entries plus six near-miss
distractors placed next to the dataset each imitates. Each looks like a
relevant feed, often with the same columns, and only its description says why
it is not usable: retired, from the lab network, an old tariff, a 12-month
planning horizon, closed tickets, or relations not yet configured.

Run after the other generators, including generate_maintenance.py
(generate_all.py does this last). It uses its
own random generator, so it never changes the other datasets.

Output: data/synthetic/catalog.yaml and catalog_hard.yaml (+ the distractor CSVs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.catalog import DATA_DIR, TOOL_SOURCES, profile_file, write_catalog  # noqa: E402

# Meanings for the datasets the harness tools read. Keys are catalog names;
# "files" are profiled from DATA_DIR.
SEMANTICS = {
    "historical_kpi": {
        "files": ["historical_kpi.csv"],
        "description": "Per-cell radio KPIs every 15 minutes over the last week: throughput, "
                       "PRB utilisation, latency and whether the cell was asleep.",
        "concepts": ["cell", "throughput", "PRB utilisation", "latency", "sleep state"],
        "columns": {
            "timestamp":       "start of the 15-minute interval",
            "cell_id":         "cell index",
            "throughput_mbps": "mean user throughput in the cell (Mbps)",
            "utilization_pct": "PRB utilisation of the cell (%)",
            "latency_ms":      "mean user-plane latency (ms)",
            "sleep_state":     "1 if the cell was asleep in the interval, 0 if awake",
        },
    },
    "traffic_forecast": {
        "files": ["traffic_forecast.csv"],
        "description": "Short-term traffic forecast: expected load for the next 15-minute steps, "
                       "from which upcoming load surges are flagged.",
        "concepts": ["cell", "traffic forecast", "load surge"],
        "columns": {
            "timestamp":            "time the forecast was made",
            "cell_id":              "cell index",
            "forecast_step":        "steps ahead, 15 minutes each",
            "predicted_mbps":       "forecast throughput demand (Mbps)",
            "forecast_window_mean": "mean demand over the forecast window (Mbps)",
        },
        "note": "The harness forecast tool computes surges from the traffic-profile schedule; "
                "this file is the per-cell forecast export.",
    },
    "faults": {
        "files": ["faults.csv"],
        "description": "Live alarm and fault feed: faults raised and cleared per cell, "
                       "with type and severity.",
        "concepts": ["cell", "fault", "alarm", "severity"],
        "columns": {
            "timestamp":  "time of the fault event",
            "cell_id":    "cell index",
            "fault_type": "kind of fault",
            "severity":   "fault severity",
            "active":     "True while the fault is active, False once cleared",
        },
    },
    "interference": {
        "files": ["neighbor_graph.json", "load_shift_matrix.csv"],
        "description": "Neighbour relations between cells, and the share of a sleeping cell's "
                       "load that each neighbour would absorb.",
        "concepts": ["cell", "neighbour", "load shift", "overload"],
        "columns": {
            "sleeping_cell":       "cell that goes to sleep",
            "absorbing_cell":      "neighbour that takes over part of its load",
            "load_shift_fraction": "share of the sleeping cell's load moved to this neighbour",
            "distance_km":         "distance between the two cells (km)",
        },
    },
    "maintenance_windows": {
        "files": ["maintenance_windows.csv"],
        "description": "Planned maintenance schedule from the change-management system: "
                       "work booked on a cell, with its time window. The cell must stay on air "
                       "during the window for the work to be done.",
        "concepts": ["cell", "planned maintenance", "change ticket", "time window"],
        "columns": {
            "ticket_id":    "change ticket identifier",
            "cell_id":      "cell index",
            "window_start": "start of the maintenance window",
            "window_end":   "end of the maintenance window",
            "work_type":    "kind of work booked",
        },
    },
    "energy_pricing": {
        "files": ["energy_pricing.csv"],
        "description": "Electricity price every 15 minutes under a time-of-use tariff.",
        "concepts": ["energy price", "tariff tier"],
        "columns": {
            "timestamp":     "start of the 15-minute interval",
            "tier":          "tariff tier",
            "price_per_kwh": "electricity price (USD/kWh)",
        },
    },
}

DISTRACTORS = {
    "subscriber_billing": {
        "files": ["subscriber_billing.csv"],
        "description": "Monthly bill per subscriber: plan, data used and amount charged.",
        "concepts": ["subscriber", "billing", "tariff plan"],
        "columns": {
            "subscriber_id": "anonymised subscriber identifier",
            "month":         "billing month",
            "plan":          "subscription plan",
            "data_used_gb":  "mobile data used in the month (GB)",
            "amount_usd":    "amount billed (USD)",
        },
    },
    "site_inventory": {
        "files": ["site_inventory.csv"],
        "description": "Static inventory of radio sites: location, equipment vendor and install date.",
        "concepts": ["site", "cell", "equipment", "vendor"],
        "columns": {
            "cell_id":      "cell index",
            "site_name":    "site name",
            "vendor":       "radio equipment vendor",
            "install_date": "date the equipment was installed",
            "latitude":     "site latitude",
            "longitude":    "site longitude",
        },
    },
    "legacy_alarm_archive": {
        "files": ["legacy_alarm_archive.csv"],
        "description": "Alarm export from the previous OSS, frozen in March 2023 when the OSS was "
                       "replaced. Superseded by the live fault feed.",
        "concepts": ["cell", "alarm", "fault"],
        "columns": {
            "alarm_id":   "alarm identifier in the old OSS",
            "raised_at":  "time the alarm was raised",
            "cell_ref":   "cell name in the old OSS naming scheme",
            "alarm_text": "alarm description",
            "cleared":    "True once the alarm was cleared",
        },
    },
}


def write_distractors(data_dir: Path, n_cells: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    n_subs = 200
    months = pd.period_range("2024-01", periods=3, freq="M").astype(str)
    billing = pd.DataFrame([
        {
            "subscriber_id": f"S{sid:05d}",
            "month":         m,
            "plan":          rng.choice(["prepaid", "postpaid_basic", "postpaid_unlimited"]),
            "data_used_gb":  round(float(rng.gamma(2.0, 6.0)), 2),
            "amount_usd":    round(float(rng.uniform(10, 90)), 2),
        }
        for sid in range(n_subs) for m in months
    ])
    billing.to_csv(data_dir / "subscriber_billing.csv", index=False)

    inventory = pd.DataFrame([
        {
            "cell_id":      c,
            "site_name":    f"SITE-{c // 3:02d}",
            "vendor":       rng.choice(["vendor_a", "vendor_b"]),
            "install_date": str(pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 1800))))[:10],
            "latitude":     round(float(51.50 + rng.normal(0, 0.02)), 5),
            "longitude":    round(float(-0.12 + rng.normal(0, 0.03)), 5),
        }
        for c in range(n_cells)
    ])
    inventory.to_csv(data_dir / "site_inventory.csv", index=False)

    raised = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 80 * 24 * 60, 150), unit="min")
    legacy = pd.DataFrame({
        "alarm_id":   [f"A{i:06d}" for i in range(150)],
        "raised_at":  sorted(raised),
        "cell_ref":   [f"CELL-{int(c)}" for c in rng.integers(0, n_cells, 150)],
        "alarm_text": rng.choice(["LINK DOWN", "VSWR HIGH", "TEMP HIGH", "SYNC LOSS"], 150),
        "cleared":    rng.random(150) < 0.9,
    })
    legacy.to_csv(data_dir / "legacy_alarm_archive.csv", index=False)


HARD_DISTRACTORS = {
    "traffic_forecast_2023": {
        "after": "traffic_forecast",
        "files": ["traffic_forecast_2023.csv"],
        "description": "Per-cell traffic forecast from the forecasting model retired in December 2023. "
                       "Kept for audit; no longer refreshed.",
        "concepts": ["cell", "traffic forecast", "load surge"],
        "columns": SEMANTICS["traffic_forecast"]["columns"],
    },
    "capacity_plan": {
        "after": "traffic_forecast_2023",
        "files": ["capacity_plan.csv"],
        "description": "Busy-hour demand forecast per cell for each of the next 12 months, used by "
                       "network planning to schedule capacity upgrades.",
        "concepts": ["cell", "demand forecast", "capacity planning"],
        "columns": {
            "cell_id":               "cell index",
            "month":                 "forecast month",
            "busy_hour_demand_mbps": "forecast busy-hour demand (Mbps)",
            "upgrade_needed":        "True if planning expects the cell to need more capacity",
        },
    },
    "fault_feed_lab": {
        "after": "faults",
        "files": ["fault_feed_lab.csv"],
        "description": "Alarm and fault events from the lab test network, injected by the OSS "
                       "integration tests. Same format as the live feed; not from the production network.",
        "concepts": ["cell", "fault", "alarm", "severity"],
        "columns": SEMANTICS["faults"]["columns"],
    },
    "neighbour_plan": {
        "after": "interference",
        "files": ["neighbour_plan.csv"],
        "description": "Neighbour relations and load-shift estimates proposed for the next network "
                       "expansion. Not yet configured on the network.",
        "concepts": ["cell", "neighbour", "load shift"],
        "columns": SEMANTICS["interference"]["columns"],
    },
    "change_history": {
        "after": "maintenance_windows",
        "files": ["change_history.csv"],
        "description": "Closed change tickets from the last quarter: maintenance work completed on "
                       "cells, with completion dates.",
        "concepts": ["cell", "change ticket", "maintenance"],
        "columns": {
            "ticket_id":    "change ticket identifier",
            "cell_id":      "cell index",
            "completed_at": "time the work was completed",
            "work_type":    "kind of work done",
        },
    },
    "energy_pricing_2022": {
        "after": "energy_pricing",
        "files": ["energy_pricing_2022.csv"],
        "description": "Electricity price every 15 minutes under the 2022 time-of-use tariff, "
                       "replaced in January 2023.",
        "concepts": ["energy price", "tariff tier"],
        "columns": SEMANTICS["energy_pricing"]["columns"],
    },
}


def write_hard_distractors(data_dir: Path, seed: int) -> None:
    """Near-copies of the relevant files, so the columns give nothing away."""
    rng = np.random.default_rng(seed + 1)

    def shift_years(df: pd.DataFrame, col: str, years: int) -> pd.DataFrame:
        df[col] = (pd.to_datetime(df[col]) - pd.DateOffset(years=years)).astype(str)
        return df

    tf = pd.read_csv(data_dir / "traffic_forecast.csv")
    tf["predicted_mbps"] = (tf["predicted_mbps"] * rng.uniform(0.7, 1.3, len(tf))).round(2)
    shift_years(tf, "timestamp", 1).to_csv(data_dir / "traffic_forecast_2023.csv", index=False)

    n_cells = int(pd.read_csv(data_dir / "historical_kpi.csv", usecols=["cell_id"]).cell_id.nunique())
    months = pd.period_range("2024-02", periods=12, freq="M").astype(str)
    pd.DataFrame([
        {"cell_id": c, "month": m, "busy_hour_demand_mbps": round(float(rng.uniform(40, 120)), 1),
         "upgrade_needed": bool(rng.random() < 0.2)}
        for c in range(n_cells) for m in months
    ]).to_csv(data_dir / "capacity_plan.csv", index=False)

    faults = pd.read_csv(data_dir / "faults.csv")
    faults["cell_id"] = rng.permutation(faults["cell_id"].to_numpy())
    faults.to_csv(data_dir / "fault_feed_lab.csv", index=False)

    ls = pd.read_csv(data_dir / "load_shift_matrix.csv")
    ls["absorbing_cell"] = rng.permutation(ls["absorbing_cell"].to_numpy())
    ls["load_shift_fraction"] = (ls["load_shift_fraction"] * rng.uniform(0.6, 1.4, len(ls))).round(4)
    ls.to_csv(data_dir / "neighbour_plan.csv", index=False)

    done = pd.Timestamp("2023-10-01") + pd.to_timedelta(rng.integers(0, 90 * 24, 40), unit="h")
    pd.DataFrame({
        "ticket_id":    [f"CHG-{3000 + i}" for i in range(40)],
        "cell_id":      rng.integers(0, n_cells, 40),
        "completed_at": [str(t) for t in sorted(done)],
        "work_type":    rng.choice(["sw_upgrade_verification", "drive_test", "antenna_alignment"], 40),
    }).to_csv(data_dir / "change_history.csv", index=False)

    ep = pd.read_csv(data_dir / "energy_pricing.csv")
    ep["price_per_kwh"] = (ep["price_per_kwh"] * 0.72).round(4)
    shift_years(ep, "timestamp", 2).to_csv(data_dir / "energy_pricing_2022.csv", index=False)


def generate_hard(entries: dict[str, dict], data_dir: Path = DATA_DIR, seed: int = 42) -> dict[str, dict]:
    """The catalog plus the near-miss distractors, each after the entry it imitates."""
    data_dir = Path(data_dir)
    write_hard_distractors(data_dir, seed)
    out = dict(entries)
    for name, spec in HARD_DISTRACTORS.items():
        items = list(out.items())
        at = [n for n, _ in items].index(spec["after"]) + 1
        out = dict(items[:at] + [(name, build_entry(name, spec, data_dir))] + items[at:])
    return out


def build_entry(name: str, spec: dict, data_dir: Path) -> dict:
    profiles = {f: profile_file(data_dir / f) for f in spec["files"]}
    columns = {}
    for prof in profiles.values():
        for col, info in (prof.get("columns") or {}).items():
            columns[col] = {"meaning": spec["columns"].get(col, ""), **info}
    tools = {TOOL_SOURCES.get(f) for f in spec["files"]} - {None}
    entry = {
        "description": spec["description"],
        "concepts":    spec["concepts"],
        "files":       spec["files"],
        "tool":        tools.pop() if tools else None,
        "columns":     columns,
    }
    for f, prof in profiles.items():
        if prof["format"] == "json":
            entry.setdefault("structure", {})[f] = prof["shape"]
    if spec.get("note"):
        entry["note"] = spec["note"]
    return entry


def generate(data_dir: Path = DATA_DIR, seed: int = 42) -> dict[str, dict]:
    data_dir = Path(data_dir)
    n_cells = int(pd.read_csv(data_dir / "historical_kpi.csv", usecols=["cell_id"]).cell_id.nunique())
    write_distractors(data_dir, n_cells, seed)
    return {name: build_entry(name, spec, data_dir) for name, spec in {**SEMANTICS, **DISTRACTORS}.items()}


def main():
    parser = argparse.ArgumentParser(description="Generate the data catalog for the synthetic datasets")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    entries = generate(args.data_dir, args.seed)
    out = args.data_dir / "catalog.yaml"
    write_catalog(entries, out, header=(
        "Data catalog, generated by generators/generate_catalog.py. Do not edit:\n"
        "regenerated with the data. Policy fields live in data/catalog_policy.yaml."
    ))
    hard = generate_hard(entries, args.data_dir, args.seed)
    write_catalog(hard, args.data_dir / "catalog_hard.yaml", header=(
        "Harder catalog for benchmarking discovery (eval/eval_discovery.py --hard):\n"
        "catalog.yaml plus near-miss distractors. Generated by generators/generate_catalog.py."
    ))
    n_tool = sum(1 for e in entries.values() if e["tool"])
    print(f"✓ Data catalog     → {out}  ({len(entries)} datasets: {n_tool} with a tool, "
          f"1 for a role reader, {len(DISTRACTORS)} distractors)")
    print(f"✓ Hard catalog     → {args.data_dir / 'catalog_hard.yaml'}  "
          f"(+{len(HARD_DISTRACTORS)} near-miss distractors)")


if __name__ == "__main__":
    main()
