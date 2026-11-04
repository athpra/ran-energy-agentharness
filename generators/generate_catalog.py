"""
Generate the data catalog for the synthetic datasets.

The other generators create every synthetic dataset, so the meanings here are
written by the code that made the data, not guessed. Column structure (types,
ranges, values) is profiled from the files themselves. Policy fields
(certified, safety_critical, decision_use) are not set here: they live in
data/catalog_policy.yaml and are merged in by agent/catalog.py.

Also writes three distractor datasets that no tool reads, so discovery has
something to leave out:
  subscriber_billing    unrelated to cell sleep
  site_inventory        related to the network, but static
  legacy_alarm_archive  looks like the fault feed, but is a frozen 2023 export

Run after the other generators (generate_all.py does this last). It uses its
own random generator, so it never changes the other datasets.

Output: data/synthetic/catalog.yaml (+ the three distractor CSVs)
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
    n_tool = sum(1 for e in entries.values() if e["tool"])
    print(f"✓ Data catalog     → {out}  ({len(entries)} datasets: {n_tool} with a tool, "
          f"{len(entries) - n_tool} distractors)")


if __name__ == "__main__":
    main()
