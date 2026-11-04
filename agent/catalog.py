"""
Data catalog.

Describes what each dataset means, so the agent can choose its own data
instead of being wired to a fixed set of tools. Two files make up the
catalog:

  data/synthetic/catalog.yaml  generated with the data (generators/generate_catalog.py)
                               or drafted from any folder of files
                               (generators/profile_catalog.py); gitignored
  data/catalog_policy.yaml     tracked in git: the fields a person decides
                               (certified, safety_critical, decision_use)

load_catalog() merges the two. Only a certified entry can be safety-critical,
so a drafted description can never add or remove a safety rule on its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DATA_DIR     = Path(__file__).parent.parent / "data" / "synthetic"
CATALOG_PATH = DATA_DIR / "catalog.yaml"
POLICY_PATH  = Path(__file__).parent.parent / "data" / "catalog_policy.yaml"

# Which harness tool reads each source file. Files not listed here have no
# tool: the agent can see them in the catalog but cannot act on them.
TOOL_SOURCES = {
    "historical_kpi.csv":    "historical_kpi",
    "traffic_forecast.csv":  "traffic_forecast",
    "faults.csv":            "alarm_fault",
    "neighbor_graph.json":   "interference",
    "load_shift_matrix.csv": "interference",
    "energy_pricing.csv":    "energy_pricing",
}

POLICY_FIELDS = ("certified", "safety_critical", "decision_use")


# ── Profiling ─────────────────────────────────────────────────────────────────

def _column_profile(s: pd.Series, max_values: int) -> dict[str, Any]:
    if s.dtype == bool:
        return {"type": "bool"}
    if pd.api.types.is_numeric_dtype(s):
        cast = int if pd.api.types.is_integer_dtype(s) else (lambda v: round(float(v), 4))
        info: dict[str, Any] = {"type": "number", "min": cast(s.min()), "max": cast(s.max())}
        if s.nunique() <= max_values:
            info["values"] = sorted(s.dropna().unique().tolist())
        return info
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if s.notna().any() and parsed.notna().sum() == s.notna().sum():
        return {"type": "timestamp", "min": str(parsed.min()), "max": str(parsed.max())}
    info = {"type": "text"}
    if s.nunique() <= max_values:
        info["values"] = sorted(str(v) for v in s.dropna().unique())
    return info


def profile_file(path: Path, max_values: int = 8) -> dict[str, Any]:
    """Structure of one data file, read from the data alone (no LLM)."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        shape = f"object with {len(data)} keys" if isinstance(data, dict) else f"array of {len(data)} items"
        return {"format": "json", "shape": shape}
    df = pd.read_csv(path)
    return {
        "format":  "csv",
        "rows":    int(len(df)),
        "columns": {c: _column_profile(df[c], max_values) for c in df.columns},
    }


# ── Load / write ──────────────────────────────────────────────────────────────

def write_catalog(entries: dict[str, dict], path: Path, header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=100)
    path.write_text("".join(f"# {line}\n" for line in header.splitlines()) + "\n" + body)


def load_catalog(catalog_path: Path = CATALOG_PATH, policy_path: Path | None = POLICY_PATH) -> dict[str, dict]:
    """Catalog entries with the policy fields merged in.

    An entry the policy does not cover is uncertified: its safety_critical is
    forced to False and its drafted decision_use is kept only as a draft.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"No catalog at {catalog_path}. Run python generators/generate_all.py "
            f"(or generators/generate_catalog.py) first."
        )
    entries = yaml.safe_load(catalog_path.read_text()) or {}
    policy  = {}
    if policy_path is not None and Path(policy_path).exists():
        policy = yaml.safe_load(Path(policy_path).read_text()) or {}

    merged = {}
    for name, entry in entries.items():
        entry = dict(entry)
        rules = policy.get(name) or {}
        entry["certified"] = bool(rules.get("certified", False))
        entry["safety_critical"] = entry["certified"] and bool(rules.get("safety_critical", False))
        if entry["certified"] and rules.get("decision_use"):
            entry["decision_use"] = rules["decision_use"]
        merged[name] = entry
    return merged


# ── Prompt ────────────────────────────────────────────────────────────────────

def _column_line(name: str, col: dict) -> str:
    parts = [col.get("type", "")]
    if "values" in col:
        parts.append("values " + ", ".join(str(v) for v in col["values"]))
    elif "min" in col:
        parts.append(f"{col['min']} to {col['max']}")
    meaning = f": {col['meaning']}" if col.get("meaning") else ""
    return f"  - {name} ({'; '.join(p for p in parts if p)}){meaning}"


def catalog_prompt(catalog: dict[str, dict]) -> str:
    """The catalog as the discovery step sees it.

    Leaves out the tool binding and the safety flag, so discovery is judged
    on what the data means, not on labels that give the answer away. The
    decision_use line is shown only once a person has certified it.
    """
    blocks = []
    for name, e in catalog.items():
        lines = [f"## {name}", e.get("description", "").strip()]
        if e.get("concepts"):
            lines.append("Concepts: " + ", ".join(e["concepts"]))
        cols = e.get("columns") or {}
        if cols:
            lines.append("Columns:")
            lines += [_column_line(c, info) for c, info in cols.items()]
        if e.get("certified") and e.get("decision_use"):
            lines.append(f"Decision use (certified): {e['decision_use']}")
        blocks.append("\n".join(l for l in lines if l))
    return "\n\n".join(blocks)
