"""
Draft a data catalog for any folder of data files.

For data nobody wrote a generator for (an operator's exports, say): profiles
each CSV and JSON file, then asks an LLM to draft what each dataset and column
means from the profile and a few sample rows. Every entry comes out
uncertified; a person reviews it and adds the policy fields to
data/catalog_policy.yaml, keyed by the entry name (the file name without its
extension).

Run:
    python generators/profile_catalog.py                           # data/synthetic/
    python generators/profile_catalog.py --data-dir /path/to/exports --out draft.yaml
    python generators/profile_catalog.py --no-llm                  # structure only

Output: data/synthetic/catalog_draft.yaml (default)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.catalog import DATA_DIR, TOOL_SOURCES, profile_file, write_catalog  # noqa: E402

SKIP = {"catalog.yaml", "catalog_draft.yaml"}

DRAFT_PROMPT = """\
You document datasets for a mobile network operator's data catalog.
From the file name, column profile and sample rows below, write what the
dataset and each column mean. Do not guess how the data should be used in
decisions. Respond with JSON only, no markdown:
{"description": "<one or two sentences>", "concepts": ["<short noun>", ...],
 "columns": {"<column>": "<meaning, with units if known>", ...}}
"""


def _samples(path: Path, n: int = 3) -> str:
    if path.suffix == ".json":
        text = path.read_text()
        return text[:400] + ("…" if len(text) > 400 else "")
    return pd.read_csv(path, nrows=n).to_csv(index=False)


def _parse_draft(text: str) -> dict:
    text = (text or "").rsplit("</think>", 1)[-1]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def draft_entry(path: Path, llm=None) -> dict:
    """One uncertified catalog entry for one file. Without an LLM it holds
    the profiled structure only."""
    prof = profile_file(path)
    draft = {}
    if llm is not None:
        reply = llm.invoke([
            {"role": "system", "content": DRAFT_PROMPT},
            {"role": "user", "content": (
                f"File: {path.name}\n\nProfile:\n{json.dumps(prof, indent=1, default=str)}\n\n"
                f"Sample:\n{_samples(path)}"
            )},
        ])
        draft = _parse_draft(str(reply.content))
    meanings = draft.get("columns") or {}
    entry = {
        "description": str(draft.get("description", "")),
        "concepts":    [str(c) for c in draft.get("concepts", [])],
        "files":       [path.name],
        "tool":        TOOL_SOURCES.get(path.name),
        "columns":     {c: {"meaning": str(meanings.get(c, "")), **info}
                        for c, info in (prof.get("columns") or {}).items()},
        "drafted_by":  getattr(llm, "model_name", None) if llm is not None else "profile only",
    }
    if prof["format"] == "json":
        entry["structure"] = {path.name: prof["shape"]}
    return entry


def draft_catalog(data_dir: Path, llm=None) -> dict[str, dict]:
    files = sorted(p for p in Path(data_dir).iterdir()
                   if p.suffix in (".csv", ".json") and p.name not in SKIP)
    return {p.stem: draft_entry(p, llm) for p in files}


def main():
    parser = argparse.ArgumentParser(description="Draft a data catalog from a folder of data files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out",      type=Path, default=None,
                        help="Output file (default: <data-dir>/catalog_draft.yaml)")
    parser.add_argument("--model",    default=None, help="Model ID. Defaults to LLM_MODEL env var.")
    parser.add_argument("--no-llm",   action="store_true", help="Profile structure only, no descriptions")
    args = parser.parse_args()

    llm = None
    if not args.no_llm:
        from experiments.common import get_llm
        llm = get_llm(args.model)

    entries = draft_catalog(args.data_dir, llm)
    out = args.out or args.data_dir / "catalog_draft.yaml"
    write_catalog(entries, out, header=(
        "Draft data catalog, written by generators/profile_catalog.py.\n"
        "Uncertified: review it, then add policy fields to data/catalog_policy.yaml."
    ))
    print(f"✓ Draft catalog    → {out}  ({len(entries)} datasets, uncertified)")


if __name__ == "__main__":
    main()
