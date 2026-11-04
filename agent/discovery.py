"""
Data discovery.

Runs once, before the first decision cycle: the Planner LLM reads the data
catalog and the operator intent, and chooses the datasets the agent should
use. The chosen datasets map onto the harness's enabled_tools, so everything
after this step (context assembly, safety checks, Planner, Validator) runs
unchanged.

Safety rule: a certified safety-critical dataset (data/catalog_policy.yaml)
is always enabled. If discovery leaves one out, it is added back and the
miss is recorded in safety_added, so a run never loses a safety check but
the miss still shows up in the results.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field

from langchain_openai import ChatOpenAI

from agent.catalog import CATALOG_PATH, catalog_prompt
from agent.context import ALL_TOOLS

SYSTEM_PROMPT = """\
You choose the data a 5G RAN energy-saving agent reads on every decision cycle.
The agent puts capacity-layer (N1) cells to sleep when traffic is low and wakes
them again, while keeping per-user throughput at or above the operator intent.

Read the data catalog. Select every dataset that would change whether a cell can
safely sleep, or whether a sleeping cell should wake. Leave out datasets that are
unrelated, out of date, or superseded by another dataset.

Respond with JSON only, no markdown:
{"selected": [{"dataset": "<name from the catalog>", "reason": "<one short line>"}]}
"""


@dataclass
class DiscoveryResult:
    selected:      list[str]
    reasons:       dict[str, str]
    enabled_tools: frozenset[str]
    safety_added:  list[str] = field(default_factory=list)   # safety-critical datasets discovery missed
    unknown:       list[str] = field(default_factory=list)   # names not in the catalog
    raw:           str = ""
    elapsed_s:     float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["enabled_tools"] = sorted(self.enabled_tools)
        d["elapsed_s"] = round(self.elapsed_s, 2)
        return d

    def summary(self) -> str:
        lines = [f"Discovery chose {len(self.selected)} dataset(s) in {self.elapsed_s:.1f}s:"]
        lines += [f"  {name}: {self.reasons.get(name, '')}" for name in self.selected]
        if self.safety_added:
            lines.append(f"  ⚠ missed safety-critical, added back: {self.safety_added}")
        if self.unknown:
            lines.append(f"  ignored names not in the catalog: {self.unknown}")
        lines.append(f"  enabled tools: {sorted(self.enabled_tools)}")
        return "\n".join(lines)


def _parse_selection(text: str) -> list[dict]:
    """Extract the selection from the LLM response, tolerating reasoning text
    before it and a bare list instead of the wrapping object."""
    text = (text or "").rsplit("</think>", 1)[-1].strip()
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    items = data.get("selected", []) if isinstance(data, dict) else data
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            out.append({"dataset": item, "reason": ""})
        elif isinstance(item, dict) and item.get("dataset"):
            out.append({"dataset": str(item["dataset"]), "reason": str(item.get("reason", ""))})
    return out


def resolve(selection: list[dict], catalog: dict[str, dict], enforce_safety: bool = True) -> DiscoveryResult:
    """Map a selection onto enabled_tools and apply the safety rule."""
    selected, reasons, unknown = [], {}, []
    for item in selection:
        name = item["dataset"]
        if name not in catalog:
            unknown.append(name)
        elif name not in selected:
            selected.append(name)
            reasons[name] = item["reason"]

    safety_added = []
    if enforce_safety:
        for name, entry in catalog.items():
            if entry.get("safety_critical") and name not in selected:
                safety_added.append(name)

    tools = {
        catalog[name].get("tool")
        for name in selected + safety_added
    }
    return DiscoveryResult(
        selected=selected,
        reasons=reasons,
        enabled_tools=frozenset(t for t in tools if t in ALL_TOOLS),
        safety_added=safety_added,
        unknown=unknown,
    )


def discover(
    llm: ChatOpenAI,
    operator_intent: str,
    catalog: dict[str, dict],
    enforce_safety: bool = True,
) -> DiscoveryResult:
    user_message = (
        f"### Operator Intent\n{operator_intent} minimum per-user throughput\n\n"
        f"### Data Catalog\n{catalog_prompt(catalog)}\n\n"
        f"Respond with the JSON selection."
    )
    t0 = time.time()
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])
    elapsed = time.time() - t0
    raw = str(response.content)

    result = resolve(_parse_selection(raw), catalog, enforce_safety)
    result.raw = raw
    result.elapsed_s = elapsed
    return result


def add_discovery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--discover", action="store_true",
        help="Let the Planner choose its datasets from the data catalog before the "
             "first cycle (overrides --tools). Certified safety-critical data is always kept.",
    )
    parser.add_argument(
        "--catalog", default=str(CATALOG_PATH),
        help="Catalog file for --discover (default: data/synthetic/catalog.yaml).",
    )
