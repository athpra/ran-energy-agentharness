"""
Planner Agent.

Takes current network KPIs, operator intent, and assembled tool context,
returns a list of recommended sleep/wake actions.
"""
from __future__ import annotations

import json
import re
import time

from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """\
You are a 5G RAN energy optimization expert.
Your goal is to maximize energy savings while preserving the operator's QoS intent.

You will be given:
1. Current network KPIs for each cell (throughput, utilization, sleep state)
2. Operator QoS intent (minimum acceptable throughput)
3. Tool context with additional signals (historical patterns, traffic forecasts,
   active faults, inter-cell interference impacts, energy pricing)

Respond with a JSON array of actions only — no explanation, no markdown fences.
Each action must be one of:
  {"action": "sleep", "cell_id": <int>, "reason": "<short reason>"}
  {"action": "wake",  "cell_id": <int>, "reason": "<short reason>"}

Rules:
- Only propose sleep for cells where current utilization is low AND forecasts remain low
- Never propose sleep for a cell with an active fault (let the fault clear first)
- Consider inter-cell interference: avoid sleeping a cell if it would overload a neighbor
- When energy pricing is in peak tier, be more aggressive about sleeping idle cells
- When energy pricing is off-peak, apply a higher utilization bar before sleeping
- Omit cells that need no change (already asleep and should stay asleep, etc.)
- If no actions are warranted, return an empty array: []
"""


def _parse_actions(text: str) -> list[dict]:
    """Extract a JSON array from the LLM response, tolerating minor formatting."""
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        actions = json.loads(match.group())
        return [a for a in actions if isinstance(a, dict) and "action" in a and "cell_id" in a]
    except json.JSONDecodeError:
        return []


def plan(
    kpi_summary: str,
    operator_intent: str,
    tool_context_block: str,
    llm: ChatOpenAI,
) -> tuple[list[dict], str, float]:
    """
    Returns (actions, raw_response, elapsed_seconds).
    """
    user_message = (
        f"{tool_context_block}\n\n"
        f"### Current Network KPIs\n{kpi_summary}\n\n"
        f"### Operator Intent\n{operator_intent}\n\n"
        f"Respond with a JSON array of sleep/wake actions."
    )

    t0       = time.time()
    response = llm.invoke([
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ])
    elapsed  = time.time() - t0
    raw      = response.content

    return _parse_actions(raw), raw, elapsed
