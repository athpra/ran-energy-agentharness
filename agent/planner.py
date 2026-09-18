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
You are a 5G RAN energy optimization controller.
Your job is to output sleep/wake actions for N1 cells to save energy while meeting the QoS threshold.

Respond with a JSON array of actions only — no explanation, no markdown fences.
Each action: {"action": "sleep"|"wake", "cell_id": <int>, "reason": "<short reason>"}

Follow this exact procedure:

STEP 1 — Build the candidate list:
  Start with all cell_ids that appear in "Current Network KPIs".
  Remove cells listed under "Active faults" or "DO NOT sleep" in the tool context.
  These are your SLEEP CANDIDATES.

STEP 2 — Filter candidates by current KPI:
  For each SLEEP CANDIDATE, look at its N1_PRB in "Current Network KPIs".
  - If N1_PRB < 25% AND the cell is currently Awake → add a sleep action.
    NOTE: QoS=0.00 Mbps on an Awake cell means NO active UEs — that is IDEAL for sleeping,
    not a disqualifier. Only skip if active users would drop below the QoS threshold.
  - If N1_PRB > 50% AND the cell is currently Sleeping → add a wake action.
  - Otherwise → no action needed for that cell.

STEP 3 — Apply interference filter (only if interference data is present):
  Remove sleep actions where the interference data shows OVERLOAD RISK (not SAFE).

STEP 4 — Apply forecast filter (only if forecast data is present):
  For each remaining sleep action, check if that cell_id appears in the forecast block.
  If it does AND t+1 predicted_mbps > 5 × QoS_threshold → remove that sleep action.
  If the cell has NO forecast entry → keep the sleep action (no concern).

STEP 5 — Output the remaining actions as a JSON array. If empty, output [].
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
