"""
Validator Agent.

Reviews Planner actions against simulation results and tool context.
Returns approved actions, rejected actions, and a structured decision log.
"""
from __future__ import annotations

import json
import re
import time

from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """\
You are a 5G RAN safety validator.
Your role is to approve or reject energy-saving actions proposed by the Planner,
using simulation results and additional tool context as evidence.

You will be given:
1. Proposed actions from the Planner
2. Simulation results showing the impact of those actions on network KPIs
3. Tool context (same signals the Planner used)
4. Operator QoS intent (minimum acceptable throughput)

Respond with a JSON object only — no explanation, no markdown fences:
{
  "approved": [{"action": "sleep"|"wake", "cell_id": <int>}],
  "rejected": [{"action": "sleep"|"wake", "cell_id": <int>, "reason": "<why rejected>"}]
}

Approval criteria — read carefully:

PRIMARY CHECK: The first line of the simulation results shows:
  "avg_throughput=X.XX Mbps  sleeping=Y/Z"
Read the avg_throughput value. If avg_throughput >= the operator's QoS threshold, the
simulation is SAFE. In that case, approve ALL proposed actions by listing them all in
"approved" and leave "rejected" empty.

Only add an action to "rejected" when it causes ONE of these specific, factual violations:
  1. avg_throughput in the simulation result is NUMERICALLY BELOW the operator threshold
     (e.g., threshold "5 Mbps" means reject only if avg_throughput < 5.0 — if avg is
     7.0 Mbps that is ABOVE 5 Mbps and is NOT a violation)
  2. The tool context lists an active fault on that specific cell (reject only that cell)
  3. Interference data shows sleeping that cell specifically causes a neighbor to exceed
     85% PRB (only apply if interference data is present in tool context)

NEVER reject based on:
- A sleeping cell's per-cell QoS (sleeping cells show QoS=0.00 — that is expected)
- High N12_PRB% on a cell (even 100% N12 utilization is NOT a rejection reason unless
  avg_throughput drops below the threshold)
- Traffic forecast concerns (only if forecast data is present in tool context)

If avg_throughput >= threshold: put ALL proposed actions in "approved", leave "rejected" empty.
"""


def _parse_decision(text: str) -> dict:
    text  = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"approved": [], "rejected": []}
    try:
        d = json.loads(match.group())
        return {
            "approved": d.get("approved", []),
            "rejected": d.get("rejected", []),
        }
    except json.JSONDecodeError:
        return {"approved": [], "rejected": []}


def validate(
    proposed_actions: list[dict],
    sim_result_summary: str,
    operator_intent: str,
    tool_context_block: str,
    llm: ChatOpenAI,
) -> tuple[list[dict], list[dict], str, float]:
    """
    Returns (approved_actions, rejected_actions, raw_response, elapsed_seconds).
    """
    if not proposed_actions:
        return [], [], "[]", 0.0

    actions_json = json.dumps(proposed_actions, indent=2)
    user_message = (
        f"{tool_context_block}\n\n"
        f"### Proposed Actions\n```json\n{actions_json}\n```\n\n"
        f"### Simulation Results\n{sim_result_summary}\n\n"
        f"### Operator Intent\n{operator_intent}\n\n"
        f"Respond with the approve/reject JSON object."
    )

    t0       = time.time()
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])
    elapsed  = time.time() - t0
    raw      = response.content

    decision = _parse_decision(raw)
    return decision["approved"], decision["rejected"], raw, elapsed
