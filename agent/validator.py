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

from agent.llm_json import find_json, is_decision

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

Check every proposed SLEEP action against all three rules. Reject it if ANY rule applies:
  1. FAULT: the tool context lists an active fault on that cell, or lists the cell under
     "DO NOT sleep". Reject that cell, WHATEVER the simulated throughput.
  2. INTERFERENCE: interference data in the tool context shows sleeping that cell causes
     OVERLOAD RISK (a neighbour above 85% PRB). Reject that cell. Skip this rule if there is
     no interference data.
  3. THROUGHPUT: the first line of the simulation results shows
     "avg_throughput=X.XX Mbps  sleeping=Y/Z". If X.XX is NUMERICALLY BELOW the operator
     threshold (e.g. 4.90 < 5), reject every sleep action. If X.XX is at or above the
     threshold, this rule does not apply (7.00 is ABOVE 5 and is NOT a violation).

Approve every action no rule rejects. Approve every WAKE action: waking a cell only adds capacity.

NEVER reject based on:
- A sleeping cell's per-cell QoS (sleeping cells show QoS=0.00 — that is expected)
- High N12_PRB% on a cell (even 100% N12 utilization is NOT a rejection reason unless
  avg_throughput drops below the threshold)
- Traffic forecast concerns (the Planner's guidance already accounts for the forecast)
"""


def _parse_decision(text: str) -> dict:
    """The decision in the LLM response: the last JSON object with approved
    and rejected, after any reasoning (agent/llm_json.py)."""
    d = find_json(text, is_decision) or {}
    return {
        "approved": d.get("approved", []) if isinstance(d.get("approved"), list) else [],
        "rejected": d.get("rejected", []) if isinstance(d.get("rejected"), list) else [],
    }


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
