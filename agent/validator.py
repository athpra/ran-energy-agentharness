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

Approval criteria:
- Post-simulation throughput must remain at or above the operator's QoS threshold
- Cells with active faults must not be put to sleep (reject regardless of simulation)
- Actions that would push any neighbor above 85% utilization must be rejected
- When traffic forecast shows a spike in the next 1-2 intervals, reject sleep actions
  even if current simulation looks acceptable

Be conservative: reject any action where the evidence is ambiguous.
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
