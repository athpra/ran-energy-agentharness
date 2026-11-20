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

from agent.llm_json import find_json, is_action_list

SYSTEM_PROMPT = """\
You are a 5G RAN energy optimization controller.
Your job is to output sleep/wake actions for N1 cells to save energy while meeting the QoS threshold.

Respond with a JSON array of actions only — no explanation, no markdown fences.
Each action: {"action": "sleep"|"wake", "cell_id": <int>, "reason": "<short reason>"}

CASE A — the tool context has a "### Pre-computed Action Guidance" section.
  The harness has already combined faults, forecast, interference, pricing, maintenance and
  the throughput guard into that section. It is binding:
  - Sleep the cells in its "SLEEP these cells" list, except a cell whose N12_PRB > 60% in
    "Current Network KPIs" (its neighbours are already overloaded).
  - Never sleep a cell that is not in that list, even one with low N1_PRB: it was left out
    for a reason (fault, surge, interference, throughput guard, maintenance, PRB threshold).
    If the list is empty, propose no sleep actions.
  - Wake every cell in its "WAKE these cells" lines.
  - Also wake any Sleeping cell whose N12_PRB > 60% in "Current Network KPIs".

CASE B — there is no "### Pre-computed Action Guidance" section.
  - Leave out cells listed under "Active faults" or "DO NOT sleep" in the tool context.
  - Sleep each remaining Awake cell whose N1_PRB < 12%.
    QoS=0.00 Mbps on an Awake cell means no active UEs: that is ideal for sleeping.
  - Wake each Sleeping cell whose N12_PRB > 60% (a sleeping cell's own N1_PRB is always 0;
    N12_PRB shows the load its neighbours carry for it).

Output the actions as a JSON array. If there are none, output [].
"""

# Blueprint-equivalent prompt: mirrors the PRB thresholds from the blueprint
# notebook's SQL decision rules. Used for baseline_digital_twin and
# baseline_openloop so they replicate the existing PoC's decision logic.
BLUEPRINT_SYSTEM_PROMPT = """\
You are a 5G RAN energy optimization controller.
Your ONLY output is a JSON array of sleep/wake actions. No explanation, no markdown, nothing else.

DECISION RULES — apply to every cell line in ### Current Network KPIs:
  • [Awake]   cell with N1_PRB value < 12  → sleep action
  • [Sleeping] cell with N12_PRB value > 60 → wake action
  • Otherwise → skip that cell

EXAMPLE INPUT cells:
  cell_id=0  A/N1/1: N1_PRB=0.0%  N12_PRB=1.2%  QoS=0.00 Mbps  [Awake]
  cell_id=1  B/N1/1: N1_PRB=4.5%  N12_PRB=18.0%  QoS=2.1 Mbps  [Awake]
  cell_id=2  C/N1/1: N1_PRB=38.0%  N12_PRB=55.0%  QoS=7.8 Mbps  [Awake]
  cell_id=3  D/N1/1: N1_PRB=0.0%  N12_PRB=71.2%  QoS=6.5 Mbps  [Sleeping]

CORRECT OUTPUT for the example (cells 0 and 1 have N1_PRB < 12; cell 3 is Sleeping with N12_PRB > 60):
[{"action": "sleep", "cell_id": 0, "reason": "N1_PRB=0.0% < 12%"}, {"action": "sleep", "cell_id": 1, "reason": "N1_PRB=4.5% < 12%"}, {"action": "wake", "cell_id": 3, "reason": "N12_PRB=71.2% > 60%"}]

Now produce the JSON array for the cells below. Output ONLY [...] — nothing before or after.
"""


def _parse_actions(text: str) -> list[dict]:
    """The action list in the LLM response: the last JSON array of objects,
    after any reasoning (agent/llm_json.py)."""
    actions = find_json(text, is_action_list) or []
    return [a for a in actions if "action" in a and "cell_id" in a]


def plan(
    kpi_summary: str,
    operator_intent: str,
    tool_context_block: str,
    llm: ChatOpenAI,
    sleep_candidates: list[int] | None = None,
    system_prompt: str | None = None,
) -> tuple[list[dict], str, float]:
    """
    Returns (actions, raw_response, elapsed_seconds).
    """
    # If tool analysis pre-computed sleep candidates, add them as a final
    # directive immediately before the response request so the model sees
    # them as the most recent instruction.
    if sleep_candidates:
        example = json.dumps(
            [{"action": "sleep", "cell_id": cid, "reason": "low N1_PRB, tool signals clear"}
             for cid in sleep_candidates[:2]]
        )
        candidates_directive = (
            f"\n### Final Instruction\n"
            f"Tool analysis has determined the following cells are safe to sleep: {sleep_candidates}\n"
            f"Output a sleep action for each of these cell_ids. Example format: {example}, ...\n"
        )
    else:
        candidates_directive = ""

    user_message = (
        f"{tool_context_block}\n\n"
        f"### Current Network KPIs\n{kpi_summary}\n\n"
        f"### Operator Intent\n{operator_intent}\n"
        f"{candidates_directive}\n"
        f"Respond with a JSON array of sleep/wake actions."
    )

    active_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    t0       = time.time()
    response = llm.invoke([
        {"role": "system",  "content": active_system_prompt},
        {"role": "user",    "content": user_message},
    ])
    elapsed  = time.time() - t0
    raw      = response.content

    return _parse_actions(raw), raw, elapsed
