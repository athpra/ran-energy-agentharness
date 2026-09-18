"""
Agent Harness.

Orchestrates one closed-loop iteration:
  1. Assemble tool context
  2. Planner proposes actions
  3. VIAVI AI RSG simulates proposed actions  (Sim 1)
  4. Validator approves / rejects
  5. VIAVI AI RSG applies approved actions    (Sim 2)
  6. Log everything

The `enabled_tools` parameter controls which tools are active,
enabling ablation experiments without changing any other code.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from langchain_openai import ChatOpenAI

from agent.context   import assemble, ALL_TOOLS
from agent.planner   import plan
from agent.validator import validate


# ── Simulation interface ──────────────────────────────────────────────────────
# The harness calls these two functions; in experiments they are wired to
# the VIAVI AI RSG SDK. In unit tests they can be replaced with stubs.

SimFn = Callable[[list[dict]], tuple[str, dict]]
# signature: sim_fn(actions) -> (human_readable_summary, raw_kpi_dict)


# ── Per-iteration result ──────────────────────────────────────────────────────

@dataclass
class IterationResult:
    iteration:          int
    timestamp:          pd.Timestamp
    tools_used:         list[str]
    tool_context:       dict[str, Any]
    proposed_actions:   list[dict]
    approved_actions:   list[dict]
    rejected_actions:   list[dict]
    sim1_summary:       str          # test simulation result
    sim2_summary:       str          # apply simulation result
    planner_raw:        str
    validator_raw:      str
    planner_elapsed_s:  float
    validator_elapsed_s: float
    total_elapsed_s:    float
    post_kpis:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iteration":            self.iteration,
            "timestamp":            str(self.timestamp),
            "tools_used":           self.tools_used,
            "n_proposed":           len(self.proposed_actions),
            "n_approved":           len(self.approved_actions),
            "n_rejected":           len(self.rejected_actions),
            "proposed_actions":     self.proposed_actions,
            "approved_actions":     self.approved_actions,
            "rejected_actions":     self.rejected_actions,
            "planner_elapsed_s":    round(self.planner_elapsed_s,  2),
            "validator_elapsed_s":  round(self.validator_elapsed_s, 2),
            "total_elapsed_s":      round(self.total_elapsed_s,    2),
            "post_kpis":            self.post_kpis,
        }


# ── Harness ───────────────────────────────────────────────────────────────────

class AgentHarness:
    def __init__(
        self,
        llm: ChatOpenAI,
        sim_test_fn: SimFn,
        sim_apply_fn: SimFn,
        operator_intent: str,
        enabled_tools: frozenset[str] = ALL_TOOLS,
    ):
        self.llm             = llm
        self.sim_test_fn     = sim_test_fn
        self.sim_apply_fn    = sim_apply_fn
        self.operator_intent = operator_intent
        self.enabled_tools   = enabled_tools
        self.history: list[IterationResult] = []

    def run_iteration(
        self,
        iteration: int,
        timestamp: pd.Timestamp,
        kpi_summary: str,
        cell_ids: list[int],
        current_utilization: dict[int, float],
    ) -> IterationResult:
        t_start = time.time()

        # Step 1 — assemble tool context
        tool_ctx, tool_block = assemble(
            timestamp=timestamp,
            cell_ids=cell_ids,
            current_utilization=current_utilization,
            enabled_tools=self.enabled_tools,
        )

        # Step 2 — planner proposes actions
        proposed, planner_raw, t_plan = plan(
            kpi_summary=kpi_summary,
            operator_intent=self.operator_intent,
            tool_context_block=tool_block,
            llm=self.llm,
        )

        # Step 3 — simulate proposed actions (Sim 1)
        if proposed:
            sim1_summary, _ = self.sim_test_fn(proposed)
        else:
            sim1_summary = "No actions proposed — simulation skipped."

        # Step 4 — validator approves / rejects
        validator_approved, rejected, validator_raw, t_val = validate(
            proposed_actions=proposed,
            sim_result_summary=sim1_summary,
            operator_intent=self.operator_intent,
            tool_context_block=tool_block,
            llm=self.llm,
        )
        # Veto-only semantics: approved = proposed minus explicitly rejected.
        # This is robust against the model omitting safe actions from "approved".
        rejected_ids = {r["cell_id"] for r in rejected}
        approved = [a for a in proposed if a["cell_id"] not in rejected_ids]

        # Step 5 — apply approved actions (Sim 2)
        if approved:
            sim2_summary, post_kpis = self.sim_apply_fn(approved)
        else:
            sim2_summary = "No actions approved — nothing applied."
            post_kpis    = {}

        result = IterationResult(
            iteration=iteration,
            timestamp=timestamp,
            tools_used=list(self.enabled_tools),
            tool_context=tool_ctx,
            proposed_actions=proposed,
            approved_actions=approved,
            rejected_actions=rejected,
            sim1_summary=sim1_summary,
            sim2_summary=sim2_summary,
            planner_raw=planner_raw,
            validator_raw=validator_raw,
            planner_elapsed_s=t_plan,
            validator_elapsed_s=t_val,
            total_elapsed_s=time.time() - t_start,
            post_kpis=post_kpis,
        )
        self.history.append(result)
        return result

    def summary(self) -> list[dict]:
        return [r.to_dict() for r in self.history]
