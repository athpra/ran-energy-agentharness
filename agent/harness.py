"""
Agent Harness.

Orchestrates one closed-loop iteration:
  1. Assemble tool context
  2. Planner proposes actions
     (then two rules in code, not an LLM: the guidance rule, sleep only the
      harness's sleep candidates and add the wakes it asks for; and the
      certified-blocker rule, no sleep on a cell a role-bound cell_blocker
      dataset lists, and wake it if asleep)
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
from agent.policy    import Policy
from agent.planner   import plan
from agent.validator import validate


# ── Simulation interface ──────────────────────────────────────────────────────
# The harness calls these two functions; in experiments they are wired to
# the VIAVI AI RSG SDK. In unit tests they can be replaced with stubs.

SimFn = Callable[[list[dict]], tuple[str, dict]]
# signature: sim_fn(actions) -> (human_readable_summary, raw_kpi_dict)


def _reply_note(reply: str) -> str:
    """The Planner's own reply, appended to a fallback note so the log shows
    whether it answered [] or gave an answer that could not be read."""
    return f"\n--- Planner reply ---\n{reply}" if reply else ""


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
    pre_kpis:           dict[str, Any] = field(default_factory=dict)
    post_kpis:          dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iteration":            self.iteration,
            "timestamp":            str(self.timestamp),
            "tools_used":           self.tools_used,
            "tool_blocks":          self.tool_context.get("tool_blocks", {}),
            "pricing_tier":         (self.tool_context.get("tool_blocks") or {}).get("pricing_tier", ""),
            "n_proposed":           len(self.proposed_actions),
            "n_approved":           len(self.approved_actions),
            "n_rejected":           len(self.rejected_actions),
            "proposed_actions":     self.proposed_actions,
            "approved_actions":     self.approved_actions,
            "rejected_actions":     self.rejected_actions,
            "planner_raw":          self.planner_raw,      # the replies, so parsing can be audited
            "validator_raw":        self.validator_raw,    # (eval/audit_parsing.py)
            "planner_elapsed_s":    round(self.planner_elapsed_s,  2),
            "validator_elapsed_s":  round(self.validator_elapsed_s, 2),
            "total_elapsed_s":      round(self.total_elapsed_s,    2),
            "pre_kpis":             self.pre_kpis,
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
        planner_system_prompt: str | None = None,
        validator_llm: ChatOpenAI | None = None,
        policy: Policy | None = None,
        bindings: dict[str, dict] | None = None,
    ):
        self.llm                    = llm
        self.validator_llm          = validator_llm or llm
        self.sim_test_fn            = sim_test_fn
        self.sim_apply_fn           = sim_apply_fn
        self.operator_intent        = operator_intent
        self.enabled_tools          = enabled_tools
        self.planner_system_prompt  = planner_system_prompt
        self.policy                 = policy
        self.bindings               = bindings or {}
        self.history: list[IterationResult] = []

    def run_iteration(
        self,
        iteration: int,
        timestamp: pd.Timestamp,
        kpi_summary: str,
        cell_ids: list[int],
        current_utilization: dict[int, float],
        static_sleep_candidates:  list[int] | None = None,
        static_wake_candidates:   list[int] | None = None,
        sleeping_cell_ids:        list[int] | None = None,
        current_throughput_mbps:  float | None = None,
        pre_kpis:                 dict | None = None,
    ) -> IterationResult:
        t_start = time.time()

        intent_mbps = float(self.operator_intent.split()[0])

        # Step 1 — assemble tool context
        tool_ctx, tool_block = assemble(
            timestamp=timestamp,
            cell_ids=cell_ids,
            current_utilization=current_utilization,
            enabled_tools=self.enabled_tools,
            sleeping_cell_ids=sleeping_cell_ids,
            current_throughput_mbps=current_throughput_mbps,
            intent_mbps=intent_mbps,
            policy=self.policy,
            bindings=self.bindings,
        )

        # Step 2 — planner proposes actions
        proposed, planner_raw, t_plan = plan(
            kpi_summary=kpi_summary,
            operator_intent=self.operator_intent,
            tool_context_block=tool_block,
            llm=self.llm,
            sleep_candidates=tool_ctx.get("sleep_candidates"),
            system_prompt=self.planner_system_prompt,
        )
        llm_reply = planner_raw   # kept in the log even when a fallback replaces the actions

        # Strip invalid LLM actions: sleep on already-sleeping cells and wake on
        # already-awake cells are no-ops that inflate n_proposed and confuse the
        # Validator.  sleeping_cell_ids is the ground truth for current state.
        if proposed and sleeping_cell_ids is not None:
            sleeping_set = set(sleeping_cell_ids)
            proposed = [
                a for a in proposed
                if (a["action"] == "sleep" and a["cell_id"] not in sleeping_set)
                or (a["action"] == "wake"  and a["cell_id"]     in sleeping_set)
            ]

        # Guidance rule: when the tool context has the pre-computed guidance, the
        # Planner may sleep only the harness's sleep candidates (they already
        # account for faults, surges, interference, pricing, maintenance and the
        # throughput guard). Enforced in code, like the certified-blocker rule.
        rule_rejected: list[dict] = []
        if proposed and "sleep_candidates" in tool_ctx:
            allowed = set(tool_ctx.get("sleep_candidates") or [])
            rule_rejected = [
                {"action": "sleep", "cell_id": a["cell_id"],
                 "reason": "guidance rule: not in the harness's sleep candidates"}
                for a in proposed if a["action"] == "sleep" and a["cell_id"] not in allowed
            ]
            proposed = [a for a in proposed if not (a["action"] == "sleep" and a["cell_id"] not in allowed)]
        planner_answered = bool(proposed)   # the Planner's own actions survived the rule

        # Fallback: if the LLM proposes nothing but candidates are available
        # (from tool pre-analysis for full_harness, or from static PRB rules for
        # baseline_digital_twin), use those directly.
        fallback_candidates = tool_ctx.get("sleep_candidates") or static_sleep_candidates
        if not proposed and fallback_candidates:
            candidates = fallback_candidates
            source = "tool analysis" if tool_ctx.get("sleep_candidates") else "blueprint PRB rule: N1_PRB < 12%"
            proposed = [
                {"action": "sleep", "cell_id": cid, "reason": source}
                for cid in candidates
            ]
            planner_raw = (f"[fallback] LLM returned []; using {len(proposed)} candidates: {candidates}"
                           f"{_reply_note(llm_reply)}")

        # Wake fallback: tool-derived surge warning takes priority over reactive
        # QoS-based wake, since it acts before throughput has already dropped.
        if not proposed:
            tool_wake = tool_ctx.get("wake_candidates") or []
            reactive_wake = static_wake_candidates or []
            wake_source = tool_wake or reactive_wake
            if wake_source:
                reason = (
                    "surge imminent: proactive capacity restore"
                    if tool_wake else "QoS below intent: restoring capacity"
                )
                proposed = [
                    {"action": "wake", "cell_id": cid, "reason": reason}
                    for cid in wake_source
                ]
                planner_raw = (
                    f"[{'surge' if tool_wake else 'qos'}-wake-fallback] LLM returned []; "
                    f"waking {len(proposed)} cells: {wake_source}{_reply_note(llm_reply)}"
                )

        # Wakes the harness asks for are added when the Planner's own answer left
        # them out: waking only adds capacity, so this never makes a step riskier
        if planner_answered:
            have_wake = {a["cell_id"] for a in proposed if a["action"] == "wake"}
            surge_wake = tool_ctx.get("wake_candidates") or []
            qos_wake = [] if any(a["action"] == "sleep" for a in proposed) else (static_wake_candidates or [])
            for cids, reason in ((surge_wake, "surge imminent: proactive capacity restore"),
                                 (qos_wake, "QoS below intent: restoring capacity")):
                for cid in cids:
                    if cid not in have_wake:
                        proposed.append({"action": "wake", "cell_id": cid, "reason": reason})
                        have_wake.add(cid)

        # Certified-blocker rule: enforced here, before Sim 1 and the Validator,
        # so neither LLM can sleep a cell that must be on air
        role_blocked = tool_ctx.get("role_blocked") or {}
        if role_blocked:
            source = {cid: name for name, cells in role_blocked.items() for cid in cells}
            rule_rejected += [
                {"action": "sleep", "cell_id": a["cell_id"],
                 "reason": f"certified rule: {source[a['cell_id']]} lists the cell as active"}
                for a in proposed if a["action"] == "sleep" and a["cell_id"] in source
            ]
            proposed = [a for a in proposed
                        if not (a["action"] == "sleep" and a["cell_id"] in source)]
            have_wake = {a["cell_id"] for a in proposed if a["action"] == "wake"}
            proposed += [
                {"action": "wake", "cell_id": cid, "reason": f"must be on air: {source[cid]}"}
                for cid in tool_ctx.get("required_wake") or [] if cid not in have_wake
            ]

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
            llm=self.validator_llm,
        )
        # Veto-only semantics: approved = proposed minus explicitly rejected.
        # This is robust against the model omitting safe actions from "approved".
        #
        # Wake actions are monotonically safe — restoring a sleeping cell can
        # only add capacity, never reduce throughput. Restricting the validator's
        # veto to sleep proposals prevents false rejections during capacity-
        # constrained periods (e.g. Evening Peak) where Sim 1 shows sub-intent
        # throughput regardless of whether cells are woken or not.
        rejected = [r for r in rejected if r.get("action") == "sleep"]
        rejected_ids = {r["cell_id"] for r in rejected}
        approved = [a for a in proposed if a["cell_id"] not in rejected_ids]
        rejected = rule_rejected + rejected

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
            pre_kpis=pre_kpis or {},
            post_kpis=post_kpis,
        )
        self.history.append(result)
        return result

    def summary(self) -> list[dict]:
        return [r.to_dict() for r in self.history]
