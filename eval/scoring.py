"""
Scoring functions for the eval pipeline.

All functions are pure (no LLM calls) and operate on the raw outputs
returned by the planner and validator agents.
"""
from __future__ import annotations

import json
import re
from typing import Any


# ── Planner scoring ───────────────────────────────────────────────────────────

def score_planner(
    proposed: list[dict],
    raw_response: str,
    ground_truth_sleep: list[int],
    ground_truth_wake: list[int],
    all_cell_ids: list[int],
) -> dict[str, Any]:
    """
    Returns a dict with precision, recall, f1, format_valid, no_hallucinations.

    Treats sleep and wake actions together as binary classification:
      TP = proposed action matches ground truth (correct cell + correct direction)
      FP = proposed action not in ground truth
      FN = ground truth action not proposed
    """
    proposed_sleep = {a["cell_id"] for a in proposed if a.get("action") == "sleep"}
    proposed_wake  = {a["cell_id"] for a in proposed if a.get("action") == "wake"}
    gt_sleep       = set(ground_truth_sleep)
    gt_wake        = set(ground_truth_wake)

    tp = len((proposed_sleep & gt_sleep) | (proposed_wake & gt_wake))
    fp = len((proposed_sleep - gt_sleep) | (proposed_wake - gt_wake))
    fn = len((gt_sleep - proposed_sleep) | (gt_wake - proposed_wake))

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Format validity: response must parse to a non-None list
    format_valid = _is_format_valid(raw_response)

    # Hallucination check: any cell_id in proposed that is not in all_cell_ids
    known_ids        = set(all_cell_ids)
    proposed_ids     = {a["cell_id"] for a in proposed}
    no_hallucinations = proposed_ids.issubset(known_ids)

    return {
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "tp":                tp,
        "fp":                fp,
        "fn":                fn,
        "format_valid":      format_valid,
        "no_hallucinations": no_hallucinations,
        "n_proposed":        len(proposed),
        "n_gt_sleep":        len(gt_sleep),
        "n_gt_wake":         len(gt_wake),
    }


def _is_format_valid(raw: str) -> bool:
    """True if the raw response contains a parseable JSON array."""
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return False
    try:
        result = json.loads(match.group())
        return isinstance(result, list)
    except json.JSONDecodeError:
        return False


# ── Validator scoring ─────────────────────────────────────────────────────────

def score_validator(
    approved: list[dict],
    rejected: list[dict],
    raw_response: str,
    risky_proposal_ids: list[int],
    safe_proposal_ids: list[int],
) -> dict[str, Any]:
    """
    Returns risky_rejection_rate, safe_approval_rate, format_valid.

    risky_rejection_rate: fraction of truly risky proposals the validator rejected
    safe_approval_rate:   fraction of truly safe proposals the validator approved
    """
    approved_ids = {a["cell_id"] for a in approved}
    rejected_ids = {r["cell_id"] for r in rejected}
    risky        = set(risky_proposal_ids)
    safe         = set(safe_proposal_ids)

    correctly_rejected = risky & rejected_ids
    correctly_approved = safe & approved_ids

    risky_rejection_rate = (
        len(correctly_rejected) / len(risky) if risky else 1.0
    )
    safe_approval_rate = (
        len(correctly_approved) / len(safe) if safe else 1.0
    )

    format_valid = _is_validator_format_valid(raw_response)

    return {
        "risky_rejection_rate": round(risky_rejection_rate, 4),
        "safe_approval_rate":   round(safe_approval_rate, 4),
        "n_risky":              len(risky),
        "n_safe":               len(safe),
        "n_correctly_rejected": len(correctly_rejected),
        "n_correctly_approved": len(correctly_approved),
        "format_valid":         format_valid,
    }


def _is_validator_format_valid(raw: str) -> bool:
    """True if the raw response contains a parseable JSON object with approved/rejected keys."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False
    try:
        d = json.loads(match.group())
        return isinstance(d, dict) and "approved" in d and "rejected" in d
    except json.JSONDecodeError:
        return False


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def aggregate_planner_scores(per_scenario: list[dict]) -> dict[str, float]:
    """Mean of each numeric metric across all scenarios."""
    if not per_scenario:
        return {}
    keys = ["precision", "recall", "f1"]
    agg  = {}
    for k in keys:
        vals = [s[k] for s in per_scenario if k in s]
        agg[f"{k}_mean"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    agg["format_valid_rate"]      = round(
        sum(1 for s in per_scenario if s.get("format_valid")) / len(per_scenario), 4
    )
    agg["no_hallucinations_rate"] = round(
        sum(1 for s in per_scenario if s.get("no_hallucinations")) / len(per_scenario), 4
    )
    return agg


def aggregate_validator_scores(per_scenario: list[dict]) -> dict[str, float]:
    """Mean of each numeric metric across all validator scenarios."""
    if not per_scenario:
        return {}
    agg = {}
    for k in ["risky_rejection_rate", "safe_approval_rate"]:
        vals = [s[k] for s in per_scenario if k in s]
        agg[f"{k}_mean"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    agg["format_valid_rate"] = round(
        sum(1 for s in per_scenario if s.get("format_valid")) / len(per_scenario), 4
    )
    return agg
