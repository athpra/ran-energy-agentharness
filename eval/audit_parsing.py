"""
Audit how harness runs read their LLM replies.

Until agent/llm_json.py, the Planner and Validator replies were read from the
first bracket to the last, which fails when a reasoning model's thinking
contains brackets. A lost Planner answer made the harness fall back to its
tool-derived candidates; a lost Validator answer approved every proposal.

Per run, this reports:
  acting / own answer / fallback
      iterations with actions; of those, how many acted on the Planner's own
      answer and how many on the harness's fallback (told apart by the
      actions' reasons, so this works on every log)
  sleep_vetoes
      sleep actions the Validator rejected (it can reject wakes too; the
      harness ignores those)
  unread (planner, validator), answers lost, vetoes lost
      only for runs that log the raw replies (from this fix on): replies the
      old parser could not read, and of those, Planner replies whose actions
      were dropped and Validator replies whose rejections were dropped

Run:
    python -m eval.audit_parsing output/run_*full_harness*
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.planner import _parse_actions  # noqa: E402
from agent.validator import _parse_decision  # noqa: E402


def _old_actions(text: str) -> list[dict] | None:
    """The pre-fix Planner parser; None when it could not read the reply."""
    m = re.search(r"\[.*\]", (text or "").strip(), re.DOTALL)
    if not m:
        return None
    try:
        return [a for a in json.loads(m.group()) if isinstance(a, dict) and "action" in a and "cell_id" in a]
    except (json.JSONDecodeError, TypeError):
        return None


def _old_rejections(text: str) -> list | None:
    """The pre-fix Validator parser's rejections; None when it could not read the reply."""
    m = re.search(r"\{.*\}", (text or "").strip(), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group()).get("rejected", [])
    except (json.JSONDecodeError, AttributeError):
        return None


def _key(actions: list[dict]) -> list[tuple]:
    return sorted((str(a.get("action")), a.get("cell_id")) for a in actions or [])


FALLBACK_REASONS = ("tool analysis", "blueprint PRB rule", "surge imminent", "QoS below intent")


def audit_run(run_dir: Path) -> dict | None:
    path = run_dir / "iterations.jsonl"
    if not path.exists():
        return None
    its = [json.loads(line) for line in open(path) if line.strip()]
    out = {"run": run_dir.name, "planner_calls": 0, "acting": 0, "own_answer": 0, "fallback": 0,
           "sleep_vetoes": 0, "raw_logged": "no", "planner_unread": "—", "answers_lost": "—",
           "validator_unread": "—", "vetoes_lost": "—"}
    raw_logged = any("validator_raw" in it for it in its)
    if raw_logged:
        out.update(raw_logged="yes", planner_unread=0, answers_lost=0, validator_unread=0, vetoes_lost=0)
    for it in its:
        if not (it.get("planner_elapsed_s") or 0) > 0:
            continue                                      # no Planner call (e.g. a rules baseline)
        out["planner_calls"] += 1
        out["sleep_vetoes"] += sum(1 for r in it.get("rejected_actions") or [] if r.get("action") == "sleep")
        acts = it.get("proposed_actions") or []
        if acts:
            out["acting"] += 1
            if all(str(a.get("reason", "")).startswith(FALLBACK_REASONS) for a in acts):
                out["fallback"] += 1
            else:
                out["own_answer"] += 1
        if not raw_logged:
            continue
        p = it.get("planner_raw") or ""
        if "\n--- Planner reply ---\n" in p:            # a fallback note with the reply under it
            reply = p.split("\n--- Planner reply ---\n", 1)[1]
        elif p.startswith("[") and "fallback]" in p.split("\n", 1)[0]:
            reply = ""                                   # a fallback note without the reply
        else:
            reply = p
        if reply:
            if _old_actions(reply) is None:
                out["planner_unread"] += 1
                out["answers_lost"] += bool(_parse_actions(reply))
        v = it.get("validator_raw") or ""
        if v and v != "[]" and _old_rejections(v) is None:
            out["validator_unread"] += 1
            out["vetoes_lost"] += any(isinstance(r, dict) and r.get("action") == "sleep"
                                      for r in _parse_decision(v)["rejected"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Audit how harness runs read their LLM replies")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [r for d in sorted(args.run_dirs) if (r := audit_run(d)) and r["planner_calls"]]
    if not rows:
        sys.exit("No harness runs with Planner calls found.")
    cols = ["run", "planner_calls", "acting", "own_answer", "fallback", "sleep_vetoes", "raw_logged",
            "planner_unread", "answers_lost", "validator_unread", "vetoes_lost"]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    print("\nown_answer: iterations that acted on the Planner's own answer; fallback: on the harness's\n"
          "candidates because the Planner's answer was empty or unreadable. The unread and lost columns\n"
          "need runs that log the raw replies (made after the parser fix).")


if __name__ == "__main__":
    main()
