"""
Tests for reading JSON answers out of LLM replies (agent/llm_json.py) and the
parsers built on it. The replies are shaped like Nemotron's: reasoning, then
</think>, then the answer, with JSON-like text in the reasoning.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_llm_json.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.discovery import _parse_selection  # noqa: E402
from agent.llm_json import find_json, is_action_list, is_decision  # noqa: E402
from agent.planner import _parse_actions  # noqa: E402
from agent.validator import _parse_decision  # noqa: E402
from eval.scoring import _is_format_valid, _is_validator_format_valid  # noqa: E402

# The end of Nemotron's reply to eval scenario S02 (benchmark, 2026-11-19)
S02 = """We need to follow steps.
cell3: 78% -> >50 but Awake -> no.
Thus no sleep actions from step2.
Thus output [].

Return JSON array.

</think>

[]"""

SLEEPS = """Check N1_PRB values: cell0: 2.0% (<25%), cells [0, 1, 4] qualify.
Each action: {"action":"sleep","cell_id":<int>,"reason":"<short reason>"}
</think>
[{"action": "sleep", "cell_id": 0, "reason": "low load"},
 {"action": "sleep", "cell_id": 4, "reason": "low load"}]"""


def test_the_answer_after_the_reasoning_is_read_not_the_brackets_inside_it():
    assert _parse_actions(S02) == [] and _is_format_valid(S02)
    assert [a["cell_id"] for a in _parse_actions(SLEEPS)] == [0, 4] and _is_format_valid(SLEEPS)


def test_without_a_think_tag_the_last_answer_of_the_right_shape_wins():
    reply = 'Maybe [3, 7]? Draft: [{"action": "sleep", "cell_id": 3}]. Final:\n' \
            '[{"action": "sleep", "cell_id": 7, "reason": "low"}]'
    assert [a["cell_id"] for a in _parse_actions(reply)] == [7]
    assert find_json("cells [3, 7]", is_action_list) is None   # a list of numbers is not an answer


def test_validator_reads_the_decision_object_not_an_action_inside_it():
    reply = ('Proposal {"action": "sleep", "cell_id": 2} looks safe.\n</think>\n'
             '{"approved": [{"action": "sleep", "cell_id": 2}], '
             '"rejected": [{"action": "sleep", "cell_id": 5, "reason": "fault"}]}')
    d = _parse_decision(reply)
    assert [a["cell_id"] for a in d["approved"]] == [2] and d["rejected"][0]["cell_id"] == 5
    assert _is_validator_format_valid(reply)
    assert find_json('{"action": "sleep"}', is_decision) is None


def test_a_reply_cut_off_before_its_answer_is_still_unreadable():
    cut = 'Thinking about cells [0, 1] ... Each action: {"action":"sleep","cell_id":'
    assert _parse_actions(cut) == [] and not _is_format_valid(cut)
    assert _parse_decision(cut) == {"approved": [], "rejected": []} and not _is_validator_format_valid(cut)


def test_discovery_selection_after_reasoning_that_quotes_json():
    reply = ('Format is {"selected": [{"dataset": "<name>"}]}; maybe ["faults"].\n</think>\n'
             '{"selected": [{"dataset": "faults", "reason": "safety"}, '
             '{"dataset": "maintenance_windows", "reason": "on air"}]}')
    assert [s["dataset"] for s in _parse_selection(reply)] == ["faults", "maintenance_windows"]


def test_audit_counts_lost_answers_and_vetoes_in_a_run_that_logs_replies():
    import json
    import tempfile
    from eval.audit_parsing import audit_run
    veto = ('Cell {"cell_id": 4} risky.\n</think>\n'
            '{"approved": [], "rejected": [{"action": "sleep", "cell_id": 4, "reason": "fault"}]}')
    its = [
        {"planner_elapsed_s": 1.0, "planner_raw": SLEEPS, "validator_raw": veto,
         "proposed_actions": [{"action": "sleep", "cell_id": 0, "reason": "low load"}],
         "rejected_actions": [{"action": "sleep", "cell_id": 4}]},
        {"planner_elapsed_s": 1.0, "validator_raw": "[]",
         "planner_raw": "[fallback] LLM returned []; using 1 candidates: [2]\n--- Planner reply ---\n" + S02,
         "proposed_actions": [{"action": "sleep", "cell_id": 2, "reason": "tool analysis"}],
         "rejected_actions": []},
    ]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "iterations.jsonl").write_text("".join(json.dumps(i) + "\n" for i in its))
        r = audit_run(Path(d))
    assert (r["acting"], r["own_answer"], r["fallback"], r["sleep_vetoes"]) == (2, 1, 1, 1)
    # SLEEPS and S02 were unreadable before; SLEEPS lost two actions; the veto reply lost its veto
    assert (r["planner_unread"], r["answers_lost"], r["validator_unread"], r["vetoes_lost"]) == (2, 1, 1, 1)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
