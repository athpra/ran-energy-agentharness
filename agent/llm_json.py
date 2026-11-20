"""
Read the JSON answer out of an LLM reply.

Reasoning models write their thinking before the answer, often ending it with
</think>, and the thinking can contain JSON-like text ("output [].", "cells
[3, 7] are low"). Matching from the first bracket to the last then spans
thinking and answer, and the reply looks unparseable although its answer is
fine. Instead: take the last complete JSON value of the expected shape,
looking after the last </think> first and in the whole reply only if
nothing is found there.
"""
from __future__ import annotations

import json
from typing import Any, Callable

_DECODER = json.JSONDecoder()


def _values(text: str):
    """Every top-level JSON array or object in text, in order."""
    i, n = 0, len(text)
    while i < n:
        if text[i] in "[{":
            try:
                value, end = _DECODER.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            yield value
            i = end
        else:
            i += 1


def find_json(text: str | None, accept: Callable[[Any], bool]) -> Any | None:
    """The last JSON value in text for which accept() is true, or None."""
    text = text or ""
    answer = text.rsplit("</think>", 1)[-1]
    for part in ([answer, text] if answer is not text else [text]):
        found = None
        for value in _values(part):
            if accept(value):
                found = value
        if found is not None:
            return found
    return None


def is_action_list(value: Any) -> bool:
    """A Planner answer: a list of action objects (possibly empty)."""
    return isinstance(value, list) and all(isinstance(a, dict) for a in value)


def is_decision(value: Any) -> bool:
    """A Validator answer: an object with approved and rejected."""
    return isinstance(value, dict) and "approved" in value and "rejected" in value
