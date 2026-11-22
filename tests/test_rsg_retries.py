"""
Tests for the RSG retry policy (experiments/common.with_rsg_retries). No RSG
needed: attempts are stubs and sleeping is recorded, not done.

Run from the project root:
    python -m pytest tests/        (or: python tests/test_rsg_retries.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import (  # noqa: E402
    MAX_QUOTA_WAITS, SIM_MAX_ATTEMPTS, SIM_RETRY_DELAYS_S, with_rsg_retries,
)


class _Response:
    status_code = 429

    def json(self):
        return {"cooldown_remaining_seconds": 1795, "error": "Quota exceeded"}


class HTTPError(Exception):
    def __init__(self):
        super().__init__("429 Client Error: TOO MANY REQUESTS")
        self.response = _Response()


INFLUX = RuntimeError("Query timeout reached. Data not written in Influx.")


def _attempts(*outcomes):
    """An attempt function that raises each exception in turn, then returns 'ok'."""
    it = iter(outcomes)

    def attempt():
        o = next(it, "ok")
        if isinstance(o, Exception):
            raise o
        return o
    return attempt


def test_the_run_that_failed_on_2026_11_20_now_survives():
    """Four Influx timeouts, then a 429 on what used to be the last attempt."""
    waits = []
    assert with_rsg_retries(_attempts(INFLUX, INFLUX, INFLUX, INFLUX, HTTPError()), sleep=waits.append) == "ok"
    assert waits == [60, 120, 240, 480, 1795 + 60]


def test_a_quota_error_does_not_use_up_an_attempt():
    waits = []
    outcomes = [HTTPError()] * MAX_QUOTA_WAITS + [INFLUX] * (SIM_MAX_ATTEMPTS - 1)
    assert with_rsg_retries(_attempts(*outcomes), sleep=waits.append) == "ok"
    assert waits == [1855] * MAX_QUOTA_WAITS + list(SIM_RETRY_DELAYS_S)


def test_a_wrapped_quota_error_is_recognised():
    try:
        raise RuntimeError("query failed") from HTTPError()
    except RuntimeError as e:
        wrapped = e
    waits = []
    with_rsg_retries(_attempts(wrapped), sleep=waits.append)
    assert waits == [1855]


def test_retries_and_quota_waits_are_bounded():
    for outcomes, why in [([INFLUX] * SIM_MAX_ATTEMPTS, f"after {SIM_MAX_ATTEMPTS} attempts"),
                          ([HTTPError()] * (MAX_QUOTA_WAITS + 1), "quota still exceeded")]:
        try:
            with_rsg_retries(_attempts(*outcomes), sleep=lambda s: None)
        except RuntimeError as e:
            assert why in str(e)
        else:
            raise AssertionError("did not give up")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
