"""
Harness policy.

The numeric thresholds that decide which cells are sleep candidates, moved out
of the code into a versioned file so they can be tuned
(experiments/tune_policy.py) and recorded with every run. The defaults are the
values the harness has always used, so loading no policy changes nothing.

Deliberately NOT in the policy, and never tuned: fault blocking, the
digital-twin test, the Validator's veto, and wake actions always being
allowed. An optimiser rewarded on energy would otherwise learn to weaken them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml

POLICIES_DIR = Path(__file__).parent.parent / "policies"


@dataclass(frozen=True)
class Policy:
    name: str = "default"
    # Awake N1 cells below this PRB % are sleep candidates
    prb_sleep_threshold: float = 12.0
    # The same gate during peak pricing, when each sleeping cell saves more
    prb_sleep_threshold_peak: float = 25.0
    # Block all new sleeps when network throughput < intent × this
    throughput_headroom: float = 1.1
    # A forecast step is a surge when its load ≥ current load × this
    surge_ratio: float = 1.5
    # A sleep is unsafe if any neighbour would exceed this PRB %
    overload_threshold_pct: float = 85.0


DEFAULT_POLICY = Policy()

# Tuning bounds. The safety floors are part of the bounds: the headroom never
# goes below 1.0 (sleeping while already under intent), and neighbours are
# never allowed past 95% load.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "prb_sleep_threshold":      (4.0, 30.0),
    "prb_sleep_threshold_peak": (8.0, 40.0),
    "throughput_headroom":      (1.0, 1.5),
    "surge_ratio":              (1.2, 2.5),
    "overload_threshold_pct":   (60.0, 95.0),
}


def tunable_params(policy: Policy) -> dict[str, float]:
    return {k: getattr(policy, k) for k in SEARCH_SPACE}


def check(policy: Policy) -> Policy:
    """Raise if a policy is outside the tuning bounds (and so the safety floors)."""
    for key, (lo, hi) in SEARCH_SPACE.items():
        value = getattr(policy, key)
        if not lo <= value <= hi:
            raise ValueError(f"policy {policy.name!r}: {key}={value} is outside [{lo}, {hi}]")
    if policy.prb_sleep_threshold_peak < policy.prb_sleep_threshold:
        raise ValueError(f"policy {policy.name!r}: prb_sleep_threshold_peak is below prb_sleep_threshold")
    return policy


def load_policy(path: str | Path | None) -> Policy:
    """The policy in a YAML file, or the default when path is None."""
    if path is None:
        return DEFAULT_POLICY
    data = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(Policy)}
    unknown = set(data) - known - {"meta"}
    if unknown:
        raise ValueError(f"{path}: unknown policy fields {sorted(unknown)}")
    values = {k: v for k, v in data.items() if k in known}
    values.setdefault("name", Path(path).stem)
    return check(replace(DEFAULT_POLICY, **values))


def save_policy(policy: Policy, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(policy)
    if meta:
        data["meta"] = meta
    path.write_text(yaml.safe_dump(data, sort_keys=False, width=100))
