"""
Hand-crafted KPI scenarios for offline eval.

Each EvalScenario has:
  - cells: list of CellState describing network state
  - ground_truth_sleep: cell IDs an ideal planner SHOULD propose to sleep
  - ground_truth_wake:  cell IDs an ideal planner SHOULD propose to wake
  - kpi_summary: the text string fed to the Planner LLM (same format as production)
  - validator_proposals / sim_result_summary: populated on validator scenarios
  - tool_context / sleep_candidates: optional; what the harness would show the
    Planner. Without them the Planner sees no tool context, and the expected
    answers follow the blueprint rules (agent/planner.py, CASE B). With a
    "Pre-computed Action Guidance" section, expected_sleep / expected_wake give
    the answer the guidance makes binding (CASE A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Blueprint thresholds (mirrors agent/planner.py CASE B and common.py)
SLEEP_PRB_THRESHOLD = 12.0   # N1_PRB < threshold → sleep candidate
WAKE_N12_THRESHOLD  = 60.0   # N12_PRB > threshold → wake candidate


# ── Cell state ────────────────────────────────────────────────────────────────

@dataclass
class CellState:
    cell_id:     int
    site:        str
    n1_prb:      float
    n12_prb:     float
    avg_qos:     float
    n1_sleeping: bool

    @property
    def should_sleep(self) -> bool:
        return not self.n1_sleeping and self.n1_prb < SLEEP_PRB_THRESHOLD

    @property
    def should_wake(self) -> bool:
        return self.n1_sleeping and self.n12_prb > WAKE_N12_THRESHOLD


# ── Eval scenario ─────────────────────────────────────────────────────────────

@dataclass
class EvalScenario:
    id:          str
    description: str
    cells:       list[CellState]
    tags:        list[str] = field(default_factory=list)

    # Validator-specific fields (None → planner-only scenario)
    validator_proposals:  Optional[list[dict]] = None   # proposed actions to validate
    sim_result_summary:   Optional[str]        = None   # Sim 1 output shown to validator
    # Cell IDs in validator_proposals that are "risky" (should be rejected)
    risky_proposal_ids:   Optional[list[int]]  = None
    # Cell IDs in validator_proposals that are "safe" (should be approved)
    safe_proposal_ids:    Optional[list[int]]  = None
    # What the harness shows: tool context (None → no tool context) and the
    # sleep candidates it passes to plan() as the final instruction
    tool_context:         Optional[str]        = None
    sleep_candidates:     Optional[list[int]]  = None
    # Expected answers when the guidance, not the blueprint rules, decides
    expected_sleep:       Optional[list[int]]  = None
    expected_wake:        Optional[list[int]]  = None
    # The network throughput shown in the first KPI line. None → the mean
    # QoS of the awake cells, which near-idle cells drag far below intent.
    shown_throughput_mbps: Optional[float]     = None

    @property
    def ground_truth_sleep(self) -> list[int]:
        if self.expected_sleep is not None:
            return list(self.expected_sleep)
        return [c.cell_id for c in self.cells if c.should_sleep]

    @property
    def ground_truth_wake(self) -> list[int]:
        if self.expected_wake is not None:
            return list(self.expected_wake)
        return [c.cell_id for c in self.cells if c.should_wake]

    @property
    def kpi_summary(self) -> str:
        sleeping = sum(1 for c in self.cells if c.n1_sleeping)
        awake    = [c for c in self.cells if not c.n1_sleeping]
        avg_tp   = sum(c.avg_qos for c in awake) / len(awake) if awake else 0.0
        if self.shown_throughput_mbps is not None:
            avg_tp = self.shown_throughput_mbps
        lines    = [f"avg_throughput={avg_tp:.2f} Mbps  sleeping={sleeping}/{len(self.cells)}"]
        for c in sorted(self.cells, key=lambda x: x.cell_id):
            state = "Sleeping" if c.n1_sleeping else "Awake"
            lines.append(
                f"  cell_id={c.cell_id}  {c.site}: "
                f"N1_PRB={c.n1_prb:.1f}%  N12_PRB={c.n12_prb:.1f}%  "
                f"QoS={c.avg_qos:.2f} Mbps  [{state}]"
            )
        return "\n".join(lines)

    @property
    def is_validator_scenario(self) -> bool:
        return self.validator_proposals is not None


# ── Helper: build a sim-result string for validator scenarios ─────────────────

def _sim_summary(cells: list[CellState], avg_throughput_mbps: float) -> str:
    sleeping = sum(1 for c in cells if c.n1_sleeping)
    lines = [f"avg_throughput={avg_throughput_mbps:.2f} Mbps  sleeping={sleeping}/{len(cells)}"]
    for c in sorted(cells, key=lambda x: x.cell_id):
        state = "Sleeping" if c.n1_sleeping else "Awake"
        lines.append(
            f"  cell_id={c.cell_id}  {c.site}: "
            f"N1_PRB={c.n1_prb:.1f}%  N12_PRB={c.n12_prb:.1f}%  "
            f"QoS={c.avg_qos:.2f} Mbps  [{state}]"
        )
    return "\n".join(lines)


# ── Scenario catalogue ────────────────────────────────────────────────────────

SCENARIOS: list[EvalScenario] = []

# ── S01: Night — all awake, zero traffic → sleep everything ──────────────────
_s01_cells = [
    CellState(0, "S1/C1", n1_prb=0.0,  n12_prb=0.0, avg_qos=0.50, n1_sleeping=False),
    CellState(1, "S1/C2", n1_prb=1.2,  n12_prb=0.0, avg_qos=0.60, n1_sleeping=False),
    CellState(2, "S2/C1", n1_prb=0.0,  n12_prb=0.0, avg_qos=0.40, n1_sleeping=False),
    CellState(3, "S2/C2", n1_prb=2.5,  n12_prb=0.0, avg_qos=0.70, n1_sleeping=False),
    CellState(4, "S3/C1", n1_prb=0.0,  n12_prb=0.0, avg_qos=0.30, n1_sleeping=False),
    CellState(5, "S3/C2", n1_prb=1.8,  n12_prb=0.0, avg_qos=0.55, n1_sleeping=False),
]
SCENARIOS.append(EvalScenario(
    id="S01_night_all_low",
    description="Night: all 6 cells awake with N1_PRB 0–2.5%. Planner should sleep all 6.",
    cells=_s01_cells,
    tags=["planner", "night", "sleep_all"],
))

# ── S02: Peak — all awake, heavy traffic → sleep nothing ─────────────────────
SCENARIOS.append(EvalScenario(
    id="S02_peak_all_high",
    description="Evening peak: all 6 cells awake with N1_PRB 35–78%. Planner should propose nothing.",
    cells=[
        CellState(0, "S1/C1", n1_prb=45.0, n12_prb=0.0, avg_qos=7.8, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=62.0, n12_prb=0.0, avg_qos=7.5, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=38.0, n12_prb=0.0, avg_qos=7.6, n1_sleeping=False),
        CellState(3, "S2/C2", n1_prb=78.0, n12_prb=0.0, avg_qos=7.9, n1_sleeping=False),
        CellState(4, "S3/C1", n1_prb=55.0, n12_prb=0.0, avg_qos=7.7, n1_sleeping=False),
        CellState(5, "S3/C2", n1_prb=35.0, n12_prb=0.0, avg_qos=7.4, n1_sleeping=False),
    ],
    tags=["planner", "peak", "no_action"],
))

# ── S03: Morning — mixed traffic, partial sleep eligible ─────────────────────
SCENARIOS.append(EvalScenario(
    id="S03_morning_mixed",
    description="Morning: cells 0,1,4,5 have N1_PRB < 12% (eligible); cells 2,3 do not.",
    cells=[
        CellState(0, "S1/C1", n1_prb=2.0,  n12_prb=0.0, avg_qos=1.2, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=8.5,  n12_prb=0.0, avg_qos=2.1, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=25.0, n12_prb=0.0, avg_qos=6.5, n1_sleeping=False),
        CellState(3, "S2/C2", n1_prb=44.0, n12_prb=0.0, avg_qos=7.2, n1_sleeping=False),
        CellState(4, "S3/C1", n1_prb=5.0,  n12_prb=0.0, avg_qos=1.5, n1_sleeping=False),
        CellState(5, "S3/C2", n1_prb=11.9, n12_prb=0.0, avg_qos=3.8, n1_sleeping=False),
    ],
    tags=["planner", "morning", "partial_sleep"],
))

# ── S04: Sleeping cells need waking (N12 overloaded) ─────────────────────────
SCENARIOS.append(EvalScenario(
    id="S04_wake_overloaded",
    description="All cells sleeping. Cells 0,1,2 have N12_PRB > 60% — must wake them.",
    cells=[
        CellState(0, "S1/C1", n1_prb=0.0, n12_prb=85.0, avg_qos=4.2, n1_sleeping=True),
        CellState(1, "S1/C2", n1_prb=0.0, n12_prb=92.0, avg_qos=3.8, n1_sleeping=True),
        CellState(2, "S2/C1", n1_prb=0.0, n12_prb=71.0, avg_qos=5.1, n1_sleeping=True),
        CellState(3, "S2/C2", n1_prb=0.0, n12_prb=40.0, avg_qos=7.5, n1_sleeping=True),
        CellState(4, "S3/C1", n1_prb=0.0, n12_prb=20.0, avg_qos=7.8, n1_sleeping=True),
        CellState(5, "S3/C2", n1_prb=0.0, n12_prb=55.0, avg_qos=7.3, n1_sleeping=True),
    ],
    tags=["planner", "wake", "n12_overload"],
))

# ── S05: Threshold boundary — tests exact rule application ───────────────────
SCENARIOS.append(EvalScenario(
    id="S05_boundary_sleep_threshold",
    description="Tests N1_PRB boundary: 11.9% → sleep, 12.0% → no sleep, 12.1% → no sleep.",
    cells=[
        CellState(0, "S1/C1", n1_prb=0.0,  n12_prb=0.0, avg_qos=0.1, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=11.9, n12_prb=0.0, avg_qos=3.5, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=12.0, n12_prb=0.0, avg_qos=3.8, n1_sleeping=False),
        CellState(3, "S2/C2", n1_prb=12.1, n12_prb=0.0, avg_qos=3.9, n1_sleeping=False),
        CellState(4, "S3/C1", n1_prb=15.0, n12_prb=0.0, avg_qos=4.5, n1_sleeping=False),
        CellState(5, "S3/C2", n1_prb=30.0, n12_prb=0.0, avg_qos=6.2, n1_sleeping=False),
    ],
    tags=["planner", "boundary", "precision"],
))

# ── S06: Wake threshold boundary ─────────────────────────────────────────────
SCENARIOS.append(EvalScenario(
    id="S06_boundary_wake_threshold",
    description="Tests N12_PRB boundary: 60.1% → wake, 60.0% → no wake, 59.9% → no wake.",
    cells=[
        CellState(0, "S1/C1", n1_prb=0.0, n12_prb=100.0, avg_qos=2.1, n1_sleeping=True),
        CellState(1, "S1/C2", n1_prb=0.0, n12_prb=60.1,  avg_qos=5.5, n1_sleeping=True),
        CellState(2, "S2/C1", n1_prb=0.0, n12_prb=60.0,  avg_qos=6.5, n1_sleeping=True),
        CellState(3, "S2/C2", n1_prb=0.0, n12_prb=59.9,  avg_qos=7.0, n1_sleeping=True),
        CellState(4, "S3/C1", n1_prb=0.0, n12_prb=30.0,  avg_qos=7.5, n1_sleeping=True),
        CellState(5, "S3/C2", n1_prb=0.0, n12_prb=10.0,  avg_qos=7.8, n1_sleeping=True),
    ],
    tags=["planner", "boundary", "wake", "precision"],
))

# ── S07: Mixed — simultaneous sleep and wake candidates ──────────────────────
SCENARIOS.append(EvalScenario(
    id="S07_mixed_sleep_and_wake",
    description="Some awake cells eligible for sleep, some sleeping cells need waking.",
    cells=[
        CellState(0, "S1/C1", n1_prb=3.0,  n12_prb=0.0,  avg_qos=0.8, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=35.0, n12_prb=0.0,  avg_qos=7.2, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=0.0,  n12_prb=88.0, avg_qos=3.5, n1_sleeping=True),
        CellState(3, "S2/C2", n1_prb=0.0,  n12_prb=28.0, avg_qos=7.6, n1_sleeping=True),
        CellState(4, "S3/C1", n1_prb=7.0,  n12_prb=0.0,  avg_qos=1.9, n1_sleeping=False),
        CellState(5, "S3/C2", n1_prb=0.0,  n12_prb=75.0, avg_qos=4.8, n1_sleeping=True),
    ],
    tags=["planner", "mixed", "sleep_and_wake"],
))

# ── S08: No-op — nothing to do ───────────────────────────────────────────────
SCENARIOS.append(EvalScenario(
    id="S08_noop",
    description="All awake cells have N1_PRB 15–60%. No eligible cells. Expect empty output [].",
    cells=[
        CellState(0, "S1/C1", n1_prb=15.0, n12_prb=0.0, avg_qos=5.0, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=28.0, n12_prb=0.0, avg_qos=6.5, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=42.0, n12_prb=0.0, avg_qos=7.0, n1_sleeping=False),
        CellState(3, "S2/C2", n1_prb=60.0, n12_prb=0.0, avg_qos=7.5, n1_sleeping=False),
        CellState(4, "S3/C1", n1_prb=18.0, n12_prb=0.0, avg_qos=5.5, n1_sleeping=False),
        CellState(5, "S3/C2", n1_prb=33.0, n12_prb=0.0, avg_qos=6.8, n1_sleeping=False),
    ],
    tags=["planner", "noop", "empty_output"],
))

# ── S09: Scale test — 21 cells, realistic evening profile ────────────────────
_s09_cells = []
for _i in range(21):
    _site = f"S{_i // 3 + 1}/C{_i % 3 + 1}"
    # Alternate between low-PRB (sleep eligible) and moderate/high-PRB cells
    if _i % 3 == 0:
        _s09_cells.append(CellState(_i, _site, n1_prb=2.0,  n12_prb=0.0, avg_qos=0.8, n1_sleeping=False))
    elif _i % 3 == 1:
        _s09_cells.append(CellState(_i, _site, n1_prb=32.0, n12_prb=0.0, avg_qos=6.5, n1_sleeping=False))
    else:
        _s09_cells.append(CellState(_i, _site, n1_prb=9.0,  n12_prb=0.0, avg_qos=2.5, n1_sleeping=False))
SCENARIOS.append(EvalScenario(
    id="S09_scale_21_cells",
    description="21 cells (full VIAVI scenario size). Tests that the LLM handles the full prompt length.",
    cells=_s09_cells,
    tags=["planner", "scale", "21_cells"],
))

# ── S10: Format stress — ambiguous QoS values near boundary ──────────────────
SCENARIOS.append(EvalScenario(
    id="S10_format_stress",
    description="Tests JSON format compliance when cell values are non-round numbers.",
    # The prompt shows one decimal place, so values just inside a threshold
    # must stay inside it once rounded: 11.94 shows as 11.9 (< 12, sleep) and
    # 60.06 as 60.1 (> 60, wake). 11.999 and 60.001 showed as 12.0 and 60.0,
    # which the rules leave alone.
    cells=[
        CellState(0, "S1/C1", n1_prb=0.123,  n12_prb=0.0,    avg_qos=0.001, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=11.94,  n12_prb=0.0,    avg_qos=3.777, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=0.0,    n12_prb=60.06,  avg_qos=0.0,   n1_sleeping=True),
        CellState(3, "S2/C2", n1_prb=50.5,   n12_prb=0.0,    avg_qos=7.123, n1_sleeping=False),
    ],
    tags=["planner", "format", "boundary"],
))

# ── V01: Validator — safe proposals, sim passes → approve all ────────────────
_v01_cells = [
    CellState(0, "S1/C1", n1_prb=0.0, n12_prb=12.0, avg_qos=0.0, n1_sleeping=True),
    CellState(1, "S1/C2", n1_prb=0.0, n12_prb=8.0,  avg_qos=0.0, n1_sleeping=True),
    CellState(2, "S2/C1", n1_prb=5.0, n12_prb=0.0,  avg_qos=7.8, n1_sleeping=False),
    CellState(3, "S2/C2", n1_prb=8.0, n12_prb=0.0,  avg_qos=7.5, n1_sleeping=False),
    CellState(4, "S3/C1", n1_prb=3.0, n12_prb=0.0,  avg_qos=7.6, n1_sleeping=False),
    CellState(5, "S3/C2", n1_prb=0.0, n12_prb=25.0, avg_qos=0.0, n1_sleeping=True),
]
SCENARIOS.append(EvalScenario(
    id="V01_validator_safe_approve_all",
    description="Validator: conservative sleep proposals, sim shows avg_throughput=7.5 Mbps. Should approve all.",
    cells=_v01_cells,
    validator_proposals=[
        {"action": "sleep", "cell_id": 2, "reason": "N1_PRB=5.0% < 12%"},
        {"action": "sleep", "cell_id": 3, "reason": "N1_PRB=8.0% < 12%"},
        {"action": "sleep", "cell_id": 4, "reason": "N1_PRB=3.0% < 12%"},
    ],
    sim_result_summary=_sim_summary(
        [CellState(2, "S2/C1", 0.0, 15.0, 0.0, True),
         CellState(3, "S2/C2", 0.0, 10.0, 0.0, True),
         CellState(4, "S3/C1", 0.0, 8.0,  0.0, True)],
        avg_throughput_mbps=7.5,
    ),
    risky_proposal_ids=[],
    safe_proposal_ids=[2, 3, 4],
    tags=["validator", "approve_all", "safe"],
))

# ── V02: Validator — throughput drops below threshold → reject ────────────────
_v02_cells = [
    CellState(0, "S1/C1", n1_prb=5.0, n12_prb=0.0, avg_qos=7.8, n1_sleeping=False),
    CellState(1, "S1/C2", n1_prb=4.0, n12_prb=0.0, avg_qos=7.5, n1_sleeping=False),
    CellState(2, "S2/C1", n1_prb=6.0, n12_prb=0.0, avg_qos=7.6, n1_sleeping=False),
]
SCENARIOS.append(EvalScenario(
    id="V02_validator_throughput_fail",
    description="Validator: sim shows avg_throughput=2.1 Mbps (below 5 Mbps intent). Should reject.",
    cells=_v02_cells,
    validator_proposals=[
        {"action": "sleep", "cell_id": 0, "reason": "N1_PRB=5.0% < 12%"},
        {"action": "sleep", "cell_id": 1, "reason": "N1_PRB=4.0% < 12%"},
        {"action": "sleep", "cell_id": 2, "reason": "N1_PRB=6.0% < 12%"},
    ],
    sim_result_summary=_sim_summary(
        [CellState(0, "S1/C1", 0.0, 90.0, 0.0, True),
         CellState(1, "S1/C2", 0.0, 95.0, 0.0, True),
         CellState(2, "S2/C1", 0.0, 88.0, 0.0, True)],
        avg_throughput_mbps=2.1,
    ),
    risky_proposal_ids=[0, 1, 2],
    safe_proposal_ids=[],
    tags=["validator", "reject_throughput"],
))

# ── V03: Validator — mixed: sim passes but tool context flags a fault ─────────
_v03_cells = [
    CellState(0, "S1/C1", n1_prb=3.0, n12_prb=0.0, avg_qos=7.8, n1_sleeping=False),
    CellState(1, "S1/C2", n1_prb=5.0, n12_prb=0.0, avg_qos=7.5, n1_sleeping=False),
]
_v03_tool_ctx = (
    "### Tool Context\n"
    "**Alarm/Fault Monitor**\n"
    "  cell_id=0  S1/C1: ACTIVE FAULT — radio unit temperature critical (alarm_id=A-001)\n"
    "  cell_id=1  S1/C2: no active faults\n"
)
SCENARIOS.append(EvalScenario(
    id="V03_validator_fault_rejection",
    description="Validator: sim passes (7.8 Mbps) but cell 0 has an active fault. Should reject cell 0, approve cell 1.",
    cells=_v03_cells,
    validator_proposals=[
        {"action": "sleep", "cell_id": 0, "reason": "N1_PRB=3.0% < 12%"},
        {"action": "sleep", "cell_id": 1, "reason": "N1_PRB=5.0% < 12%"},
    ],
    sim_result_summary=_sim_summary(
        [CellState(0, "S1/C1", 0.0, 10.0, 0.0, True),
         CellState(1, "S1/C2", 0.0, 8.0,  0.0, True)],
        avg_throughput_mbps=7.8,
    ),
    risky_proposal_ids=[0],
    safe_proposal_ids=[1],
    tool_context=_v03_tool_ctx,
    tags=["validator", "fault_rejection", "tool_context"],
))


# ── Guidance scenarios: the tool context the full harness shows ──────────────
# The text follows agent/context.py. The guidance is binding (agent/planner.py
# CASE A), so the expected answers come from it, not from the blueprint rules.

def _guidance(blocked: list[int], sleep: list[int], extra: str = "", faults: str = "") -> str:
    lines = ["### Tool Context\n"]
    if faults:
        lines += ["**Active faults (DO NOT sleep these cells):**", faults, ""]
    lines += ["### Pre-computed Action Guidance",
              f"Blocked cells (faults/forecast/interference): {blocked}"]
    if extra.startswith("⚠"):
        lines.append(extra)
    lines.append(f"SLEEP these cells (Awake, N1_PRB < 12%, all tool signals clear): {sleep}")
    if extra and not extra.startswith("⚠"):
        lines.append(extra)
    return "\n".join(lines) + "\n"


# ── G01: Guidance narrows the low-PRB cells ───────────────────────────────────
SCENARIOS.append(EvalScenario(
    id="G01_guidance_subset",
    description="Cells 0-3 have low N1_PRB, but the guidance clears only 0 and 1 (3 faulted, 2 interference).",
    cells=[
        CellState(0, "S1/C1", n1_prb=2.0,  n12_prb=10.0, avg_qos=1.0, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=4.0,  n12_prb=12.0, avg_qos=1.4, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=3.0,  n12_prb=15.0, avg_qos=1.1, n1_sleeping=False),
        CellState(3, "S2/C2", n1_prb=1.0,  n12_prb=9.0,  avg_qos=0.6, n1_sleeping=False),
        CellState(4, "S3/C1", n1_prb=40.0, n12_prb=20.0, avg_qos=7.2, n1_sleeping=False),
    ],
    tool_context=_guidance([2, 3], [0, 1], faults="  Cell 3: hardware_degradation (severity=high)"),
    sleep_candidates=[0, 1],
    expected_sleep=[0, 1], expected_wake=[],
    tags=["planner", "guidance", "subset"],
))

# ── G02: Throughput guard suppresses every sleep ──────────────────────────────
SCENARIOS.append(EvalScenario(
    id="G02_guidance_throughput_guard",
    description="Low-PRB cells, but throughput is near intent and the guard suppresses all sleeps.",
    cells=[
        CellState(0, "S1/C1", n1_prb=3.0, n12_prb=30.0, avg_qos=4.9, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=6.0, n12_prb=35.0, avg_qos=5.0, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=9.0, n12_prb=28.0, avg_qos=5.2, n1_sleeping=False),
    ],
    tool_context=_guidance([], [], extra="⚠ THROUGHPUT GUARD: current tp=5.03 Mbps < intent×1.1 (5.50 Mbps) — "
                                        "all sleep candidates suppressed; focus on waking cells or holding state."),
    expected_sleep=[], expected_wake=[],
    tags=["planner", "guidance", "throughput_guard", "empty_output"],
))

# ── G03: Surge ahead — wake sleeping cells before the load arrives ────────────
SCENARIOS.append(EvalScenario(
    id="G03_guidance_surge_wake",
    description="Sleeping cells with quiet neighbours today, but a surge is two steps away: wake them now.",
    cells=[
        CellState(0, "S1/C1", n1_prb=5.0, n12_prb=20.0, avg_qos=6.0, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=0.0, n12_prb=40.0, avg_qos=0.0, n1_sleeping=True),
        CellState(2, "S2/C1", n1_prb=0.0, n12_prb=30.0, avg_qos=0.0, n1_sleeping=True),
    ],
    tool_context=_guidance([0, 1, 2], [], extra="WAKE these cells (Sleeping, surge in t+2): [1, 2]"),
    expected_sleep=[], expected_wake=[1, 2],
    tags=["planner", "guidance", "surge", "wake"],
))

# ── G04: A cell that must be on air, and a normal sleep ───────────────────────
SCENARIOS.append(EvalScenario(
    id="G04_guidance_must_be_on_air",
    description="Cell 2 is asleep inside a maintenance window: wake it. Cell 0 is cleared to sleep.",
    cells=[
        CellState(0, "S1/C1", n1_prb=2.0,  n12_prb=10.0, avg_qos=0.9, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=30.0, n12_prb=15.0, avg_qos=6.8, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=0.0,  n12_prb=18.0, avg_qos=0.0, n1_sleeping=True),
    ],
    tool_context=("### Tool Context\n\n"
                  "**maintenance_windows (certified; DO NOT sleep these cells, wake them if asleep):**\n"
                  "  Cell 2: drive_test\n\n"
                  + _guidance([2], [0], extra="WAKE these cells (Sleeping, must be on air): [2]")
                  .replace("### Tool Context\n\n", "")),
    sleep_candidates=[0],
    expected_sleep=[0], expected_wake=[2],
    tags=["planner", "guidance", "maintenance", "sleep_and_wake"],
))

# ── V04: Validator — fault in the harness's own format, sim passes ───────────
SCENARIOS.append(EvalScenario(
    id="V04_validator_fault_harness_format",
    description="Validator: sim passes (7.9 Mbps), but the harness's fault block lists cell 0. Reject 0, approve 1.",
    cells=[
        CellState(0, "S1/C1", n1_prb=2.0, n12_prb=0.0, avg_qos=7.9, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=4.0, n12_prb=0.0, avg_qos=7.6, n1_sleeping=False),
    ],
    validator_proposals=[
        {"action": "sleep", "cell_id": 0, "reason": "low N1_PRB"},
        {"action": "sleep", "cell_id": 1, "reason": "low N1_PRB"},
    ],
    sim_result_summary=_sim_summary(
        [CellState(0, "S1/C1", 0.0, 12.0, 0.0, True),
         CellState(1, "S1/C2", 0.0, 9.0,  0.0, True)],
        avg_throughput_mbps=7.9,
    ),
    risky_proposal_ids=[0],
    safe_proposal_ids=[1],
    tool_context=("### Tool Context\n\n**Active faults (DO NOT sleep these cells):**\n"
                  "  Cell 0: power_anomaly (severity=high)\n"
                  "  Sleep candidates (no active fault): [1]\n"),
    tags=["validator", "fault_rejection", "tool_context"],
))


# ── Throughput shown in scenarios that expect sleeps ─────────────────────────
# The mean per-cell QoS put these below the 5 Mbps intent (0.5-3.9 Mbps),
# which no network the harness would sleep looks like: below intent×1.1 its
# throughput guard blocks every sleep (G02). Nemotron read it that way once
# (S05, 2026-11-21) and slept nothing. These scenarios test the PRB rules and
# the guidance, so they show a network comfortably above intent.
_ABOVE_INTENT_MBPS = 6.2
for _s in SCENARIOS:
    if not _s.is_validator_scenario and _s.ground_truth_sleep:
        _s.shown_throughput_mbps = _ABOVE_INTENT_MBPS
