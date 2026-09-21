"""
Hand-crafted KPI scenarios for offline eval.

Each EvalScenario has:
  - cells: list of CellState describing network state
  - ground_truth_sleep: cell IDs an ideal planner SHOULD propose to sleep
  - ground_truth_wake:  cell IDs an ideal planner SHOULD propose to wake
  - kpi_summary: the text string fed to the Planner LLM (same format as production)
  - validator_proposals / sim_result_summary: populated on validator scenarios
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Blueprint thresholds (mirrors agent/planner.py and common.py)
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

    @property
    def ground_truth_sleep(self) -> list[int]:
        return [c.cell_id for c in self.cells if c.should_sleep]

    @property
    def ground_truth_wake(self) -> list[int]:
        return [c.cell_id for c in self.cells if c.should_wake]

    @property
    def kpi_summary(self) -> str:
        sleeping = sum(1 for c in self.cells if c.n1_sleeping)
        awake    = [c for c in self.cells if not c.n1_sleeping]
        avg_tp   = sum(c.avg_qos for c in awake) / len(awake) if awake else 0.0
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
    cells=[
        CellState(0, "S1/C1", n1_prb=0.123,  n12_prb=0.0,    avg_qos=0.001, n1_sleeping=False),
        CellState(1, "S1/C2", n1_prb=11.999, n12_prb=0.0,    avg_qos=3.777, n1_sleeping=False),
        CellState(2, "S2/C1", n1_prb=0.0,    n12_prb=60.001, avg_qos=0.0,   n1_sleeping=True),
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
    tags=["validator", "fault_rejection", "tool_context"],
))
