"""
Drop-in replacement for viavi.rsg.Scenario.

Implements the exact call surface that experiments/common.py uses — no HTTP,
no VIAVI SDK dependency.  Activate by setting USE_MOCK_RSG=1 in the job
environment (connect_scenario in common.py checks this flag).

Radio model calibrated to match VIAVI observed values:
    Night   (0.1× → 20 UEs,  21 N1 cells awake): tp ≈ 7.7 Mbps/UE
    Morning (1.0× → 200 UEs, 21 N1 cells awake): tp ≈ 5.8 Mbps/UE
    Evening (2.0× → 400 UEs, 17 cells awake):    tp ≈ 4.1 Mbps/UE
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# ── Network layout ────────────────────────────────────────────────────────────
# 7 macro sites × 3 sectors × 2 bands (N1 + N12) = 42 cells, matching the
# real scenario ("Scenario with 2 layers, static + indoor UE v5").
N_SITES   = 7
N_SECTORS = 3
BANDS     = ("N1", "N12")

_ALL_CELL_NAMES: list[str] = [
    f"gNB{s:02d}/{b}/C{c}"
    for s in range(1, N_SITES + 1)
    for b in BANDS
    for c in range(1, N_SECTORS + 1)
]
_N1_NAMES  = [n for n in _ALL_CELL_NAMES if "/N1/"  in n]   # 21 sleepable cells
_N12_NAMES = [n for n in _ALL_CELL_NAMES if "/N12/" in n]   # 21 always-on cells

# ── Radio model ───────────────────────────────────────────────────────────────
_TARGET_MBPS     = 8.0
_BETA            = 0.040   # load/interference factor
_PRB_SAT_UES     = 24.0    # UEs per awake N1 cell → 100% PRB
_N12_PRB_SAT_UES = 10.0
_N12_EFFICIENCY  = 0.20    # N12 provides 20% of N1 per-cell capacity for displaced UEs

# Per-cell load factors: heterogeneous traffic demand across 21 N1 cells.
# At Night  (base PRB≈4%):  cells with factor<3.0 qualify for sleep (<12% PRB) → 16 candidates
# At Morning(base PRB≈40%): only cells with factor<0.30 qualify              →  5 candidates
# At Evening(base PRB≈80%): only cells with factor<0.15 qualify              →  2 candidates
_CELL_LOAD_FACTORS = [
    # 5 very cold cells — always sleepable
    0.10, 0.15, 0.20, 0.25, 0.28,
    # 7 normal cells — sleepable at Night, not Morning
    0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40,
    # 4 moderately loaded cells
    1.60, 1.90, 2.20, 2.60,
    # 5 hot cells — never qualify for sleep (PRB always ≥ 12% at Night)
    3.10, 3.60, 4.20, 5.00, 6.00,
]
assert len(_CELL_LOAD_FACTORS) == len(_N1_NAMES)

# Per-UE throughput noise (±σ fraction of mean)
# Per-UE throughput noise (±σ fraction of mean)
_TP_NOISE_FRAC = 0.06

# Simulation-level load jitter: models stochastic UE arrival variation within
# a 10-second window.  Applied once per MockSimulation instance so Sim-1 and
# Sim-2 see different effective loads — making near-threshold proposals
# genuinely uncertain rather than always returning the same value.
_LOAD_JITTER_FRAC = 0.12


def _tp_per_ue(n_ues: int, n_awake_n1: int) -> float:
    """Network-average per-UE DL throughput.

    Sleeping N1 cells displace their UEs to N12 at reduced efficiency, so
    excessive sleeping degrades throughput — giving the Validator real signal
    to reject unsafe proposals at high load.
    """
    n1_total  = len(_N1_NAMES)
    n12_equiv = (n1_total - n_awake_n1) * _N12_EFFICIENCY
    effective = max(0.5, n_awake_n1 + n12_equiv)
    if n_ues == 0:
        return _TARGET_MBPS
    return _TARGET_MBPS * effective / (effective + _BETA * n_ues)


def _n1_prb(n_ues: int, n_awake_n1: int) -> float:
    if n_awake_n1 == 0 or n_ues == 0:
        return 0.0
    return round(min(100.0, (n_ues / n_awake_n1) / _PRB_SAT_UES * 100.0), 1)


def _n12_prb(n_ues: int, n_awake_n1: int) -> float:
    n1_total  = len(_N1_NAMES)   # 21
    n12_total = len(_N12_NAMES)  # 21
    if n12_total == 0 or n_ues == 0:
        return 0.0
    sleeping_n1 = n1_total - n_awake_n1
    # UEs displaced from sleeping N1 cells all fall back to N12
    displaced = n_ues * sleeping_n1 / n1_total if n1_total > 0 else n_ues
    # Normal secondary N12 usage from awake N1 cells (~40% offload)
    normal    = n_ues * n_awake_n1 / n1_total * 0.4 if n1_total > 0 else 0.0
    return round(min(100.0, (displaced + normal) / n12_total / _N12_PRB_SAT_UES * 100.0), 1)


# ── MockSimulation ─────────────────────────────────────────────────────────────

class MockSimulation:
    """Single simulation session — one-shot: start → commands → run_for → query → finish."""

    def __init__(self, sim_id: str, n_ues: int) -> None:
        self._id       = sim_id
        self._sleeping: set[str] = set()   # N1 cell names currently off
        self._t        = 0                  # simulated seconds elapsed
        self._pause_no = 0
        self._rng      = random.Random(sim_id)
        # Stochastic load: models UE arrival variance within the simulation window.
        # Each sim instance gets a different effective UE count so Sim-1 / Sim-2
        # diverge slightly, giving the Validator genuine uncertainty on proposals
        # that sit near the QoS threshold.
        jitter = self._rng.gauss(1.0, _LOAD_JITTER_FRAC)
        self._n_ues = max(1, int(n_ues * jitter))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("Started simulation '%s'", self._id)
        log.info(
            "Simulation %s reached status \"PAUSED ADK Simulation ready\" after 0 seconds",
            self._id,
        )

    def finish(self) -> None:
        log.info("Simulation %s finished.", self._id)

    # ── Control ───────────────────────────────────────────────────────────────

    def command(self, cmd: str, cell: str, reason: str = "") -> None:
        if cmd == "turn_off":
            self._sleeping.add(cell)
        elif cmd == "turn_on":
            self._sleeping.discard(cell)

    def run_for(self, duration: str) -> None:
        secs = int(duration.rstrip("s"))
        self._t += secs
        self._pause_no += 1
        log.info("Running for %d seconds", secs)
        log.info(
            "Simulation %s reached status \"PAUSED ADK Scheduled Pause #%d\" after 0 seconds",
            self._id, self._pause_no,
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, table: str, start: int = 0, stop: int = 10) -> pd.DataFrame:
        # Reset RNG seed per query so results are deterministic per sim+time
        self._rng.seed(f"{self._id}:{self._t}:{table}")
        n_awake_n1 = len(_N1_NAMES) - sum(1 for c in self._sleeping if "/N1/" in c)

        if table == "UEReports":
            return self._ue_reports(n_awake_n1)
        if table == "CellReports":
            return self._cell_reports(n_awake_n1)
        return pd.DataFrame()

    def _ue_reports(self, n_awake_n1: int) -> pd.DataFrame:
        n_ues = self._n_ues
        if n_ues == 0:
            return pd.DataFrame(columns=["DRB.UEThpDl", "Viavi.Cell.Name"])

        tp_mean = _tp_per_ue(n_ues, n_awake_n1)
        awake_n1 = [c for c in _N1_NAMES if c not in self._sleeping] or _N1_NAMES[:1]

        rows = []
        for i in range(n_ues):
            cell = awake_n1[i % len(awake_n1)]
            noise = self._rng.gauss(0, tp_mean * _TP_NOISE_FRAC)
            rows.append({
                "DRB.UEThpDl":    max(0.1, tp_mean + noise),
                "Viavi.Cell.Name": cell,
            })
        return pd.DataFrame(rows)

    def _cell_reports(self, n_awake_n1: int) -> pd.DataFrame:
        n_ues    = self._n_ues
        rows     = []
        base_prb = _n1_prb(n_ues, n_awake_n1)   # network-average PRB for awake cells

        for i, name in enumerate(_N1_NAMES):
            sleeping = name in self._sleeping
            if sleeping:
                prb = 0.0
            else:
                # Per-cell PRB scaled by cell load factor — creates spatial variation
                # so only a subset of cells qualify for sleep at any given time.
                prb = round(min(100.0, base_prb * _CELL_LOAD_FACTORS[i]), 1)
            rows.append({
                "Viavi.Cell.Name":      name,
                "RRU.PrbTotDl":         prb,
                "Viavi.isEnergySaving": int(sleeping),
            })

        n12_prb = _n12_prb(n_ues, n_awake_n1)
        for name in _N12_NAMES:
            rows.append({
                "Viavi.Cell.Name":      name,
                "RRU.PrbTotDl":         n12_prb,
                "Viavi.isEnergySaving": 0,
            })

        return pd.DataFrame(rows)


# ── Scenario ──────────────────────────────────────────────────────────────────

class Scenario:
    """
    Drop-in for viavi.rsg.Scenario.

    Usage (same as real VIAVI):
        scenario = Scenario(config_path, rsg_address)
        scenario.config["UE_Configuration"]["UE_Groups"][0]["distribution"][0]["ues"] = 5
        sim = scenario.simulation(force_start=True, adk_pace=True)
        sim.start()
        sim.command("turn_off", cell="gNB01/N1/C1", reason="...")
        sim.run_for("10s")
        df_ue   = sim.query("UEReports",   start=9, stop=10)
        df_cell = sim.query("CellReports", start=9, stop=10)
        sim.finish()
    """

    def __init__(self, config_path: str, rsg_address: str = "mock://") -> None:
        self._sim_counter = 0
        self.config = self._load_config(config_path)
        log.info("Mock RSG ready. rsg_address=%s  config=%s", rsg_address, config_path)

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
        # Minimal valid config for tests that don't have the real file
        return {
            "System": {"batch_mode": True, "duration": 10},
            "UE_Configuration": {
                "UE_Groups": [
                    {"distribution": [{"ues": 50}],  "serviceConfig": [{"targetTput_Mbps": 8}]},
                    {"distribution": [{"ues": 150}], "serviceConfig": [{"targetTput_Mbps": 8}]},
                ]
            },
        }

    def _total_ues(self) -> int:
        total = 0
        for grp in self.config.get("UE_Configuration", {}).get("UE_Groups", []):
            for dist in grp.get("distribution", []):
                total += dist.get("ues", 0)
        return max(1, total)

    def simulation(self, force_start: bool = True, adk_pace: bool = True) -> MockSimulation:
        self._sim_counter += 1
        sim_id = f"mock-{self._sim_counter:04d}"
        n_ues  = self._total_ues()
        log.info(
            "Starting a new simulation for Scenario with 2 layers, static + indoor UE v5"
        )
        return MockSimulation(sim_id, n_ues)
