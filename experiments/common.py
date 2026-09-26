"""
Shared helpers for all experiment runners.

- LLM factory (reads .env, same pattern as the PoC)
- VIAVI RSG simulation wrappers (Sim 1 test, Sim 2 apply)
- Output logging helpers
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"


# ── CML job arguments helper ──────────────────────────────────────────────────

def apply_job_arguments() -> None:
    """Inject CML's JOB_ARGUMENTS env var into sys.argv.

    CML's PBJ Workbench passes the job Arguments field as the JOB_ARGUMENTS
    environment variable rather than directly into sys.argv, so argparse never
    sees them unless we do this explicitly.  Call once at the top of main().
    """
    import shlex
    job_args = os.getenv("JOB_ARGUMENTS", "").strip()
    if job_args:
        sys.argv.extend(shlex.split(job_args))


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(model: str | None = None, api_base: str | None = None) -> ChatOpenAI:
    load_dotenv(find_dotenv(), override=True)
    auth_mode = os.getenv("AUTH_MODE", "jwt").strip().lower()

    if auth_mode == "jwt":
        try:
            import json as _json
            token = _json.load(open("/tmp/jwt")).get("access_token", "").strip()
            if token:
                api_key = token
            else:
                raise ValueError
        except Exception:
            api_key = os.getenv("CDSW_API_KEY", "").strip()
    else:
        api_key = os.getenv("CDSW_API_KEY", "").strip()

    if not api_key:
        sys.exit("Error: no API key available. Set CDSW_API_KEY in .env or ensure /tmp/jwt exists.")

    return ChatOpenAI(
        model=(model or os.getenv("LLM_MODEL", "")).strip(),
        openai_api_key=api_key,
        openai_api_base=(api_base or os.getenv("CDSW_API_URL", "")).strip(),
        temperature=0.1,
        max_tokens=4096,
        request_timeout=120,
    )


def get_validator_llm(model: str | None = None, api_base: str | None = None) -> ChatOpenAI | None:
    """Return a separate LLM for the Validator, or None if no validator config is set.

    Reads VALIDATOR_MODEL and VALIDATOR_API_URL from .env.  CLI overrides
    (model, api_base) take precedence.  Returns None when neither CLI nor env
    provides a validator model, so the caller can fall back to the planner LLM.
    """
    load_dotenv(find_dotenv(), override=True)
    resolved_model    = (model    or os.getenv("VALIDATOR_MODEL",   "")).strip()
    resolved_api_base = (api_base or os.getenv("VALIDATOR_API_URL", "")).strip()
    if not resolved_model:
        return None
    return get_llm(model=resolved_model, api_base=resolved_api_base or None)


# ── VIAVI RSG connection ──────────────────────────────────────────────────────

def connect_scenario(rsg_host: str = "", scenario_conf: str | None = None):
    """
    Connect to VIAVI AI RSG and return a Scenario object.

    Derives a stable client hash from the machine's public IP + username,
    constructing the proxy URL http://<rsg_host>:8000/c/<hash>/ — the same
    pattern used in the blueprint PoC notebook.
    """
    load_dotenv(find_dotenv(), override=True)
    if not rsg_host:
        rsg_host = os.getenv("RSG_HOST", "").strip()

    if scenario_conf is None:
        scenario_conf = str(PROJECT_ROOT / "ai_rsg_config" / "config.conf")

    if os.getenv("USE_MOCK_RSG", "").strip().lower() in ("1", "true", "yes"):
        from mock_rsg.scenario import Scenario as MockScenario
        print("Mock RSG active (USE_MOCK_RSG=1)")
        return MockScenario(scenario_conf, "mock://")

    from viavi.rsg import Scenario

    # Check for a full URL override (e.g. RSG_ADDRESS env var)
    rsg_address_override = os.getenv("RSG_ADDRESS", "").strip()
    if rsg_address_override:
        print("Using RSG_ADDRESS override:", rsg_address_override)
        return Scenario(scenario_conf, rsg_address_override)

    if not rsg_host:
        sys.exit("Error: RSG_HOST is not set. Pass --rsg-host or set the RSG_HOST env var.")

    # Verify TCP reachability
    try:
        socket.create_connection((rsg_host, 8000), timeout=5).close()
    except Exception as ex:
        sys.exit(f"Error: cannot reach RSG host {rsg_host}:8000 — {ex}")

    # Derive a stable hash from public IP + username (same method as PoC)
    try:
        public_ip = requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        public_ip = "unknownip"
    user_name = os.getenv("HADOOP_USER_NAME", os.getenv("USER", "cdsw"))
    h = hashlib.sha256(f"{public_ip}-{user_name}".encode()).hexdigest()[:8]
    rsg_addr = f"http://{rsg_host}:8000/c/{h}/"
    print(f"RSG addr: {rsg_addr}")

    return Scenario(scenario_conf, rsg_addr)


# ── Traffic profiles (mirrors PoC VTZ model) ──────────────────────────────────

_TRAFFIC_PROFILES = [
    {"name": "Night",      "start_hour": 0,  "duration_hours": 8,  "ue_ratio": 0.1},
    {"name": "Early",      "start_hour": 8,  "duration_hours": 1,  "ue_ratio": 1.5},
    {"name": "Morning",    "start_hour": 9,  "duration_hours": 4,  "ue_ratio": 1.0},
    {"name": "Afternoon",  "start_hour": 13, "duration_hours": 5,  "ue_ratio": 0.5},
    {"name": "Evening_1",  "start_hour": 18, "duration_hours": 2,  "ue_ratio": 2.0},
    {"name": "Evening_2",  "start_hour": 20, "duration_hours": 2,  "ue_ratio": 1.0},
    {"name": "Late Night", "start_hour": 22, "duration_hours": 2,  "ue_ratio": 0.1},
]

DEFAULT_THROUGHPUT_MBPS = 8.0
SIM_DURATION            = 10   # seconds per simulation run
SIM_MAX_RETRIES         = 5
SIM_RETRY_DELAY_S       = 60


def _get_profile(ts: pd.Timestamp) -> dict:
    hour = ts.hour
    for p in _TRAFFIC_PROFILES:
        if p["start_hour"] <= hour < p["start_hour"] + p["duration_hours"]:
            return p
    return _TRAFFIC_PROFILES[0]


def _compute_kpis(sim_ue: pd.DataFrame, sim_cell: pd.DataFrame) -> dict:
    """Convert VIAVI DataFrames to a summary dict the agent can consume."""
    avg_thp = float(sim_ue["DRB.UEThpDl"].mean()) if len(sim_ue) > 0 else 0.0
    per_site: dict = {}

    for _, row in sim_cell.iterrows():
        parts = row["Viavi.Cell.Name"].split("/")
        site  = f"{parts[0]}/{parts[2]}"
        band  = parts[1]
        if site not in per_site:
            per_site[site] = {"n1_prb": 0.0, "n12_prb": 0.0, "avg_qos": 0.0, "n1_sleeping": False}
        if band == "N1":
            per_site[site]["n1_prb"]      = float(row.get("RRU.PrbTotDl", 0))
            per_site[site]["n1_sleeping"] = bool(row.get("Viavi.isEnergySaving", 0))
        elif band == "N12":
            per_site[site]["n12_prb"] = float(row.get("RRU.PrbTotDl", 0))

    for _, row in sim_ue.iterrows():
        parts = row["Viavi.Cell.Name"].split("/")
        site  = f"{parts[0]}/{parts[2]}"
        if site in per_site:
            per_site[site].setdefault("_qos", []).append(float(row["DRB.UEThpDl"]))

    for s in per_site.values():
        if "_qos" in s:
            qos = s.pop("_qos")
            s["avg_qos"] = sum(qos) / len(qos)

    sleeping = sum(1 for s in per_site.values() if s["n1_sleeping"])
    return {
        "avg_throughput_mbps": round(avg_thp, 3),
        "sleeping_cells":      sleeping,
        "total_cells":         len(per_site) * 2,
        "per_site":            per_site,
    }


# ── VIAVI RSG simulation wrappers ─────────────────────────────────────────────

def make_sim_fns(scenario):
    """
    Return (sim_test_fn, sim_apply_fn, set_virtual_ts).

    Both sim functions have signature:
        fn(actions: list[dict]) -> (summary_str, kpi_dict)

    set_virtual_ts(ts: pd.Timestamp) updates the timestamp used to select the
    traffic profile for the next sim run.  Call it once per iteration before
    the harness runs, so UE counts track the virtual 24-hour cycle rather than
    the wall clock (which would always return the same profile for a run that
    completes within a single hour).

    Internally:
    - cell_id integers are mapped to VIAVI full cell names ("Site/N1/Sector")
      on the first simulation run (names discovered from CellReports output).
    - Accumulated sleep state is tracked here and replayed at the start of
      every simulation — the SDK is stateless, so we must reapply history.
    - sim_test_fn does NOT update accumulated state (probe only).
    - sim_apply_fn DOES update accumulated state (confirmed actions).
    - Traffic profile (UE count) is set from current_vts (set via set_virtual_ts).
    """
    cell_sleep_state:   dict[str, bool]       = {}   # cell_name -> is_sleeping
    cell_name_map:      dict[int, str]         = {}   # int id -> full VIAVI name
    original_ue_dist:   list[list[int]]        = []   # cached from scenario config
    current_vts:        list                   = [None]  # mutable slot for virtual ts

    def set_virtual_ts(ts: pd.Timestamp) -> None:
        """Update the virtual timestamp used to select the traffic profile."""
        current_vts[0] = ts

    def _cache_ue_dist() -> list[list[int]]:
        if not original_ue_dist:
            for grp in scenario.config["UE_Configuration"]["UE_Groups"]:
                original_ue_dist.append([d["ues"] for d in grp.get("distribution", [])])
        return original_ue_dist

    def _apply_profile(profile: dict) -> None:
        orig = _cache_ue_dist()
        for g, grp in enumerate(scenario.config["UE_Configuration"]["UE_Groups"]):
            for d, dist in enumerate(grp.get("distribution", [])):
                dist["ues"] = max(1, int(orig[g][d] * profile["ue_ratio"]))
            for svc in grp.get("serviceConfig", []):
                svc["targetTput_Mbps"] = DEFAULT_THROUGHPUT_MBPS

    def _run(actions: list[dict], update_state: bool) -> tuple[str, dict]:
        import time as _time

        ts = current_vts[0] if current_vts[0] is not None else pd.Timestamp.now()
        profile = _get_profile(ts)
        _apply_profile(profile)

        scenario.config["System"]["batch_mode"] = True
        scenario.config["System"]["duration"]   = SIM_DURATION

        last_exc: Exception | None = None
        for attempt in range(SIM_MAX_RETRIES):
            try:
                sim = scenario.simulation(force_start=True, adk_pace=True)
                sim.start()

                # Replay accumulated sleep state so each fresh simulation starts from
                # the current network configuration, not a clean slate.
                # Small delay between commands avoids 409 CONFLICT from the SBA
                # endpoint when many cells need to be replayed simultaneously.
                for cell_name, is_sleeping in cell_sleep_state.items():
                    cmd = "turn_off" if is_sleeping else "turn_on"
                    sim.command(cmd, cell=cell_name, reason="accumulated state")
                    _time.sleep(0.15)

                # Apply new actions for this iteration
                for a in (actions or []):
                    cell_name = cell_name_map.get(a["cell_id"], str(a["cell_id"]))
                    if a["action"] == "sleep":
                        sim.command("turn_off", cell=cell_name, reason=a.get("reason", "energy optimization"))
                    elif a["action"] == "wake":
                        sim.command("turn_on",  cell=cell_name, reason=a.get("reason", "capacity needed"))

                sim.run_for(f"{SIM_DURATION}s")
                sim_ue   = sim.query("UEReports",   start=SIM_DURATION - 1, stop=SIM_DURATION)
                sim_cell = sim.query("CellReports", start=SIM_DURATION - 1, stop=SIM_DURATION)
                sim.finish()
                break  # success

            except (RuntimeError, Exception) as exc:
                last_exc = exc
                # 429 quota exceeded — wait for the cooldown the server reports
                wait_s = SIM_RETRY_DELAY_S
                cause = exc.__cause__ or exc
                if hasattr(cause, "response") and getattr(cause.response, "status_code", None) == 429:
                    try:
                        cooldown = cause.response.json().get("cooldown_remaining_seconds", 1800)
                        wait_s = int(cooldown) + 60  # add 1 min buffer
                    except Exception:
                        wait_s = 1860
                if attempt < SIM_MAX_RETRIES - 1:
                    print(f"  RSG error (attempt {attempt + 1}/{SIM_MAX_RETRIES}): {exc}. "
                          f"Retrying in {wait_s}s...")
                    _time.sleep(wait_s)
                else:
                    raise RuntimeError(
                        f"RSG simulation failed after {SIM_MAX_RETRIES} attempts"
                    ) from last_exc

        sim_cell = sim_cell.drop_duplicates(subset=["Viavi.Cell.Name"], keep="last").reset_index(drop=True)

        # Build integer → name map from first results (N1 cells only — only N1 can sleep)
        if not cell_name_map and len(sim_cell) > 0:
            n1_names = sorted(
                sim_cell.loc[sim_cell["Viavi.Cell.Name"].str.contains("/N1/"), "Viavi.Cell.Name"].unique()
            )
            cell_name_map.update({idx: name for idx, name in enumerate(n1_names)})

        kpis = _compute_kpis(sim_ue, sim_cell)

        # Persist state only for the apply simulation
        if update_state:
            for a in (actions or []):
                cell_name = cell_name_map.get(a["cell_id"], str(a["cell_id"]))
                cell_sleep_state[cell_name] = (a["action"] == "sleep")

        return _kpi_to_text_viavi(kpis), kpis

    def sim_test_fn(actions):  return _run(actions, update_state=False)
    def sim_apply_fn(actions): return _run(actions, update_state=True)

    # Warm-up: run one no-action sim so cell_name_map is populated before
    # the first real call, which would otherwise use invalid integer-as-string
    # cell names and silently apply no sleep commands.
    print("Initialising cell name map...")
    _run([], update_state=False)
    print(f"Cell map ready: {len(cell_name_map)} N1 cells found.")

    # Validate VTZ traffic variation: confirm that force_start picks up
    # scenario.config mutations so UE counts actually change per profile.
    # Uses Night (02:00, ue_ratio=0.1) vs Morning (09:00, ue_ratio=1.0).
    # If force_start re-reads the disk config, both sims return identical
    # throughput and we print a clear warning so the issue is obvious.
    print("Validating VTZ traffic variation (Night vs Morning)...")
    _today = pd.Timestamp.now().normalize()
    current_vts[0] = _today + pd.Timedelta(hours=2)   # 02:00 Night   (0.1×)
    _, _night_kpis   = _run([], update_state=False)
    current_vts[0] = _today + pd.Timedelta(hours=9)   # 09:00 Morning (1.0×)
    _, _morning_kpis = _run([], update_state=False)
    current_vts[0] = None  # reset; caller sets it via set_virtual_ts each iteration

    _night_thp   = _night_kpis.get("avg_throughput_mbps", 0)
    _morning_thp = _morning_kpis.get("avg_throughput_mbps", 0)
    # avg_throughput_mbps is mean PER-UE throughput. Night has 0.1× UEs so each
    # UE gets more bandwidth → Night per-UE throughput > Morning per-UE throughput.
    _ratio = _night_thp / max(_morning_thp, 0.001)
    if _ratio > 1.1:
        print(
            f"VTZ variation confirmed: Night={_night_thp:.2f} Mbps/UE > "
            f"Morning={_morning_thp:.2f} Mbps/UE  "
            f"(Night has ~10× fewer UEs → higher per-UE throughput, ratio={_ratio:.2f}×)"
        )
    elif abs(_ratio - 1.0) < 0.05:
        print(
            f"WARNING: VTZ variation NOT detected. Night={_night_thp:.2f} Mbps, "
            f"Morning={_morning_thp:.2f} Mbps (ratio={_ratio:.2f}) — "
            f"force_start may be re-reading the disk config and discarding "
            f"scenario.config UE-count mutations."
        )
    else:
        print(
            f"VTZ variation detected (ratio={_ratio:.2f}×): "
            f"Night={_night_thp:.2f} Mbps/UE, Morning={_morning_thp:.2f} Mbps/UE"
        )

    return sim_test_fn, sim_apply_fn, set_virtual_ts


# ── Continuous VTZ-cycle simulation ──────────────────────────────────────────

VTZ_STEP_S   = 900   # simulated seconds per agent iteration (15 VTZ minutes)
VTZ_SETTLE_S = 30    # simulated seconds to stabilise after commands
VTZ_QUERY_W  = 10    # query window width (last N simulated seconds)


def make_sim_fns_continuous(scenario, step_s: int = VTZ_STEP_S,
                             settle_s: int = VTZ_SETTLE_S,
                             query_window_s: int = VTZ_QUERY_W):
    """
    Return (advance_step, sim_test_fn, sim_apply_fn, finish_sim) for a
    single continuous VIAVI simulation.

    Unlike make_sim_fns(), this starts ONE simulation that runs for the
    full experiment.  VIAVI's internal VTZ clock cycles through Night /
    Morning / Evening traffic profiles naturally, producing the dynamic
    throughput variation visible in the blueprint.

    Call order each iteration:
        kpis = advance_step()          # advance VTZ clock, get pre-action KPIs
        # harness calls sim_test_fn and sim_apply_fn internally
        ...
    Call finish_sim() once at the end.

    sim_test_fn(actions) returns the current (pre-action) KPIs as the
    Sim-1 evidence — no extra simulation time consumed.  The Validator
    therefore sees the live network state and decides whether applying the
    proposed changes is safe.

    sim_apply_fn(actions) issues commands to the running simulation,
    advances settle_s seconds for stabilisation, and returns post-action KPIs.
    """
    cell_sleep_state: dict[str, bool] = {}
    cell_name_map:    dict[int, str]  = {}
    cumulative_t = [0]
    last_kpis:    list               = [None]

    # Start the simulation with enough duration for a full VTZ cycle + buffer
    total_s = step_s * 110 + settle_s * 110 + 60   # ~29 h of headroom
    scenario.config["System"]["batch_mode"] = True
    scenario.config["System"]["duration"]   = total_s

    sim = scenario.simulation(force_start=True, adk_pace=True)
    sim.start()

    # Warmup: advance briefly so VIAVI populates CellReports
    _WU = SIM_DURATION
    sim.run_for(f"{_WU}s")
    cumulative_t[0] = _WU
    wu_ue   = sim.query("UEReports",   start=_WU - 1, stop=_WU)
    wu_cell = sim.query("CellReports", start=_WU - 1, stop=_WU)
    wu_cell = (wu_cell
               .drop_duplicates(subset=["Viavi.Cell.Name"], keep="last")
               .reset_index(drop=True))
    n1_names = sorted(
        wu_cell.loc[wu_cell["Viavi.Cell.Name"].str.contains("/N1/"),
                    "Viavi.Cell.Name"].unique()
    )
    cell_name_map.update({idx: name for idx, name in enumerate(n1_names)})
    print(f"Continuous VTZ sim started. Cell map: {len(cell_name_map)} N1 cells.")

    def _query_now() -> dict:
        t  = cumulative_t[0]
        ue = sim.query("UEReports",   start=max(0, t - query_window_s), stop=t)
        cl = sim.query("CellReports", start=max(0, t - query_window_s), stop=t)
        cl = cl.drop_duplicates(subset=["Viavi.Cell.Name"], keep="last").reset_index(drop=True)
        return _compute_kpis(ue, cl)

    def advance_step() -> dict:
        """Advance one VTZ time step; return current (pre-action) KPIs."""
        sim.run_for(f"{step_s}s")
        cumulative_t[0] += step_s
        last_kpis[0] = _query_now()
        return last_kpis[0]

    def sim_test_fn(actions: list[dict]) -> tuple[str, dict]:
        """Return the pre-action network state as Sim-1 evidence."""
        kpis = last_kpis[0] or {}
        return _kpi_to_text_viavi(kpis), kpis

    def sim_apply_fn(actions: list[dict]) -> tuple[str, dict]:
        """Issue approved commands, settle, return post-action KPIs."""
        for a in (actions or []):
            cell_name = cell_name_map.get(a["cell_id"], str(a["cell_id"]))
            cmd = "turn_off" if a["action"] == "sleep" else "turn_on"
            sim.command(cmd, cell=cell_name, reason=a.get("reason", ""))
            cell_sleep_state[cell_name] = (a["action"] == "sleep")
        sim.run_for(f"{settle_s}s")
        cumulative_t[0] += settle_s
        kpis = _query_now()
        return _kpi_to_text_viavi(kpis), kpis

    def finish_sim() -> None:
        sim.finish()

    return advance_step, sim_test_fn, sim_apply_fn, finish_sim


# ── KPI formatting ────────────────────────────────────────────────────────────

def _kpi_to_text_viavi(kpis: dict) -> str:
    """Format VIAVI KPI dict (from _compute_kpis) as a prompt-ready string.

    Integer cell_id is included in each line so the Planner can reference
    it directly in JSON actions.
    """
    lines = [f"avg_throughput={kpis['avg_throughput_mbps']:.2f} Mbps  "
             f"sleeping={kpis['sleeping_cells']}/{kpis['total_cells']}"]
    for idx, (site, s) in enumerate(sorted(kpis.get("per_site", {}).items())):
        state = "Sleeping" if s["n1_sleeping"] else "Awake"
        lines.append(
            f"  cell_id={idx}  {site}: N1_PRB={s['n1_prb']:.1f}%  N12_PRB={s['n12_prb']:.1f}%  "
            f"QoS={s['avg_qos']:.2f} Mbps  [{state}]"
        )
    return "\n".join(lines)


def _kpi_to_text(kpis: dict) -> str:
    """Format KPI dict for the Planner prompt — works for both VIAVI and simple dicts."""
    if "per_site" in kpis:
        return _kpi_to_text_viavi(kpis)
    lines = []
    for cid, k in kpis.items():
        lines.append(
            f"Cell {cid}: throughput={k.get('throughput_mbps','?')} Mbps  "
            f"util={k.get('utilization_pct','?')}%  sleep={k.get('sleep_state','?')}"
        )
    return "\n".join(lines)


def get_current_kpis(scenario, timestamp: pd.Timestamp | None = None) -> dict:
    """
    Run a short no-action simulation to read the current network state.
    Used at the start of each iteration to give the Planner fresh KPIs.

    Pass `timestamp` (the virtual iteration time) so traffic profiles advance
    through a 24-hour cycle across iterations rather than staying fixed at the
    current wall-clock time.  Falls back to now() when omitted.
    """
    profile = _get_profile(timestamp if timestamp is not None else pd.Timestamp.now())

    orig = []
    for grp in scenario.config["UE_Configuration"]["UE_Groups"]:
        orig.append([d["ues"] for d in grp.get("distribution", [])])
    for g, grp in enumerate(scenario.config["UE_Configuration"]["UE_Groups"]):
        for d, dist in enumerate(grp.get("distribution", [])):
            dist["ues"] = max(1, int(orig[g][d] * profile["ue_ratio"]))
        for svc in grp.get("serviceConfig", []):
            svc["targetTput_Mbps"] = DEFAULT_THROUGHPUT_MBPS

    scenario.config["System"]["batch_mode"] = True
    scenario.config["System"]["duration"]   = SIM_DURATION

    import time as _time
    last_exc: Exception | None = None
    for attempt in range(SIM_MAX_RETRIES):
        try:
            sim = scenario.simulation(force_start=True, adk_pace=True)
            sim.start()
            sim.run_for(f"{SIM_DURATION}s")
            sim_ue   = sim.query("UEReports",   start=SIM_DURATION - 1, stop=SIM_DURATION)
            sim_cell = sim.query("CellReports", start=SIM_DURATION - 1, stop=SIM_DURATION)
            sim.finish()
            break
        except Exception as exc:
            last_exc = exc
            wait_s = SIM_RETRY_DELAY_S
            cause = exc.__cause__ or exc
            if hasattr(cause, "response") and getattr(cause.response, "status_code", None) == 429:
                try:
                    cooldown = cause.response.json().get("cooldown_remaining_seconds", 1800)
                    wait_s = int(cooldown) + 60
                except Exception:
                    wait_s = 1860
            if attempt < SIM_MAX_RETRIES - 1:
                print(f"  RSG error in get_current_kpis (attempt {attempt + 1}/{SIM_MAX_RETRIES}): {exc}. "
                      f"Retrying in {wait_s}s...")
                _time.sleep(wait_s)
            else:
                raise RuntimeError(
                    f"get_current_kpis failed after {SIM_MAX_RETRIES} attempts"
                ) from last_exc

    sim_cell = sim_cell.drop_duplicates(subset=["Viavi.Cell.Name"], keep="last").reset_index(drop=True)
    return _compute_kpis(sim_ue, sim_cell)


def make_run_dir(condition_name: str, model_name: str) -> Path:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{ts}_{condition_name}_{model_name.replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(run_dir: Path, config: dict) -> None:
    """Write run_config.json to the run directory for provenance tracking."""
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


def append_result(run_dir: Path, result: dict) -> None:
    """Append a single iteration result to iterations.jsonl (creates file if needed)."""
    with open(run_dir / "iterations.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


def save_results(run_dir: Path, results: list[dict]) -> None:
    log_path = run_dir / "iterations.jsonl"
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_rows = [
        {
            "iteration":    r["iteration"],
            "condition":    r.get("condition", ""),
            "n_proposed":   r["n_proposed"],
            "n_approved":   r["n_approved"],
            "n_rejected":   r["n_rejected"],
            "planner_s":    r.get("planner_elapsed_s", 0),
            "validator_s":  r.get("validator_elapsed_s", 0),
            "total_s":      r["total_elapsed_s"],
        }
        for r in results
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    print(f"✓ Results saved to {run_dir}")
