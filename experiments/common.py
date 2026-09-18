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


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
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
        model=os.getenv("LLM_MODEL", "").strip(),
        openai_api_key=api_key,
        openai_api_base=os.getenv("CDSW_API_URL", "").strip(),
        temperature=0.1,
        max_tokens=4096,
        request_timeout=120,
    )


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

    from viavi.rsg import Scenario

    if scenario_conf is None:
        scenario_conf = str(PROJECT_ROOT / "ai_rsg_config" / "config.conf")

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
SIM_MAX_RETRIES         = 3
SIM_RETRY_DELAY_S       = 30


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
    Return (sim_test_fn, sim_apply_fn).

    Both functions have signature:
        fn(actions: list[dict]) -> (summary_str, kpi_dict)

    where each action is {"action": "sleep"|"wake", "cell_id": int, ...}

    Internally:
    - cell_id integers are mapped to VIAVI full cell names ("Site/N1/Sector")
      on the first simulation run (names discovered from CellReports output).
    - Accumulated sleep state is tracked here and replayed at the start of
      every simulation — the SDK is stateless, so we must reapply history.
    - sim_test_fn does NOT update accumulated state (probe only).
    - sim_apply_fn DOES update accumulated state (confirmed actions).
    - Traffic profile (UE count) is set from current wall-clock time.
    """
    cell_sleep_state:   dict[str, bool]       = {}   # cell_name -> is_sleeping
    cell_name_map:      dict[int, str]         = {}   # int id -> full VIAVI name
    original_ue_dist:   list[list[int]]        = []   # cached from scenario config

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

        profile = _get_profile(pd.Timestamp.now())
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
                for cell_name, is_sleeping in cell_sleep_state.items():
                    cmd = "turn_off" if is_sleeping else "turn_on"
                    sim.command(cmd, cell=cell_name, reason="accumulated state")

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
                if attempt < SIM_MAX_RETRIES - 1:
                    print(f"  RSG error (attempt {attempt + 1}/{SIM_MAX_RETRIES}): {exc}. "
                          f"Retrying in {SIM_RETRY_DELAY_S}s...")
                    _time.sleep(SIM_RETRY_DELAY_S)
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

    return sim_test_fn, sim_apply_fn


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


def get_current_kpis(scenario) -> dict:
    """
    Run a short no-action simulation to read the current network state.
    Used at the start of each iteration to give the Planner fresh KPIs.
    """
    profile = _get_profile(pd.Timestamp.now())

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
            if attempt < SIM_MAX_RETRIES - 1:
                print(f"  RSG error in get_current_kpis (attempt {attempt + 1}/{SIM_MAX_RETRIES}): {exc}. "
                      f"Retrying in {SIM_RETRY_DELAY_S}s...")
                _time.sleep(SIM_RETRY_DELAY_S)
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
