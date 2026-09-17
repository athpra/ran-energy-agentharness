"""
Shared helpers for all experiment runners.

- LLM factory (reads .env, same pattern as the PoC)
- VIAVI RSG simulation wrappers (Sim 1 test, Sim 2 apply)
- Output logging helpers
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
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


# ── VIAVI RSG simulation wrappers ─────────────────────────────────────────────

def make_sim_fns(scenario):
    """
    Return (sim_test_fn, sim_apply_fn) wired to a VIAVI AI RSG Scenario object.

    sim_test_fn  — sends actions to Sim 1 (test/probe), returns (summary, kpi_dict)
    sim_apply_fn — sends actions to Sim 2 (apply),       returns (summary, kpi_dict)
    """
    def _run(sim_index: int, actions: list[dict]) -> tuple[str, dict]:
        sim = scenario.simulators[sim_index]
        for a in actions:
            cell = sim.get_cell(a["cell_id"])
            if a["action"] == "sleep":
                cell.sleep()
            elif a["action"] == "wake":
                cell.wake()
        sim.step()
        kpis = sim.get_kpis()

        lines = [f"Sim {sim_index + 1} post-action KPIs:"]
        for cell_id, k in kpis.items():
            lines.append(
                f"  Cell {cell_id}: throughput={k.get('throughput_mbps', 'N/A')} Mbps  "
                f"util={k.get('utilization_pct', 'N/A')}%  "
                f"sleep={k.get('sleep_state', 'N/A')}"
            )
        return "\n".join(lines), kpis

    def sim_test_fn(actions):  return _run(0, actions)
    def sim_apply_fn(actions): return _run(1, actions)

    return sim_test_fn, sim_apply_fn


# ── Output logging ────────────────────────────────────────────────────────────

def _kpi_to_text(kpis: dict) -> str:
    lines = []
    for cid, k in kpis.items():
        lines.append(
            f"Cell {cid}: throughput={k.get('throughput_mbps','?')} Mbps  "
            f"util={k.get('utilization_pct','?')}%  sleep={k.get('sleep_state','?')}"
        )
    return "\n".join(lines)


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
            "n_proposed":   r["n_proposed"],
            "n_approved":   r["n_approved"],
            "n_rejected":   r["n_rejected"],
            "planner_s":    r["planner_elapsed_s"],
            "validator_s":  r["validator_elapsed_s"],
            "total_s":      r["total_elapsed_s"],
        }
        for r in results
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    print(f"✓ Results saved to {run_dir}")
