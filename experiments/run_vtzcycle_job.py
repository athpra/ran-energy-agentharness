"""
CML Job entry point — run all four VTZ-cycle conditions sequentially.

Set this file as the job script in CML. Configure via environment variables
in the job settings (or .env file on the CML project):

  INTENT          operator QoS intent  (default: "5 Mbps")
  ITERATIONS      iterations per run   (default: 96)
  SKIP            comma-separated conditions to skip
                  choices: rules, openloop, digital_twin, full_harness
  LLM_MODEL       planner model ID
  VALIDATOR_MODEL validator model ID (full_harness only)
  RSG_HOST        VIAVI AI RSG host URL
"""
from __future__ import annotations

import os
import sys
import time
import traceback

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

INTENT     = os.environ.get("INTENT",     "5 Mbps")
ITERATIONS = int(os.environ.get("ITERATIONS", 96))
SKIP       = {s.strip() for s in os.environ.get("SKIP", "").split(",") if s.strip()}

print(f"VTZ-cycle sweep  intent='{INTENT}'  iterations={ITERATIONS}")
if SKIP:
    print(f"  Skipping: {SKIP}")
print()


def _run(label: str, module: str):
    """Import module.main(), patch sys.argv, run it, catch failures."""
    if label in SKIP:
        print(f"⏭  Skipping {label}")
        return

    print(f"\n{'='*60}")
    print(f"▶  {label}")
    print(f"{'='*60}")

    # Patch argv so each script's argparse picks up INTENT/ITERATIONS
    sys.argv = [
        label,
        "--intent",     INTENT,
        "--iterations", str(ITERATIONS),
    ]

    t0 = time.time()
    try:
        mod = __import__(module, fromlist=["main"])
        mod.main()
        print(f"✓  {label} completed in {time.time()-t0:.0f}s")
    except Exception:
        print(f"✗  {label} FAILED after {time.time()-t0:.0f}s")
        traceback.print_exc()
        print("   Continuing with next condition...")


_run("rules",        "experiments.baseline_rules")
_run("openloop",     "experiments.baseline_openloop_vtzcycle")
_run("digital_twin", "experiments.baseline_digital_twin_vtzcycle")
_run("full_harness", "experiments.full_harness_vtzcycle")

print(f"\n{'='*60}")
print("All conditions complete.")
