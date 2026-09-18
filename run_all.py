"""
Run all four experimental conditions sequentially.
Designed to be launched as a CML Job (Workbench / Python 3.10).

Environment overrides:
  ITER   — number of iterations per condition (default 96)
  INTENT — operator QoS intent string       (default "5 Mbps")
"""
import os
import subprocess
import sys

# Install dependencies before importing project modules
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)

ITER   = os.environ.get("ITER",   "96")
INTENT = os.environ.get("INTENT", "5 Mbps")

CONDITIONS = [
    ("baseline_rules",        "experiments/baseline_rules.py"),
    ("baseline_openloop",     "experiments/baseline_openloop.py"),
    ("baseline_digital_twin", "experiments/baseline_digital_twin.py"),
    ("full_harness",          "experiments/full_harness.py"),
]

print(f"Starting full experiment run: iterations={ITER}  intent='{INTENT}'")
print()

for i, (name, script) in enumerate(CONDITIONS, 1):
    print(f"=== [{i}/{len(CONDITIONS)}] {name} ===")
    result = subprocess.run(
        [sys.executable, script, "--intent", INTENT, "--iterations", ITER],
        check=True,
    )
    print()

print("=== All conditions complete ===")
subprocess.run(["ls", "-lh", "output/"])
