"""
Run all synthetic data generators in dependency order.

Historical KPI must be generated before traffic forecast and faults,
since those read from it. Interference and pricing are independent.

Usage:
    python generators/generate_all.py
    python generators/generate_all.py --n-cells 6 --n-days 14 --seed 99
"""
import argparse
import subprocess
import sys
from pathlib import Path

GENERATORS_DIR = Path(__file__).parent


def run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(GENERATORS_DIR / script)] + extra_args
    print(f"\n→ {script}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Failed: {script}")


def main():
    parser = argparse.ArgumentParser(description="Generate all synthetic datasets")
    parser.add_argument("--n-cells",   type=int,   default=6)
    parser.add_argument("--n-days",    type=int,   default=7)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--fault-prob",type=float, default=0.05)
    args = parser.parse_args()

    common  = ["--n-days",  str(args.n_days),  "--seed", str(args.seed)]
    cell    = ["--n-cells", str(args.n_cells), "--seed", str(args.seed)]

    # Order matters: historical KPI first
    run("generate_historical_kpi.py",   cell + ["--n-days", str(args.n_days)])
    run("generate_traffic_forecast.py", ["--seed", str(args.seed)])
    run("generate_faults.py",           ["--fault-prob", str(args.fault_prob),
                                         "--seed", str(args.seed)])
    run("generate_interference.py",     cell)
    run("generate_energy_pricing.py",   common)

    print("\n✓ All synthetic datasets generated in data/synthetic/")


if __name__ == "__main__":
    main()
