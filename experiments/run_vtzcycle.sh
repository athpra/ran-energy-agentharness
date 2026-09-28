#!/usr/bin/env bash
# Run all four VTZ-cycle experimental conditions sequentially.
#
# Usage:
#   bash experiments/run_vtzcycle.sh
#   INTENT="3 Mbps" ITERATIONS=48 bash experiments/run_vtzcycle.sh
#
# Environment variables (all optional — defaults shown):
#   INTENT      operator QoS intent passed to each script  (default: "5 Mbps")
#   ITERATIONS  number of iterations per run               (default: 96)
#   SKIP        space-separated list of conditions to skip (default: none)
#               e.g. SKIP="rules openloop" to run only digital_twin and full_harness
#
# Each run's stdout+stderr is tee'd to output/logs/<timestamp>_<condition>.log
# so you can check what happened even if a run crashes.

set -uo pipefail

INTENT="${INTENT:-5 Mbps}"
ITERATIONS="${ITERATIONS:-96}"
SKIP="${SKIP:-}"

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")

run() {
    local condition="$1"
    local cmd="$2"

    # Check SKIP list
    for s in $SKIP; do
        if [[ "$condition" == *"$s"* ]]; then
            echo "⏭  Skipping $condition (in SKIP list)"
            return 0
        fi
    done

    local log="$LOG_DIR/${TS}_${condition}.log"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "▶  $condition"
    echo "   intent='$INTENT'  iterations=$ITERATIONS"
    echo "   log → $log"
    echo "════════════════════════════════════════════════════════"

    local start=$(date +%s)
    if eval "$cmd" 2>&1 | tee "$log"; then
        local elapsed=$(( $(date +%s) - start ))
        echo "✓  $condition completed in ${elapsed}s"
    else
        local elapsed=$(( $(date +%s) - start ))
        echo "✗  $condition FAILED after ${elapsed}s — see $log"
        echo "   Continuing with next condition..."
    fi
}

echo "Starting VTZ-cycle experiment sweep"
echo "  intent='$INTENT'  iterations=$ITERATIONS  started=$(date)"

run "rules" \
    "python experiments/baseline_rules.py \
        --intent \"$INTENT\" \
        --iterations $ITERATIONS"

run "openloop" \
    "python experiments/baseline_openloop_vtzcycle.py \
        --intent \"$INTENT\" \
        --iterations $ITERATIONS"

run "digital_twin" \
    "python experiments/baseline_digital_twin_vtzcycle.py \
        --intent \"$INTENT\" \
        --iterations $ITERATIONS"

run "full_harness" \
    "python experiments/full_harness_vtzcycle.py \
        --intent \"$INTENT\" \
        --iterations $ITERATIONS"

echo ""
echo "════════════════════════════════════════════════════════"
echo "All conditions done — $(date)"
echo "Logs in $LOG_DIR/"
echo "Results in output/"
