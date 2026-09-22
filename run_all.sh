#!/bin/bash
set -e

ITER=${ITER:-96}
INTENT=${INTENT:-"5 Mbps"}

echo "Starting full VTZ-cycle experiment run: iterations=$ITER  intent='$INTENT'"
echo ""

echo "=== [1/4] baseline_rules ==="
python experiments/baseline_rules.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [2/4] baseline_openloop_vtzcycle ==="
python experiments/baseline_openloop_vtzcycle.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [3/4] baseline_digital_twin_vtzcycle ==="
python experiments/baseline_digital_twin_vtzcycle.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [4/4] full_harness_vtzcycle ==="
python experiments/full_harness_vtzcycle.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== All conditions complete ==="
ls -lh output/
