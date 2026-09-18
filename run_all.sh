#!/bin/bash
set -e

ITER=${ITER:-20}
INTENT=${INTENT:-"5 Mbps"}

echo "Starting full experiment run: iterations=$ITER  intent='$INTENT'"
echo ""

echo "=== [1/4] baseline_rules ==="
python experiments/baseline_rules.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [2/4] baseline_openloop ==="
python experiments/baseline_openloop.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [3/4] baseline_digital_twin ==="
python experiments/baseline_digital_twin.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== [4/4] full_harness ==="
python experiments/full_harness.py --intent "$INTENT" --iterations $ITER

echo ""
echo "=== All conditions complete ==="
ls -lh output/
