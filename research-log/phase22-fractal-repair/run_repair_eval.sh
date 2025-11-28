#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase22-fractal-repair/results_repair.jsonl"

echo "Starting Phase 22: Fractal Repair Loop (Representative Set)..."

# Run Repair Loop
python research-log/phase22-fractal-repair/fractal_repair_loop.py --representative

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
