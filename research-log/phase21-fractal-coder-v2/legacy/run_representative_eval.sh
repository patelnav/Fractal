#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase21-fractal-coder-v2/results_v2_representative.jsonl"

# Clear existing results
if [ -f "$RESULTS_FILE" ]; then
    rm "$RESULTS_FILE"
fi

echo "Starting Representative Structural Evaluation (15 Problems)..."

# 1. Run Baseline
echo "----------------------------------------"
echo "Running Baseline Mode..."
echo "----------------------------------------"
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode baseline --representative

# 2. Run Fractal
echo "----------------------------------------"
echo "Running Fractal Mode..."
echo "----------------------------------------"
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode fractal --representative

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
