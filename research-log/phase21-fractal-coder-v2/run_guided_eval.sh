#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase21-fractal-coder-v2/results_v3_guided.jsonl"

# Clear existing results
if [ -f "$RESULTS_FILE" ]; then
    rm "$RESULTS_FILE"
fi

echo "Starting Guided Evaluation (Representative Set)..."

# Run Guided Mode
python research-log/phase21-fractal-coder-v2/fractal_guided_v3.py --representative

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
