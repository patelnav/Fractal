#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase21-fractal-coder-v2/results_v2_hierarchical.jsonl"

# Clear existing results
if [ -f "$RESULTS_FILE" ]; then
    rm "$RESULTS_FILE"
fi

echo "Starting Hierarchical Evaluation (Representative Set)..."

# Run Hierarchical Mode
python research-log/phase21-fractal-coder-v2/fractal_tree_v2.5.py --representative

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
