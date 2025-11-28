#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase24-fractal-critic/results_critic.jsonl"

# Clear existing results
if [ -f "$RESULTS_FILE" ]; then
    rm "$RESULTS_FILE"
fi

echo "Starting Phase 24: Fractal Critic Evaluation..."

# Run Critic Eval
python research-log/phase24-fractal-critic/critic_eval.py --representative

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
