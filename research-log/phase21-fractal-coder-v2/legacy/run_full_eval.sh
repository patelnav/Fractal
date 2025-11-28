#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase21-fractal-coder-v2/results_v2.jsonl"
BACKUP_FILE="research-log/phase21-fractal-coder-v2/results_v2.backup.jsonl"

# Backup existing results if they exist
if [ -f "$RESULTS_FILE" ]; then
    echo "Backing up existing results to $BACKUP_FILE"
    mv "$RESULTS_FILE" "$BACKUP_FILE"
fi

echo "Starting Full Evaluation..."

# 1. Run Baseline (Standard Generation)
echo "----------------------------------------"
echo "Running Baseline Mode (164 Problems)..."
echo "----------------------------------------"
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode baseline

# 2. Run Fractal (Sketch-then-Fill)
echo "----------------------------------------"
echo "Running Fractal Mode (164 Problems)..."
echo "----------------------------------------"
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode fractal

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
