#!/bin/bash

# Set up paths
RESULTS_FILE="research-log/phase23-flash-flood/results_flood.jsonl"

echo "Starting Phase 23: Flash Flood (Diversity Sampling)..."

# Run Both Modes on Representative Set
python research-log/phase23-flash-flood/flash_flood_sampling.py --representative --mode both

echo "----------------------------------------"
echo "Evaluation Complete. Results saved to $RESULTS_FILE"
echo "----------------------------------------"
