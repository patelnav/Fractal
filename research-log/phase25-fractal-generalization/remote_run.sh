#!/bin/bash
# Install dependencies
pip install torch tqdm matplotlib

# Run the benchmark
python research-log/phase25-fractal-generalization/train_compare.py
