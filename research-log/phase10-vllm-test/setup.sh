#!/bin/bash
# Phase 10 vLLM Test Setup
# Run this on Lambda A100 instance to set up vLLM with Gemma-3-1B
#
# IMPORTANT: Run this in a SEPARATE terminal from the current eval run!

set -e

echo "=============================================="
echo "Phase 10: vLLM + Gemma-3-1B Setup"
echo "=============================================="

# Navigate to this directory
cd "$(dirname "$0")"

# Check CUDA version
echo ""
echo "Checking CUDA version..."
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv
nvcc --version 2>/dev/null || echo "nvcc not in PATH (ok if using container)"

# Create fresh virtual environment with uv (recommended by vLLM docs)
echo ""
echo "Creating virtual environment..."

# Check if uv is available, otherwise use python venv
if command -v uv &> /dev/null; then
    echo "Using uv for environment setup..."
    uv venv --python 3.12 --seed .venv
    source .venv/bin/activate
    uv pip install vllm
else
    echo "uv not found, using standard venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install vllm
fi

# Install additional dependencies
pip install huggingface_hub

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the test:"
echo "  python test_vllm_gemma.py"
echo "=============================================="
