#!/bin/bash
# Stage 2: Hybrid-Aware Retraining
# Run on cloud GPU (CUDA)
#
# Key change: 50% of samples use AR-prefix corruption
# - Normal corruption: σ ∈ [0.1, 0.9] with mixed corruption
# - AR-prefix: First 3 tokens clean, rest masked (simulates hybrid generation)

set -e

# Navigate to phase31 directory
cd "$(dirname "$0")"

# Activate virtual environment (adjust path as needed)
# source /path/to/venv/bin/activate

echo "=========================================="
echo "Stage 2: Hybrid-Aware Retraining"
echo "=========================================="
echo ""
echo "Config:"
echo "  - 50% AR-prefix corruption (ar_prefix_ratio=0.5)"
echo "  - 3 tokens AR prefix (ar_prefix_len=3)"
echo "  - 3000 iterations"
echo "  - 20K samples"
echo "  - Output: checkpoints_hybrid/"
echo ""

# Run training
python train_universal.py \
    --num_samples 20000 \
    --max_iters 3000 \
    --batch_size 64 \
    --lr 3e-4 \
    --sigma_min 0.1 \
    --sigma_max 0.9 \
    --ar_prefix_ratio 0.5 \
    --ar_prefix_len 3 \
    --checkpoint_dir checkpoints_hybrid \
    --device auto \
    2>&1 | tee /tmp/phase31_hybrid_train.txt

echo ""
echo "Training complete!"
echo "Output log: /tmp/phase31_hybrid_train.txt"
echo "Checkpoint: checkpoints_hybrid/best.pt"
echo ""
echo "Next: Run benchmark with hybrid generation:"
echo "  python benchmark.py --checkpoint checkpoints_hybrid/best.pt --use_hybrid --num_ar_tokens 3 --num_samples 200"
