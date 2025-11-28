#!/bin/bash
# Stage 3: Two-Stage Coarse→Fine Training
# Run on cloud GPU (CUDA)
#
# Step 1: Train skeleton generator (~15 min)
# Step 2: Train digit filler (~15 min)
# Step 3: Benchmark two-stage generation

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "Stage 3: Two-Stage Coarse→Fine Training"
echo "=========================================="

# Step 1: Train Skeleton Generator
echo ""
echo "Step 1/3: Training Skeleton Generator"
echo "--------------------------------------"
echo "Skeleton vocab: <PAD>, <MASK>, <BOS>, <EOS>, <DIGIT>, (, ), +, *, ="
echo "Output: checkpoints_skeleton/"
echo ""

python train_twostage.py \
    --stage skeleton \
    --num_samples 20000 \
    --max_iters 1500 \
    --batch_size 64 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints_skeleton \
    --device auto \
    2>&1 | tee /tmp/phase31_skeleton_train.txt

echo ""
echo "Skeleton training complete!"
echo ""

# Step 2: Train Digit Filler
echo "Step 2/3: Training Digit Filler"
echo "--------------------------------"
echo "Full vocab with digits"
echo "Output: checkpoints_filler/"
echo ""

python train_twostage.py \
    --stage filler \
    --num_samples 20000 \
    --max_iters 1500 \
    --batch_size 64 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints_filler \
    --device auto \
    2>&1 | tee /tmp/phase31_filler_train.txt

echo ""
echo "Filler training complete!"
echo ""

# Step 3: Benchmark
echo "Step 3/3: Benchmarking Two-Stage Generation"
echo "--------------------------------------------"

python benchmark_twostage.py \
    --skeleton_checkpoint checkpoints_skeleton/best.pt \
    --filler_checkpoint checkpoints_filler/best.pt \
    --num_samples 200 \
    --device auto \
    2>&1 | tee /tmp/phase31_twostage_benchmark.txt

echo ""
echo "=========================================="
echo "Stage 3 Complete!"
echo "=========================================="
echo ""
echo "Logs:"
echo "  Skeleton: /tmp/phase31_skeleton_train.txt"
echo "  Filler:   /tmp/phase31_filler_train.txt"
echo "  Benchmark: /tmp/phase31_twostage_benchmark.txt"
echo ""
echo "Checkpoints:"
echo "  checkpoints_skeleton/best.pt"
echo "  checkpoints_filler/best.pt"
