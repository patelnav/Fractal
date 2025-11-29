#!/bin/bash
# A100 Training Script for Phase 32 JSON Repair Engine
#
# Usage:
#   ./run_a100.sh launch    # Provision A100 and start training
#   ./run_a100.sh train     # Just run training (if instance already up)
#   ./run_a100.sh status    # Check training progress
#   ./run_a100.sh pull      # Pull results back to local
#   ./run_a100.sh terminate # Shut down instance (ASKS FOR CONFIRMATION)

set -e

PHASE_DIR="research-log/phase32-json-planner-refiner"
REMOTE_ROOT="~/Fractal"
VENV_PATH="research-log/phase11-fractal-logic/.venv"
LOG_FILE="/tmp/phase32_train.log"

cd "$(dirname "$0")/../.."  # Go to fractal root

case "$1" in
    launch)
        echo "=== Launching A100 instance ==="
        python lambda_helper.py launch

        echo "=== Waiting for SSH ==="
        python lambda_helper.py wait

        echo "=== Setting up SSH config ==="
        python lambda_helper.py setup-ssh

        echo "=== Syncing Phase 32 code ==="
        python lambda_helper.py sync $PHASE_DIR

        echo "=== Installing dependencies ==="
        ssh lambda "source ~/.local/bin/env && \
            source $REMOTE_ROOT/$VENV_PATH/bin/activate && \
            uv pip install torch tqdm"

        echo ""
        echo "=== Ready! Run './run_a100.sh train' to start training ==="
        ;;

    train)
        echo "=== Starting A100 training ==="
        echo "Logging to: $LOG_FILE"
        echo "Monitor with: tail -f $LOG_FILE"
        echo ""

        ssh lambda "source ~/.local/bin/env && \
            cd $REMOTE_ROOT && \
            source $VENV_PATH/bin/activate && \
            python $PHASE_DIR/train_denoiser_a100.py \
                --device cuda \
                --epochs 100 \
                --batch_size 256 \
                --train_samples 500000 \
                --val_samples 5000 \
                --n_layer 8 \
                --n_head 8 \
                --n_embd 512 \
                --max_len 256 \
                --lr 1e-3 \
                --save_path $PHASE_DIR/checkpoints_a100 \
                --compile" 2>&1 | tee $LOG_FILE
        ;;

    train-small)
        # Quick test run with smaller config
        echo "=== Starting A100 training (SMALL TEST) ==="
        echo "Logging to: $LOG_FILE"

        ssh lambda "source ~/.local/bin/env && \
            cd $REMOTE_ROOT && \
            source $VENV_PATH/bin/activate && \
            python $PHASE_DIR/train_denoiser_a100.py \
                --device cuda \
                --epochs 20 \
                --batch_size 256 \
                --train_samples 50000 \
                --val_samples 1000 \
                --n_layer 6 \
                --n_head 8 \
                --n_embd 384 \
                --max_len 256 \
                --lr 1e-3 \
                --save_path $PHASE_DIR/checkpoints_a100" 2>&1 | tee $LOG_FILE
        ;;

    status)
        echo "=== Checking training status ==="
        ssh lambda "tail -50 $REMOTE_ROOT/$PHASE_DIR/checkpoints_a100/training_history.json 2>/dev/null || echo 'No training history yet'"
        echo ""
        ssh lambda "ls -la $REMOTE_ROOT/$PHASE_DIR/checkpoints_a100/ 2>/dev/null || echo 'No checkpoints yet'"
        ;;

    pull)
        echo "=== Pulling results from A100 ==="
        mkdir -p checkpoints_a100
        scp -r lambda:$REMOTE_ROOT/$PHASE_DIR/checkpoints_a100/* $PHASE_DIR/checkpoints_a100/
        echo "Results saved to $PHASE_DIR/checkpoints_a100/"
        ;;

    benchmark)
        echo "=== Running benchmark on A100 ==="
        ssh lambda "source ~/.local/bin/env && \
            cd $REMOTE_ROOT && \
            source $VENV_PATH/bin/activate && \
            python $PHASE_DIR/benchmark.py \
                --model $PHASE_DIR/checkpoints_a100/best_denoiser.pt \
                --device cuda \
                --num_samples 500" 2>&1 | tee /tmp/phase32_benchmark.log
        ;;

    terminate)
        echo ""
        echo "WARNING: This will terminate the A100 instance!"
        echo "Make sure you've run './run_a100.sh pull' first!"
        echo ""
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python lambda_helper.py terminate
        else
            echo "Cancelled."
        fi
        ;;

    *)
        echo "Usage: $0 {launch|train|train-small|status|pull|benchmark|terminate}"
        echo ""
        echo "Commands:"
        echo "  launch      - Provision A100 and sync code"
        echo "  train       - Start full training (500K samples, 100 epochs)"
        echo "  train-small - Quick test (50K samples, 20 epochs)"
        echo "  status      - Check training progress"
        echo "  pull        - Download checkpoints to local"
        echo "  benchmark   - Run benchmark with trained model"
        echo "  terminate   - Shut down A100 (asks confirmation)"
        exit 1
        ;;
esac
