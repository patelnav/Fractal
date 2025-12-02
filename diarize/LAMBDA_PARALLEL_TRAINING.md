# Parallel Training on 3 Lambda A100 Instances

**Goal:** Train MLP, Transformer, Contrastive in parallel (~5-6 hours each)
**Cost:** 3 × $1.29/hr × 6 hours = ~$23

**Baseline:** 4.52% DER (with overlaps, no collar) - better than published SOTA 5.2%

---

## Instance Setup (Same for All 3)

Run this on EACH Lambda A100 instance:

```bash
#!/bin/bash
set -e

# Clone and setup
cd ~
git clone https://github.com/BUTSpeechFIT/DiariZen.git
cd DiariZen
git submodule init && git submodule update

# ============================================
# DOWNLOAD DATA DIRECTLY ON LAMBDA (NOT RSYNC)
# ============================================

# VoxConverse dev audio (~4GB)
mkdir -p data/voxconverse
cd data/voxconverse
curl -L -o voxconverse_dev_wav.zip https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
unzip voxconverse_dev_wav.zip
mv voxconverse_dev_wav dev
rm voxconverse_dev_wav.zip

# VoxConverse annotations (from repo - already included)
# Annotations are at: data/voxconverse/annotations/dev/*.rttm

cd ~/DiariZen
echo "Data download complete: $(ls data/voxconverse/dev/*.wav | wc -l) audio files"

# Install deps
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
cd pyannote-audio && pip install -e . && cd ..
pip install 'numpy<2.0'
pip install pytorch-lightning tensorboard

# Fix dscore
sed -i 's/dtype=np.int)/dtype=int)/g' dscore/scorelib/metrics.py

# Fix evaluate script (remove --ignore_overlaps)
sed -i '/--ignore_overlaps/d' evaluate_voxconverse.py

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo "Setup complete!"
```

---

## Instance-Specific Commands

### Instance 1: Transformer (PRIMARY)

```bash
cd ~/DiariZen/boundary_refinement
python scripts/train.py --config configs/transformer.toml 2>&1 | tee /tmp/train_transformer.txt

# When done, evaluate
cd ~/DiariZen
python evaluate_voxconverse.py 2>&1 | tee /tmp/eval_baseline.txt
```

### Instance 2: MLP

```bash
cd ~/DiariZen/boundary_refinement
python scripts/train.py --config configs/mlp.toml 2>&1 | tee /tmp/train_mlp.txt
```

### Instance 3: Contrastive

```bash
cd ~/DiariZen/boundary_refinement
python scripts/train.py --config configs/contrastive.toml 2>&1 | tee /tmp/train_contrastive.txt
```

---

## Monitoring

### Problem: Log files are flooded with warnings
The training logs (`/tmp/train_*.log`) are 90% "Warning: Using dummy embeddings" messages (one per example). PyTorch Lightning progress bars don't appear in file redirects.

### Solution: Parse TensorBoard event files directly

```bash
# Get metrics from TensorBoard logs (run on each instance)
ssh lambda "cd ~/DiariZen && python3 -c \"
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for exp in ['transformer', 'mlp', 'contrastive']:
    exp_path = f'exp/{exp}'
    if not os.path.exists(exp_path): continue
    logs_path = os.path.join(exp_path, 'logs')
    if not os.path.exists(logs_path): continue
    for version in sorted(os.listdir(logs_path)):
        version_path = os.path.join(logs_path, version)
        for f in os.listdir(version_path):
            if f.startswith('events'):
                ea = EventAccumulator(os.path.join(version_path, f))
                ea.Reload()
                print(f'=== {exp} ===')
                for tag in sorted(ea.Tags().get('scalars', [])):
                    vals = ea.Scalars(tag)
                    if vals:
                        print(f'  {tag}: {vals[-1].value:.6f} (epoch {len(vals)-1})')
\""
```

### Quick status check
```bash
# Check checkpoints (shows completed epochs)
for host in lambda lambda-mlp lambda-contrastive; do
  echo "=== $host ==="
  ssh $host "ls ~/DiariZen/exp/*/checkpoints/ 2>/dev/null"
done

# Check GPU utilization
for host in lambda lambda-mlp lambda-contrastive; do
  echo "=== $host ==="
  ssh $host "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
done
```

### Watch raw logs (mostly warnings)
```bash
ssh ubuntu@<IP> "tail -f /tmp/train_*.log | grep -v 'dummy embeddings'"
```

---

## After Training

### 1. Get Results
```bash
# From each instance
scp ubuntu@<IP>:/tmp/train_*.txt ./results/
scp ubuntu@<IP>:~/DiariZen/boundary_refinement/checkpoints/*.pt ./checkpoints/
```

### 2. Compare Models
```bash
# Best checkpoint from each
ls -la checkpoints/
```

### 3. Terminate Instances
Only after downloading all checkpoints and logs!

---

## Expected Results

| Model | Expected MAE | Training Time |
|-------|--------------|---------------|
| Transformer | <50ms | ~5 hours |
| MLP | ~100-200ms | ~3 hours |
| Contrastive | ~75-150ms | ~4 hours |

**Baseline:** DiariZen = 923ms boundary error
**Target:** <100ms (10x improvement)

---

## Cost Tracking

| Instance | Model | Hours | Cost |
|----------|-------|-------|------|
| 1 | Transformer | ~6 | ~$7.75 |
| 2 | MLP | ~4 | ~$5.15 |
| 3 | Contrastive | ~5 | ~$6.45 |
| **Total** | | ~15 GPU-hrs | **~$19.35** |

Well under $1000 budget ($999.28 remaining).
