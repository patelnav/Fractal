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

From local machine:
```bash
# Watch logs (replace IP)
ssh ubuntu@<IP1> "tail -f /tmp/train_transformer.txt"
ssh ubuntu@<IP2> "tail -f /tmp/train_mlp.txt"
ssh ubuntu@<IP3> "tail -f /tmp/train_contrastive.txt"
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
