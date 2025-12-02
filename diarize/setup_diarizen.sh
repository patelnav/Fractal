#!/bin/bash
set -e

echo "=== DiariZen Setup Script ==="
echo "Started at: $(date)"

# Clone and setup
cd ~
if [ -d "DiariZen" ]; then
    echo "DiariZen already exists, pulling latest..."
    cd DiariZen && git pull
else
    echo "Cloning DiariZen..."
    git clone https://github.com/BUTSpeechFIT/DiariZen.git
    cd DiariZen
fi

echo "Initializing submodules..."
git submodule init && git submodule update

# Download VoxConverse data
echo "=== Downloading VoxConverse data ==="
mkdir -p data/voxconverse
cd data/voxconverse

if [ -d "dev" ] && [ "$(ls dev/*.wav 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "VoxConverse data already downloaded: $(ls dev/*.wav | wc -l) files"
else
    echo "Downloading VoxConverse dev audio (~4GB)..."
    curl -L -o voxconverse_dev_wav.zip https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
    unzip -o voxconverse_dev_wav.zip
    mv voxconverse_dev_wav dev 2>/dev/null || true
    rm -f voxconverse_dev_wav.zip
fi

cd ~/DiariZen
echo "Data status: $(ls data/voxconverse/dev/*.wav 2>/dev/null | wc -l) audio files"

# Install dependencies
echo "=== Installing dependencies ==="
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
cd pyannote-audio && pip install -e . && cd ..
pip install 'numpy<2.0'
pip install pytorch-lightning tensorboard

# Apply fixes
echo "=== Applying fixes ==="
sed -i 's/dtype=np.int)/dtype=int)/g' dscore/scorelib/metrics.py 2>/dev/null || true
sed -i '/--ignore_overlaps/d' evaluate_voxconverse.py 2>/dev/null || true

# Verify
echo "=== Verification ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo "Setup complete at: $(date)"
