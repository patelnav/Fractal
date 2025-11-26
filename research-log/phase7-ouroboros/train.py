"""
Ouroboros Training Script

nanoGPT-style training loop for energy-based reasoning verification.
Supports single GPU (CUDA/MPS) and distributed training.

Usage:
    # Single GPU
    python train.py config/train_m2.py

    # A100
    python train.py config/train_a100.py

    # Override params
    python train.py config/train_m2.py --batch_size=32 --max_iters=1000
"""

import os
import sys
import time
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import OuroborosModel, OuroborosConfig, compute_contrastive_loss


# -----------------------------------------------------------------------------
# Default config values
# -----------------------------------------------------------------------------

# I/O
out_dir = 'checkpoints'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'ouroboros'
wandb_run_name = 'run'

# data
data_dir = 'data/processed'
batch_size = 32

# model
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
max_seq_len = 512

# optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5

# system
device = 'cuda'
dtype = 'bfloat16'
compile_model = True

# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# Load config file if provided
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if config_file.endswith('.py'):
        print(f"Loading config from {config_file}")
        with open(config_file) as f:
            exec(f.read())

# Override with command line args
for arg in sys.argv[2:]:
    if '=' in arg:
        key, val = arg.lstrip('-').split('=', 1)
        if key in config_keys:
            # Parse value
            if val.lower() in ('true', 'false'):
                val = val.lower() == 'true'
            elif val.replace('.', '').replace('-', '').isdigit():
                val = float(val) if '.' in val else int(val)
            globals()[key] = val
            print(f"Override: {key}={val}")

config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Device setup
if device == 'cuda' and not torch.cuda.is_available():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"CUDA not available, using {device}")

device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')

# Set seeds
torch.manual_seed(42 + seed_offset)
np.random.seed(42 + seed_offset)

# Mixed precision setup
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
elif device_type == 'mps':
    ptdtype = torch.float32  # MPS doesn't support bfloat16
    ctx = nullcontext()
else:
    ptdtype = torch.float32
    ctx = nullcontext()

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Config: {config}")
    print(f"Device: {device}, dtype: {ptdtype}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(split='train'):
    """Load contrastive data from .npz file."""
    path = Path(data_dir) / f'{split}.npz'
    data = np.load(path)
    return {
        'contexts': torch.from_numpy(data['contexts'].astype(np.int64)),
        'targets': torch.from_numpy(data['targets'].astype(np.int64)),
        'context_lens': torch.from_numpy(data['context_lens'].astype(np.int64)),
        'target_lens': torch.from_numpy(data['target_lens'].astype(np.int64)),
        'labels': torch.from_numpy(data['labels'].astype(np.int64)),
        'domains': torch.from_numpy(data['domains'].astype(np.int64)),
    }


# Load data
print(f"Loading data from {data_dir}...")
train_data = load_data('train')
val_data = load_data('val')
print(f"Train: {len(train_data['labels'])} samples, Val: {len(val_data['labels'])} samples")


def get_batch(split='train'):
    """Get a random batch of contrastive samples."""
    data = train_data if split == 'train' else val_data
    n = len(data['labels'])

    # Random indices
    idx = torch.randint(0, n, (batch_size,))

    contexts = data['contexts'][idx]
    targets = data['targets'][idx]
    context_lens = data['context_lens'][idx]
    target_lens = data['target_lens'][idx]
    labels = data['labels'][idx]

    # Move to device
    if device_type == 'cuda':
        contexts = contexts.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
        context_lens = context_lens.pin_memory().to(device, non_blocking=True)
        target_lens = target_lens.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        contexts = contexts.to(device)
        targets = targets.to(device)
        context_lens = context_lens.to(device)
        target_lens = target_lens.to(device)
        labels = labels.to(device)

    return contexts, targets, context_lens, target_lens, labels


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------

print("Initializing model...")
model_config = OuroborosConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    max_seq_len=max_seq_len,
    vocab_size=50257,  # tiktoken GPT-2
)

iter_num = 0
best_val_loss = float('inf')

if init_from == 'scratch':
    model = OuroborosModel(model_config)
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = OuroborosConfig(**checkpoint['config'])
    model = OuroborosModel(model_config)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model = model.to(device)

# Compile model (PyTorch 2.0)
if compile_model and device_type == 'cuda':
    print("Compiling model...")
    model = torch.compile(model)

# DDP wrapper
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2)
)

if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])

# GradScaler for fp16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))


def get_lr(it):
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """Estimate loss and metrics on train/val splits."""
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = []
        metrics_list = []

        for _ in range(eval_iters):
            contexts, targets, context_lens, target_lens, labels = get_batch(split)

            with ctx:
                loss, metrics = compute_contrastive_loss(
                    model, contexts, targets, labels,
                    context_lens, target_lens
                )

            losses.append(loss.item())
            metrics_list.append(metrics)

        out[split] = {
            'loss': np.mean(losses),
            'energy_correct': np.mean([m['energy_correct'] for m in metrics_list]),
            'energy_wrong': np.mean([m['energy_wrong'] for m in metrics_list]),
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
        }

    model.train()
    return out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

print(f"\nStarting training from iter {iter_num}...")
print(f"Max iters: {max_iters}, batch size: {batch_size}")

# Initialize wandb
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

t0 = time.time()
running_loss = 0.0

while iter_num < max_iters:
    # Learning rate schedule
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"\nIter {iter_num}: train loss {losses['train']['loss']:.4f}, "
              f"val loss {losses['val']['loss']:.4f}")
        print(f"  Train: E_correct={losses['train']['energy_correct']:.4f}, "
              f"E_wrong={losses['train']['energy_wrong']:.4f}, "
              f"acc={losses['train']['accuracy']:.3f}")
        print(f"  Val:   E_correct={losses['val']['energy_correct']:.4f}, "
              f"E_wrong={losses['val']['energy_wrong']:.4f}, "
              f"acc={losses['val']['accuracy']:.3f}")

        if wandb_log:
            wandb.log({
                'iter': iter_num,
                'train/loss': losses['train']['loss'],
                'val/loss': losses['val']['loss'],
                'train/energy_correct': losses['train']['energy_correct'],
                'train/energy_wrong': losses['train']['energy_wrong'],
                'train/accuracy': losses['train']['accuracy'],
                'val/energy_correct': losses['val']['energy_correct'],
                'val/energy_wrong': losses['val']['energy_wrong'],
                'val/accuracy': losses['val']['accuracy'],
                'lr': lr,
            })

        # Save best checkpoint
        if losses['val']['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']['loss']
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'n_layer': model_config.n_layer,
                    'n_head': model_config.n_head,
                    'n_embd': model_config.n_embd,
                    'dropout': model_config.dropout,
                    'max_seq_len': model_config.max_seq_len,
                    'vocab_size': model_config.vocab_size,
                },
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'metrics': losses,
            }
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

        if eval_only:
            break

    # Forward + backward
    contexts, targets, context_lens, target_lens, labels = get_batch('train')

    with ctx:
        loss, metrics = compute_contrastive_loss(
            model, contexts, targets, labels,
            context_lens, target_lens
        )

    scaler.scale(loss).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    running_loss += loss.item()
    if iter_num % log_interval == 0 and master_process:
        dt = time.time() - t0
        avg_loss = running_loss / log_interval
        print(f"iter {iter_num}: loss {avg_loss:.4f}, "
              f"E_c={metrics['energy_correct']:.3f}, E_w={metrics['energy_wrong']:.3f}, "
              f"acc={metrics['accuracy']:.2f}, lr={lr:.2e}, time={dt*1000:.0f}ms")
        running_loss = 0.0
        t0 = time.time()

    iter_num += 1

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

if ddp:
    destroy_process_group()

print("\nTraining complete!")
print(f"Best val loss: {best_val_loss:.4f}")
