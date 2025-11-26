#!/usr/bin/env python3
"""
Ouroboros Adversarial Hardening Training Script

Fine-tunes Phase 7 checkpoint on Hard Negatives mixed with original data.
"""

import os
import sys
import time
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import OuroborosModel, OuroborosConfig
from tokenizer import OuroborosTokenizer
from utils import parse_gsm8k_answer

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# I/O
out_dir = 'checkpoints'
data_dir = '../phase7-ouroboros/data/processed'
hard_negatives_path = 'data/hard_negatives.jsonl'
phase7_checkpoint = '../phase7-ouroboros/checkpoints/ckpt.pt'

eval_interval = 50
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

# Training
batch_size = 32
max_iters = 1000
learning_rate = 1e-5  # Lower LR for fine-tuning
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 100
decay_lr = True
lr_decay_iters = 1000
min_lr = 1e-6

# Model
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
max_seq_len = 512

# System
device = 'cuda'
dtype = 'bfloat16'
compile_model = False  # Disabled due to InductorError

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
    device = 'cpu'
    print("CUDA not available, using CPU")

device_type = 'cuda' if 'cuda' in device else 'cpu'

torch.manual_seed(42 + seed_offset)
np.random.seed(42 + seed_offset)

if device_type == 'cuda':
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
else:
    ptdtype = torch.float32
    ctx = nullcontext()

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def process_hard_negatives(tokenizer, max_context_len=256, max_target_len=256):
    """Load and tokenize hard negatives."""
    if not os.path.exists(hard_negatives_path):
        print(f"Warning: Hard negatives file not found at {hard_negatives_path}")
        return None

    print(f"Processing hard negatives from {hard_negatives_path}...")
    
    contexts = []
    correct_targets = []
    wrong_targets = []
    
    with open(hard_negatives_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} hard negatives.")
    
    for line in lines:
        item = json.loads(line)
        question = item['question']
        wrong_sol = item['wrong_solution']
        gt_full = item['ground_truth']
        
        # Parse GT to get clean answer
        # But wait, Ouroboros needs FULL solution sequence as target?
        # Phase 7 `prepare_data.py` does:
        # context = "Question: {q}\nSolution: "
        # target = "{solution}"
        
        # Tokenize
        # Ensure consistency with Phase 7 format
        q_text = f"Question: {question}\nSolution: "
        q_tokens = tokenizer.encode(q_text)
        
        # For correct target, we need a clean solution text.
        # The GT in hard_negatives.jsonl is raw from GSM8K.
        # We can use it directly.
        c_tokens = tokenizer.encode(gt_full)
        
        # For wrong target
        w_tokens = tokenizer.encode(wrong_sol)
        
        if len(q_tokens) > max_context_len:
            q_tokens = q_tokens[:max_context_len]
        
        if len(c_tokens) > max_target_len:
            c_tokens = c_tokens[:max_target_len]
            
        if len(w_tokens) > max_target_len:
            w_tokens = w_tokens[:max_target_len]
            
        contexts.append(q_tokens)
        correct_targets.append(c_tokens)
        wrong_targets.append(w_tokens)
        
    # Convert to numpy arrays with padding
    n = len(contexts)
    pad_id = 50256 # EOT
    
    np_contexts = np.full((n, max_context_len), pad_id, dtype=np.uint16)
    np_correct = np.full((n, max_target_len), pad_id, dtype=np.uint16)
    np_wrong = np.full((n, max_target_len), pad_id, dtype=np.uint16)
    
    np_c_lens = np.zeros(n, dtype=np.uint16)
    np_ct_lens = np.zeros(n, dtype=np.uint16)
    np_wt_lens = np.zeros(n, dtype=np.uint16)
    np_domains = np.zeros(n, dtype=np.uint8) # 0 for math
    
    for i in range(n):
        c_len = len(contexts[i])
        ct_len = len(correct_targets[i])
        wt_len = len(wrong_targets[i])
        
        np_contexts[i, :c_len] = contexts[i]
        np_correct[i, :ct_len] = correct_targets[i]
        np_wrong[i, :wt_len] = wrong_targets[i]
        
        np_c_lens[i] = c_len
        np_ct_lens[i] = ct_len
        np_wt_lens[i] = wt_len
        
    return {
        'contexts': np_contexts,
        'correct_targets': np_correct,
        'wrong_targets': np_wrong,
        'context_lens': np_c_lens,
        'correct_target_lens': np_ct_lens,
        'wrong_target_lens': np_wt_lens,
        'domains': np_domains
    }

def load_mixed_data(split='train'):
    """Load original data and mix with hard negatives (for train split)."""
    path = Path(data_dir) / f'{split}.npz'
    print(f"Loading original data from {path}...")
    data = np.load(path)
    
    contexts = data['contexts']
    correct_targets = data['correct_targets']
    wrong_targets = data['wrong_targets']
    context_lens = data['context_lens']
    correct_target_lens = data['correct_target_lens']
    wrong_target_lens = data['wrong_target_lens']
    domains = data['domains']
    
    if split == 'train':
        tokenizer = OuroborosTokenizer()
        hn_data = process_hard_negatives(tokenizer)
        
        if hn_data:
            print(f"Mixing in {len(hn_data['contexts'])} hard negatives...")
            contexts = np.concatenate([contexts, hn_data['contexts']])
            correct_targets = np.concatenate([correct_targets, hn_data['correct_targets']])
            wrong_targets = np.concatenate([wrong_targets, hn_data['wrong_targets']])
            context_lens = np.concatenate([context_lens, hn_data['context_lens']])
            correct_target_lens = np.concatenate([correct_target_lens, hn_data['correct_target_lens']])
            wrong_target_lens = np.concatenate([wrong_target_lens, hn_data['wrong_target_lens']])
            domains = np.concatenate([domains, hn_data['domains']])
            
            print(f"Total training samples: {len(contexts)}")

    return {
        'contexts': torch.from_numpy(contexts.astype(np.int64)),
        'correct_targets': torch.from_numpy(correct_targets.astype(np.int64)),
        'wrong_targets': torch.from_numpy(wrong_targets.astype(np.int64)),
        'context_lens': torch.from_numpy(context_lens.astype(np.int64)),
        'correct_target_lens': torch.from_numpy(correct_target_lens.astype(np.int64)),
        'wrong_target_lens': torch.from_numpy(wrong_target_lens.astype(np.int64)),
        'domains': torch.from_numpy(domains.astype(np.int64)),
    }

print("Loading data...")
train_data = load_mixed_data('train')
val_data = load_mixed_data('val')

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    n = len(data['contexts'])
    idx = torch.randint(0, n, (batch_size,))

    contexts = data['contexts'][idx]
    correct_targets = data['correct_targets'][idx]
    wrong_targets = data['wrong_targets'][idx]
    context_lens = data['context_lens'][idx]
    correct_target_lens = data['correct_target_lens'][idx]
    wrong_target_lens = data['wrong_target_lens'][idx]

    if device_type == 'cuda':
        contexts = contexts.pin_memory().to(device, non_blocking=True)
        correct_targets = correct_targets.pin_memory().to(device, non_blocking=True)
        wrong_targets = wrong_targets.pin_memory().to(device, non_blocking=True)
        context_lens = context_lens.pin_memory().to(device, non_blocking=True)
        correct_target_lens = correct_target_lens.pin_memory().to(device, non_blocking=True)
        wrong_target_lens = wrong_target_lens.pin_memory().to(device, non_blocking=True)
    else:
        contexts = contexts.to(device)
        correct_targets = correct_targets.to(device)
        wrong_targets = wrong_targets.to(device)
        context_lens = context_lens.to(device)
        correct_target_lens = correct_target_lens.to(device)
        wrong_target_lens = wrong_target_lens.to(device)

    return contexts, correct_targets, wrong_targets, context_lens, correct_target_lens, wrong_target_lens

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

print("Initializing model...")
model_config = OuroborosConfig(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
    max_seq_len=max_seq_len, vocab_size=50257
)

model = OuroborosModel(model_config)

if os.path.exists(phase7_checkpoint):
    print(f"Loading Phase 7 checkpoint from {phase7_checkpoint}...")
    try:
        checkpoint = torch.load(phase7_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older torch versions
        checkpoint = torch.load(phase7_checkpoint, map_location=device)
    
    # Handle state dict mismatch if any
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove prefix if from DDP
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
else:
    print(f"WARNING: Phase 7 checkpoint not found at {phase7_checkpoint}!")

model.to(device)

if compile_model and device_type == 'cuda':
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2)
)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def compute_paired_loss(model, contexts, correct_targets, wrong_targets,
                        context_lens, correct_target_lens, wrong_target_lens):
    energy_correct, _ = model(contexts, correct_targets, context_lens, correct_target_lens)
    energy_wrong, _ = model(contexts, wrong_targets, context_lens, wrong_target_lens)

    target_correct = torch.zeros_like(energy_correct)
    target_wrong = torch.ones_like(energy_wrong)

    loss = F.mse_loss(energy_correct, target_correct) + F.mse_loss(energy_wrong, target_wrong)

    with torch.no_grad():
        detection_rate = (energy_wrong > energy_correct).float().mean().item()
        metrics = {
            'loss': loss.item(),
            'energy_correct': energy_correct.mean().item(),
            'energy_wrong': energy_wrong.mean().item(),
            'detection_rate': detection_rate,
        }
    return loss, metrics

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        metrics_list = []
        for _ in range(eval_iters):
            batch = get_batch(split)
            with ctx:
                loss, metrics = compute_paired_loss(model, *batch)
            losses.append(loss.item())
            metrics_list.append(metrics)
        out[split] = {
            'loss': np.mean(losses),
            'energy_correct': np.mean([m['energy_correct'] for m in metrics_list]),
            'energy_wrong': np.mean([m['energy_wrong'] for m in metrics_list]),
            'detection_rate': np.mean([m['detection_rate'] for m in metrics_list]),
        }
    model.train()
    return out

print(f"Starting adversarial hardening for {max_iters} iterations...")

iter_num = 0
best_val_loss = float('inf')

while iter_num < max_iters:
    # LR Schedule
    lr = learning_rate # Fixed or cosine? Plan says "fast fine-tuning". Constant small LR is fine.
    if decay_lr:
        if iter_num < warmup_iters:
            lr = learning_rate * iter_num / warmup_iters
        else:
            decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = min_lr + coeff * (learning_rate - min_lr)
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Iter {iter_num}: train loss {losses['train']['loss']:.4f}, val loss {losses['val']['loss']:.4f}, det {losses['val']['detection_rate']:.1%}")
        
        if losses['val']['loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']['loss']
            ckpt_path = os.path.join(out_dir, 'ckpt_hardened.pt')
            torch.save({
                'model': raw_model.state_dict(),
                'config': model_config.__dict__,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }, ckpt_path)

    batch = get_batch('train')
    with ctx:
        loss, metrics = compute_paired_loss(model, *batch)

    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {loss.item():.4f}, det {metrics['detection_rate']:.1%}, lr {lr:.2e}")

    iter_num += 1

if ddp:
    destroy_process_group()

print("Hardening complete.")
