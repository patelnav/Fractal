"""
Training script for Iterative JSON Repair.

Key differences from train_denoiser_a100.py:
1. Uses JSONIterativeDataset (multi-corruption samples, target = clean)
2. Iterative inference loop with convergence detection
3. Progress-based REINFORCE rewards
4. Optimized for local MPS/CPU training

Philosophy:
- Train model on (heavily_corrupted -> clean) pairs
- Model learns to fix as much as it can per pass
- At inference: iterate until no changes or valid JSON
- Like json-repair, but neural

Usage:
    python train_iterative.py --device mps --epochs 30 --train_samples 100000

Expected training time on M1/M2 Mac: ~2-4 hours for 30 epochs
"""

import os
import json
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer_json import JSONTokenizer
from data_json import JSONIterativeDataset, JSONEvalDataset, collate_fn
from model_denoiser import JSONDenoiser, JSONDenoiserConfig


# =============================================================================
# ITERATIVE INFERENCE
# =============================================================================

def repair_single_pass(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    device: str,
    max_len: int = 64,
) -> str:
    """
    Single forward pass of repair.

    Args:
        model: Trained denoiser model
        tokenizer: JSON tokenizer
        broken_json: Input JSON string (possibly invalid)
        device: Device to run on
        max_len: Maximum sequence length

    Returns:
        Repaired JSON string (may still be invalid)
    """
    model.eval()

    # Tokenize input
    input_ids = tokenizer.tokenize(broken_json)

    # Pad/truncate
    if len(input_ids) < max_len:
        input_ids = input_ids + [tokenizer.pad_id] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len - 1] + [tokenizer.eos_id]

    # Forward pass
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    sigma_tensor = torch.tensor([0.5], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor, sigma_tensor)
        pred_ids = logits.argmax(dim=-1)[0].tolist()

    return tokenizer.detokenize(pred_ids)


def repair_iterative(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    device: str,
    max_len: int = 64,
    max_iters: int = 10,
) -> tuple[str, int, str]:
    """
    Iteratively repair JSON until valid, converged, or max iterations.

    This is the key innovation: instead of single-pass, we iterate
    like json-repair does with its heuristics.

    Args:
        model: Trained denoiser model
        tokenizer: JSON tokenizer
        broken_json: Input JSON string
        device: Device to run on
        max_len: Maximum sequence length
        max_iters: Maximum repair iterations

    Returns:
        (repaired_json, num_iterations, exit_reason)
        exit_reason: 'valid', 'converged', 'loop_detected', 'max_iters'
    """
    current = broken_json
    history = [current]

    for i in range(max_iters):
        # Check if already valid
        try:
            json.loads(current)
            return current, i + 1, 'valid'
        except json.JSONDecodeError:
            pass

        # One repair pass
        repaired = repair_single_pass(model, tokenizer, current, device, max_len)

        # Check for convergence (no change)
        if repaired == current:
            return current, i + 1, 'converged'

        # Check for loops (seen this before)
        if repaired in history:
            return current, i + 1, 'loop_detected'

        history.append(repaired)
        current = repaired

    return current, max_iters, 'max_iters'


# =============================================================================
# PROGRESS-BASED REINFORCE
# =============================================================================

def count_json_errors(text: str) -> int:
    """
    Estimate number of JSON errors by trying to parse and counting issues.

    This is a heuristic - we count:
    - Parse failure = 1 base error
    - Unclosed brackets/braces
    - Quote issues
    etc.
    """
    errors = 0

    try:
        json.loads(text)
        return 0  # Valid JSON
    except json.JSONDecodeError:
        errors += 1

    # Count structural issues
    errors += abs(text.count('{') - text.count('}'))
    errors += abs(text.count('[') - text.count(']'))
    errors += text.count("'")  # Single quotes
    errors += text.count('```')  # Markdown fences

    # Check for Python literals
    if 'True' in text or 'False' in text or 'None' in text:
        errors += 1

    return errors


def compute_progress_reward(
    original: str,
    repaired: str,
) -> float:
    """
    Compute reward based on progress toward valid JSON.

    Rewards:
    - +1.0 if repaired parses as valid JSON
    - +0.2 for each error reduced
    - -0.5 if we made things worse

    Args:
        original: Input to model
        repaired: Model output

    Returns:
        Reward value
    """
    # Check if valid
    try:
        json.loads(repaired)
        return 1.0  # Full reward for valid JSON
    except json.JSONDecodeError:
        pass

    # Count errors
    orig_errors = count_json_errors(original)
    new_errors = count_json_errors(repaired)

    # Progress reward
    if new_errors < orig_errors:
        return 0.2 * (orig_errors - new_errors)  # Positive for improvement
    elif new_errors > orig_errors:
        return -0.5  # Penalty for making things worse
    else:
        return 0.0  # No change


_reinforce_baseline = 0.5  # Running average


def compute_reinforce_loss_progress(
    logits: torch.Tensor,
    input_texts: list[str],
    tokenizer: JSONTokenizer,
    device: str,
    temperature: float = 1.0,
    num_samples: int = 8,
) -> tuple[torch.Tensor, float, float]:
    """
    REINFORCE loss with progress-based rewards.

    Args:
        logits: Model output [batch, seq, vocab]
        input_texts: Original input strings (for progress comparison)
        tokenizer: For decoding
        device: Tensor device
        temperature: Sampling temperature
        num_samples: Number of samples to evaluate

    Returns:
        (reinforce_loss, parse_rate, avg_reward)
    """
    global _reinforce_baseline

    batch_size, seq_len, vocab_size = logits.shape

    # Sample subset
    sample_indices = torch.randperm(batch_size)[:min(num_samples, batch_size)]
    sampled_logits = logits[sample_indices]

    # Sample from distribution
    probs = F.softmax(sampled_logits / temperature, dim=-1)
    dist = torch.distributions.Categorical(probs)
    samples = dist.sample()
    log_probs = dist.log_prob(samples)

    # Compute progress-based rewards
    rewards = []
    parse_success = 0

    for i, idx in enumerate(sample_indices):
        sample_tokens = samples[i].tolist()
        decoded = tokenizer.detokenize(sample_tokens)
        original = input_texts[idx.item()]

        # Progress reward
        reward = compute_progress_reward(original, decoded)
        rewards.append(reward)

        # Track parse success
        try:
            json.loads(decoded)
            parse_success += 1
        except:
            pass

    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    parse_rate = parse_success / len(rewards)
    avg_reward = rewards.mean().item()

    # Update baseline
    _reinforce_baseline = 0.99 * _reinforce_baseline + 0.01 * avg_reward

    # REINFORCE gradient
    advantages = rewards - _reinforce_baseline
    seq_log_probs = log_probs.sum(dim=-1)
    reinforce_loss = -(advantages * seq_log_probs).mean()

    return reinforce_loss, parse_rate, avg_reward


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: JSONDenoiser,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    tokenizer: JSONTokenizer,
    reinforce_weight: float = 0.1,
    reinforce_start_epoch: int = 5,
) -> tuple[float, float, float]:
    """Train for one epoch with progress-based REINFORCE."""
    model.train()
    total_ce_loss = 0.0
    total_rl_loss = 0.0
    total_parse_rate = 0.0
    rl_batches = 0
    num_batches = 0

    use_reinforce = (epoch >= reinforce_start_epoch) and (reinforce_weight > 0)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (clean, corrupted, n_corrupt) in enumerate(pbar):
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        sigma = torch.full((clean.size(0),), 0.5, device=device)  # Fixed sigma for iterative

        optimizer.zero_grad()

        # Forward pass
        logits, ce_loss = model(corrupted, sigma, targets=clean)

        # REINFORCE (every 4th batch for speed)
        if use_reinforce and batch_idx % 4 == 0:
            # Get input texts for progress comparison
            input_texts = [tokenizer.detokenize(corrupted[i].tolist()) for i in range(corrupted.size(0))]

            rl_loss, parse_rate, avg_reward = compute_reinforce_loss_progress(
                logits, input_texts, tokenizer, device,
                temperature=1.0, num_samples=8
            )
            total_rl_loss += rl_loss.item()
            total_parse_rate += parse_rate
            rl_batches += 1
        else:
            rl_loss = torch.tensor(0.0, device=device)
            parse_rate = 0.0

        # Combined loss
        loss = ce_loss + reinforce_weight * rl_loss

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_ce_loss += ce_loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'ce': f'{ce_loss.item():.4f}',
                'rl': f'{rl_loss.item():.4f}' if use_reinforce else 'off',
                'parse': f'{parse_rate:.2f}' if parse_rate > 0 else '-',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

    avg_ce = total_ce_loss / num_batches
    avg_rl = total_rl_loss / max(rl_batches, 1) if use_reinforce else 0
    avg_parse = total_parse_rate / max(rl_batches, 1) if use_reinforce else 0

    return avg_ce, avg_rl, avg_parse


@torch.no_grad()
def evaluate_iterative(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    device: str,
    max_len: int,
    num_samples: int = 100,
    max_iters: int = 10,
) -> dict:
    """
    Evaluate with iterative repair.

    Tests parse success at different iteration counts.
    """
    model.eval()

    # Generate test samples with various corruption levels
    from data_json import generate_random_json, generate_multi_corruption_sample

    results = {
        'parse_at_1': 0,
        'parse_at_3': 0,
        'parse_at_5': 0,
        'parse_at_10': 0,
        'avg_iterations': 0,
        'exit_reasons': {'valid': 0, 'converged': 0, 'loop_detected': 0, 'max_iters': 0},
    }

    for _ in range(num_samples):
        # Generate corrupted sample
        clean_json = generate_random_json(max_depth=2)
        corrupted_json, _, n_corrupt = generate_multi_corruption_sample(
            clean_json, max_corruptions=5, tokenizer=tokenizer
        )

        # Iterative repair
        repaired, iters, reason = repair_iterative(
            model, tokenizer, corrupted_json, device, max_len, max_iters
        )

        results['avg_iterations'] += iters
        results['exit_reasons'][reason] += 1

        # Check parse success at various iteration counts
        # (We only have final result, so approximate)
        try:
            json.loads(repaired)
            results['parse_at_10'] += 1
            if iters <= 5:
                results['parse_at_5'] += 1
            if iters <= 3:
                results['parse_at_3'] += 1
            if iters <= 1:
                results['parse_at_1'] += 1
        except:
            pass

    # Normalize
    results['parse_at_1'] /= num_samples
    results['parse_at_3'] /= num_samples
    results['parse_at_5'] /= num_samples
    results['parse_at_10'] /= num_samples
    results['avg_iterations'] /= num_samples

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Iterative JSON Repair (Local)')

    # Training params
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')

    # REINFORCE params
    parser.add_argument('--reinforce_weight', type=float, default=0.1, help='REINFORCE weight')
    parser.add_argument('--reinforce_start_epoch', type=int, default=5, help='Start REINFORCE after')

    # Model params (smaller for local training)
    parser.add_argument('--n_layer', type=int, default=6, help='Transformer layers')
    parser.add_argument('--n_head', type=int, default=6, help='Attention heads')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=64, help='Max sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')

    # Data params
    parser.add_argument('--train_samples', type=int, default=100000, help='Training samples')
    parser.add_argument('--eval_samples', type=int, default=500, help='Eval samples (more = less variance)')
    parser.add_argument('--max_corruptions', type=int, default=5, help='Max corruptions per sample')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')

    # System params
    parser.add_argument('--device', type=str, default='mps', help='Device (cpu/cuda/mps)')
    parser.add_argument('--save_path', type=str, default='checkpoints_iterative', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)

    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = JSONTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Model config (smaller for local)
    config = JSONDenoiserConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        block_size=args.max_len,
    )

    model = JSONDenoiser(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}, resuming from epoch {start_epoch}")

    # Dataset
    print(f"\nCreating dataset with {args.train_samples:,} samples...")
    t0 = time.time()

    train_dataset = JSONIterativeDataset(
        num_samples=args.train_samples,
        max_len=args.max_len,
        max_corruptions=args.max_corruptions,
        seed=args.seed,
    )

    print(f"Dataset created in {time.time() - t0:.1f}s")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print(f"Batches per epoch: {len(train_loader):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Step scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Restored optimizer state")

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Save config
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump({
            'model': config.__dict__,
            'training': vars(args),
        }, f, indent=2)

    # Training loop
    best_parse_rate = 0.0
    history = []

    print(f"\n{'='*70}")
    print(f"Starting iterative training for {args.epochs} epochs...")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # Train
        ce_loss, rl_loss, train_parse = train_epoch(
            model, train_loader, optimizer, device, epoch,
            tokenizer=tokenizer,
            reinforce_weight=args.reinforce_weight,
            reinforce_start_epoch=args.reinforce_start_epoch,
        )

        scheduler.step()

        # Evaluate with iterative repair
        iter_metrics = evaluate_iterative(
            model, tokenizer, device, args.max_len,
            num_samples=args.eval_samples, max_iters=10
        )

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train CE: {ce_loss:.4f}, RL: {rl_loss:.4f}, Parse: {train_parse:.3f}")
        print(f"  Iterative eval:")
        print(f"    Parse @1: {iter_metrics['parse_at_1']:.3f}")
        print(f"    Parse @3: {iter_metrics['parse_at_3']:.3f}")
        print(f"    Parse @5: {iter_metrics['parse_at_5']:.3f}")
        print(f"    Parse @10: {iter_metrics['parse_at_10']:.3f}")
        print(f"    Avg iterations: {iter_metrics['avg_iterations']:.2f}")
        print(f"    Exit reasons: {iter_metrics['exit_reasons']}")

        history.append({
            'epoch': epoch,
            'train_ce_loss': ce_loss,
            'train_rl_loss': rl_loss,
            'train_parse': train_parse,
            **iter_metrics,
            'time': elapsed,
        })

        # Save best model
        if iter_metrics['parse_at_10'] > best_parse_rate:
            best_parse_rate = iter_metrics['parse_at_10']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'iter_metrics': iter_metrics,
            }, os.path.join(args.save_path, 'best_iterative.pt'))
            print(f"  -> Saved best model (parse@10={best_parse_rate:.3f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, os.path.join(args.save_path, f'checkpoint_epoch{epoch}.pt'))

            with open(os.path.join(args.save_path, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(args.save_path, 'final_iterative.pt'))

    with open(os.path.join(args.save_path, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best parse@10: {best_parse_rate:.3f}")
    print(f"Models saved to: {args.save_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
