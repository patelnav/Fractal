"""
Training script for JSON Denoiser - A100 Scale with REINFORCE.

Optimized for large-scale training on A100 GPUs:
- Mixed precision (bf16/fp16)
- Gradient accumulation
- Larger model (8 layers, 512 dim)
- 500K+ samples, 100 epochs
- REINFORCE loss for direct parse success optimization

Usage:
    python train_denoiser_a100.py --device cuda --epochs 100 --train_samples 500000

Expected training time on A100: ~2-4 hours
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from tokenizer_json import JSONTokenizer
from data_json import JSONRepairDataset, JSONEvalDataset, collate_fn
from model_denoiser import JSONDenoiser, JSONDenoiserConfig


# Global tokenizer for REINFORCE (initialized in main)
_tokenizer = None
_reinforce_baseline = 0.5  # Running average of rewards


def compute_reinforce_loss(
    logits: torch.Tensor,
    tokenizer,
    device: str,
    temperature: float = 1.0,
    num_samples: int = 16,  # Number of samples per batch to check (for speed)
) -> tuple[torch.Tensor, float]:
    """
    Compute REINFORCE loss based on parse success.

    Args:
        logits: Model output logits [batch, seq, vocab]
        tokenizer: JSON tokenizer for decoding
        device: Device for tensors
        temperature: Sampling temperature
        num_samples: How many samples from batch to check (for speed)

    Returns:
        reinforce_loss: Scalar loss tensor
        parse_rate: Fraction that parsed successfully
    """
    global _reinforce_baseline

    batch_size, seq_len, vocab_size = logits.shape

    # Only sample a subset for speed
    sample_indices = torch.randperm(batch_size)[:min(num_samples, batch_size)]
    sampled_logits = logits[sample_indices]  # [num_samples, seq, vocab]

    # Sample from distribution (with temperature)
    probs = F.softmax(sampled_logits / temperature, dim=-1)
    dist = torch.distributions.Categorical(probs)
    samples = dist.sample()  # [num_samples, seq]
    log_probs = dist.log_prob(samples)  # [num_samples, seq]

    # Compute rewards (parse success)
    rewards = []
    for i in range(samples.size(0)):
        sample_tokens = samples[i].tolist()
        decoded = tokenizer.detokenize(sample_tokens)
        try:
            json.loads(decoded)
            rewards.append(1.0)
        except json.JSONDecodeError:
            rewards.append(0.0)

    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    parse_rate = rewards.mean().item()

    # Update baseline with exponential moving average
    _reinforce_baseline = 0.99 * _reinforce_baseline + 0.01 * parse_rate

    # REINFORCE gradient: -E[(reward - baseline) * log_prob]
    # We want to maximize reward, so we minimize negative
    advantages = rewards - _reinforce_baseline  # [num_samples]

    # Sum log probs over sequence (product of token probs)
    seq_log_probs = log_probs.sum(dim=-1)  # [num_samples]

    # REINFORCE loss
    reinforce_loss = -(advantages * seq_log_probs).mean()

    return reinforce_loss, parse_rate


def train_epoch(
    model: JSONDenoiser,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    tokenizer,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    reinforce_weight: float = 0.1,
    reinforce_start_epoch: int = 5,  # Start REINFORCE after warmup
) -> tuple[float, float, float]:
    """Train for one epoch with mixed precision, gradient accumulation, and REINFORCE."""
    model.train()
    total_ce_loss = 0.0
    total_rl_loss = 0.0
    total_parse_rate = 0.0
    num_batches = 0

    use_reinforce = (epoch >= reinforce_start_epoch) and (reinforce_weight > 0)

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (clean, corrupted, sigma) in enumerate(pbar):
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        sigma = sigma.to(device)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            logits, ce_loss = model(corrupted, sigma, targets=clean)

            # REINFORCE loss (only on subset for speed)
            if use_reinforce and batch_idx % 4 == 0:  # Every 4th batch
                rl_loss, parse_rate = compute_reinforce_loss(
                    logits, tokenizer, device,
                    temperature=1.0, num_samples=16
                )
                total_rl_loss += rl_loss.item()
                total_parse_rate += parse_rate
            else:
                rl_loss = torch.tensor(0.0, device=device)
                parse_rate = 0.0

            # Combined loss
            loss = ce_loss + reinforce_weight * rl_loss
            loss = loss / accumulation_steps

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_ce_loss += ce_loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'ce': f'{ce_loss.item():.4f}',
                'rl': f'{rl_loss.item():.4f}' if use_reinforce else 'off',
                'parse': f'{parse_rate:.2f}' if parse_rate > 0 else '-',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

    avg_ce = total_ce_loss / num_batches
    avg_rl = total_rl_loss / max(num_batches // 4, 1) if use_reinforce else 0
    avg_parse = total_parse_rate / max(num_batches // 4, 1) if use_reinforce else 0

    return avg_ce, avg_rl, avg_parse


@torch.no_grad()
def evaluate(
    model: JSONDenoiser,
    dataloader: DataLoader,
    tokenizer: JSONTokenizer,
    device: str,
    use_amp: bool = True,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    parse_success = 0
    num_samples = 0

    for clean, corrupted, sigma in tqdm(dataloader, desc="Evaluating"):
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        sigma = sigma.to(device)

        with autocast(enabled=use_amp):
            logits, loss = model(corrupted, sigma, targets=clean)
        total_loss += loss.item()

        # Token accuracy (excluding PAD)
        pred = logits.argmax(dim=-1)
        mask = clean != 0  # Non-PAD
        correct = (pred == clean) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

        # Parse success (sample subset for speed)
        if num_samples < 500:
            for i in range(min(pred.size(0), 500 - num_samples)):
                pred_json = tokenizer.detokenize(pred[i].tolist())
                try:
                    json.loads(pred_json)
                    parse_success += 1
                except json.JSONDecodeError:
                    pass
                num_samples += 1

    return {
        'loss': total_loss / len(dataloader),
        'token_accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'parse_success': parse_success / num_samples if num_samples > 0 else 0,
    }


@torch.no_grad()
def evaluate_repair(
    model: JSONDenoiser,
    eval_dataset: JSONEvalDataset,
    tokenizer: JSONTokenizer,
    device: str,
    num_samples: int = 200,
    use_amp: bool = True,
) -> dict:
    """Evaluate repair quality on controlled corruption types."""
    model.eval()

    results = {}
    for i in range(min(num_samples, len(eval_dataset))):
        clean, corrupted, sigma, ctype = eval_dataset[i]

        clean = clean.unsqueeze(0).to(device)
        corrupted = corrupted.unsqueeze(0).to(device)
        sigma = sigma.unsqueeze(0).to(device)

        with autocast(enabled=use_amp):
            logits, _ = model(corrupted, sigma)
        pred = logits.argmax(dim=-1)

        # Metrics
        pred_json = tokenizer.detokenize(pred[0].tolist())

        try:
            json.loads(pred_json)
            parse_ok = 1
        except json.JSONDecodeError:
            parse_ok = 0

        # Token accuracy
        mask = clean[0] != 0
        token_acc = ((pred[0] == clean[0]) & mask).float().mean().item()

        if ctype not in results:
            results[ctype] = {'parse_success': [], 'token_accuracy': []}
        results[ctype]['parse_success'].append(parse_ok)
        results[ctype]['token_accuracy'].append(token_acc)

    # Aggregate
    summary = {}
    for ctype, metrics in results.items():
        summary[ctype] = {
            'parse_success': sum(metrics['parse_success']) / len(metrics['parse_success']),
            'token_accuracy': sum(metrics['token_accuracy']) / len(metrics['token_accuracy']),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Train JSON Denoiser (A100 Scale with REINFORCE)')

    # Training params
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')

    # REINFORCE params
    parser.add_argument('--reinforce_weight', type=float, default=0.1, help='Weight for REINFORCE loss')
    parser.add_argument('--reinforce_start_epoch', type=int, default=5, help='Epoch to start REINFORCE')

    # Model params
    parser.add_argument('--n_layer', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Data params
    parser.add_argument('--train_samples', type=int, default=500000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of validation samples')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # System params
    parser.add_argument('--device', type=str, default='cuda', help='Device (cpu/cuda/mps)')
    parser.add_argument('--save_path', type=str, default='checkpoints_a100', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    use_amp = (device == 'cuda') and not args.no_amp
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_amp}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer
    tokenizer = JSONTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Model config
    config = JSONDenoiserConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        block_size=args.max_len,
    )

    # Model
    model = JSONDenoiser(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optional: torch.compile for faster training
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    best_parse_success = 0.0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_parse_success = checkpoint.get('best_parse_success', 0.0)

    # Datasets
    print(f"\nCreating datasets...")
    print(f"This may take a few minutes for {args.train_samples:,} samples...")

    t0 = time.time()
    train_dataset = JSONRepairDataset(
        num_samples=args.train_samples,
        max_len=args.max_len,
        sigma_min=0.1,
        sigma_max=0.5,
        seed=args.seed,
    )
    print(f"Train dataset created in {time.time() - t0:.1f}s")

    val_dataset = JSONRepairDataset(
        num_samples=args.val_samples,
        max_len=args.max_len,
        sigma_min=0.1,
        sigma_max=0.5,
        seed=args.seed + 1000,
    )

    eval_dataset = JSONEvalDataset(
        num_samples=500,
        max_len=args.max_len,
        seed=args.seed + 2000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Batches per epoch: {len(train_loader):,}")

    effective_batch = args.batch_size * args.accumulation_steps
    print(f"Effective batch size: {effective_batch}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler(enabled=use_amp)

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Save config
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump({
            'model': config.__dict__,
            'training': vars(args),
        }, f, indent=2)

    # Training loop
    history = []

    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # Train with REINFORCE
        ce_loss, rl_loss, train_parse = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            tokenizer=tokenizer,
            accumulation_steps=args.accumulation_steps,
            use_amp=use_amp,
            reinforce_weight=args.reinforce_weight,
            reinforce_start_epoch=args.reinforce_start_epoch,
        )

        # Validate every epoch
        val_metrics = evaluate(model, val_loader, tokenizer, device, use_amp=use_amp)

        # Update scheduler
        scheduler.step()

        # Log
        elapsed = time.time() - start_time
        samples_per_sec = len(train_dataset) / elapsed

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s, {samples_per_sec:.0f} samples/sec)")
        print(f"  Train CE loss: {ce_loss:.4f}, RL loss: {rl_loss:.4f}, Train parse: {train_parse:.3f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        print(f"  Val token acc: {val_metrics['token_accuracy']:.3f}")
        print(f"  Val parse success: {val_metrics['parse_success']:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_ce_loss': ce_loss,
            'train_rl_loss': rl_loss,
            'train_parse_rate': train_parse,
            'val_loss': val_metrics['loss'],
            'val_token_accuracy': val_metrics['token_accuracy'],
            'val_parse_success': val_metrics['parse_success'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed,
        })

        # Save best model (prioritize parse success, not loss!)
        current_parse_success = val_metrics['parse_success']
        if current_parse_success > best_parse_success:
            best_parse_success = current_parse_success
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'best_parse_success': best_parse_success,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.save_path, 'best_denoiser.pt'))
            print(f"  ✓ Saved best model (parse_success={best_parse_success:.3f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.save_path, f'checkpoint_epoch{epoch}.pt'))

            # Save history
            with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        # Detailed repair evaluation every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"\n  Repair evaluation by corruption type:")
            repair_metrics = evaluate_repair(
                model, eval_dataset, tokenizer, device,
                num_samples=500, use_amp=use_amp
            )
            for ctype, metrics in sorted(repair_metrics.items()):
                print(f"    {ctype}: parse={metrics['parse_success']:.2f}, acc={metrics['token_accuracy']:.2f}")

    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'best_val_loss': best_val_loss,
    }, os.path.join(args.save_path, 'final_denoiser.pt'))

    with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best parse success: {best_parse_success:.3f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.save_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
