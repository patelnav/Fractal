"""
Training script for JSON Denoiser.

Stage 1: Train denoiser on (corrupted_json, clean_json, sigma) triples.

Training objective:
- Given corrupted tokens and noise level sigma
- Predict clean tokens at all positions
- Cross-entropy loss (ignoring PAD tokens)
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer_json import JSONTokenizer
from data_json import JSONRepairDataset, JSONEvalDataset, collate_fn
from model_denoiser import JSONDenoiser, JSONDenoiserConfig


def train_epoch(
    model: JSONDenoiser,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for clean, corrupted, sigma in pbar:
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        sigma = sigma.to(device)

        # Forward pass: predict clean from corrupted
        logits, loss = model(corrupted, sigma, targets=clean)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: JSONDenoiser,
    dataloader: DataLoader,
    tokenizer: JSONTokenizer,
    device: str,
) -> dict:
    """
    Evaluate model on validation set.

    Metrics:
    - Loss: Cross-entropy loss
    - Token accuracy: Fraction of correctly predicted tokens
    - Parse success: Fraction of outputs that parse as valid JSON
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    parse_success = 0
    num_samples = 0

    for clean, corrupted, sigma in dataloader:
        clean = clean.to(device)
        corrupted = corrupted.to(device)
        sigma = sigma.to(device)

        logits, loss = model(corrupted, sigma, targets=clean)
        total_loss += loss.item()

        # Token accuracy (excluding PAD)
        pred = logits.argmax(dim=-1)
        mask = clean != 0  # Non-PAD
        correct = (pred == clean) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

        # Parse success
        for i in range(pred.size(0)):
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
    num_samples: int = 100,
) -> dict:
    """
    Evaluate repair quality on controlled corruption types.

    Returns per-corruption-type metrics.
    """
    model.eval()

    results = {}
    for i in range(min(num_samples, len(eval_dataset))):
        clean, corrupted, sigma, ctype = eval_dataset[i]

        clean = clean.unsqueeze(0).to(device)
        corrupted = corrupted.unsqueeze(0).to(device)
        sigma = sigma.unsqueeze(0).to(device)

        logits, _ = model(corrupted, sigma)
        pred = logits.argmax(dim=-1)

        # Metrics
        pred_json = tokenizer.detokenize(pred[0].tolist())
        clean_json = tokenizer.detokenize(clean[0].tolist())

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
    parser = argparse.ArgumentParser(description='Train JSON Denoiser')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=500, help='Number of validation samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)

    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = JSONTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Model config
    config = JSONDenoiserConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
        block_size=args.max_len,
    )

    # Model
    model = JSONDenoiser(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Datasets
    print(f"\nCreating datasets...")
    train_dataset = JSONRepairDataset(
        num_samples=args.train_samples,
        max_len=args.max_len,
        sigma_min=0.1,
        sigma_max=0.5,
        seed=args.seed,
    )
    val_dataset = JSONRepairDataset(
        num_samples=args.val_samples,
        max_len=args.max_len,
        sigma_min=0.1,
        sigma_max=0.5,
        seed=args.seed + 1000,
    )
    eval_dataset = JSONEvalDataset(
        num_samples=200,
        max_len=args.max_len,
        seed=args.seed + 2000,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, tokenizer, device)

        # Update scheduler
        scheduler.step()

        # Log
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        print(f"  Val token acc: {val_metrics['token_accuracy']:.3f}")
        print(f"  Val parse success: {val_metrics['parse_success']:.3f}")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_token_accuracy': val_metrics['token_accuracy'],
            'val_parse_success': val_metrics['parse_success'],
        })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_path, 'best_denoiser.pt'))
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic repair evaluation
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"\n  Repair evaluation by corruption type:")
            repair_metrics = evaluate_repair(model, eval_dataset, tokenizer, device)
            for ctype, metrics in repair_metrics.items():
                print(f"    {ctype}: parse={metrics['parse_success']:.2f}, acc={metrics['token_accuracy']:.2f}")

    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, os.path.join(args.save_path, 'final_denoiser.pt'))

    with open(os.path.join(args.save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
