"""
Training script for Two-Stage Generation.

Stage 1: Skeleton Generator - learns structure (parens, ops, =)
Stage 2: Digit Filler - fills in digits given skeleton
"""

import os
import random
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UniversalDenoiser, Config
from data_twostage import SkeletonDataset, DigitFillerDataset, collate_fn


def train(args):
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Dataset
    print(f"Creating {args.stage} dataset...")
    if args.stage == 'skeleton':
        train_ds = SkeletonDataset(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            max_len=args.max_len,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
    else:  # 'filler'
        train_ds = DigitFillerDataset(
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            max_len=args.max_len,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Dataset size: {len(train_ds)}")
    print(f"Vocab size: {len(train_ds.vocab)}")
    print(f"Vocab: {train_ds.vocab}")

    # Model
    config = Config()
    config.vocab_size = len(train_ds.vocab)
    config.block_size = args.max_len
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.n_embd = args.n_embd
    config.dropout = args.dropout

    model = UniversalDenoiser(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_iters,
        eta_min=args.lr / 10,
    )

    # Training loop
    model.train()
    iter_num = 0
    best_loss = float('inf')
    losses = []

    print(f"\nStarting {args.stage} training for {args.max_iters} iterations...")
    pbar = tqdm(total=args.max_iters, desc=f"Training ({args.stage})")

    while iter_num < args.max_iters:
        for clean, corrupted, sigma in train_dl:
            if iter_num >= args.max_iters:
                break

            clean = clean.to(device)
            corrupted = corrupted.to(device)
            sigma = sigma.to(device)

            # Targets: predict clean, ignore PAD
            targets = clean.clone()
            targets[clean == train_ds.pad_id] = -1

            # Forward
            logits, loss = model(corrupted, sigma, targets, K_iter=args.k_iter)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            iter_num += 1
            pbar.update(1)

            if iter_num % args.log_interval == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

            if iter_num % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample_clean = clean[:4]
                    sample_corrupted = corrupted[:4]
                    sample_sigma = sigma[:4]

                    logits, _ = model(sample_corrupted, sample_sigma, K_iter=args.k_iter)
                    pred = logits.argmax(dim=-1)

                    mask = sample_clean != train_ds.pad_id
                    correct = (pred == sample_clean) & mask
                    acc = correct.sum().item() / mask.sum().item()

                    print(f"\n[Iter {iter_num}] Eval accuracy: {acc:.2%}")
                    print(f"  Clean:     {train_ds.decode(sample_clean[0])}")
                    print(f"  Input:     {train_ds.decode(sample_corrupted[0])}")
                    print(f"  Predicted: {train_ds.decode(pred[0])}")

                model.train()

            if iter_num % args.save_interval == 0:
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)

                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter_num': iter_num,
                    'config': config.__dict__,
                    'args': vars(args),
                    'loss': avg_loss,
                    'vocab': train_ds.vocab,
                    'stoi': train_ds.stoi,
                }

                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pt'))

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pt'))
                    print(f"\n[Iter {iter_num}] New best loss: {best_loss:.4f}")

    pbar.close()

    # Final save
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    final_checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'config': config.__dict__,
        'args': vars(args),
        'loss': sum(losses[-100:]) / min(100, len(losses)),
        'vocab': train_ds.vocab,
        'stoi': train_ds.stoi,
    }
    torch.save(final_checkpoint, os.path.join(args.checkpoint_dir, 'final.pt'))
    print(f"\nTraining complete. Checkpoint saved to {args.checkpoint_dir}/")

    return model, train_ds


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stage Models')

    # Stage selection
    parser.add_argument('--stage', type=str, required=True,
                        choices=['skeleton', 'filler'],
                        help='Which stage to train: skeleton or filler')

    # Data
    parser.add_argument('--num_samples', type=int, default=20000)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--sigma_min', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=0.9)

    # Model
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--k_iter', type=int, default=2)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Logging
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_twostage')

    # Device
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Set checkpoint dir based on stage
    if args.checkpoint_dir == 'checkpoints_twostage':
        args.checkpoint_dir = f'checkpoints_{args.stage}'

    train(args)


if __name__ == "__main__":
    main()
