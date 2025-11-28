"""
Stage 2: Train Energy Head on frozen denoiser trunk.

Energy head learns to distinguish:
- Clean sequences → low energy (label = 0)
- Corrupted sequences → high energy (label = 1)

Training: Binary cross-entropy on (clean, corrupted) pairs.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

from model import UniversalDenoiser, UniversalDenoiserWithEnergy, Config
from data import UniversalDenoiserDataset
from mutations import corrupt_sequence


class EnergyDataset(Dataset):
    """
    Dataset for energy head training.

    Returns (sequence, label) pairs where:
    - label=0: clean sequence
    - label=1: corrupted sequence
    """

    def __init__(
        self,
        base_dataset: UniversalDenoiserDataset,
        corrupt_ratio: float = 0.5,
        sigma_min: float = 0.2,
        sigma_max: float = 0.6,
    ):
        self.base = base_dataset
        self.corrupt_ratio = corrupt_ratio
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self):
        return len(self.base) * 2  # Each sample can be clean or corrupted

    def __getitem__(self, idx):
        # Get base sample
        base_idx = idx % len(self.base)
        clean, _, _ = self.base[base_idx]

        # Decide if this sample should be corrupted
        is_corrupted = random.random() < self.corrupt_ratio

        if is_corrupted:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            sequence = corrupt_sequence(
                clean,
                sigma=sigma,
                vocab_size=len(self.base.vocab),
                special_tokens=self.base.special_tokens,
                mask_token_id=self.base.mask_id,
                pad_token_id=self.base.pad_id,
                max_len=self.base.max_len,
            )
            label = 1.0  # High energy for corrupted
        else:
            sequence = clean
            label = 0.0  # Low energy for clean

        return sequence, torch.tensor(label, dtype=torch.float32)


def train_energy(args):
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load pretrained denoiser
    print(f"Loading pretrained denoiser from {args.denoiser_checkpoint}...")
    checkpoint = torch.load(args.denoiser_checkpoint, map_location=device)

    config = Config()
    for k, v in checkpoint['config'].items():
        setattr(config, k, v)

    denoiser = UniversalDenoiser(config)
    denoiser.load_state_dict(checkpoint['model'])
    print(f"Loaded denoiser (iter {checkpoint.get('iter_num', '?')}, loss {checkpoint.get('loss', '?'):.4f})")

    # Create model with energy head
    model = UniversalDenoiserWithEnergy(config, pretrained_denoiser=denoiser).to(device)

    # Freeze denoiser trunk
    model.freeze_denoiser()
    print("Denoiser trunk frozen. Training energy head only.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Dataset
    print("Creating dataset...")
    base_ds = UniversalDenoiserDataset(
        num_samples=args.num_samples,
        max_depth=6,
        max_len=args.max_len,
        corruption_mode='mask_only',
    )
    energy_ds = EnergyDataset(
        base_ds,
        corrupt_ratio=0.5,
        sigma_min=0.2,
        sigma_max=0.6,
    )

    dataloader = DataLoader(
        energy_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer (only energy head parameters)
    optimizer = torch.optim.AdamW(
        model.energy_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    model.train()
    # But keep denoiser in eval mode
    model.denoiser.eval()

    iter_num = 0
    losses = []
    accuracies = []

    print(f"\nStarting energy head training for {args.max_iters} iterations...")

    pbar = tqdm(total=args.max_iters, desc="Training Energy")

    while iter_num < args.max_iters:
        for sequences, labels in dataloader:
            if iter_num >= args.max_iters:
                break

            sequences = sequences.to(device)
            labels = labels.to(device)
            B = sequences.size(0)

            # Forward pass
            # Use sigma=0.3 as a moderate noise level for the hidden state computation
            sigma = torch.full((B,), 0.3, device=device)

            with torch.no_grad():
                # Get hidden states from frozen denoiser
                hidden = model.denoiser.get_hidden(sequences, sigma, K_iter=2)

            # Compute energy (this is trainable)
            mask = sequences != base_ds.pad_id
            energy = model.energy_head(hidden, mask)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(energy, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            preds = (torch.sigmoid(energy) > 0.5).float()
            acc = (preds == labels).float().mean().item()

            losses.append(loss.item())
            accuracies.append(acc)
            iter_num += 1
            pbar.update(1)

            if iter_num % args.log_interval == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                avg_acc = sum(accuracies[-100:]) / min(100, len(accuracies))
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.1%}',
                })

    pbar.close()

    # Final stats
    final_loss = sum(losses[-100:]) / min(100, len(losses))
    final_acc = sum(accuracies[-100:]) / min(100, len(accuracies))
    print(f"\nFinal loss: {final_loss:.4f}, accuracy: {final_acc:.1%}")

    # Save
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    save_dict = {
        'model': model.state_dict(),
        'energy_head': model.energy_head.state_dict(),
        'config': config.__dict__,
        'iter_num': iter_num,
        'loss': final_loss,
        'accuracy': final_acc,
    }
    torch.save(save_dict, os.path.join(args.checkpoint_dir, 'energy_model.pt'))
    print(f"Saved to {args.checkpoint_dir}/energy_model.pt")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Energy Head')

    # Model
    parser.add_argument('--denoiser_checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to pretrained denoiser')

    # Data
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--max_len', type=int, default=64)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Logging
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # Device
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()
    train_energy(args)


if __name__ == "__main__":
    main()
