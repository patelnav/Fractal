"""
Training script for Energy Head.

Stage 2: Train energy head on (valid_json, invalid_json) pairs.

The energy head learns to distinguish:
- Positives (low energy): Valid JSON, successful repairs
- Negatives (high energy): Corrupted JSON, failed repairs

Training objective: Binary classification with BCE loss.
"""

import os
import json
import time
import argparse
import random
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer_json import JSONTokenizer
from data_json import JSONRepairDataset, generate_random_json, JSONCorruptionEngine
from model_denoiser import JSONDenoiser, JSONDenoiserConfig, JSONDenoiserWithEnergy


class EnergyDataset(Dataset):
    """
    Dataset for energy head training.

    Generates (tokens, label) pairs where:
    - label=0: Valid JSON (low energy target)
    - label=1: Invalid JSON (high energy target)
    """

    def __init__(
        self,
        num_samples: int,
        max_len: int = 128,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.max_len = max_len
        self.tokenizer = JSONTokenizer()
        self.corruption_engine = JSONCorruptionEngine(self.tokenizer)

        self.samples = []

        # Generate half valid, half invalid
        samples_per_class = num_samples // 2

        # Valid JSON (label=0)
        for _ in range(samples_per_class):
            json_str = generate_random_json(max_depth=2)
            ids = self.tokenizer.tokenize(json_str)
            ids = self._pad_or_truncate(ids)
            self.samples.append((ids, 0))

        # Invalid JSON (label=1)
        for _ in range(samples_per_class):
            # Start with valid JSON
            json_str = generate_random_json(max_depth=2)
            ids = self.tokenizer.tokenize(json_str)

            # Corrupt it
            sigma = random.uniform(0.2, 0.5)
            corrupted_ids, _ = self.corruption_engine.corrupt(ids, sigma)
            corrupted_ids = self._pad_or_truncate(corrupted_ids)

            # Verify it's actually invalid
            corrupted_json = self.tokenizer.detokenize(corrupted_ids)
            try:
                json.loads(corrupted_json)
                # Still valid - corrupt more
                corrupted_ids, _ = self.corruption_engine.corrupt(corrupted_ids, 0.3)
                corrupted_ids = self._pad_or_truncate(corrupted_ids)
            except json.JSONDecodeError:
                pass

            self.samples.append((corrupted_ids, 1))

        random.shuffle(self.samples)

        if seed is not None:
            random.seed()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

    def _pad_or_truncate(self, ids):
        if len(ids) < self.max_len:
            return ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        else:
            return ids[:self.max_len - 1] + [self.tokenizer.eos_id]


class RepairResultDataset(Dataset):
    """
    Dataset using denoiser repair attempts.

    - Successful repairs (parses OK): label=0
    - Failed repairs (doesn't parse): label=1

    This teaches the energy head to distinguish good repairs from bad.
    """

    def __init__(
        self,
        model: JSONDenoiser,
        num_samples: int,
        max_len: int = 128,
        device: str = 'cpu',
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.max_len = max_len
        self.tokenizer = JSONTokenizer()
        self.corruption_engine = JSONCorruptionEngine(self.tokenizer)
        self.samples = []

        model.eval()

        print(f"Generating repair-based training data...")
        for _ in tqdm(range(num_samples)):
            # Generate and corrupt
            json_str = generate_random_json(max_depth=2)
            clean_ids = self.tokenizer.tokenize(json_str)

            sigma = random.uniform(0.2, 0.5)
            corrupted_ids, _ = self.corruption_engine.corrupt(clean_ids, sigma)
            corrupted_ids = self._pad_or_truncate(corrupted_ids)

            # Run denoiser
            with torch.no_grad():
                input_tensor = torch.tensor([corrupted_ids], dtype=torch.long, device=device)
                sigma_tensor = torch.tensor([sigma], device=device)

                logits, _ = model(input_tensor, sigma_tensor)
                pred_ids = logits.argmax(dim=-1)[0].tolist()

            # Check if valid
            pred_json = self.tokenizer.detokenize(pred_ids)
            try:
                json.loads(pred_json)
                label = 0  # Valid = low energy
            except json.JSONDecodeError:
                label = 1  # Invalid = high energy

            self.samples.append((pred_ids, label))

        # Balance if needed
        positives = [s for s in self.samples if s[1] == 0]
        negatives = [s for s in self.samples if s[1] == 1]
        print(f"Valid repairs: {len(positives)}, Invalid repairs: {len(negatives)}")

        if seed is not None:
            random.seed()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

    def _pad_or_truncate(self, ids):
        if len(ids) < self.max_len:
            return ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        else:
            return ids[:self.max_len - 1] + [self.tokenizer.eos_id]


def train_epoch(
    model: JSONDenoiserWithEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> float:
    """Train energy head for one epoch."""
    model.train()
    model.freeze_denoiser()  # Only train energy head

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for ids, labels in pbar:
        ids = ids.to(device)
        labels = labels.to(device)

        # Get energy scores
        sigma = torch.zeros(ids.size(0), device=device)
        energy = model.compute_energy(ids, sigma)

        # BCE loss
        loss = criterion(energy, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.energy_head.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        pred = (energy > 0).float()
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(
    model: JSONDenoiserWithEnergy,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate energy head."""
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_energies = []

    for ids, labels in dataloader:
        ids = ids.to(device)
        labels = labels.to(device)

        sigma = torch.zeros(ids.size(0), device=device)
        energy = model.compute_energy(ids, sigma)

        loss = criterion(energy, labels)
        total_loss += loss.item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend((energy > 0).float().cpu().tolist())
        all_energies.extend(energy.cpu().tolist())

    # Metrics
    accuracy = sum(1 for l, p in zip(all_labels, all_preds) if l == p) / len(all_labels)

    # AUC approximation (simple)
    valid_energies = [e for e, l in zip(all_energies, all_labels) if l == 0]
    invalid_energies = [e for e, l in zip(all_energies, all_labels) if l == 1]

    if valid_energies and invalid_energies:
        # Valid should have lower energy
        valid_mean = sum(valid_energies) / len(valid_energies)
        invalid_mean = sum(invalid_energies) / len(invalid_energies)
        separation = invalid_mean - valid_mean
    else:
        separation = 0.0

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'valid_energy_mean': sum(valid_energies) / len(valid_energies) if valid_energies else 0,
        'invalid_energy_mean': sum(invalid_energies) / len(invalid_energies) if invalid_energies else 0,
        'separation': separation,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Energy Head')
    parser.add_argument('--denoiser_path', type=str, required=True, help='Path to trained denoiser')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--val_samples', type=int, default=500, help='Validation samples')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Save path')
    parser.add_argument('--use_repair_data', action='store_true', help='Use repair-based data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    # Load denoiser
    print(f"Loading denoiser from {args.denoiser_path}...")
    checkpoint = torch.load(args.denoiser_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    denoiser = JSONDenoiser(config).to(device)
    denoiser.load_state_dict(checkpoint['model_state_dict'])

    # Create model with energy head
    model = JSONDenoiserWithEnergy(config, pretrained_denoiser=denoiser).to(device)
    model.freeze_denoiser()

    energy_params = sum(p.numel() for p in model.energy_head.parameters())
    print(f"Energy head parameters: {energy_params:,}")

    # Datasets
    print(f"\nCreating datasets...")
    if args.use_repair_data:
        train_dataset = RepairResultDataset(
            model=denoiser,
            num_samples=args.train_samples,
            max_len=args.max_len,
            device=device,
            seed=args.seed,
        )
        val_dataset = RepairResultDataset(
            model=denoiser,
            num_samples=args.val_samples,
            max_len=args.max_len,
            device=device,
            seed=args.seed + 1000,
        )
    else:
        train_dataset = EnergyDataset(
            num_samples=args.train_samples,
            max_len=args.max_len,
            seed=args.seed,
        )
        val_dataset = EnergyDataset(
            num_samples=args.val_samples,
            max_len=args.max_len,
            seed=args.seed + 1000,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Optimizer (only for energy head)
    optimizer = torch.optim.AdamW(model.energy_head.parameters(), lr=args.lr)

    # Training
    best_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Training energy head for {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.3f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.3f}")
        print(f"  Valid energy: {val_metrics['valid_energy_mean']:.3f}, "
              f"Invalid energy: {val_metrics['invalid_energy_mean']:.3f}, "
              f"Separation: {val_metrics['separation']:.3f}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_path, 'best_energy.pt'))
            print(f"  Saved best model (acc={best_acc:.3f})")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(args.save_path, 'final_energy.pt'))

    print(f"\n{'='*60}")
    print(f"Training complete! Best accuracy: {best_acc:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
