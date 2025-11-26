#!/usr/bin/env python3
"""
Ouroboros Evaluation Script

Evaluates the trained energy head on held-out data.
Reports discrimination accuracy, energy distributions, and per-domain metrics.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import OuroborosModel, OuroborosConfig


def load_checkpoint(ckpt_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = OuroborosConfig(**checkpoint['config'])
    model = OuroborosModel(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_data(data_dir: str, split: str = 'val'):
    """Load evaluation data."""
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


@torch.no_grad()
def evaluate(model, data, device: str = 'cpu', batch_size: int = 32):
    """Run full evaluation."""
    n = len(data['labels'])
    all_energies = []
    all_labels = []
    all_domains = []

    for i in tqdm(range(0, n, batch_size), desc="Evaluating"):
        end_idx = min(i + batch_size, n)

        contexts = data['contexts'][i:end_idx].to(device)
        targets = data['targets'][i:end_idx].to(device)
        context_lens = data['context_lens'][i:end_idx].to(device)
        target_lens = data['target_lens'][i:end_idx].to(device)
        labels = data['labels'][i:end_idx]
        domains = data['domains'][i:end_idx]

        energy, _ = model(contexts, targets, context_lens, target_lens)

        all_energies.extend(energy.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_domains.extend(domains.tolist())

    return np.array(all_energies), np.array(all_labels), np.array(all_domains)


def compute_metrics(energies, labels, domains):
    """Compute detailed metrics."""
    results = {}

    # Overall metrics
    correct_mask = labels == 1
    wrong_mask = labels == 0

    e_correct = energies[correct_mask]
    e_wrong = energies[wrong_mask]

    results['overall'] = {
        'n_correct': int(correct_mask.sum()),
        'n_wrong': int(wrong_mask.sum()),
        'energy_correct_mean': float(e_correct.mean()),
        'energy_correct_std': float(e_correct.std()),
        'energy_wrong_mean': float(e_wrong.mean()),
        'energy_wrong_std': float(e_wrong.std()),
        'energy_separation': float(e_wrong.mean() - e_correct.mean()),
    }

    # Accuracy at different thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        predictions = (energies > threshold).astype(int)
        expected = (labels == 0).astype(int)  # wrong = 1, correct = 0
        accuracy = (predictions == expected).mean()
        results['overall'][f'accuracy_t{threshold}'] = float(accuracy)

    # Pair discrimination rate (for each correct-wrong pair, is wrong energy higher?)
    # This is the key metric - what % of wrong samples have higher energy than correct
    pair_correct = (e_wrong > e_correct.mean()).mean()
    results['overall']['pair_discrimination'] = float(pair_correct)

    # Per-domain metrics
    for domain_id, domain_name in [(0, 'math'), (1, 'code')]:
        domain_mask = domains == domain_id
        if domain_mask.sum() == 0:
            continue

        domain_energies = energies[domain_mask]
        domain_labels = labels[domain_mask]

        d_correct = domain_energies[domain_labels == 1]
        d_wrong = domain_energies[domain_labels == 0]

        if len(d_correct) == 0 or len(d_wrong) == 0:
            continue

        results[domain_name] = {
            'n_correct': int((domain_labels == 1).sum()),
            'n_wrong': int((domain_labels == 0).sum()),
            'energy_correct_mean': float(d_correct.mean()),
            'energy_correct_std': float(d_correct.std()),
            'energy_wrong_mean': float(d_wrong.mean()),
            'energy_wrong_std': float(d_wrong.std()),
            'energy_separation': float(d_wrong.mean() - d_correct.mean()),
        }

        for threshold in [0.5]:
            predictions = (domain_energies > threshold).astype(int)
            expected = (domain_labels == 0).astype(int)
            accuracy = (predictions == expected).mean()
            results[domain_name][f'accuracy_t{threshold}'] = float(accuracy)

    return results


def print_results(results, checkpoint_info=None):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("OUROBOROS EVALUATION RESULTS")
    print("=" * 60)

    if checkpoint_info:
        print(f"\nCheckpoint: iter {checkpoint_info.get('iter_num', 'N/A')}")
        print(f"Training val loss: {checkpoint_info.get('best_val_loss', 'N/A'):.4f}")

    print("\n--- OVERALL ---")
    o = results['overall']
    print(f"Samples: {o['n_correct']} correct, {o['n_wrong']} wrong")
    print(f"\nEnergy distributions:")
    print(f"  Correct: {o['energy_correct_mean']:.4f} +/- {o['energy_correct_std']:.4f}")
    print(f"  Wrong:   {o['energy_wrong_mean']:.4f} +/- {o['energy_wrong_std']:.4f}")
    print(f"  Separation: {o['energy_separation']:.4f}")

    print(f"\nAccuracy at threshold 0.5: {o['accuracy_t0.5']*100:.1f}%")
    print(f"Pair discrimination rate: {o['pair_discrimination']*100:.1f}%")

    # Success criteria check
    print("\n--- SUCCESS CRITERIA ---")
    criteria = [
        ('Energy correct < 0.2', o['energy_correct_mean'] < 0.2),
        ('Energy wrong > 0.8', o['energy_wrong_mean'] > 0.8),
        ('Accuracy > 90%', o['accuracy_t0.5'] > 0.9),
        ('Separation > 0.6', o['energy_separation'] > 0.6),
    ]
    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    # Per-domain
    for domain in ['math', 'code']:
        if domain in results:
            print(f"\n--- {domain.upper()} ---")
            d = results[domain]
            print(f"Samples: {d['n_correct']} correct, {d['n_wrong']} wrong")
            print(f"Energy correct: {d['energy_correct_mean']:.4f} +/- {d['energy_correct_std']:.4f}")
            print(f"Energy wrong: {d['energy_wrong_mean']:.4f} +/- {d['energy_wrong_std']:.4f}")
            print(f"Separation: {d['energy_separation']:.4f}")
            print(f"Accuracy: {d['accuracy_t0.5']*100:.1f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ouroboros model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ckpt.pt',
                        help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to evaluate (train/val)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda/mps)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Device: {args.device}")

    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)

    # Load data
    data = load_data(args.data_dir, args.split)
    print(f"Loaded {len(data['labels'])} samples from {args.split} split")

    # Evaluate
    energies, labels, domains = evaluate(model, data, args.device, args.batch_size)

    # Compute metrics
    results = compute_metrics(energies, labels, domains)

    # Print results
    print_results(results, checkpoint)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
