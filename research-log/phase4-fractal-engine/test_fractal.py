"""
Test Suite for Phase 4: Fractal Engine Hallucination Detection

Tests whether the trained model can detect hallucinations at BOTH hierarchy levels:
- Level 0: Root -> Chunks (Can it detect wrong chunk expansions?)
- Level 1: Chunk -> Chars (Can it detect wrong character sequences?)

Success criteria: >95% detection rate at BOTH levels using the same shared weights.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from hierarchical_bpe import (
    HierarchicalBPE,
    FractalDataset,
    load_fractal_dataset
)
from fractal_loader import get_level0_batch, get_level1_batch
from run_fractal_engine import (
    FractalDiffusionModel,
    FractalModelConfig,
    DiscreteDiffusion
)


@dataclass
class LevelTestResults:
    """Results for a single hierarchy level."""
    level: int
    level_name: str
    correct_energy_mean: float
    correct_energy_std: float
    wrong_energy_mean: float
    wrong_energy_std: float
    correct_below_threshold: float  # % of correct with energy < 0.5
    wrong_above_threshold: float  # % of wrong with energy > 0.5
    pair_detection_rate: float  # % where wrong > correct
    num_samples: int
    passed: bool


def test_level_detection(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    dataset: FractalDataset,
    level: int,
    device: torch.device,
    num_samples: int = 100,
    batch_size: int = 50
) -> LevelTestResults:
    """
    Test hallucination detection at a single hierarchy level.

    Uses clean targets (t=0) for testing - no noise added.
    Compares energy of correct vs wrong pairings.
    """
    model.eval()
    config = model.config

    level_name = "Root->Chunks" if level == 0 else "Chunk->Chars"
    print(f"\n{'=' * 60}")
    print(f"TESTING LEVEL {level}: {level_name}")
    print(f"{'=' * 60}")

    correct_energies = []
    wrong_energies = []

    with torch.no_grad():
        for batch_idx in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - batch_idx)

            if level == 0:
                # Level 0: Root -> Chunks
                cond, targets, lens, wrong = get_level0_batch(
                    dataset, current_batch, device
                )

                # Build correct input (no noise - t=0)
                x_correct = torch.cat([
                    (cond + config.root_offset).unsqueeze(1),
                    targets + config.chunk_offset
                ], dim=1)

                # Build wrong input
                x_wrong = torch.cat([
                    (cond + config.root_offset).unsqueeze(1),
                    wrong + config.chunk_offset
                ], dim=1)

            else:
                # Level 1: Chunk -> Chars
                cond, targets, lens, wrong = get_level1_batch(
                    dataset, current_batch, device
                )

                # Build correct input
                x_correct = torch.cat([
                    (cond + config.chunk_offset).unsqueeze(1),
                    targets
                ], dim=1)

                # Build wrong input
                x_wrong = torch.cat([
                    (cond + config.chunk_offset).unsqueeze(1),
                    wrong
                ], dim=1)

            # Get energies at t=0 (clean data)
            t = torch.zeros(current_batch, dtype=torch.long, device=device)

            _, energy_correct = model(x_correct, t, level=level, return_energy=True)
            _, energy_wrong = model(x_wrong, t, level=level, return_energy=True)

            correct_energies.extend(energy_correct.cpu().tolist())
            wrong_energies.extend(energy_wrong.cpu().tolist())

            if batch_idx % 50 == 0:
                print(f"  Processing sample {batch_idx}/{num_samples}...")

    # Compute statistics
    correct_energies = torch.tensor(correct_energies)
    wrong_energies = torch.tensor(wrong_energies)

    correct_mean = correct_energies.mean().item()
    correct_std = correct_energies.std().item()
    wrong_mean = wrong_energies.mean().item()
    wrong_std = wrong_energies.std().item()

    threshold = 0.5
    correct_below = (correct_energies < threshold).float().mean().item() * 100
    wrong_above = (wrong_energies > threshold).float().mean().item() * 100
    pair_detection = (wrong_energies > correct_energies).float().mean().item() * 100

    # Print results
    print(f"\nResults:")
    print(f"  Correct Energy: {correct_mean:.4f} ± {correct_std:.4f} (target: 0)")
    print(f"  Wrong Energy:   {wrong_mean:.4f} ± {wrong_std:.4f} (target: 1)")
    print(f"  Threshold:      {threshold}")
    print(f"\n  Correct < threshold:  {correct_below:.1f}%")
    print(f"  Wrong > threshold:    {wrong_above:.1f}%")
    print(f"  Wrong > Correct:      {pair_detection:.1f}%")

    passed = pair_detection >= 95.0
    status = "SUCCESS" if passed else "FAILURE"
    print(f"\n  {'✓' if passed else '✗'} {status}: Detection rate {'>' if passed else '<'}= 95%")

    return LevelTestResults(
        level=level,
        level_name=level_name,
        correct_energy_mean=correct_mean,
        correct_energy_std=correct_std,
        wrong_energy_mean=wrong_mean,
        wrong_energy_std=wrong_std,
        correct_below_threshold=correct_below,
        wrong_above_threshold=wrong_above,
        pair_detection_rate=pair_detection,
        num_samples=num_samples,
        passed=passed
    )


def test_cross_level_interference(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    dataset: FractalDataset,
    device: torch.device,
    num_samples: int = 50
) -> Dict:
    """
    Test that the model distinguishes between levels.

    The level embedding should prevent cross-level confusion.
    """
    model.eval()
    config = model.config

    print(f"\n{'=' * 60}")
    print(f"CROSS-LEVEL INTERFERENCE TEST")
    print(f"{'=' * 60}")

    # Get Level 0 data
    l0_cond, l0_targets, l0_lens, _ = get_level0_batch(dataset, num_samples, device)

    # Get Level 1 data
    l1_cond, l1_targets, l1_lens, _ = get_level1_batch(dataset, num_samples, device)

    with torch.no_grad():
        t = torch.zeros(num_samples, dtype=torch.long, device=device)

        # Level 0 data through Level 0 head (correct)
        x0 = torch.cat([
            (l0_cond + config.root_offset).unsqueeze(1),
            l0_targets + config.chunk_offset
        ], dim=1)
        _, e0_correct = model(x0, t, level=0, return_energy=True)

        # Level 0 data through Level 1 head (wrong level)
        _, e0_wrong_level = model(x0, t, level=1, return_energy=True)

        # Level 1 data through Level 1 head (correct)
        x1 = torch.cat([
            (l1_cond + config.chunk_offset).unsqueeze(1),
            l1_targets
        ], dim=1)
        _, e1_correct = model(x1, t, level=1, return_energy=True)

        # Level 1 data through Level 0 head (wrong level)
        _, e1_wrong_level = model(x1, t, level=0, return_energy=True)

    results = {
        "level0_correct_level_energy": e0_correct.mean().item(),
        "level0_wrong_level_energy": e0_wrong_level.mean().item(),
        "level1_correct_level_energy": e1_correct.mean().item(),
        "level1_wrong_level_energy": e1_wrong_level.mean().item()
    }

    print(f"\nLevel 0 data:")
    print(f"  Through Level 0 (correct): {results['level0_correct_level_energy']:.4f}")
    print(f"  Through Level 1 (wrong):   {results['level0_wrong_level_energy']:.4f}")

    print(f"\nLevel 1 data:")
    print(f"  Through Level 1 (correct): {results['level1_correct_level_energy']:.4f}")
    print(f"  Through Level 0 (wrong):   {results['level1_wrong_level_energy']:.4f}")

    return results


def run_all_tests(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    dataset: FractalDataset,
    device: torch.device,
    num_samples: int = 100
) -> Dict:
    """Run all tests and return combined results."""

    print("\n" + "#" * 70)
    print("FRACTAL ENGINE HALLUCINATION DETECTION TESTS (Phase 4)")
    print("#" * 70)

    # Test Level 0
    level0_results = test_level_detection(
        model, diffusion, dataset, level=0, device=device, num_samples=num_samples
    )

    # Test Level 1
    level1_results = test_level_detection(
        model, diffusion, dataset, level=1, device=device, num_samples=num_samples
    )

    # Cross-level test
    cross_level = test_cross_level_interference(
        model, diffusion, dataset, device, num_samples=num_samples // 2
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Test':<40} {'Detection Rate':<15} {'Status'}")
    print("-" * 70)
    print(f"{'Level 0 (Root -> Chunks)':<40} {level0_results.pair_detection_rate:.1f}%{'':<10} "
          f"{'✓ SUCCESS' if level0_results.passed else '✗ FAILURE'}")
    print(f"{'Level 1 (Chunk -> Chars)':<40} {level1_results.pair_detection_rate:.1f}%{'':<10} "
          f"{'✓ SUCCESS' if level1_results.passed else '✗ FAILURE'}")

    both_passed = level0_results.passed and level1_results.passed
    print(f"\n{'=' * 70}")
    if both_passed:
        print("✓ PHASE 4 SUCCESS: Both levels achieve >95% hallucination detection!")
        print("  The shared-weight fractal model successfully handles BOTH abstraction levels.")
    else:
        print("✗ PHASE 4 INCOMPLETE: One or more levels below 95% threshold.")
    print(f"{'=' * 70}")

    # Combine results
    all_results = {
        "level0": asdict(level0_results),
        "level1": asdict(level1_results),
        "cross_level": cross_level,
        "both_passed": both_passed
    }

    return all_results


if __name__ == "__main__":
    # Load dataset
    data_path = Path("data/fractal_hierarchy.pkl")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Run run_fractal_engine.py first to build the dataset.")
        exit(1)

    print("Loading dataset...")
    tokenizer, dataset, bpe_config = load_fractal_dataset(str(data_path))
    print(f"  Loaded {len(dataset.root_ids):,} root samples, {len(dataset.chunk_ids):,} chunk samples")

    # Load model
    model_path = Path("checkpoints/best_model.pt")
    if not model_path.exists():
        model_path = Path("checkpoints/final_model.pt")

    if not model_path.exists():
        print(f"Error: No model found in checkpoints/")
        print("Run run_fractal_engine.py first to train the model.")
        exit(1)

    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                         if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Recreate model config
    saved_config = checkpoint['config']
    print(f"  Model trained for {checkpoint.get('iter', 'unknown')} iterations")
    if 'det0' in checkpoint:
        print(f"  Training detection: L0={checkpoint['det0']:.1f}% L1={checkpoint['det1']:.1f}%")

    # Build model
    model = FractalDiffusionModel(saved_config).to(device)
    model.load_state_dict(checkpoint['model'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = DiscreteDiffusion(saved_config)

    # Run tests
    results = run_all_tests(model, diffusion, dataset, device, num_samples=100)

    # Save results
    results_path = Path("fractal_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
