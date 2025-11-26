"""
Hallucination Detection Test for BPE Decompression (Phase 3)

Three approaches to detect hallucinations:

1. ENERGY HEAD (Trained Discriminator) - NEW in Phase 3:
   - Use the trained energy_head to directly predict if pairing is valid
   - Trained contrastively: correct pairs -> 0, wrong pairs -> 1
   - Threshold at 0.5: if energy > 0.5, reject as hallucination

2. ENERGY METRIC (Chen's Lemma 7) - Legacy from Phase 2.5:
   - Compute E[∫ ||score||² dt] for correct vs wrong decompression
   - Hypothesis: Wrong decompressions have higher energy

3. CROSS-TOKEN ENERGY - Legacy from Phase 2.5:
   - Token A -> Chars of Token A (correct) vs Token A -> Chars of Token B (wrong)
   - Tests if model learned specific token-to-char mappings
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from bpe_tokenizer import load_decompression_dataset, MinimalBPE, DecompressionDataset, BPEConfig
from run_bpe_diffusion import (
    DecompressionDiffusion,
    DiscreteDiffusion,
    ModelConfig,
    compute_generation_energy,
    generate_decompression
)


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class TestConfig:
    num_samples: int = 200  # Number of test samples
    num_generations: int = 5  # Generations per sample for diversity
    energy_integration_steps: int = 50
    generation_steps: int = 50  # Reduced timesteps for faster generation
    temperature: float = 1.0
    energy_threshold: float = 0.5  # Threshold for energy head detection


# ============================================================================
# Energy Head Detection (Phase 3 - Trained Discriminator)
# ============================================================================

def test_energy_head_detection(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    tokenizer: MinimalBPE,
    dataset: DecompressionDataset,
    config: TestConfig,
    device: torch.device
) -> dict:
    """
    Test hallucination detection using the trained energy head.

    This is the Phase 3 approach: use the energy head directly to predict
    whether a (BPE token, char sequence) pairing is valid.

    For each sample:
    1. Compute energy_head output for CORRECT pairing (should be ~0)
    2. Compute energy_head output for WRONG pairing (should be ~1)
    3. Check if wrong_energy > threshold AND correct_energy < threshold
    """
    print("\n" + "=" * 60)
    print("ENERGY HEAD DETECTION (Phase 3 - Trained Discriminator)")
    print("=" * 60)

    model_config = model.config
    bpe_offset = model_config.char_vocab_size + 1

    correct_energies = []
    wrong_energies = []
    correct_detections = []  # Should be detected as VALID (energy < 0.5)
    wrong_detections = []    # Should be detected as INVALID (energy > 0.5)
    pair_detections = []     # wrong_energy > correct_energy

    # Sample random indices
    indices = torch.randperm(len(dataset))[:config.num_samples]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i % 50 == 0:
                print(f"Processing sample {i}/{config.num_samples}...")

            idx = idx.item()
            bpe_id = dataset.bpe_token_ids[idx].item()
            correct_chars = dataset.char_sequences[idx].unsqueeze(0).to(device)
            seq_len = dataset.sequence_lengths[idx].item()

            # Sample a different random index for wrong chars
            wrong_idx = torch.randint(len(dataset), (1,)).item()
            while wrong_idx == idx:
                wrong_idx = torch.randint(len(dataset), (1,)).item()
            wrong_chars = dataset.char_sequences[wrong_idx].unsqueeze(0).to(device)

            # Use t=0 (clean input) for energy evaluation
            t = torch.zeros(1, dtype=torch.long, device=device)

            # --- CORRECT PAIRING ---
            x_correct = torch.cat([
                torch.tensor([[bpe_id + bpe_offset]], device=device),
                correct_chars
            ], dim=1)
            _, energy_correct = model(x_correct, t, return_energy=True)
            energy_correct = energy_correct.item()

            # --- WRONG PAIRING ---
            x_wrong = torch.cat([
                torch.tensor([[bpe_id + bpe_offset]], device=device),
                wrong_chars
            ], dim=1)
            _, energy_wrong = model(x_wrong, t, return_energy=True)
            energy_wrong = energy_wrong.item()

            correct_energies.append(energy_correct)
            wrong_energies.append(energy_wrong)

            # Detection tests
            correct_detections.append(1 if energy_correct < config.energy_threshold else 0)
            wrong_detections.append(1 if energy_wrong > config.energy_threshold else 0)
            pair_detections.append(1 if energy_wrong > energy_correct else 0)

    # Statistics
    correct_mean = np.mean(correct_energies)
    correct_std = np.std(correct_energies)
    wrong_mean = np.mean(wrong_energies)
    wrong_std = np.std(wrong_energies)

    correct_accuracy = np.mean(correct_detections) * 100  # % of correct pairs with energy < 0.5
    wrong_accuracy = np.mean(wrong_detections) * 100      # % of wrong pairs with energy > 0.5
    pair_accuracy = np.mean(pair_detections) * 100        # % where wrong_energy > correct_energy

    # Combined detection: both must be correct
    combined_detections = [c * w for c, w in zip(correct_detections, wrong_detections)]
    combined_accuracy = np.mean(combined_detections) * 100

    results = {
        'correct_energy_mean': correct_mean,
        'correct_energy_std': correct_std,
        'wrong_energy_mean': wrong_mean,
        'wrong_energy_std': wrong_std,
        'correct_accuracy': correct_accuracy,
        'wrong_accuracy': wrong_accuracy,
        'pair_accuracy': pair_accuracy,
        'combined_accuracy': combined_accuracy,
        'threshold': config.energy_threshold,
        'num_samples': config.num_samples
    }

    print(f"\nResults:")
    print(f"  Correct Energy: {correct_mean:.4f} ± {correct_std:.4f} (target: 0)")
    print(f"  Wrong Energy:   {wrong_mean:.4f} ± {wrong_std:.4f} (target: 1)")
    print(f"  Threshold:      {config.energy_threshold}")
    print(f"\n  Correct < threshold:  {correct_accuracy:.1f}%")
    print(f"  Wrong > threshold:    {wrong_accuracy:.1f}%")
    print(f"  Wrong > Correct:      {pair_accuracy:.1f}%")
    print(f"  Combined (both OK):   {combined_accuracy:.1f}%")

    if pair_accuracy >= 95:
        print("\n  ✓ SUCCESS: Energy head achieves >95% detection rate!")
    elif pair_accuracy >= 80:
        print("\n  ~ GOOD: Energy head works well (>80%)")
    elif pair_accuracy >= 60:
        print("\n  ~ PARTIAL: Some discrimination, needs more training")
    else:
        print("\n  ✗ FAILURE: Energy head not discriminating effectively")

    return results


# ============================================================================
# Energy-Based Hallucination Detection (Legacy Phase 2.5)
# ============================================================================

def test_energy_detection(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    tokenizer: MinimalBPE,
    dataset: DecompressionDataset,
    config: TestConfig,
    device: torch.device
) -> dict:
    """
    Test hallucination detection via energy comparison.

    For each sample:
    1. Compute energy of CORRECT decompression
    2. Compute energy of WRONG (random) decompression
    3. Check if wrong_energy > correct_energy
    """
    print("\n" + "=" * 60)
    print("ENERGY-BASED HALLUCINATION DETECTION")
    print("=" * 60)

    correct_energies = []
    wrong_energies = []
    energy_gaps = []
    detections = []

    # Sample random indices
    indices = torch.randperm(len(dataset))[:config.num_samples]

    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"Processing sample {i}/{config.num_samples}...")

        bpe_id = dataset.bpe_token_ids[idx].item()
        correct_chars = dataset.char_sequences[idx]
        seq_len = dataset.sequence_lengths[idx].item()

        # Energy of correct decompression
        correct_energy = compute_generation_energy(
            model, diffusion, bpe_id,
            correct_chars.unsqueeze(0).to(device),
            seq_len, device,
            config.energy_integration_steps
        )

        # Generate WRONG decompression: random characters of same length
        wrong_chars = torch.randint(0, dataset.num_chars, (1, correct_chars.shape[0]))

        wrong_energy = compute_generation_energy(
            model, diffusion, bpe_id,
            wrong_chars.to(device),
            seq_len, device,
            config.energy_integration_steps
        )

        correct_energies.append(correct_energy)
        wrong_energies.append(wrong_energy)
        energy_gaps.append(wrong_energy - correct_energy)
        detections.append(1 if wrong_energy > correct_energy else 0)

    # Statistics
    correct_mean = np.mean(correct_energies)
    correct_std = np.std(correct_energies)
    wrong_mean = np.mean(wrong_energies)
    wrong_std = np.std(wrong_energies)
    gap_mean = np.mean(energy_gaps)
    gap_std = np.std(energy_gaps)
    detection_rate = np.mean(detections) * 100

    results = {
        'correct_energy_mean': correct_mean,
        'correct_energy_std': correct_std,
        'wrong_energy_mean': wrong_mean,
        'wrong_energy_std': wrong_std,
        'energy_gap_mean': gap_mean,
        'energy_gap_std': gap_std,
        'detection_rate': detection_rate,
        'num_samples': config.num_samples
    }

    print(f"\nResults:")
    print(f"  Correct Energy: {correct_mean:.2f} ± {correct_std:.2f}")
    print(f"  Wrong Energy:   {wrong_mean:.2f} ± {wrong_std:.2f}")
    print(f"  Energy Gap:     {gap_mean:.2f} ± {gap_std:.2f}")
    print(f"  Detection Rate: {detection_rate:.1f}%")

    if detection_rate > 70:
        print("\n  ✓ SUCCESS: Energy metric discriminates hallucinations")
    elif detection_rate > 55:
        print("\n  ~ PARTIAL: Some discrimination, but noisy")
    else:
        print("\n  ✗ FAILURE: Detection rate near random chance")

    return results


# ============================================================================
# Diversity-Based Hallucination Detection
# ============================================================================

def levenshtein_distance(s1: List[int], s2: List[int]) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_diversity(generations: List[List[int]]) -> float:
    """
    Compute average pairwise Levenshtein distance.
    Normalized by average sequence length.
    """
    if len(generations) < 2:
        return 0.0

    distances = []
    for i in range(len(generations)):
        for j in range(i + 1, len(generations)):
            dist = levenshtein_distance(generations[i], generations[j])
            # Normalize by average length
            avg_len = (len(generations[i]) + len(generations[j])) / 2
            if avg_len > 0:
                distances.append(dist / avg_len)

    return np.mean(distances) if distances else 0.0


def test_diversity_detection(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    tokenizer: MinimalBPE,
    dataset: DecompressionDataset,
    config: TestConfig,
    device: torch.device
) -> dict:
    """
    Test hallucination detection via generation diversity.

    For each token:
    1. Generate N decompressions
    2. Compute pairwise Levenshtein distance
    3. Compare diversity of REAL tokens vs FAKE tokens
    """
    print("\n" + "=" * 60)
    print("DIVERSITY-BASED HALLUCINATION DETECTION")
    print("=" * 60)

    real_diversities = []
    fake_diversities = []
    detections = []

    # Sample unique BPE tokens
    unique_bpe_ids = torch.unique(dataset.bpe_token_ids)
    sample_ids = unique_bpe_ids[torch.randperm(len(unique_bpe_ids))[:config.num_samples]]

    for i, bpe_id in enumerate(sample_ids):
        if i % 20 == 0:
            print(f"Processing token {i}/{config.num_samples}...")

        bpe_id = bpe_id.item()

        # Get expected length for this token
        token_chars = tokenizer.get_token_chars(bpe_id)
        seq_len = len(token_chars)

        # Generate N decompressions for REAL token
        real_generations = []
        for _ in range(config.num_generations):
            gen = generate_decompression(
                model, diffusion, bpe_id, seq_len,
                device, config.temperature, config.generation_steps
            )
            real_generations.append(gen.tolist())

        real_div = compute_diversity(real_generations)
        real_diversities.append(real_div)

        # Generate N decompressions for FAKE token (out-of-distribution)
        # Use a random BPE ID that's likely out of training distribution
        # or use a completely random "token" that was never trained
        fake_bpe_id = dataset.num_bpe_tokens - 1  # Last token (rare)

        fake_generations = []
        for _ in range(config.num_generations):
            gen = generate_decompression(
                model, diffusion, fake_bpe_id, seq_len,
                device, config.temperature, config.generation_steps
            )
            fake_generations.append(gen.tolist())

        fake_div = compute_diversity(fake_generations)
        fake_diversities.append(fake_div)

        # Detection: fake should have higher diversity
        detections.append(1 if fake_div > real_div else 0)

    # Statistics
    real_mean = np.mean(real_diversities)
    real_std = np.std(real_diversities)
    fake_mean = np.mean(fake_diversities)
    fake_std = np.std(fake_diversities)
    detection_rate = np.mean(detections) * 100

    results = {
        'real_diversity_mean': real_mean,
        'real_diversity_std': real_std,
        'fake_diversity_mean': fake_mean,
        'fake_diversity_std': fake_std,
        'detection_rate': detection_rate,
        'num_samples': config.num_samples,
        'num_generations': config.num_generations
    }

    print(f"\nResults:")
    print(f"  Real Token Diversity: {real_mean:.4f} ± {real_std:.4f}")
    print(f"  Fake Token Diversity: {fake_mean:.4f} ± {fake_std:.4f}")
    print(f"  Detection Rate:       {detection_rate:.1f}%")

    if detection_rate > 70:
        print("\n  ✓ SUCCESS: Diversity metric discriminates hallucinations")
    elif detection_rate > 55:
        print("\n  ~ PARTIAL: Some discrimination, but noisy")
    else:
        print("\n  ✗ FAILURE: Detection rate near random chance")

    return results


# ============================================================================
# Cross-Token Test (Mismatched BPE -> Chars)
# ============================================================================

def test_cross_token_energy(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    tokenizer: MinimalBPE,
    dataset: DecompressionDataset,
    config: TestConfig,
    device: torch.device
) -> dict:
    """
    More rigorous test: compare energy of correct pairing vs mismatched pairing.

    Token A -> Chars of Token A (correct) vs Token A -> Chars of Token B (wrong)

    This tests if the model learned actual token-to-char mappings, not just
    which characters are plausible in general.
    """
    print("\n" + "=" * 60)
    print("CROSS-TOKEN ENERGY TEST")
    print("=" * 60)

    correct_energies = []
    wrong_energies = []
    energy_gaps = []
    detections = []

    # Get pairs of different tokens
    unique_bpe_ids = torch.unique(dataset.bpe_token_ids)
    n_pairs = min(config.num_samples, len(unique_bpe_ids) // 2)

    shuffled = unique_bpe_ids[torch.randperm(len(unique_bpe_ids))]
    token_pairs = [(shuffled[i].item(), shuffled[i + n_pairs].item())
                   for i in range(n_pairs)]

    for i, (bpe_a, bpe_b) in enumerate(token_pairs):
        if i % 20 == 0:
            print(f"Processing pair {i}/{n_pairs}...")

        # Get character sequences for both tokens
        chars_a = tokenizer.get_token_char_ids(bpe_a)
        chars_b = tokenizer.get_token_char_ids(bpe_b)

        # Pad to dataset max_len
        def pad_chars(chars):
            padded = chars + [dataset.pad_id] * (dataset.max_len - len(chars))
            return torch.tensor(padded).unsqueeze(0).to(device)

        chars_a_padded = pad_chars(chars_a)
        chars_b_padded = pad_chars(chars_b)

        # Energy of CORRECT pairing: Token A -> Chars A
        correct_energy = compute_generation_energy(
            model, diffusion, bpe_a,
            chars_a_padded,
            len(chars_a), device,
            config.energy_integration_steps
        )

        # Energy of WRONG pairing: Token A -> Chars B
        wrong_energy = compute_generation_energy(
            model, diffusion, bpe_a,
            chars_b_padded,
            len(chars_b), device,
            config.energy_integration_steps
        )

        correct_energies.append(correct_energy)
        wrong_energies.append(wrong_energy)
        energy_gaps.append(wrong_energy - correct_energy)
        detections.append(1 if wrong_energy > correct_energy else 0)

    # Statistics
    correct_mean = np.mean(correct_energies)
    correct_std = np.std(correct_energies)
    wrong_mean = np.mean(wrong_energies)
    wrong_std = np.std(wrong_energies)
    gap_mean = np.mean(energy_gaps)
    gap_std = np.std(energy_gaps)
    detection_rate = np.mean(detections) * 100

    results = {
        'correct_energy_mean': correct_mean,
        'correct_energy_std': correct_std,
        'wrong_energy_mean': wrong_mean,
        'wrong_energy_std': wrong_std,
        'energy_gap_mean': gap_mean,
        'energy_gap_std': gap_std,
        'detection_rate': detection_rate,
        'num_pairs': n_pairs
    }

    print(f"\nResults:")
    print(f"  Correct Pairing Energy: {correct_mean:.2f} ± {correct_std:.2f}")
    print(f"  Wrong Pairing Energy:   {wrong_mean:.2f} ± {wrong_std:.2f}")
    print(f"  Energy Gap:             {gap_mean:.2f} ± {gap_std:.2f}")
    print(f"  Detection Rate:         {detection_rate:.1f}%")

    if detection_rate > 70:
        print("\n  ✓ SUCCESS: Model learned specific token-char mappings")
    elif detection_rate > 55:
        print("\n  ~ PARTIAL: Some token-specific learning")
    else:
        print("\n  ✗ FAILURE: Model didn't learn token-specific mappings")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    # Load dataset and model
    print("Loading dataset...")
    data_path = Path("data/bpe_decompression.pkl")
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please run: python bpe_tokenizer.py")
        return

    tokenizer, dataset, bpe_config = load_decompression_dataset(str(data_path))
    print(f"  Loaded {len(dataset):,} samples, {dataset.num_bpe_tokens} BPE tokens")

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path("checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Model not found at {checkpoint_path}")
        print("Please run: python run_bpe_diffusion.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                         if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']
    print(f"  Model trained for {checkpoint['iter']} iterations, loss={checkpoint['loss']:.4f}")

    model = DecompressionDiffusion(model_config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    diffusion = DiscreteDiffusion(model_config)

    # Test configuration
    test_config = TestConfig(
        num_samples=100,
        num_generations=10,
        energy_integration_steps=50,
        temperature=1.0
    )

    # Run tests
    print("\n" + "#" * 60)
    print("HALLUCINATION DETECTION TESTS (Phase 3)")
    print("#" * 60)

    # Test 0: ENERGY HEAD (Phase 3 - Primary Test)
    energy_head_results = test_energy_head_detection(
        model, diffusion, tokenizer, dataset, test_config, device
    )

    # Test 1: Energy (random wrong chars) - Legacy
    energy_results = test_energy_detection(
        model, diffusion, tokenizer, dataset, test_config, device
    )

    # Test 2: Cross-token energy (mismatched real tokens) - Legacy
    cross_results = test_cross_token_energy(
        model, diffusion, tokenizer, dataset, test_config, device
    )

    # Skip diversity test (slow and less relevant for Phase 3)
    # diversity_results = test_diversity_detection(
    #     model, diffusion, tokenizer, dataset, test_config, device
    # )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Test':<35} {'Detection Rate':<15} {'Status'}")
    print("-" * 65)

    def status(rate):
        if rate >= 95:
            return "✓ SUCCESS (>95%)"
        elif rate >= 80:
            return "~ GOOD (>80%)"
        elif rate > 55:
            return "~ PARTIAL"
        else:
            return "✗ FAILURE"

    print(f"{'Energy Head (Phase 3)':<35} {energy_head_results['pair_accuracy']:.1f}%{'':<10} {status(energy_head_results['pair_accuracy'])}")
    print(f"{'Legacy: Energy (random wrong)':<35} {energy_results['detection_rate']:.1f}%{'':<10} {status(energy_results['detection_rate'])}")
    print(f"{'Legacy: Cross-token (mismatched)':<35} {cross_results['detection_rate']:.1f}%{'':<10} {status(cross_results['detection_rate'])}")

    # Save results
    results = {
        'energy_head': energy_head_results,
        'energy_legacy': energy_results,
        'cross_token': cross_results
    }

    import json
    with open("hallucination_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to hallucination_results.json")


if __name__ == "__main__":
    main()
