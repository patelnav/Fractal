"""
Hallucination Energy Test (The Popperian Engine)

Tests whether the discrete diffusion model can detect hallucinations
by measuring generation energy.

Hypothesis: When conditioned on "To be or no", generating "I like pizz"
should have significantly HIGHER energy than generating the correct text.

If this holds, we have built a self-verifying AI that can reject
its own hallucinations.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
import numpy as np

from fractal_shakespeare import ShakespeareHierarchy, HierarchyConfig
from run_shakespeare import (
    ShakespeareConfig,
    ShakespeareDiffusionModel,
    ShakespeareDiffusion,
    compute_generation_energy,
    generate_from_condition
)


def load_trained_model(
    checkpoint_path: str,
    hierarchy: ShakespeareHierarchy,
    device: torch.device
) -> Tuple[ShakespeareDiffusionModel, ShakespeareDiffusion]:
    """Load a trained model from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.vocab_size = hierarchy.total_vocab_size

    model = ShakespeareDiffusionModel(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    diffusion = ShakespeareDiffusion(config, hierarchy)

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Best loss: {checkpoint.get('best_loss', 'N/A')}")
    print(f"  Iteration: {checkpoint.get('iter_num', 'N/A')}")

    return model, diffusion


@torch.no_grad()
def compute_energy_batch(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    conditions: torch.Tensor,      # (B,)
    targets: torch.Tensor,         # (B, 4)
    device: torch.device,
    num_integration_steps: int = 50
) -> torch.Tensor:
    """
    Compute generation energy for a batch of (condition, target) pairs.

    Returns energy for each sample in the batch.
    """
    model.eval()
    B = conditions.size(0)

    total_energy = torch.zeros(B, device=device)
    dt = 1.0 / num_integration_steps

    for step in range(num_integration_steps):
        t_val = int(step * diffusion.num_timesteps / num_integration_steps)
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Add noise at this timestep
        noisy_targets = diffusion.add_noise(targets, t, device)

        # Build input
        x = torch.cat([conditions.unsqueeze(1), noisy_targets], dim=1)

        # Get scores
        scores = model.get_scores(x, t)  # (B, 4, vocab_size)

        # Energy at this timestep: mean squared score per sample
        energy_t = (scores ** 2).mean(dim=(1, 2))  # (B,)
        total_energy += energy_t * dt

    return total_energy


def run_hallucination_test(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    hierarchy: ShakespeareHierarchy,
    device: torch.device,
    n_tests: int = 100
) -> Dict:
    """
    Run the hallucination detection test.

    For each test:
    1. Pick a valid (root, chunks) pair from training data
    2. Compute energy for the CORRECT expansion
    3. Compute energy for a WRONG expansion (random chunks)
    4. Check if correct < wrong

    Returns:
        Dictionary with test statistics
    """
    print("\n" + "=" * 70)
    print("HALLUCINATION ENERGY TEST")
    print("Testing whether energy can distinguish correct vs hallucinated text")
    print("=" * 70)

    correct_energies = []
    wrong_energies = []
    energy_gaps = []

    # Test Root -> Chunks level
    print("\n--- Level: Root -> Chunks ---")

    for i in tqdm(range(n_tests), desc="Root->Chunk tests"):
        # Get a valid (root, chunks) pair
        root_id, chunk_ids = random.choice(hierarchy.root_to_chunks_samples)
        root_global = root_id + hierarchy.root_offset
        chunks_global = [cid + hierarchy.chunk_offset for cid in chunk_ids]

        # Compute energy for correct expansion
        correct_energy = compute_generation_energy(
            model, diffusion, root_global, chunks_global, device, num_integration_steps=30
        )

        # Generate a WRONG expansion (random chunks, different from correct)
        wrong_chunks = []
        for _ in range(4):
            random_chunk = random.randint(0, hierarchy.chunk_vocab_size - 2)  # Exclude UNK
            wrong_chunks.append(random_chunk + hierarchy.chunk_offset)

        # Compute energy for wrong expansion
        wrong_energy = compute_generation_energy(
            model, diffusion, root_global, wrong_chunks, device, num_integration_steps=30
        )

        correct_energies.append(correct_energy)
        wrong_energies.append(wrong_energy)
        energy_gaps.append(wrong_energy - correct_energy)

    # Test Chunk -> Chars level
    print("\n--- Level: Chunk -> Chars ---")

    for i in tqdm(range(n_tests), desc="Chunk->Char tests"):
        # Get a valid (chunk, chars) pair
        chunk_id, char_ids = random.choice(hierarchy.chunk_to_chars_samples)
        chunk_global = chunk_id + hierarchy.chunk_offset

        # Compute energy for correct expansion
        correct_energy = compute_generation_energy(
            model, diffusion, chunk_global, char_ids, device, num_integration_steps=30
        )

        # Generate a WRONG expansion (random chars)
        wrong_chars = [random.randint(0, hierarchy.char_vocab_size - 1) for _ in range(4)]

        # Compute energy for wrong expansion
        wrong_energy = compute_generation_energy(
            model, diffusion, chunk_global, wrong_chars, device, num_integration_steps=30
        )

        correct_energies.append(correct_energy)
        wrong_energies.append(wrong_energy)
        energy_gaps.append(wrong_energy - correct_energy)

    # Analyze results
    correct_energies = np.array(correct_energies)
    wrong_energies = np.array(wrong_energies)
    energy_gaps = np.array(energy_gaps)

    # How often is wrong > correct?
    detection_rate = (energy_gaps > 0).mean()

    # Energy ratio
    energy_ratio = wrong_energies.mean() / correct_energies.mean()

    results = {
        'correct_energy_mean': correct_energies.mean(),
        'correct_energy_std': correct_energies.std(),
        'wrong_energy_mean': wrong_energies.mean(),
        'wrong_energy_std': wrong_energies.std(),
        'energy_gap_mean': energy_gaps.mean(),
        'energy_gap_std': energy_gaps.std(),
        'detection_rate': detection_rate,
        'energy_ratio': energy_ratio,
    }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nCorrect expansions:")
    print(f"  Mean energy: {results['correct_energy_mean']:.4f} +/- {results['correct_energy_std']:.4f}")

    print(f"\nWrong expansions (hallucinations):")
    print(f"  Mean energy: {results['wrong_energy_mean']:.4f} +/- {results['wrong_energy_std']:.4f}")

    print(f"\nEnergy gap (wrong - correct):")
    print(f"  Mean: {results['energy_gap_mean']:.4f} +/- {results['energy_gap_std']:.4f}")

    print(f"\nDetection rate: {results['detection_rate']:.1%}")
    print(f"Energy ratio (wrong/correct): {results['energy_ratio']:.2f}x")

    print("\n" + "=" * 70)
    if results['detection_rate'] > 0.9:
        print("*** SUCCESS: The model can reliably detect hallucinations! ***")
        print("The Popperian Engine works - the model can self-verify.")
    elif results['detection_rate'] > 0.7:
        print("*** PARTIAL SUCCESS: The model shows hallucination detection ability ***")
        print("Energy separation is present but not perfect.")
    else:
        print("*** FAILURE: The model cannot reliably distinguish correct from wrong ***")
        print("The energy metric needs refinement or more training is needed.")
    print("=" * 70)

    return results


def demo_specific_examples(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    hierarchy: ShakespeareHierarchy,
    device: torch.device
):
    """Show specific examples of hallucination detection."""

    print("\n" + "=" * 70)
    print("SPECIFIC EXAMPLES")
    print("=" * 70)

    # Pick a few specific roots and show correct vs wrong
    examples = random.sample(hierarchy.root_to_chunks_samples, min(5, len(hierarchy.root_to_chunks_samples)))

    for root_id, chunk_ids in examples:
        root_global = root_id + hierarchy.root_offset
        root_text = hierarchy.decode_root(root_id)

        # Correct expansion
        chunks_global = [cid + hierarchy.chunk_offset for cid in chunk_ids]
        correct_chunks_text = [hierarchy.decode_chunk(cid) for cid in chunk_ids]

        correct_energy = compute_generation_energy(
            model, diffusion, root_global, chunks_global, device, num_integration_steps=30
        )

        # Wrong expansion
        wrong_chunks = []
        for _ in range(4):
            random_chunk = random.randint(0, hierarchy.chunk_vocab_size - 2)
            wrong_chunks.append(random_chunk + hierarchy.chunk_offset)

        wrong_chunks_text = [hierarchy.decode_chunk(c - hierarchy.chunk_offset) for c in wrong_chunks]

        wrong_energy = compute_generation_energy(
            model, diffusion, root_global, wrong_chunks, device, num_integration_steps=30
        )

        gap = wrong_energy - correct_energy
        detected = "YES" if gap > 0 else "NO"

        print(f"\nCondition: '{root_text}'")
        print(f"  Correct: {correct_chunks_text} (energy: {correct_energy:.2f})")
        print(f"  Wrong:   {wrong_chunks_text} (energy: {wrong_energy:.2f})")
        print(f"  Gap: {gap:+.2f} | Detected: {detected}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load hierarchy
    hierarchy_path = "data/shakespeare_hierarchy.pkl"
    if not os.path.exists(hierarchy_path):
        print("ERROR: Hierarchy not found. Run fractal_shakespeare.py first.")
        exit(1)

    hierarchy = ShakespeareHierarchy.load(hierarchy_path)

    # Load model
    checkpoint_path = "checkpoints/best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/final.pt"
        if not os.path.exists(checkpoint_path):
            print("ERROR: No trained model found. Run run_shakespeare.py first.")
            exit(1)

    model, diffusion = load_trained_model(checkpoint_path, hierarchy, device)

    # Run tests
    results = run_hallucination_test(model, diffusion, hierarchy, device, n_tests=100)

    # Show specific examples
    demo_specific_examples(model, diffusion, hierarchy, device)
