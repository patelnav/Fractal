"""
generate_fractal.py
Phase 5: The Dreamer Demo

Uses the Phase 4 Fractal Engine to recursively expand a "Seed Idea" into text.
Implements Energy-Based Rejection Sampling to self-correct hallucinations.

This is a "Vertical Language Model" - it expands Top → Down (abstract → concrete),
unlike standard LLMs which expand Left → Right (token by token).
"""
import sys
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple

# Add phase4 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))

from hierarchical_bpe import (
    HierarchicalBPE,
    FractalDataset,
    load_fractal_dataset
)
from run_fractal_engine import (
    FractalDiffusionModel,
    FractalModelConfig
)

# Phase 4 directory for loading data/models
PHASE4_DIR = Path(__file__).parent.parent / "phase4-fractal-engine"


# Configuration
MAX_RETRIES = 10
ENERGY_THRESHOLD = 0.5


def load_system(device: torch.device) -> Tuple[FractalDiffusionModel, HierarchicalBPE, FractalDataset, FractalModelConfig]:
    """Load the trained model and tokenizer."""
    print(f"Loading system on {device}...")

    # Load tokenizer and dataset from phase4
    data_path = PHASE4_DIR / "data/fractal_hierarchy.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run phase4's run_fractal_engine.py first.")

    tokenizer, dataset, _ = load_fractal_dataset(str(data_path))
    print(f"  Loaded tokenizer: {len(tokenizer.root_vocab)} roots, {len(tokenizer.chunk_vocab)} chunks")

    # Load model from phase4
    model_path = PHASE4_DIR / "checkpoints/best_model.pt"
    if not model_path.exists():
        model_path = PHASE4_DIR / "checkpoints/final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("No model found in phase4 checkpoints/")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = FractalDiffusionModel(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Trained for {checkpoint.get('iter', 'unknown')} iterations")

    return model, tokenizer, dataset, config


def generate_with_rejection(
    model: FractalDiffusionModel,
    config: FractalModelConfig,
    condition_id: int,
    level: int,
    device: torch.device,
    verbose: bool = True,
    use_sampling: bool = True,
    temperature: float = 1.0
) -> Tuple[Optional[torch.Tensor], float, int]:
    """
    Generate tokens at a given level using the model's predictions + rejection sampling.

    The key insight: use the model's diffusion head to PROPOSE candidates,
    then use the energy head to VERIFY them.

    Args:
        model: The trained FractalDiffusionModel
        config: Model configuration
        condition_id: The conditioning token (root_id for level 0, chunk_id for level 1)
        level: 0 for Root→Chunks, 1 for Chunk→Chars
        device: Torch device
        verbose: Print progress
        use_sampling: If True, sample from distribution; if False, use greedy
        temperature: Sampling temperature (higher = more random)

    Returns:
        (generated_ids, energy, num_attempts) or (None, best_energy, MAX_RETRIES)
    """
    if level == 0:
        # Root → Chunks
        target_vocab = config.num_chunks
        seq_len = config.expansion_size  # 4 chunks
        cond_offset = config.root_offset
        target_offset = config.chunk_offset
        task_name = "Chunks"
    else:
        # Chunk → Chars
        target_vocab = config.num_chars
        seq_len = config.max_char_len  # 16 chars
        cond_offset = config.chunk_offset
        target_offset = 0  # Chars have no offset
        task_name = "Chars"

    best_candidate = None
    best_energy = float('inf')

    for attempt in range(MAX_RETRIES):
        # Initialize with random noise (simulating diffusion start)
        noise = torch.randint(0, target_vocab, (1, seq_len), device=device)

        # Build input: [condition + offset, noisy_targets + offset]
        cond_tensor = torch.tensor([[condition_id + cond_offset]], device=device)
        target_tensor = noise + target_offset
        x = torch.cat([cond_tensor, target_tensor], dim=1)

        # Forward pass - model predicts clean tokens from noisy input
        # Use t close to max (high noise) so model does full denoising
        t = torch.tensor([config.num_timesteps - 1], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(x, t, level=level, return_energy=False)

        # Get candidate from model's predictions
        if use_sampling and temperature > 0:
            # Sample from the distribution with temperature
            probs = F.softmax(logits / temperature, dim=-1)
            candidate = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, seq_len)
        else:
            # Greedy decode
            candidate = torch.argmax(logits, dim=-1)  # (1, seq_len)

        # Now CHECK the candidate's energy by running it through again at t=0
        candidate_with_offset = candidate + target_offset
        x_check = torch.cat([cond_tensor, candidate_with_offset], dim=1)
        t_check = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            _, energy = model(x_check, t_check, level=level, return_energy=True)

        energy_val = energy.item()

        # Track best
        if energy_val < best_energy:
            best_energy = energy_val
            best_candidate = candidate

        # Check threshold
        if energy_val < ENERGY_THRESHOLD:
            if verbose:
                print(f"Accepted! (E={energy_val:.4f}, attempt {attempt + 1})")
            return candidate, energy_val, attempt + 1

        if verbose and attempt == 0:
            print(f"Retry (E={energy_val:.4f})...", end=" ")
        elif verbose:
            print(f"(E={energy_val:.4f})...", end=" ")

    if verbose:
        print(f"Using best after {MAX_RETRIES} attempts (E={best_energy:.4f})")

    return best_candidate, best_energy, MAX_RETRIES


def dream_from_root(
    model: FractalDiffusionModel,
    tokenizer: HierarchicalBPE,
    config: FractalModelConfig,
    root_id: int,
    device: torch.device,
    verbose: bool = True,
    use_gt_length: bool = True
) -> Tuple[str, dict]:
    """
    Recursively expand a root into text.

    Level 0: Root → 4 Chunks
    Level 1: Each Chunk → up to 16 Chars (but only take the known length)

    Returns (generated_text, stats_dict)
    """
    stats = {
        'root_id': root_id,
        'ground_truth': tokenizer.decode_root(root_id),
        'level0_energy': None,
        'level0_attempts': 0,
        'level1_energies': [],
        'level1_attempts': [],
        'total_rejections': 0
    }

    if verbose:
        print(f"\n  Ground truth: \"{stats['ground_truth']}\"")

    # Level 0: Root → Chunks
    if verbose:
        print(f"  Thinking ({config.expansion_size} Chunks)... ", end="")

    chunk_ids, energy, attempts = generate_with_rejection(
        model, config, root_id, level=0, device=device, verbose=verbose
    )

    stats['level0_energy'] = energy
    stats['level0_attempts'] = attempts
    if attempts > 1:
        stats['total_rejections'] += attempts - 1

    if chunk_ids is None:
        return "[Level 0 failed]", stats

    # Level 1: Each Chunk → Chars
    all_text_parts = []
    chunk_ids_list = chunk_ids[0].tolist()

    for i, chunk_id in enumerate(chunk_ids_list):
        # Skip padding tokens
        if chunk_id == config.pad_chunk_id or chunk_id >= config.num_chunks:
            continue

        # Get the KNOWN length of this chunk from the tokenizer
        # This is key: the model outputs 16 positions but only N are valid
        if use_gt_length and chunk_id in tokenizer.chunk_vocab:
            chunk_text = tokenizer.chunk_vocab[chunk_id].decode('utf-8', errors='replace')
            chunk_len = len(chunk_text)
        else:
            chunk_len = config.max_char_len  # Fallback to full length

        if verbose:
            print(f"  Thinking (Chars for chunk {i}, len={chunk_len})... ", end="")

        char_ids, energy, attempts = generate_with_rejection(
            model, config, chunk_id, level=1, device=device, verbose=verbose
        )

        stats['level1_energies'].append(energy)
        stats['level1_attempts'].append(attempts)
        if attempts > 1:
            stats['total_rejections'] += attempts - 1

        if char_ids is not None:
            # Only take the first `chunk_len` characters
            valid_char_ids = char_ids[0][:chunk_len].tolist()
            text_part = ''.join(tokenizer.id_to_char.get(c, '?') for c in valid_char_ids if c < config.num_chars)
            all_text_parts.append(text_part)

    return ''.join(all_text_parts), stats


def test_decompression_mode(
    model: FractalDiffusionModel,
    tokenizer: HierarchicalBPE,
    config: FractalModelConfig,
    dataset: FractalDataset,
    device: torch.device,
    num_samples: int = 5
) -> None:
    """
    Test the model in DECOMPRESSION mode - give it the correct condition
    and see if it recovers the correct expansion.

    This is what the model was actually trained to do!
    """
    print("\n" + "=" * 70)
    print("DECOMPRESSION MODE (What the model was trained to do)")
    print("=" * 70)

    # Get unique root IDs
    unique_roots = dataset.root_ids.unique().tolist()
    sample_roots = random.sample(unique_roots, min(num_samples, len(unique_roots)))

    for root_id in sample_roots:
        print(f"\n{'─' * 50}")
        print(f"ROOT ID: {root_id}")

        # Get ground truth
        gt_text = tokenizer.decode_root(root_id)
        gt_chunks = tokenizer.get_root_chunks(root_id)

        print(f"  Ground Truth: \"{gt_text}\"")
        print(f"  GT Chunk IDs: {gt_chunks}")

        # Test Level 0: Root → Chunks
        # Give model NOISY chunks, see if it predicts the correct ones
        noise = torch.randint(0, config.num_chunks, (1, config.expansion_size), device=device)
        cond = torch.tensor([[root_id + config.root_offset]], device=device)
        x = torch.cat([cond, noise + config.chunk_offset], dim=1)
        t = torch.tensor([config.num_timesteps - 1], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(x, t, level=0, return_energy=False)

        pred_chunks = torch.argmax(logits, dim=-1)[0].tolist()
        print(f"  Predicted:    {pred_chunks}")

        # Check accuracy
        matches = sum(1 for p, g in zip(pred_chunks[:len(gt_chunks)], gt_chunks) if p == g)
        print(f"  Chunk Match:  {matches}/{len(gt_chunks)}")

        # Decode predicted chunks using tokenizer
        pred_text_parts = []
        for cid in pred_chunks:
            if cid < len(tokenizer.chunk_vocab):
                try:
                    pred_text_parts.append(tokenizer.chunk_vocab[cid].decode('utf-8', errors='replace'))
                except:
                    pred_text_parts.append(f"[{cid}]")
            else:
                pred_text_parts.append("[PAD]")

        print(f"  Decoded:      \"{''.join(pred_text_parts)}\"")

        # Also test Level 1: Chunk → Chars for one of the chunks
        if gt_chunks:
            test_chunk = gt_chunks[0]
            gt_chars = tokenizer.get_chunk_chars(test_chunk)
            gt_char_ids = tokenizer.get_chunk_char_ids(test_chunk)

            # Give model noisy chars, see if it predicts correct
            char_noise = torch.randint(0, config.num_chars, (1, config.max_char_len), device=device)
            chunk_cond = torch.tensor([[test_chunk + config.chunk_offset]], device=device)
            x_char = torch.cat([chunk_cond, char_noise], dim=1)

            with torch.no_grad():
                char_logits, _ = model(x_char, t, level=1, return_energy=False)

            pred_char_ids = torch.argmax(char_logits, dim=-1)[0].tolist()
            pred_chars = ''.join(tokenizer.id_to_char.get(c, '?') for c in pred_char_ids if c < config.num_chars)

            print(f"  --- Level 1 Test (Chunk {test_chunk}) ---")
            print(f"    GT Chars:  \"{gt_chars}\" -> {gt_char_ids[:8]}...")
            print(f"    Predicted: \"{pred_chars[:16]}\" -> {pred_char_ids[:8]}...")
            char_matches = sum(1 for p, g in zip(pred_char_ids[:len(gt_char_ids)], gt_char_ids) if p == g)
            print(f"    Match:     {char_matches}/{len(gt_char_ids)}")


def compare_with_ground_truth(
    model: FractalDiffusionModel,
    tokenizer: HierarchicalBPE,
    config: FractalModelConfig,
    dataset: FractalDataset,
    device: torch.device,
    num_samples: int = 5
) -> None:
    """
    Show generation vs ground truth for sample roots.
    Also shows the energy of ground truth for comparison.
    """
    print("\n" + "=" * 70)
    print("GENERATION VS GROUND TRUTH (Rejection Sampling)")
    print("=" * 70)

    # Get unique root IDs
    unique_roots = dataset.root_ids.unique().tolist()
    sample_roots = random.sample(unique_roots, min(num_samples, len(unique_roots)))

    for root_id in sample_roots:
        print(f"\n{'─' * 50}")
        print(f"ROOT ID: {root_id}")

        # Get ground truth
        gt_text = tokenizer.decode_root(root_id)
        gt_chunks = tokenizer.get_root_chunks(root_id)

        # Compute energy of ground truth
        gt_chunk_tensor = torch.tensor([gt_chunks[:config.expansion_size]], device=device)
        if len(gt_chunks) < config.expansion_size:
            # Pad
            padding = torch.full(
                (1, config.expansion_size - len(gt_chunks)),
                config.pad_chunk_id,
                device=device
            )
            gt_chunk_tensor = torch.cat([gt_chunk_tensor, padding], dim=1)

        cond = torch.tensor([[root_id + config.root_offset]], device=device)
        x_gt = torch.cat([cond, gt_chunk_tensor + config.chunk_offset], dim=1)
        t = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            _, gt_energy = model(x_gt, t, level=0, return_energy=True)

        print(f"  Ground Truth: \"{gt_text}\" (E={gt_energy.item():.4f})")

        # Generate
        gen_text, stats = dream_from_root(
            model, tokenizer, config, root_id, device, verbose=True
        )

        print(f"  Generated:    \"{gen_text}\"")
        print(f"  Rejections:   {stats['total_rejections']}")


def run_dream(num_dreams: int = 5):
    """Main demo function."""
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load system
    model, tokenizer, dataset, config = load_system(device)

    print("\n" + "#" * 70)
    print("THE FRACTAL DREAMER")
    print("A Vertical Language Model: Abstract → Concrete Expansion")
    print("#" * 70)

    print(f"\nConfiguration:")
    print(f"  Energy threshold: {ENERGY_THRESHOLD}")
    print(f"  Max retries:      {MAX_RETRIES}")
    print(f"  Expansion:        Root → {config.expansion_size} Chunks → {config.max_char_len} Chars each")

    # First test: Decompression mode (what the model was trained to do)
    test_decompression_mode(model, tokenizer, config, dataset, device, num_samples=num_dreams)

    # Second test: Generation with rejection sampling
    compare_with_ground_truth(model, tokenizer, config, dataset, device, num_samples=num_dreams)

    # Summary
    print("\n" + "=" * 70)
    print("WHAT YOU JUST SAW:")
    print("=" * 70)
    print("""
1. The model received a Root ID (a high-level concept from Shakespeare)
2. It expanded this into 4 Chunks using its Level 0 head
3. Each Chunk was then expanded into characters using its Level 1 head
4. At each step, the Energy Head evaluated if the expansion was valid
5. If energy was too high (>0.5), it REJECTED and tried again

This is "System 2 Thinking" - the model checks its own work before committing.
The shared weights handle BOTH levels of abstraction, demonstrating that
"intelligence is scale-invariant."
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="The Fractal Dreamer Demo")
    parser.add_argument("--num-dreams", type=int, default=5, help="Number of roots to dream about")
    parser.add_argument("--threshold", type=float, default=0.5, help="Energy threshold for rejection")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries before accepting best")
    args = parser.parse_args()

    ENERGY_THRESHOLD = args.threshold
    MAX_RETRIES = args.max_retries

    run_dream(num_dreams=args.num_dreams)
