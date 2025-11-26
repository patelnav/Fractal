"""
generate_hybrid.py
Phase 6: The Full Stack Demo

Combines two models for complete text generation:
1. Manager (AR GPT): Dreams a "plot" as a sequence of Root IDs
2. Fractal Engine (Diffusion): Renders each Root into text via hierarchical expansion

This is a "Hybrid Language Model" architecture:
- The Manager handles high-level structure (what comes next)
- The Fractal Engine handles low-level details (how to spell it)
- Energy-based rejection ensures quality at every level

The result: Novel Shakespeare-like text that is both coherent AND verified.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase5-dreamer"))

from hierarchical_bpe import HierarchicalBPE, load_fractal_dataset
from run_fractal_engine import FractalDiffusionModel, FractalModelConfig
from train_manager import ManagerGPT, ManagerConfig, DEVICE


# ============================================================================
# Configuration
# ============================================================================

ENERGY_THRESHOLD = 0.5
MAX_RETRIES = 10


# ============================================================================
# Fractal Rendering (from Phase 5)
# ============================================================================

def generate_with_rejection(
    model: FractalDiffusionModel,
    config: FractalModelConfig,
    condition_id: int,
    level: int,
    device: torch.device,
    temperature: float = 1.0
) -> Tuple[Optional[torch.Tensor], float, int]:
    """
    Generate tokens using the Fractal Engine with rejection sampling.

    Returns:
        (generated_ids, energy, num_attempts) or (None, best_energy, MAX_RETRIES)
    """
    if level == 0:
        target_vocab = config.num_chunks
        seq_len = config.expansion_size
        cond_offset = config.root_offset
        target_offset = config.chunk_offset
    else:
        target_vocab = config.num_chars
        seq_len = config.max_char_len
        cond_offset = config.chunk_offset
        target_offset = 0

    best_candidate = None
    best_energy = float('inf')

    for attempt in range(MAX_RETRIES):
        # Random noise
        noise = torch.randint(0, target_vocab, (1, seq_len), device=device)

        # Build input
        cond_tensor = torch.tensor([[condition_id + cond_offset]], device=device)
        target_tensor = noise + target_offset
        x = torch.cat([cond_tensor, target_tensor], dim=1)

        # Forward pass at max timestep (full denoising)
        t = torch.tensor([config.num_timesteps - 1], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(x, t, level=level, return_energy=False)

        # Sample candidate
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            candidate = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, seq_len)
        else:
            candidate = torch.argmax(logits, dim=-1)

        # Check energy at t=0
        candidate_with_offset = candidate + target_offset
        x_check = torch.cat([cond_tensor, candidate_with_offset], dim=1)
        t_check = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            _, energy = model(x_check, t_check, level=level, return_energy=True)

        energy_val = energy.item()

        if energy_val < best_energy:
            best_energy = energy_val
            best_candidate = candidate

        if energy_val < ENERGY_THRESHOLD:
            return candidate, energy_val, attempt + 1

    return best_candidate, best_energy, MAX_RETRIES


def render_root(
    fractal_model: FractalDiffusionModel,
    tokenizer: HierarchicalBPE,
    config: FractalModelConfig,
    root_id: int,
    device: torch.device
) -> Tuple[str, dict]:
    """
    Render a single root ID to text using the Fractal Engine.

    Level 0: Root -> Chunks
    Level 1: Each Chunk -> Characters
    """
    stats = {
        'root_id': root_id,
        'level0_energy': None,
        'level0_attempts': 0,
        'level1_stats': [],
        'total_rejections': 0
    }

    # Level 0: Root -> Chunks
    chunk_ids, energy, attempts = generate_with_rejection(
        fractal_model, config, root_id, level=0, device=device
    )

    stats['level0_energy'] = energy
    stats['level0_attempts'] = attempts
    if attempts > 1:
        stats['total_rejections'] += attempts - 1

    if chunk_ids is None:
        return "[Level 0 failed]", stats

    # Level 1: Each Chunk -> Chars
    text_parts = []
    chunk_ids_list = chunk_ids[0].tolist()

    for chunk_id in chunk_ids_list:
        if chunk_id == config.pad_chunk_id or chunk_id >= config.num_chunks:
            continue

        # Get known chunk length from tokenizer
        if chunk_id in tokenizer.chunk_vocab:
            chunk_text = tokenizer.chunk_vocab[chunk_id].decode('utf-8', errors='replace')
            chunk_len = len(chunk_text)
        else:
            chunk_len = config.max_char_len

        # Generate characters
        char_ids, energy, attempts = generate_with_rejection(
            fractal_model, config, chunk_id, level=1, device=device
        )

        stats['level1_stats'].append({
            'chunk_id': chunk_id,
            'energy': energy,
            'attempts': attempts
        })
        if attempts > 1:
            stats['total_rejections'] += attempts - 1

        if char_ids is not None:
            valid_char_ids = char_ids[0][:chunk_len].tolist()
            text_part = ''.join(
                tokenizer.id_to_char.get(c, '?')
                for c in valid_char_ids
                if c < config.num_chars
            )
            text_parts.append(text_part)

    return ''.join(text_parts), stats


# ============================================================================
# Hybrid Generation
# ============================================================================

def load_hybrid_system() -> Tuple[ManagerGPT, FractalDiffusionModel, HierarchicalBPE, FractalModelConfig]:
    """Load both the Manager and Fractal Engine."""
    print("=" * 60)
    print("LOADING HYBRID SYSTEM")
    print("=" * 60)

    # Paths
    phase4_dir = Path(__file__).parent.parent / "phase4-fractal-engine"
    phase6_dir = Path(__file__).parent

    # Load tokenizer and dataset
    data_path = phase4_dir / "data/fractal_hierarchy.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    tokenizer, dataset, _ = load_fractal_dataset(str(data_path))
    print(f"  Tokenizer: {dataset.num_roots} roots, {dataset.num_chunks} chunks")

    # Load Fractal Engine
    model_path = phase4_dir / "checkpoints/best_model.pt"
    if not model_path.exists():
        model_path = phase4_dir / "checkpoints/final_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("No Fractal Engine model found")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']

    fractal_model = FractalDiffusionModel(config).to(DEVICE)
    fractal_model.load_state_dict(checkpoint['model'])
    fractal_model.eval()
    print(f"  Fractal Engine: {sum(p.numel() for p in fractal_model.parameters()):,} params")

    # Load Manager
    manager_path = phase6_dir / "manager.pt"
    if not manager_path.exists():
        raise FileNotFoundError(f"Manager not found at {manager_path}. Run train_manager.py first.")

    manager_ckpt = torch.load(manager_path, map_location=DEVICE, weights_only=False)
    manager_config = manager_ckpt['config']

    manager = ManagerGPT(manager_config).to(DEVICE)
    manager.load_state_dict(manager_ckpt['model'])
    manager.eval()
    print(f"  Manager GPT: {sum(p.numel() for p in manager.parameters()):,} params")

    print("=" * 60)

    return manager, fractal_model, tokenizer, config


def run_hybrid(
    num_roots: int = 20,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = None
):
    """
    Run the full hybrid generation pipeline.

    1. Manager generates a sequence of root IDs
    2. Fractal Engine renders each root to text
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Load models
    manager, fractal_model, tokenizer, config = load_hybrid_system()

    print("\n" + "#" * 60)
    print("THE HYBRID FRACTAL ENGINE")
    print("Manager (Plot) + Fractal Engine (Render)")
    print("#" * 60)

    print(f"\nConfiguration:")
    print(f"  Roots to generate: {num_roots}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Energy threshold: {ENERGY_THRESHOLD}")
    print(f"  Max retries: {MAX_RETRIES}")

    # Step 1: Manager dreams a plot
    print("\n" + "=" * 60)
    print("[1] MANAGER: Dreaming a plot...")
    print("=" * 60)

    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated_roots = manager.generate(
            start_idx,
            max_new_tokens=num_roots,
            temperature=temperature,
            top_k=top_k
        )

    root_seq = generated_roots[0].tolist()[1:]  # Skip start token
    print(f"\nGenerated root sequence ({len(root_seq)} roots):")
    print(f"  {root_seq[:15]}{'...' if len(root_seq) > 15 else ''}")

    # Show what these roots mean (via tokenizer lookup)
    print("\nRoot preview (via tokenizer):")
    preview_parts = []
    for rid in root_seq[:10]:
        if rid < len(tokenizer.root_vocab):
            preview_parts.append(tokenizer.decode_root(rid))
    print(f"  \"{''.join(preview_parts)[:60]}...\"")

    # Step 2: Fractal Engine renders
    print("\n" + "=" * 60)
    print("[2] FRACTAL ENGINE: Rendering...")
    print("=" * 60)

    full_text = []
    total_rejections = 0

    for i, root_id in enumerate(root_seq):
        # Check if valid root
        if root_id >= len(tokenizer.root_vocab):
            print(f"  Root {i+1}/{len(root_seq)}: ID {root_id} [OOV, skipping]")
            continue

        print(f"  Root {i+1}/{len(root_seq)}: ID {root_id}", end="")

        # Render
        text, stats = render_root(
            fractal_model, tokenizer, config, root_id, DEVICE
        )

        total_rejections += stats['total_rejections']

        if stats['total_rejections'] > 0:
            print(f" (rejections: {stats['total_rejections']})", end="")

        print(f" -> \"{text[:30]}{'...' if len(text) > 30 else ''}\"")

        full_text.append(text)

    # Final output
    print("\n" + "=" * 60)
    print("FINAL GENERATED TEXT")
    print("=" * 60)
    print()
    print("".join(full_text))
    print()
    print("=" * 60)
    print(f"Stats: {len(root_seq)} roots, {total_rejections} total rejections")
    print("=" * 60)

    return "".join(full_text), root_seq


def compare_methods(num_roots: int = 10):
    """
    Compare hybrid generation vs pure tokenizer decode.

    Shows that the Fractal Engine does more than just lookup -
    it verifies each expansion with energy-based rejection.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Hybrid vs Pure Tokenizer")
    print("=" * 60)

    manager, fractal_model, tokenizer, config = load_hybrid_system()

    # Generate roots
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = manager.generate(start_idx, max_new_tokens=num_roots, temperature=0.8)
    root_seq = generated[0].tolist()[1:]

    print(f"\nGenerated roots: {root_seq}")

    # Method 1: Pure tokenizer lookup
    print("\n[Method 1] Pure tokenizer decode:")
    pure_parts = []
    for rid in root_seq:
        if rid < len(tokenizer.root_vocab):
            pure_parts.append(tokenizer.decode_root(rid))
    pure_text = "".join(pure_parts)
    print(f"  \"{pure_text[:80]}...\"")

    # Method 2: Fractal Engine with rejection
    print("\n[Method 2] Fractal Engine (with energy verification):")
    fractal_parts = []
    total_rejections = 0
    for rid in root_seq:
        if rid < len(tokenizer.root_vocab):
            text, stats = render_root(fractal_model, tokenizer, config, rid, DEVICE)
            fractal_parts.append(text)
            total_rejections += stats['total_rejections']
    fractal_text = "".join(fractal_parts)
    print(f"  \"{fractal_text[:80]}...\"")
    print(f"  (Rejections: {total_rejections})")

    # Analysis
    print("\n[Analysis]")
    match = pure_text == fractal_text
    print(f"  Texts match: {match}")
    if not match:
        print("  The Fractal Engine may produce slightly different text when")
        print("  rejection sampling leads to alternative valid expansions.")

    return pure_text, fractal_text


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Fractal Engine Demo")
    parser.add_argument("--num-roots", type=int, default=20, help="Number of roots to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Manager sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling for Manager")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Run comparison mode")
    args = parser.parse_args()

    if args.compare:
        compare_methods(num_roots=args.num_roots)
    else:
        run_hybrid(
            num_roots=args.num_roots,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed
        )
