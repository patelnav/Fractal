"""
Benchmark suite for Universal Denoiser.

Evaluates three modes:
1. Generation: Start from full mask, measure structural accuracy
2. Repair: Start from corrupted, measure recovery rate
3. Editing: Mask specific positions, measure anchor stability

Success criteria:
- Generation: >85% structural accuracy
- Repair: >90% recovery when 20% corrupted
- Editing: 100% anchor stability
"""

import argparse
import torch
import random
from tqdm import tqdm
from typing import Dict, Tuple

from model import UniversalDenoiser, Config
from data import EvalDataset
from mutations import corrupt_sequence, mask_sequence
from inference import generate, generate_maskgit, repair, edit


def load_model(checkpoint_path: str, device: str) -> Tuple[UniversalDenoiser, dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = Config()
    for k, v in checkpoint['config'].items():
        setattr(config, k, v)
    model = UniversalDenoiser(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def compute_structural_accuracy(pred: torch.Tensor, target: torch.Tensor, pad_id: int) -> float:
    """Compute token-level accuracy ignoring PAD."""
    mask = target != pad_id
    if mask.sum() == 0:
        return 0.0
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()


def is_valid_syntax(tokens: torch.Tensor, vocab: dict) -> bool:
    """Check if sequence has valid parenthesis structure."""
    itos = {v: k for k, v in vocab.items()}
    depth = 0
    for t in tokens.tolist():
        char = itos.get(t, '')
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def benchmark_generation(
    model: UniversalDenoiser,
    eval_ds: EvalDataset,
    num_samples: int = 100,
    K_steps: int = 5,
    device: str = 'cpu',
    use_maskgit: bool = True,
    maskgit_steps: int = 12,
    maskgit_schedule: str = 'cosine',
) -> Dict[str, float]:
    """
    Benchmark generation mode.

    Generates sequences from scratch and compares to ground truth.
    Note: We can't compare exact values, but we can measure:
    - Structural validity (balanced parens, correct format)
    - Token-type distribution similarity
    """
    results = {
        'syntax_valid': 0,
        'has_equals': 0,
        'has_ops': 0,
        'total': num_samples,
        'method': 'maskgit' if use_maskgit else 'naive',
    }

    vocab = eval_ds.stoi

    for i in tqdm(range(num_samples), desc="Generation"):
        # Get target length from eval set
        target = eval_ds[i % len(eval_ds)]
        length = (target != eval_ds.pad_id).sum().item()

        # Generate
        if use_maskgit:
            gen = generate_maskgit(
                model,
                length=length,
                mask_token_id=eval_ds.mask_id,
                pad_token_id=eval_ds.pad_id,
                bos_token_id=eval_ds.bos_id,
                eos_token_id=eval_ds.eos_id,
                num_steps=maskgit_steps,
                temperature=0.0,
                schedule=maskgit_schedule,
                device=device,
            )
        else:
            gen = generate(
                model,
                length=length,
                mask_token_id=eval_ds.mask_id,
                pad_token_id=eval_ds.pad_id,
                bos_token_id=eval_ds.bos_id,
                eos_token_id=eval_ds.eos_id,
                K_steps=K_steps,
                temperature=0.0,
                device=device,
            )

        # Check syntax
        if is_valid_syntax(gen, vocab):
            results['syntax_valid'] += 1

        # Check has equals sign
        if vocab['='] in gen.tolist():
            results['has_equals'] += 1

        # Check has operators
        if vocab['+'] in gen.tolist() or vocab['*'] in gen.tolist():
            results['has_ops'] += 1

    # Compute rates
    results['syntax_valid_rate'] = results['syntax_valid'] / results['total']
    results['has_equals_rate'] = results['has_equals'] / results['total']
    results['has_ops_rate'] = results['has_ops'] / results['total']

    return results


def benchmark_repair(
    model: UniversalDenoiser,
    eval_ds: EvalDataset,
    num_samples: int = 100,
    sigma: float = 0.2,  # 20% corruption
    K_steps: int = 3,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Benchmark repair mode.

    Corrupts sequences at sigma level and measures recovery.
    """
    total_accuracy = 0.0
    perfect_recovery = 0

    special_tokens = {eval_ds.pad_id, eval_ds.mask_id, eval_ds.bos_id, eval_ds.eos_id}

    for i in tqdm(range(min(num_samples, len(eval_ds))), desc="Repair"):
        clean = eval_ds[i].to(device)

        # Corrupt
        corrupted = corrupt_sequence(
            clean.cpu(),
            sigma=sigma,
            vocab_size=len(eval_ds.vocab),
            special_tokens=special_tokens,
            mask_token_id=eval_ds.mask_id,
            pad_token_id=eval_ds.pad_id,
            max_len=len(clean),
        ).to(device)

        # Repair
        repaired = repair(
            model,
            corrupted,
            pad_token_id=eval_ds.pad_id,
            K_steps=K_steps,
            sigma_estimate=sigma,
            temperature=0.0,
            device=device,
        )

        # Measure accuracy
        acc = compute_structural_accuracy(repaired, clean, eval_ds.pad_id)
        total_accuracy += acc

        if acc == 1.0:
            perfect_recovery += 1

    results = {
        'mean_accuracy': total_accuracy / num_samples,
        'perfect_recovery_rate': perfect_recovery / num_samples,
        'sigma': sigma,
        'K_steps': K_steps,
        'num_samples': num_samples,
    }

    return results


def benchmark_editing(
    model: UniversalDenoiser,
    eval_ds: EvalDataset,
    num_samples: int = 100,
    mask_ratio: float = 0.2,  # Mask 20% of tokens
    K_steps: int = 2,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Benchmark editing mode.

    Masks random positions and checks that non-masked positions stay anchored.
    Critical metric: 100% anchor stability.
    """
    total_anchor_stability = 0.0
    total_fill_rate = 0.0
    perfect_anchor = 0

    special_tokens = {eval_ds.pad_id, eval_ds.mask_id, eval_ds.bos_id, eval_ds.eos_id}

    for i in tqdm(range(min(num_samples, len(eval_ds))), desc="Editing"):
        clean = eval_ds[i].to(device)

        # Find editable positions (non-special, non-pad)
        editable_mask = torch.ones_like(clean, dtype=torch.bool)
        for st in special_tokens:
            editable_mask &= (clean != st)

        editable_positions = editable_mask.nonzero(as_tuple=True)[0].tolist()

        if len(editable_positions) == 0:
            continue

        # Sample positions to mask
        n_mask = max(1, int(mask_ratio * len(editable_positions)))
        mask_indices = random.sample(editable_positions, n_mask)

        # Create mask tensor
        mask_positions = torch.zeros_like(clean, dtype=torch.bool)
        mask_positions[mask_indices] = True

        # Edit
        edited = edit(
            model,
            clean,
            mask_positions,
            mask_token_id=eval_ds.mask_id,
            pad_token_id=eval_ds.pad_id,
            K_steps=K_steps,
            sigma=0.5,
            temperature=0.0,
            device=device,
        )

        # Check anchor stability (non-masked positions should be unchanged)
        anchor_mask = ~mask_positions & (clean != eval_ds.pad_id)
        anchor_correct = (edited == clean) & anchor_mask
        anchor_stability = anchor_correct.sum().item() / max(1, anchor_mask.sum().item())
        total_anchor_stability += anchor_stability

        if anchor_stability == 1.0:
            perfect_anchor += 1

        # Check if masked positions were filled (not still MASK)
        filled = (edited[mask_positions] != eval_ds.mask_id)
        fill_rate = filled.sum().item() / max(1, len(mask_indices))
        total_fill_rate += fill_rate

    results = {
        'mean_anchor_stability': total_anchor_stability / num_samples,
        'perfect_anchor_rate': perfect_anchor / num_samples,
        'mean_fill_rate': total_fill_rate / num_samples,
        'mask_ratio': mask_ratio,
        'K_steps': K_steps,
        'num_samples': num_samples,
    }

    return results


def run_full_benchmark(
    model: UniversalDenoiser,
    device: str = 'cpu',
    num_samples: int = 100,
    use_maskgit: bool = True,
    maskgit_steps: int = 12,
    maskgit_schedule: str = 'cosine',
) -> Dict[str, Dict]:
    """Run all benchmarks."""
    eval_ds = EvalDataset(num_samples=num_samples * 2, seed=42)

    print("\n" + "=" * 60)
    print("Universal Denoiser Benchmark Suite")
    print("=" * 60)

    # Generation
    method = "MaskGIT" if use_maskgit else "Naive"
    print(f"\n1. GENERATION BENCHMARK ({method}, {maskgit_steps} steps, {maskgit_schedule})")
    print("-" * 40)
    gen_results = benchmark_generation(
        model, eval_ds, num_samples=num_samples, K_steps=5, device=device,
        use_maskgit=use_maskgit, maskgit_steps=maskgit_steps, maskgit_schedule=maskgit_schedule,
    )
    print(f"  Syntax Valid: {gen_results['syntax_valid_rate']:.1%}")
    print(f"  Has Equals:   {gen_results['has_equals_rate']:.1%}")
    print(f"  Has Ops:      {gen_results['has_ops_rate']:.1%}")

    # Repair
    print("\n2. REPAIR BENCHMARK")
    print("-" * 40)
    repair_results = benchmark_repair(
        model, eval_ds, num_samples=num_samples, sigma=0.2, K_steps=3, device=device
    )
    print(f"  Mean Accuracy:        {repair_results['mean_accuracy']:.1%}")
    print(f"  Perfect Recovery:     {repair_results['perfect_recovery_rate']:.1%}")
    print(f"  Target: >90% accuracy")

    # Editing
    print("\n3. EDITING BENCHMARK")
    print("-" * 40)
    edit_results = benchmark_editing(
        model, eval_ds, num_samples=num_samples, mask_ratio=0.2, K_steps=2, device=device
    )
    print(f"  Mean Anchor Stability: {edit_results['mean_anchor_stability']:.1%}")
    print(f"  Perfect Anchor Rate:   {edit_results['perfect_anchor_rate']:.1%}")
    print(f"  Mean Fill Rate:        {edit_results['mean_fill_rate']:.1%}")
    print(f"  Target: 100% anchor stability")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Generation syntax valid: {gen_results['syntax_valid_rate']:.1%} (target: >85%)")
    print(f"  Repair accuracy:         {repair_results['mean_accuracy']:.1%} (target: >90%)")
    print(f"  Edit anchor stability:   {edit_results['mean_anchor_stability']:.1%} (target: 100%)")

    # Check success criteria
    success = (
        gen_results['syntax_valid_rate'] >= 0.85 and
        repair_results['mean_accuracy'] >= 0.90 and
        edit_results['mean_anchor_stability'] >= 0.99
    )
    print(f"\n  Overall Success: {'PASS' if success else 'NEEDS WORK'}")

    return {
        'generation': gen_results,
        'repair': repair_results,
        'editing': edit_results,
        'success': success,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Universal Denoiser')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per benchmark')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--use_maskgit', action='store_true', default=True,
                        help='Use MaskGIT-style generation (default: True)')
    parser.add_argument('--no_maskgit', action='store_true',
                        help='Disable MaskGIT (use naive generation)')
    parser.add_argument('--maskgit_steps', type=int, default=12,
                        help='Number of MaskGIT unmasking steps')
    parser.add_argument('--maskgit_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='MaskGIT unmasking schedule')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"Model loaded (iter {checkpoint.get('iter_num', '?')}, loss {checkpoint.get('loss', '?'):.4f})")

    # MaskGIT settings
    use_maskgit = not args.no_maskgit

    # Run benchmarks
    results = run_full_benchmark(
        model, device=device, num_samples=args.num_samples,
        use_maskgit=use_maskgit, maskgit_steps=args.maskgit_steps,
        maskgit_schedule=args.maskgit_schedule,
    )

    return results


if __name__ == "__main__":
    main()
