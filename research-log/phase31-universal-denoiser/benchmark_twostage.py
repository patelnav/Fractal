"""
Benchmark for Two-Stage Generation.
"""

import argparse
import torch
from tqdm import tqdm

from inference_twostage import (
    load_twostage_models,
    generate_twostage,
    decode_skeleton,
    decode_filler,
)
from data import EvalDataset


def is_valid_syntax(text: str) -> bool:
    """Check balanced parentheses and basic structure."""
    # Remove special tokens
    text = text.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '')
    text = text.replace('<MASK>', '').replace('<DIGIT>', '')

    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def has_equals(text: str) -> bool:
    return '=' in text


def has_ops(text: str) -> bool:
    return '+' in text or '*' in text


def benchmark_twostage(
    skeleton_model,
    filler_model,
    skel_vocab,
    filler_vocab,
    num_samples: int = 200,
    skel_steps: int = 12,
    filler_steps: int = 8,
    device: str = 'cpu',
):
    """Benchmark two-stage generation."""
    # Use EvalDataset for target lengths
    eval_ds = EvalDataset(num_samples=num_samples * 2, seed=42)

    results = {
        'syntax_valid': 0,
        'has_equals': 0,
        'has_ops': 0,
        'skeleton_valid': 0,
        'total': num_samples,
    }

    print(f"\nBenchmarking Two-Stage Generation ({num_samples} samples)")
    print("-" * 50)

    for i in tqdm(range(num_samples), desc="Two-Stage Generation"):
        target = eval_ds[i % len(eval_ds)]
        length = (target != eval_ds.pad_id).sum().item()

        skeleton, output = generate_twostage(
            skeleton_model,
            filler_model,
            length=length,
            skel_vocab=skel_vocab,
            filler_vocab=filler_vocab,
            skel_steps=skel_steps,
            filler_steps=filler_steps,
            temperature=0.0,
            device=device,
        )

        skel_text = decode_skeleton(skeleton, skel_vocab)
        out_text = decode_filler(output, filler_vocab)

        # Check skeleton validity
        if is_valid_syntax(skel_text):
            results['skeleton_valid'] += 1

        # Check final output
        if is_valid_syntax(out_text):
            results['syntax_valid'] += 1
        if has_equals(out_text):
            results['has_equals'] += 1
        if has_ops(out_text):
            results['has_ops'] += 1

        # Show first few examples
        if i < 5:
            print(f"\nExample {i}:")
            print(f"  Skeleton: {skel_text}")
            print(f"  Output:   {out_text}")
            print(f"  Valid:    {is_valid_syntax(out_text)}")

    # Compute rates
    results['syntax_valid_rate'] = results['syntax_valid'] / results['total']
    results['has_equals_rate'] = results['has_equals'] / results['total']
    results['has_ops_rate'] = results['has_ops'] / results['total']
    results['skeleton_valid_rate'] = results['skeleton_valid'] / results['total']

    print("\n" + "=" * 50)
    print("Two-Stage Generation Results")
    print("=" * 50)
    print(f"  Skeleton Valid:  {results['skeleton_valid_rate']:.1%}")
    print(f"  Final Valid:     {results['syntax_valid_rate']:.1%}")
    print(f"  Has Equals:      {results['has_equals_rate']:.1%}")
    print(f"  Has Ops:         {results['has_ops_rate']:.1%}")
    print(f"  Target: >85% syntax valid")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Two-Stage Generation')
    parser.add_argument('--skeleton_checkpoint', type=str, default='checkpoints_skeleton/best.pt')
    parser.add_argument('--filler_checkpoint', type=str, default='checkpoints_filler/best.pt')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--skel_steps', type=int, default=12)
    parser.add_argument('--filler_steps', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    print(f"Loading skeleton model from {args.skeleton_checkpoint}...")
    print(f"Loading filler model from {args.filler_checkpoint}...")

    skeleton_model, filler_model, skel_ckpt, filler_ckpt = load_twostage_models(
        args.skeleton_checkpoint,
        args.filler_checkpoint,
        device=device,
    )

    skel_vocab = skel_ckpt['vocab']
    filler_vocab = filler_ckpt['vocab']

    print(f"Skeleton vocab: {skel_vocab}")
    print(f"Filler vocab: {filler_vocab}")

    results = benchmark_twostage(
        skeleton_model,
        filler_model,
        skel_vocab,
        filler_vocab,
        num_samples=args.num_samples,
        skel_steps=args.skel_steps,
        filler_steps=args.filler_steps,
        device=device,
    )

    return results


if __name__ == "__main__":
    main()
