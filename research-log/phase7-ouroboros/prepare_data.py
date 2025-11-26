#!/usr/bin/env python3
"""
Prepare training data for Ouroboros.

Generates contrastive pairs (correct/wrong) from GSM8K and HumanEval,
then saves as train.bin/val.bin (nanoGPT-compatible format).
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm

from tokenizer import (
    OuroborosTokenizer,
    create_math_contrastive_pairs,
    create_code_contrastive_pairs,
)

DATA_DIR = Path(__file__).parent / "data"
SEED = 42


@dataclass
class ContrastiveSample:
    """A single contrastive training sample."""
    context_ids: List[int]  # Question/prompt tokens
    target_ids: List[int]   # Solution/code tokens
    is_correct: bool        # True = correct, False = wrong
    domain: str             # "math" or "code"


def load_gsm8k() -> Tuple[List[Dict], List[Dict]]:
    """Load GSM8K train and test sets."""
    train_path = DATA_DIR / "gsm8k" / "train.json"
    test_path = DATA_DIR / "gsm8k" / "test.json"

    with open(train_path) as f:
        train = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    return train, test


def load_humaneval() -> List[Dict]:
    """Load HumanEval problems."""
    path = DATA_DIR / "humaneval" / "problems.json"
    with open(path) as f:
        return json.load(f)


def generate_math_samples(
    data: List[Dict],
    tokenizer: OuroborosTokenizer,
    num_wrong_per_correct: int = 3,
    max_seq_len: int = 512
) -> List[ContrastiveSample]:
    """Generate contrastive samples from GSM8K data."""
    samples = []

    for item in tqdm(data, desc="Math samples"):
        question = item["question"]
        answer = item["answer"]

        pairs = create_math_contrastive_pairs(question, answer, num_wrong_per_correct)

        for q, sol, is_correct in pairs:
            # Tokenize
            context_ids = tokenizer.encode(f"Question: {q}\nSolution: ")
            target_ids = tokenizer.encode(sol)

            # Skip if too long
            if len(context_ids) + len(target_ids) > max_seq_len:
                continue

            samples.append(ContrastiveSample(
                context_ids=context_ids,
                target_ids=target_ids,
                is_correct=is_correct,
                domain="math"
            ))

    return samples


def generate_code_samples(
    data: List[Dict],
    tokenizer: OuroborosTokenizer,
    num_wrong_per_correct: int = 3,
    max_seq_len: int = 512
) -> List[ContrastiveSample]:
    """Generate contrastive samples from HumanEval data."""
    samples = []

    for item in tqdm(data, desc="Code samples"):
        prompt = item["prompt"]
        solution = item["canonical_solution"]
        test = item.get("test", "")

        pairs = create_code_contrastive_pairs(prompt, solution, test, num_wrong_per_correct)

        for p, code, is_correct in pairs:
            # Tokenize
            context_ids = tokenizer.encode(p)
            target_ids = tokenizer.encode(code)

            # Skip if too long
            if len(context_ids) + len(target_ids) > max_seq_len:
                continue

            samples.append(ContrastiveSample(
                context_ids=context_ids,
                target_ids=target_ids,
                is_correct=is_correct,
                domain="code"
            ))

    return samples


def samples_to_arrays(
    samples: List[ContrastiveSample],
    max_context_len: int = 256,
    max_target_len: int = 256,
    pad_id: int = 0
) -> Dict[str, np.ndarray]:
    """Convert samples to padded numpy arrays."""
    n = len(samples)

    contexts = np.full((n, max_context_len), pad_id, dtype=np.uint16)
    targets = np.full((n, max_target_len), pad_id, dtype=np.uint16)
    context_lens = np.zeros(n, dtype=np.uint16)
    target_lens = np.zeros(n, dtype=np.uint16)
    labels = np.zeros(n, dtype=np.uint8)  # 0 = wrong, 1 = correct
    domains = np.zeros(n, dtype=np.uint8)  # 0 = math, 1 = code

    for i, sample in enumerate(samples):
        ctx_len = min(len(sample.context_ids), max_context_len)
        tgt_len = min(len(sample.target_ids), max_target_len)

        contexts[i, :ctx_len] = sample.context_ids[:ctx_len]
        targets[i, :tgt_len] = sample.target_ids[:tgt_len]
        context_lens[i] = ctx_len
        target_lens[i] = tgt_len
        labels[i] = 1 if sample.is_correct else 0
        domains[i] = 0 if sample.domain == "math" else 1

    return {
        "contexts": contexts,
        "targets": targets,
        "context_lens": context_lens,
        "target_lens": target_lens,
        "labels": labels,
        "domains": domains
    }


def balance_samples(samples: List[ContrastiveSample]) -> List[ContrastiveSample]:
    """Balance samples to have equal correct and wrong examples."""
    correct = [s for s in samples if s.is_correct]
    wrong = [s for s in samples if not s.is_correct]

    # Undersample the majority class
    min_count = min(len(correct), len(wrong))
    random.shuffle(correct)
    random.shuffle(wrong)

    balanced = correct[:min_count] + wrong[:min_count]
    random.shuffle(balanced)

    return balanced


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("PREPARING OUROBOROS TRAINING DATA (BALANCED)")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = OuroborosTokenizer()
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")

    # Load datasets
    print("\nLoading datasets...")
    gsm8k_train, gsm8k_test = load_gsm8k()
    humaneval = load_humaneval()

    print(f"  GSM8K train: {len(gsm8k_train)} problems")
    print(f"  GSM8K test: {len(gsm8k_test)} problems")
    print(f"  HumanEval: {len(humaneval)} problems")

    # Generate contrastive samples with 1:1 ratio
    print("\nGenerating contrastive samples (1:1 correct:wrong)...")

    # Math samples (train) - 1 wrong per correct for balance
    math_train_samples = generate_math_samples(
        gsm8k_train,
        tokenizer,
        num_wrong_per_correct=1,  # Changed from 3 to 1 for balance
        max_seq_len=512
    )
    print(f"  Math train (raw): {len(math_train_samples)} samples")

    # Math samples (test -> validation)
    math_val_samples = generate_math_samples(
        gsm8k_test,
        tokenizer,
        num_wrong_per_correct=1,  # Changed from 3 to 1 for balance
        max_seq_len=512
    )
    print(f"  Math val (raw): {len(math_val_samples)} samples")

    # Code samples (split 80/20 for train/val since HumanEval is small)
    code_samples = generate_code_samples(
        humaneval,
        tokenizer,
        num_wrong_per_correct=1,  # Changed from 3 to 1 for balance
        max_seq_len=512
    )
    random.shuffle(code_samples)
    split_idx = int(len(code_samples) * 0.8)
    code_train_samples = code_samples[:split_idx]
    code_val_samples = code_samples[split_idx:]
    print(f"  Code train (raw): {len(code_train_samples)} samples")
    print(f"  Code val (raw): {len(code_val_samples)} samples")

    # Combine samples
    train_samples = math_train_samples + code_train_samples
    val_samples = math_val_samples + code_val_samples

    # Balance the samples (equal correct/wrong)
    train_samples = balance_samples(train_samples)
    val_samples = balance_samples(val_samples)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\nTotal train: {len(train_samples)} samples")
    print(f"Total val: {len(val_samples)} samples")

    # Check balance
    train_correct = sum(1 for s in train_samples if s.is_correct)
    train_math = sum(1 for s in train_samples if s.domain == "math")
    val_correct = sum(1 for s in val_samples if s.is_correct)
    val_math = sum(1 for s in val_samples if s.domain == "math")

    print(f"\nTrain balance:")
    print(f"  Correct: {train_correct} ({100*train_correct/len(train_samples):.1f}%)")
    print(f"  Math: {train_math} ({100*train_math/len(train_samples):.1f}%)")
    print(f"\nVal balance:")
    print(f"  Correct: {val_correct} ({100*val_correct/len(val_samples):.1f}%)")
    print(f"  Math: {val_math} ({100*val_math/len(val_samples):.1f}%)")

    # Convert to arrays
    print("\nConverting to arrays...")
    train_arrays = samples_to_arrays(train_samples)
    val_arrays = samples_to_arrays(val_samples)

    # Save
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out_dir}...")

    # Save as .npz for easy loading
    np.savez(
        out_dir / "train.npz",
        **train_arrays
    )
    np.savez(
        out_dir / "val.npz",
        **val_arrays
    )

    # Also save as .bin for nanoGPT compatibility (just contexts+targets concatenated)
    # This is simpler - just a flat array of token ids
    train_concat = []
    for i in range(len(train_samples)):
        ctx_len = train_arrays["context_lens"][i]
        tgt_len = train_arrays["target_lens"][i]
        train_concat.extend(train_arrays["contexts"][i, :ctx_len].tolist())
        train_concat.extend(train_arrays["targets"][i, :tgt_len].tolist())

    val_concat = []
    for i in range(len(val_samples)):
        ctx_len = val_arrays["context_lens"][i]
        tgt_len = val_arrays["target_lens"][i]
        val_concat.extend(val_arrays["contexts"][i, :ctx_len].tolist())
        val_concat.extend(val_arrays["targets"][i, :tgt_len].tolist())

    np.array(train_concat, dtype=np.uint16).tofile(out_dir / "train.bin")
    np.array(val_concat, dtype=np.uint16).tofile(out_dir / "val.bin")

    print(f"  train.npz: {(out_dir / 'train.npz').stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  val.npz: {(out_dir / 'val.npz').stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  train.bin: {(out_dir / 'train.bin').stat().st_size / 1024 / 1024:.1f} MB ({len(train_concat):,} tokens)")
    print(f"  val.bin: {(out_dir / 'val.bin').stat().st_size / 1024 / 1024:.1f} MB ({len(val_concat):,} tokens)")

    # Save metadata
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "max_context_len": 256,
        "max_target_len": 256,
        "train_correct_ratio": train_correct / len(train_samples),
        "train_math_ratio": train_math / len(train_samples),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    # Show examples
    print("\nExample samples:")
    for i in range(min(3, len(train_samples))):
        s = train_samples[i]
        ctx = tokenizer.decode(s.context_ids[:50])
        tgt = tokenizer.decode(s.target_ids[:50])
        print(f"\n[{s.domain.upper()}] {'CORRECT' if s.is_correct else 'WRONG'}")
        print(f"  Context: {ctx}...")
        print(f"  Target: {tgt}...")


if __name__ == "__main__":
    main()
