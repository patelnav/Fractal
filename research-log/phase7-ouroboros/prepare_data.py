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


@dataclass
class ContrastivePair:
    """A paired sample for contrastive training (same context, correct vs wrong)."""
    context_ids: List[int]         # Question/prompt tokens (shared)
    correct_target_ids: List[int]  # Correct solution
    wrong_target_ids: List[int]    # Wrong solution
    domain: str                    # "math" or "code"


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


def generate_math_pairs(
    data: List[Dict],
    tokenizer: OuroborosTokenizer,
    max_seq_len: int = 512
) -> List[ContrastivePair]:
    """Generate contrastive PAIRS from GSM8K data (same context, correct vs wrong)."""
    pairs = []

    for item in tqdm(data, desc="Math pairs"):
        question = item["question"]
        answer = item["answer"]

        # Get 1 correct and 1 wrong
        contrastive = create_math_contrastive_pairs(question, answer, num_wrong=1)

        if len(contrastive) < 2:
            continue

        _, correct_sol, _ = contrastive[0]  # First is always correct
        _, wrong_sol, _ = contrastive[1]    # Second is wrong

        # Tokenize
        context_ids = tokenizer.encode(f"Question: {question}\nSolution: ")
        correct_target_ids = tokenizer.encode(correct_sol)
        wrong_target_ids = tokenizer.encode(wrong_sol)

        # Skip if too long
        if len(context_ids) + max(len(correct_target_ids), len(wrong_target_ids)) > max_seq_len:
            continue

        pairs.append(ContrastivePair(
            context_ids=context_ids,
            correct_target_ids=correct_target_ids,
            wrong_target_ids=wrong_target_ids,
            domain="math"
        ))

    return pairs


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


def generate_code_pairs(
    data: List[Dict],
    tokenizer: OuroborosTokenizer,
    max_seq_len: int = 512
) -> List[ContrastivePair]:
    """Generate contrastive PAIRS from HumanEval data (same context, correct vs wrong)."""
    pairs = []

    for item in tqdm(data, desc="Code pairs"):
        prompt = item["prompt"]
        solution = item["canonical_solution"]
        test = item.get("test", "")

        # Get 1 correct and 1 wrong
        contrastive = create_code_contrastive_pairs(prompt, solution, test, num_wrong=1)

        if len(contrastive) < 2:
            continue

        _, correct_code, _ = contrastive[0]  # First is always correct
        _, wrong_code, _ = contrastive[1]    # Second is wrong

        # Tokenize
        context_ids = tokenizer.encode(prompt)
        correct_target_ids = tokenizer.encode(correct_code)
        wrong_target_ids = tokenizer.encode(wrong_code)

        # Skip if too long
        if len(context_ids) + max(len(correct_target_ids), len(wrong_target_ids)) > max_seq_len:
            continue

        pairs.append(ContrastivePair(
            context_ids=context_ids,
            correct_target_ids=correct_target_ids,
            wrong_target_ids=wrong_target_ids,
            domain="code"
        ))

    return pairs


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


def pairs_to_arrays(
    pairs: List[ContrastivePair],
    max_context_len: int = 256,
    max_target_len: int = 256,
    pad_id: int = 0
) -> Dict[str, np.ndarray]:
    """Convert pairs to padded numpy arrays for paired training."""
    n = len(pairs)

    # Each pair has same context, but correct and wrong targets
    contexts = np.full((n, max_context_len), pad_id, dtype=np.uint16)
    correct_targets = np.full((n, max_target_len), pad_id, dtype=np.uint16)
    wrong_targets = np.full((n, max_target_len), pad_id, dtype=np.uint16)
    context_lens = np.zeros(n, dtype=np.uint16)
    correct_target_lens = np.zeros(n, dtype=np.uint16)
    wrong_target_lens = np.zeros(n, dtype=np.uint16)
    domains = np.zeros(n, dtype=np.uint8)  # 0 = math, 1 = code

    for i, pair in enumerate(pairs):
        ctx_len = min(len(pair.context_ids), max_context_len)
        correct_len = min(len(pair.correct_target_ids), max_target_len)
        wrong_len = min(len(pair.wrong_target_ids), max_target_len)

        contexts[i, :ctx_len] = pair.context_ids[:ctx_len]
        correct_targets[i, :correct_len] = pair.correct_target_ids[:correct_len]
        wrong_targets[i, :wrong_len] = pair.wrong_target_ids[:wrong_len]
        context_lens[i] = ctx_len
        correct_target_lens[i] = correct_len
        wrong_target_lens[i] = wrong_len
        domains[i] = 0 if pair.domain == "math" else 1

    return {
        "contexts": contexts,
        "correct_targets": correct_targets,
        "wrong_targets": wrong_targets,
        "context_lens": context_lens,
        "correct_target_lens": correct_target_lens,
        "wrong_target_lens": wrong_target_lens,
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
    print("PREPARING OUROBOROS TRAINING DATA (PAIRED FORMAT)")
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

    # Generate contrastive PAIRS (same context, correct vs wrong)
    print("\nGenerating contrastive pairs...")

    # Math pairs (train)
    math_train_pairs = generate_math_pairs(gsm8k_train, tokenizer, max_seq_len=512)
    print(f"  Math train: {len(math_train_pairs)} pairs")

    # Math pairs (test -> validation)
    math_val_pairs = generate_math_pairs(gsm8k_test, tokenizer, max_seq_len=512)
    print(f"  Math val: {len(math_val_pairs)} pairs")

    # Code pairs (split 80/20 for train/val since HumanEval is small)
    code_pairs = generate_code_pairs(humaneval, tokenizer, max_seq_len=512)
    random.shuffle(code_pairs)
    split_idx = int(len(code_pairs) * 0.8)
    code_train_pairs = code_pairs[:split_idx]
    code_val_pairs = code_pairs[split_idx:]
    print(f"  Code train: {len(code_train_pairs)} pairs")
    print(f"  Code val: {len(code_val_pairs)} pairs")

    # Combine pairs
    train_pairs = math_train_pairs + code_train_pairs
    val_pairs = math_val_pairs + code_val_pairs

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    print(f"\nTotal train: {len(train_pairs)} pairs")
    print(f"Total val: {len(val_pairs)} pairs")

    # Check domain balance
    train_math = sum(1 for p in train_pairs if p.domain == "math")
    val_math = sum(1 for p in val_pairs if p.domain == "math")

    print(f"\nTrain domain:")
    print(f"  Math: {train_math} ({100*train_math/len(train_pairs):.1f}%)")
    print(f"  Code: {len(train_pairs) - train_math} ({100*(len(train_pairs)-train_math)/len(train_pairs):.1f}%)")
    print(f"\nVal domain:")
    print(f"  Math: {val_math} ({100*val_math/len(val_pairs):.1f}%)")
    print(f"  Code: {len(val_pairs) - val_math} ({100*(len(val_pairs)-val_math)/len(val_pairs):.1f}%)")

    # Convert to arrays
    print("\nConverting to arrays...")
    train_arrays = pairs_to_arrays(train_pairs)
    val_arrays = pairs_to_arrays(val_pairs)

    # Save
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out_dir}...")

    # Save as .npz for easy loading (paired format)
    np.savez(out_dir / "train.npz", **train_arrays)
    np.savez(out_dir / "val.npz", **val_arrays)

    print(f"  train.npz: {(out_dir / 'train.npz').stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  val.npz: {(out_dir / 'val.npz').stat().st_size / 1024 / 1024:.1f} MB")

    # Save metadata
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "max_context_len": 256,
        "max_target_len": 256,
        "train_math_ratio": train_math / len(train_pairs),
        "format": "paired",  # Indicates paired format for training
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    # Show example pairs
    print("\nExample pairs:")
    for i in range(min(3, len(train_pairs))):
        p = train_pairs[i]
        ctx = tokenizer.decode(p.context_ids[:60])
        correct = tokenizer.decode(p.correct_target_ids[:50])
        wrong = tokenizer.decode(p.wrong_target_ids[:50])
        print(f"\n[{p.domain.upper()}] Pair {i}")
        print(f"  Context: {ctx}...")
        print(f"  Correct: {correct}...")
        print(f"  Wrong: {wrong}...")


if __name__ == "__main__":
    main()
