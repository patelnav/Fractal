#!/usr/bin/env python3
"""
Download GSM8K and HumanEval datasets for Ouroboros training.

GSM8K: Grade school math word problems with step-by-step solutions
HumanEval: Python function synthesis with unit tests
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_gsm8k():
    """Download GSM8K dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main")

    # Save train and test splits
    gsm8k_dir = DATA_DIR / "gsm8k"
    gsm8k_dir.mkdir(parents=True, exist_ok=True)

    train_data = []
    for item in dataset["train"]:
        train_data.append({
            "question": item["question"],
            "answer": item["answer"]  # Contains step-by-step solution + final answer
        })

    test_data = []
    for item in dataset["test"]:
        test_data.append({
            "question": item["question"],
            "answer": item["answer"]
        })

    with open(gsm8k_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(gsm8k_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"  GSM8K train: {len(train_data)} problems")
    print(f"  GSM8K test: {len(test_data)} problems")
    print(f"  Saved to {gsm8k_dir}")

    return train_data, test_data


def download_humaneval():
    """Download HumanEval dataset from HuggingFace."""
    from datasets import load_dataset

    print("\nDownloading HumanEval...")
    dataset = load_dataset("openai_humaneval")

    humaneval_dir = DATA_DIR / "humaneval"
    humaneval_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    for item in dataset["test"]:
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],           # Function signature + docstring
            "canonical_solution": item["canonical_solution"],  # Reference solution
            "test": item["test"],               # Unit tests
            "entry_point": item["entry_point"]  # Function name
        })

    with open(humaneval_dir / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)

    print(f"  HumanEval: {len(problems)} problems")
    print(f"  Saved to {humaneval_dir}")

    return problems


def show_examples():
    """Show example problems from each dataset."""
    print("\n" + "="*60)
    print("EXAMPLE: GSM8K")
    print("="*60)

    gsm8k_path = DATA_DIR / "gsm8k" / "train.json"
    if gsm8k_path.exists():
        with open(gsm8k_path) as f:
            data = json.load(f)
        ex = data[0]
        print(f"\nQuestion: {ex['question'][:200]}...")
        print(f"\nAnswer: {ex['answer'][:300]}...")

    print("\n" + "="*60)
    print("EXAMPLE: HumanEval")
    print("="*60)

    humaneval_path = DATA_DIR / "humaneval" / "problems.json"
    if humaneval_path.exists():
        with open(humaneval_path) as f:
            data = json.load(f)
        ex = data[0]
        print(f"\nPrompt:\n{ex['prompt']}")
        print(f"\nSolution:\n{ex['canonical_solution'][:200]}...")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_gsm8k()
    download_humaneval()
    show_examples()

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
