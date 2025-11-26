#!/usr/bin/env python3
"""
Phase 8: GSM8K Evaluation with Ouroboros Energy Ranking

Evaluates the Ouroboros energy head as a solution ranker on GSM8K math problems.

Approach:
1. Generate N candidate solutions per problem
2. Score each candidate with the energy head
3. Compare: greedy baseline vs min-energy selection
4. Prove that verification boosts reasoning accuracy

Metrics:
- Baseline: accuracy using first candidate (greedy)
- Ouroboros: accuracy using min-energy candidate
- Oracle: at least one candidate is correct
- Lift: (ouroboros - baseline) / baseline

Usage:
    python solve_math.py --config config/eval_gsm8k.py
    python solve_math.py --max_problems 50  # Quick test
    python solve_math.py --generator_type dummy  # Use dummy generator
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
from tqdm import tqdm

from model import OuroborosModel, OuroborosConfig
from tokenizer import OuroborosTokenizer
from utils import extract_final_answer, is_correct, format_prompt, parse_gsm8k_answer
from generator import load_generator, Generator


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Paths
    checkpoint_path: str = 'checkpoints/ckpt.pt'
    data_path: str = 'data/gsm8k/test.json'
    output_dir: str = 'results/gsm8k_eval'
    cache_dir: str = 'cache/generations'

    # Evaluation settings
    n_candidates: int = 5
    max_problems: Optional[int] = None
    batch_size: int = 16

    # Generator settings
    generator_type: str = 'huggingface'
    generator_model: str = 'google/gemma-3-1b-it'  # 62.8% GSM8K, 1B params
    generator_temperature: float = 0.7
    generator_max_tokens: int = 256

    # Device
    device: str = 'auto'

    # Reproducibility
    seed: int = 42

    # Logging
    verbose: bool = True
    save_per_problem_results: bool = True


@dataclass
class ProblemResult:
    """Result for a single problem."""
    idx: int
    question: str
    ground_truth: str
    candidates: List[str]
    energies: List[float]
    candidate_answers: List[str]
    candidate_correct: List[bool]
    baseline_idx: int
    ouroboros_idx: int
    baseline_correct: bool
    ouroboros_correct: bool
    oracle_correct: bool


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    n_problems: int
    n_candidates: int
    baseline_correct: int
    ouroboros_correct: int
    oracle_correct: int
    baseline_accuracy: float
    ouroboros_accuracy: float
    oracle_accuracy: float
    lift: float
    energy_correct_mean: float
    energy_correct_std: float
    energy_wrong_mean: float
    energy_wrong_std: float
    timestamp: str
    config: Dict[str, Any]


def load_checkpoint(ckpt_path: str, device: str = 'cpu') -> Tuple[OuroborosModel, dict]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = OuroborosConfig(**checkpoint['config'])
    model = OuroborosModel(config)

    # Handle torch.compile() prefix if present
    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Stripping _orig_mod. prefix from state_dict (torch.compile artifact)")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_gsm8k(data_path: str) -> List[Dict[str, str]]:
    """Load GSM8K test set."""
    print(f"Loading GSM8K from {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")
    return data


def score_candidates_batch(
    model: OuroborosModel,
    tokenizer: OuroborosTokenizer,
    questions: List[str],
    candidates_list: List[List[str]],
    device: str,
    max_ctx_len: int = 256,
    max_tgt_len: int = 256,
) -> List[List[float]]:
    """
    Score multiple questions and their candidates in parallel (Turbo Mode).

    Args:
        model: Trained Ouroboros model
        tokenizer: Tokenizer
        questions: List of questions
        candidates_list: List of lists of candidate solutions
        device: Device to use
        max_ctx_len: Max context length
        max_tgt_len: Max target length

    Returns:
        List of lists of energy scores (one list per question)
    """
    # Flatten for batch processing
    flat_questions = []
    flat_candidates = []
    for q, cands in zip(questions, candidates_list):
        flat_questions.extend([q] * len(cands))
        flat_candidates.extend(cands)

    if not flat_candidates:
        return [[] for _ in questions]

    # Encode contexts
    context_texts = [f"Question: {q}\nSolution: " for q in flat_questions]
    ctx_ids_list = [tokenizer.encode(t)[:max_ctx_len] for t in context_texts]
    tgt_ids_list = [tokenizer.encode(c)[:max_tgt_len] for c in flat_candidates]

    # Pad
    batch_size = len(ctx_ids_list)
    ctx_padded = torch.zeros((batch_size, max_ctx_len), dtype=torch.long, device=device)
    tgt_padded = torch.zeros((batch_size, max_tgt_len), dtype=torch.long, device=device)
    ctx_lens = torch.tensor([len(ids) for ids in ctx_ids_list], dtype=torch.long, device=device)
    tgt_lens = torch.tensor([len(ids) for ids in tgt_ids_list], dtype=torch.long, device=device)

    for i, (c_ids, t_ids) in enumerate(zip(ctx_ids_list, tgt_ids_list)):
        ctx_padded[i, :len(c_ids)] = torch.tensor(c_ids, dtype=torch.long, device=device)
        tgt_padded[i, :len(t_ids)] = torch.tensor(t_ids, dtype=torch.long, device=device)

    # Score in one forward pass
    with torch.no_grad():
        energies, _ = model(ctx_padded, tgt_padded, ctx_lens, tgt_lens)

    # Reshape back
    flat_energies = energies.cpu().numpy().tolist()

    reshaped_energies = []
    idx = 0
    for cands in candidates_list:
        n = len(cands)
        reshaped_energies.append(flat_energies[idx : idx + n])
        idx += n

    return reshaped_energies


def evaluate_problems_batch(
    batch_indices: List[int],
    batch_problems: List[Dict[str, str]],
    generator: Generator,
    model: OuroborosModel,
    tokenizer: OuroborosTokenizer,
    config: EvalConfig,
    device: str,
) -> List[ProblemResult]:
    """Evaluate a batch of problems with parallel generation."""
    questions = [p['question'] for p in batch_problems]
    ground_truths = [parse_gsm8k_answer(p['answer'])[1] for p in batch_problems]

    # Generate candidates for all questions in parallel
    all_candidates = generator.generate_batch(questions, config.n_candidates)

    # Score all candidates in parallel (Turbo Mode)
    all_energies = score_candidates_batch(
        model, tokenizer, questions, all_candidates, device
    )

    # PENALTY: If candidate lacks "####", set energy to high value (100.0)
    for i in range(len(all_energies)):
        candidates = all_candidates[i]
        for j, cand in enumerate(candidates):
            if "####" not in cand:
                all_energies[i][j] = 100.0

    results = []
    for i, (idx, problem, candidates, energies, ground_truth) in enumerate(
        zip(batch_indices, batch_problems, all_candidates, all_energies, ground_truths)
    ):
        question = problem['question']

        # PENALTY: If candidate lacks "####", set energy to high value (100.0)
        for j, cand in enumerate(candidates):
            if "####" not in cand:
                energies[j] = 100.0

        # Extract answers and check correctness
        candidate_answers = [extract_final_answer(c) or "" for c in candidates]
        candidate_correct = [is_correct(c, ground_truth) for c in candidates]

        # Compute metrics
        baseline_idx = 0
        ouroboros_idx = int(np.argmin(energies))

        results.append(ProblemResult(
            idx=idx,
            question=question,
            ground_truth=ground_truth,
            candidates=candidates,
            energies=energies,
            candidate_answers=candidate_answers,
            candidate_correct=candidate_correct,
            baseline_idx=baseline_idx,
            ouroboros_idx=ouroboros_idx,
            baseline_correct=candidate_correct[baseline_idx],
            ouroboros_correct=candidate_correct[ouroboros_idx],
            oracle_correct=any(candidate_correct),
        ))

    return results


def aggregate_results(
    problem_results: List[ProblemResult],
    config: EvalConfig,
) -> EvalResults:
    """Aggregate individual problem results."""
    n = len(problem_results)

    baseline_correct = sum(1 for r in problem_results if r.baseline_correct)
    ouroboros_correct = sum(1 for r in problem_results if r.ouroboros_correct)
    oracle_correct = sum(1 for r in problem_results if r.oracle_correct)

    baseline_accuracy = baseline_correct / n
    ouroboros_accuracy = ouroboros_correct / n
    oracle_accuracy = oracle_correct / n

    # Avoid division by zero
    lift = (ouroboros_accuracy - baseline_accuracy) / max(baseline_accuracy, 0.001)

    # Collect energy statistics
    all_energies_correct = []
    all_energies_wrong = []

    for r in problem_results:
        for energy, correct in zip(r.energies, r.candidate_correct):
            if correct:
                all_energies_correct.append(energy)
            else:
                all_energies_wrong.append(energy)

    energy_correct_mean = np.mean(all_energies_correct) if all_energies_correct else 0.0
    energy_correct_std = np.std(all_energies_correct) if len(all_energies_correct) > 1 else 0.0
    energy_wrong_mean = np.mean(all_energies_wrong) if all_energies_wrong else 0.0
    energy_wrong_std = np.std(all_energies_wrong) if len(all_energies_wrong) > 1 else 0.0

    return EvalResults(
        n_problems=n,
        n_candidates=config.n_candidates,
        baseline_correct=baseline_correct,
        ouroboros_correct=ouroboros_correct,
        oracle_correct=oracle_correct,
        baseline_accuracy=baseline_accuracy,
        ouroboros_accuracy=ouroboros_accuracy,
        oracle_accuracy=oracle_accuracy,
        lift=lift,
        energy_correct_mean=energy_correct_mean,
        energy_correct_std=energy_correct_std,
        energy_wrong_mean=energy_wrong_mean,
        energy_wrong_std=energy_wrong_std,
        timestamp=datetime.now().isoformat(),
        config=asdict(config),
    )


def print_results(results: EvalResults):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("GSM8K EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDataset: {results.n_problems} problems, {results.n_candidates} candidates each")

    print(f"\n--- ACCURACY ---")
    print(f"Baseline (greedy):       {results.baseline_correct:4d}/{results.n_problems} = {results.baseline_accuracy*100:5.1f}%")
    print(f"Ouroboros (min-energy):  {results.ouroboros_correct:4d}/{results.n_problems} = {results.ouroboros_accuracy*100:5.1f}%")
    print(f"Oracle (any correct):    {results.oracle_correct:4d}/{results.n_problems} = {results.oracle_accuracy*100:5.1f}%")

    print(f"\n--- IMPROVEMENT ---")
    print(f"Absolute lift: {(results.ouroboros_accuracy - results.baseline_accuracy)*100:+.1f}%")
    print(f"Relative lift: {results.lift*100:+.1f}%")

    print(f"\n--- ENERGY STATISTICS ---")
    print(f"Correct answers: mean={results.energy_correct_mean:.3f} +/- {results.energy_correct_std:.3f}")
    print(f"Wrong answers:   mean={results.energy_wrong_mean:.3f} +/- {results.energy_wrong_std:.3f}")
    print(f"Separation:      {results.energy_wrong_mean - results.energy_correct_mean:.3f}")

    print(f"\n--- SUCCESS CRITERIA ---")
    criteria = [
        ("Ouroboros > Baseline", results.ouroboros_accuracy > results.baseline_accuracy),
        ("Lift > 10%", results.lift > 0.10),
        ("Energy separation > 0.2", results.energy_wrong_mean - results.energy_correct_mean > 0.2),
    ]
    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print("\n" + "=" * 70)


def save_results(
    results: EvalResults,
    problem_results: List[ProblemResult],
    output_dir: str,
):
    """Save results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save aggregated results
    agg_path = output_path / 'results.json'
    with open(agg_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    # Save per-problem results (without full candidate text to save space)
    problems_path = output_path / 'problems.json'
    problem_data = []
    for r in problem_results:
        problem_data.append({
            'idx': r.idx,
            'question': r.question[:200] + '...' if len(r.question) > 200 else r.question,
            'ground_truth': r.ground_truth,
            'energies': r.energies,
            'candidate_answers': r.candidate_answers,
            'candidate_correct': r.candidate_correct,
            'baseline_correct': r.baseline_correct,
            'ouroboros_correct': r.ouroboros_correct,
            'oracle_correct': r.oracle_correct,
        })

    with open(problems_path, 'w') as f:
        json.dump(problem_data, f, indent=2)
    print(f"Per-problem results saved to {problems_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ouroboros on GSM8K')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ckpt.pt',
                        help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, default='data/gsm8k/test.json',
                        help='Path to GSM8K test data')
    parser.add_argument('--output_dir', type=str, default='results/gsm8k_eval',
                        help='Output directory')
    parser.add_argument('--n_candidates', type=int, default=5,
                        help='Number of candidates per problem')
    parser.add_argument('--max_problems', type=int, default=None,
                        help='Max problems to evaluate (None = all)')
    parser.add_argument('--generator_type', type=str, default='huggingface',
                        choices=['huggingface', 'openai', 'dummy'],
                        help='Generator type')
    parser.add_argument('--generator_model', type=str, default='google/gemma-3-1b-it',
                        help='Generator model name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    # Load config file if provided
    config = EvalConfig()
    if args.config and Path(args.config).exists():
        # Import config as module
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)

        # Override defaults with config values
        for key in vars(config):
            if hasattr(cfg_module, key):
                setattr(config, key, getattr(cfg_module, key))

    # Override with command line args
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.n_candidates:
        config.n_candidates = args.n_candidates
    if args.max_problems is not None:
        config.max_problems = args.max_problems
    if args.generator_type:
        config.generator_type = args.generator_type
    if args.generator_model:
        config.generator_model = args.generator_model
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed

    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Determine device
    device = config.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Device: {device}")

    # Load model
    model, checkpoint = load_checkpoint(config.checkpoint_path, device)
    tokenizer = OuroborosTokenizer()

    # Load generator
    print(f"\nLoading {config.generator_type} generator...")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    generator_config = {
        'type': config.generator_type,
        'model_name': config.generator_model,
        'temperature': config.generator_temperature,
        'max_new_tokens': config.generator_max_tokens,
        'device': device,
        'cache_enabled': True,
        'cache_dir': config.cache_dir,
        'hf_token': hf_token,
    }
    generator = load_generator(generator_config)

    # Load data
    problems = load_gsm8k(config.data_path)
    if config.max_problems:
        problems = problems[:config.max_problems]

    print(f"\nEvaluating {len(problems)} problems with {config.n_candidates} candidates each...")

    # Evaluate in batches for parallel generation
    problem_results = []
    start_time = time.time()
    batch_size = 16  # Process 16 questions at once (16 x 5 = 80 sequences)

    # Prepare incremental results file
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / 'results_incremental.jsonl'

    # Clear previous incremental results
    if jsonl_path.exists():
        jsonl_path.unlink()

    for batch_start in tqdm(range(0, len(problems), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(problems))
        batch_indices = list(range(batch_start, batch_end))
        batch_problems = problems[batch_start:batch_end]

        batch_results = evaluate_problems_batch(
            batch_indices, batch_problems, generator, model, tokenizer, config, device
        )
        problem_results.extend(batch_results)

        # Save results incrementally after each batch (crash-safe)
        with open(jsonl_path, 'a') as f:
            for result in batch_results:
                f.write(json.dumps(asdict(result)) + '\n')

        # Print progress every 100 problems
        if config.verbose and batch_end % 100 < batch_size:
            interim = aggregate_results(problem_results, config)
            print(f"\n[{batch_end}/{len(problems)}] "
                  f"Baseline: {interim.baseline_accuracy*100:.1f}%, "
                  f"Ouroboros: {interim.ouroboros_accuracy*100:.1f}%, "
                  f"Oracle: {interim.oracle_accuracy*100:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s ({elapsed/len(problems):.2f}s/problem)")

    # Aggregate and print results
    results = aggregate_results(problem_results, config)
    print_results(results)

    # Save results
    if config.save_per_problem_results:
        save_results(results, problem_results, config.output_dir)

    return results


if __name__ == "__main__":
    main()
