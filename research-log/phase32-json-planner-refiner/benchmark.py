"""
Benchmark for JSON Repair Engine.

Evaluates:
1. ParseSuccess@1: Parse success after single repair
2. ParseSuccess@K: Parse success after K iterations
3. Locality: Fraction of tokens unchanged outside error window
4. Edit distance: Number of tokens changed

Baselines:
1. Heuristic fixer: Try inserting/removing common punctuation at error
2. Do nothing: Return original (parse success of corrupted = baseline)
"""

import json
import time
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch

from tokenizer_json import JSONTokenizer
from data_json import JSONRepairDataset, JSONEvalDataset, generate_random_json, JSONCorruptionEngine
from model_denoiser import JSONDenoiser, JSONDenoiserWithEnergy, JSONDenoiserConfig
from inference_repair import repair_json, repair_json_full_denoise, RepairResult, get_parse_error_position


@dataclass
class BenchmarkResult:
    """Results from benchmark run."""
    name: str
    parse_success_1: float      # ParseSuccess@1
    parse_success_k: float      # ParseSuccess@K (K iterations)
    locality: float             # Fraction of tokens unchanged
    avg_edits: float            # Average tokens changed
    avg_time_ms: float          # Average time per repair (ms)
    by_corruption: Dict[str, Dict[str, float]]  # Per-corruption-type metrics


# ============================================================================
# Baseline 1: Heuristic Fixer
# ============================================================================

def heuristic_repair(broken_json: str, max_attempts: int = 10) -> Tuple[str, bool]:
    """
    Simple heuristic-based JSON repair.

    Tries common fixes at the error location:
    1. Insert missing comma
    2. Insert missing colon
    3. Insert missing brace/bracket
    4. Remove extra comma
    5. Remove extra brace/bracket
    """
    # If already valid, return as-is
    if get_parse_error_position(broken_json) is None:
        return broken_json, True

    current = broken_json

    for attempt in range(max_attempts):
        error_pos = get_parse_error_position(current)
        if error_pos is None:
            return current, True

        # Clamp position
        pos = min(error_pos, len(current))

        # Try fixes in order of likelihood
        fixes = [
            # Insert fixes
            lambda s, p: s[:p] + ',' + s[p:],
            lambda s, p: s[:p] + ':' + s[p:],
            lambda s, p: s[:p] + '"' + s[p:],
            lambda s, p: s[:p] + '}' + s[p:],
            lambda s, p: s[:p] + ']' + s[p:],
            lambda s, p: s[:p] + '{' + s[p:],
            lambda s, p: s[:p] + '[' + s[p:],
            # Delete fixes (at pos-1 to remove the problematic char)
            lambda s, p: s[:max(0, p-1)] + s[p:] if p > 0 else s,
            lambda s, p: s[:p] + s[p+1:] if p < len(s) else s,
        ]

        for fix_fn in fixes:
            try:
                fixed = fix_fn(current, pos)
                if get_parse_error_position(fixed) is None:
                    return fixed, True
            except Exception:
                continue

        # No simple fix worked, try next position
        if pos < len(current) - 1:
            current = current[:pos] + current[pos+1:]  # Delete char at error
        else:
            break

    return current, False


# ============================================================================
# Baseline 2: Do Nothing
# ============================================================================

def do_nothing_repair(broken_json: str) -> Tuple[str, bool]:
    """Baseline: just return the input."""
    success = get_parse_error_position(broken_json) is None
    return broken_json, success


# ============================================================================
# Benchmark Runner
# ============================================================================

def benchmark_method(
    method_name: str,
    repair_fn,  # Function: str -> (str, bool)
    test_samples: List[Tuple[str, str, str]],  # (clean_json, corrupted_json, corruption_type)
    tokenizer: JSONTokenizer,
) -> BenchmarkResult:
    """
    Run benchmark on a repair method.

    Args:
        method_name: Name of the method
        repair_fn: Function that takes broken JSON and returns (repaired, success)
        test_samples: List of (clean_json, corrupted_json, corruption_type)
        tokenizer: JSONTokenizer

    Returns:
        BenchmarkResult
    """
    parse_success = 0
    total_locality = 0.0
    total_edits = 0
    total_time = 0.0
    by_corruption = defaultdict(lambda: {'success': 0, 'total': 0, 'edits': []})

    for clean_json, corrupted_json, ctype in test_samples:
        # Time the repair
        start = time.time()
        repaired, success = repair_fn(corrupted_json)
        elapsed = (time.time() - start) * 1000  # ms

        total_time += elapsed

        # Parse success
        if success:
            parse_success += 1
            by_corruption[ctype]['success'] += 1

        by_corruption[ctype]['total'] += 1

        # Locality and edit distance (tokenize both)
        clean_ids = tokenizer.tokenize(clean_json)
        corrupted_ids = tokenizer.tokenize(corrupted_json)
        repaired_ids = tokenizer.tokenize(repaired)

        # Pad to same length for comparison
        max_len = max(len(clean_ids), len(corrupted_ids), len(repaired_ids))
        clean_ids = clean_ids + [0] * (max_len - len(clean_ids))
        corrupted_ids = corrupted_ids + [0] * (max_len - len(corrupted_ids))
        repaired_ids = repaired_ids + [0] * (max_len - len(repaired_ids))

        # Count changes from corrupted to repaired
        edits = sum(1 for a, b in zip(corrupted_ids, repaired_ids) if a != b)
        total_edits += edits
        by_corruption[ctype]['edits'].append(edits)

        # Locality: fraction of clean tokens preserved in repair
        preserved = sum(1 for a, b in zip(clean_ids, repaired_ids) if a == b and a != 0)
        total_clean = sum(1 for a in clean_ids if a != 0)
        locality = preserved / total_clean if total_clean > 0 else 1.0
        total_locality += locality

    n = len(test_samples)

    # Aggregate by-corruption metrics
    by_corruption_summary = {}
    for ctype, metrics in by_corruption.items():
        by_corruption_summary[ctype] = {
            'parse_success': metrics['success'] / metrics['total'] if metrics['total'] > 0 else 0,
            'avg_edits': sum(metrics['edits']) / len(metrics['edits']) if metrics['edits'] else 0,
        }

    return BenchmarkResult(
        name=method_name,
        parse_success_1=parse_success / n,
        parse_success_k=parse_success / n,  # Same for single-iteration methods
        locality=total_locality / n,
        avg_edits=total_edits / n,
        avg_time_ms=total_time / n,
        by_corruption=by_corruption_summary,
    )


def benchmark_denoiser_direct(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    num_samples: int = 200,
    device: str = 'cpu',
    max_len: int = 256,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Benchmark denoiser using direct token pipeline (no re-tokenization).

    This avoids the double-tokenization issue in the string-based benchmark.
    """
    import torch
    from data_json import JSONEvalDataset

    parse_success = 0
    total_locality = 0.0
    total_edits = 0
    total_time = 0.0
    by_corruption = defaultdict(lambda: {'success': 0, 'total': 0, 'edits': []})

    eval_dataset = JSONEvalDataset(num_samples=num_samples, max_len=max_len, seed=seed)

    for i in range(len(eval_dataset)):
        clean, corrupted, sigma, ctype = eval_dataset[i]

        start = time.time()

        with torch.no_grad():
            input_t = corrupted.unsqueeze(0).to(device)
            sigma_t = sigma.unsqueeze(0).to(device)
            logits, _ = model(input_t, sigma_t)
            pred = logits.argmax(dim=-1)

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        pred_json = tokenizer.detokenize(pred[0].tolist())

        try:
            json.loads(pred_json)
            parse_success += 1
            by_corruption[ctype]['success'] += 1
        except json.JSONDecodeError:
            pass

        by_corruption[ctype]['total'] += 1

        # Token-level metrics
        mask = clean != 0
        edits = ((pred[0].cpu() != clean) & mask).sum().item()
        total_edits += edits
        by_corruption[ctype]['edits'].append(edits)

        # Locality
        preserved = ((pred[0].cpu() == clean) & mask).sum().item()
        total_clean = mask.sum().item()
        locality = preserved / total_clean if total_clean > 0 else 1.0
        total_locality += locality

    n = len(eval_dataset)

    by_corruption_summary = {}
    for ctype, metrics in by_corruption.items():
        by_corruption_summary[ctype] = {
            'parse_success': metrics['success'] / metrics['total'] if metrics['total'] > 0 else 0,
            'avg_edits': sum(metrics['edits']) / len(metrics['edits']) if metrics['edits'] else 0,
        }

    return BenchmarkResult(
        name='Denoiser-Direct',
        parse_success_1=parse_success / n,
        parse_success_k=parse_success / n,
        locality=total_locality / n,
        avg_edits=total_edits / n,
        avg_time_ms=total_time / n,
        by_corruption=by_corruption_summary,
    )


def benchmark_denoiser_full(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    test_samples: List[Tuple[str, str, str]],
    device: str = 'cpu',
    max_len: int = 128,
) -> BenchmarkResult:
    """
    Benchmark the denoiser model using full-sequence denoising (string-based).

    Note: This uses a string-based pipeline which may have lower accuracy than
    benchmark_denoiser_direct due to tokenization round-trip issues.
    """
    parse_success = 0
    total_locality = 0.0
    total_edits = 0
    total_time = 0.0
    by_corruption = defaultdict(lambda: {'success': 0, 'total': 0, 'edits': []})

    for clean_json, corrupted_json, ctype in test_samples:
        start = time.time()
        result = repair_json_full_denoise(
            model=model,
            tokenizer=tokenizer,
            broken_json=corrupted_json,
            sigma=0.2,
            device=device,
            max_len=max_len,
        )
        elapsed = (time.time() - start) * 1000

        total_time += elapsed

        if result.success:
            parse_success += 1
            by_corruption[ctype]['success'] += 1

        by_corruption[ctype]['total'] += 1

        # Edit distance
        total_edits += result.tokens_changed
        by_corruption[ctype]['edits'].append(result.tokens_changed)

        # Locality
        clean_ids = tokenizer.tokenize(clean_json)
        repaired_ids = tokenizer.tokenize(result.repaired)
        max_len_cmp = max(len(clean_ids), len(repaired_ids))
        clean_ids = clean_ids + [0] * (max_len_cmp - len(clean_ids))
        repaired_ids = repaired_ids + [0] * (max_len_cmp - len(repaired_ids))

        preserved = sum(1 for a, b in zip(clean_ids, repaired_ids) if a == b and a != 0)
        total_clean = sum(1 for a in clean_ids if a != 0)
        locality = preserved / total_clean if total_clean > 0 else 1.0
        total_locality += locality

    n = len(test_samples)

    by_corruption_summary = {}
    for ctype, metrics in by_corruption.items():
        by_corruption_summary[ctype] = {
            'parse_success': metrics['success'] / metrics['total'] if metrics['total'] > 0 else 0,
            'avg_edits': sum(metrics['edits']) / len(metrics['edits']) if metrics['edits'] else 0,
        }

    return BenchmarkResult(
        name='Denoiser-Full',
        parse_success_1=parse_success / n,
        parse_success_k=parse_success / n,
        locality=total_locality / n,
        avg_edits=total_edits / n,
        avg_time_ms=total_time / n,
        by_corruption=by_corruption_summary,
    )


def benchmark_denoiser(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    test_samples: List[Tuple[str, str, str]],
    max_iterations: int = 3,
    device: str = 'cpu',
    max_len: int = 128,
) -> BenchmarkResult:
    """
    Benchmark the denoiser model.

    Returns both ParseSuccess@1 and ParseSuccess@K.
    """
    parse_success_1 = 0
    parse_success_k = 0
    total_locality = 0.0
    total_edits = 0
    total_time = 0.0
    by_corruption = defaultdict(lambda: {'success_1': 0, 'success_k': 0, 'total': 0, 'edits': []})

    for clean_json, corrupted_json, ctype in test_samples:
        # Try 1 iteration
        start = time.time()
        result_1 = repair_json(
            model=model,
            tokenizer=tokenizer,
            broken_json=corrupted_json,
            max_iterations=1,
            initial_window_size=5,
            sigma=0.3,
            device=device,
            max_len=max_len,
        )
        time_1 = (time.time() - start) * 1000

        # Try K iterations
        start = time.time()
        result_k = repair_json(
            model=model,
            tokenizer=tokenizer,
            broken_json=corrupted_json,
            max_iterations=max_iterations,
            initial_window_size=5,
            sigma=0.3,
            device=device,
            max_len=max_len,
        )
        time_k = (time.time() - start) * 1000

        total_time += time_k

        if result_1.success:
            parse_success_1 += 1
            by_corruption[ctype]['success_1'] += 1

        if result_k.success:
            parse_success_k += 1
            by_corruption[ctype]['success_k'] += 1

        by_corruption[ctype]['total'] += 1

        # Edit distance
        total_edits += result_k.tokens_changed
        by_corruption[ctype]['edits'].append(result_k.tokens_changed)

        # Locality
        clean_ids = tokenizer.tokenize(clean_json)
        repaired_ids = tokenizer.tokenize(result_k.repaired)
        max_len = max(len(clean_ids), len(repaired_ids))
        clean_ids = clean_ids + [0] * (max_len - len(clean_ids))
        repaired_ids = repaired_ids + [0] * (max_len - len(repaired_ids))

        preserved = sum(1 for a, b in zip(clean_ids, repaired_ids) if a == b and a != 0)
        total_clean = sum(1 for a in clean_ids if a != 0)
        locality = preserved / total_clean if total_clean > 0 else 1.0
        total_locality += locality

    n = len(test_samples)

    by_corruption_summary = {}
    for ctype, metrics in by_corruption.items():
        by_corruption_summary[ctype] = {
            'parse_success_1': metrics['success_1'] / metrics['total'] if metrics['total'] > 0 else 0,
            'parse_success_k': metrics['success_k'] / metrics['total'] if metrics['total'] > 0 else 0,
            'avg_edits': sum(metrics['edits']) / len(metrics['edits']) if metrics['edits'] else 0,
        }

    return BenchmarkResult(
        name='Denoiser',
        parse_success_1=parse_success_1 / n,
        parse_success_k=parse_success_k / n,
        locality=total_locality / n,
        avg_edits=total_edits / n,
        avg_time_ms=total_time / n,
        by_corruption=by_corruption_summary,
    )


def generate_test_samples(
    num_samples: int = 200,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Generate test samples for benchmarking."""
    import random
    random.seed(seed)

    tokenizer = JSONTokenizer()
    corruption_engine = JSONCorruptionEngine(tokenizer)

    corruption_types = [
        'delete_comma',
        'insert_comma',
        'delete_colon',
        'delete_brace',
        'delete_bracket',
        'swap_adjacent',
        'mask_token',
    ]

    samples_per_type = num_samples // len(corruption_types)
    samples = []

    for ctype in corruption_types:
        for _ in range(samples_per_type):
            # Generate clean JSON
            clean_json = generate_random_json(max_depth=2)

            # Tokenize and corrupt
            clean_ids = tokenizer.tokenize(clean_json)
            corrupted_ids, _ = corruption_engine.corrupt(clean_ids, sigma=0.2, corruption_type=ctype)

            # Decode back to JSON
            corrupted_json = tokenizer.detokenize(corrupted_ids)

            samples.append((clean_json, corrupted_json, ctype))

    random.seed()
    return samples


def run_benchmark(
    model_path: Optional[str] = None,
    num_samples: int = 200,
    device: str = 'cpu',
) -> Dict[str, BenchmarkResult]:
    """
    Run full benchmark comparing methods.

    Args:
        model_path: Path to trained model checkpoint (optional)
        num_samples: Number of test samples
        device: Device for model inference

    Returns:
        Dict mapping method name to BenchmarkResult
    """
    print("Generating test samples...")
    samples = generate_test_samples(num_samples)
    print(f"Generated {len(samples)} test samples")

    tokenizer = JSONTokenizer()
    results = {}

    # Baseline 1: Do Nothing
    print("\nBenchmarking: Do Nothing...")
    results['do_nothing'] = benchmark_method(
        'Do Nothing',
        do_nothing_repair,
        samples,
        tokenizer,
    )

    # Baseline 2: Heuristic
    print("Benchmarking: Heuristic...")
    results['heuristic'] = benchmark_method(
        'Heuristic',
        heuristic_repair,
        samples,
        tokenizer,
    )

    # Denoiser (if model provided)
    if model_path:
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        max_len = config.block_size  # Use model's block_size as max_len

        model = JSONDenoiser(config).to(device)

        # Handle torch.compile() prefix in state dict
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        print(f"Model config: {config.n_layer} layers, {config.n_embd} dim, block_size={config.block_size}")

        print("Benchmarking: Denoiser (direct - training pipeline)...")
        results['denoiser_direct'] = benchmark_denoiser_direct(
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            device=device,
            max_len=max_len,
        )

        print("Benchmarking: Denoiser (full - string pipeline)...")
        results['denoiser_full'] = benchmark_denoiser_full(
            model=model,
            tokenizer=tokenizer,
            test_samples=samples,
            device=device,
            max_len=max_len,
        )

        print("Benchmarking: Denoiser (window - iterative)...")
        results['denoiser_window'] = benchmark_denoiser(
            model=model,
            tokenizer=tokenizer,
            test_samples=samples,
            max_iterations=3,
            device=device,
            max_len=max_len,
        )

    return results


def print_results(results: Dict[str, BenchmarkResult]):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Summary table
    print(f"\n{'Method':<15} {'Parse@1':<10} {'Parse@K':<10} {'Locality':<10} {'Avg Edits':<10} {'Time (ms)':<10}")
    print("-" * 65)

    for name, result in results.items():
        print(f"{result.name:<15} {result.parse_success_1:<10.3f} {result.parse_success_k:<10.3f} "
              f"{result.locality:<10.3f} {result.avg_edits:<10.1f} {result.avg_time_ms:<10.2f}")

    # Per-corruption breakdown
    print("\n" + "-" * 80)
    print("BY CORRUPTION TYPE")
    print("-" * 80)

    for name, result in results.items():
        print(f"\n{result.name}:")
        for ctype, metrics in sorted(result.by_corruption.items()):
            if 'parse_success_1' in metrics:
                print(f"  {ctype:<20} Parse@1={metrics['parse_success_1']:.2f}, "
                      f"Parse@K={metrics['parse_success_k']:.2f}, Edits={metrics['avg_edits']:.1f}")
            else:
                print(f"  {ctype:<20} Parse={metrics['parse_success']:.2f}, Edits={metrics['avg_edits']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark JSON Repair')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        num_samples=args.num_samples,
        device=args.device,
    )

    print_results(results)


if __name__ == "__main__":
    main()
