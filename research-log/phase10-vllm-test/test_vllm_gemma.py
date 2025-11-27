#!/usr/bin/env python3
"""
Phase 10: vLLM + Gemma-3-1B Test Script

Tests vLLM with Gemma-3-1B-it for fast batch generation.
This is meant to replace HuggingFaceGenerator for ~3-5x speedup.

Usage:
    python test_vllm_gemma.py          # Quick test (1, 2, 4 gens)
    python test_vllm_gemma.py --full   # Full benchmark (up to 128 gens)

Expected output on A100:
    - Model loads in ~10-30 seconds
    - Single generation in <1 second (vs ~2s with HF)
    - If single gen takes >5s, something is wrong - abort early
"""

import os
import sys
import time
import json

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fail-fast threshold: if 1 generation takes longer than this, abort
SINGLE_GEN_TIMEOUT = 10.0  # seconds


def test_single_generation():
    """Test 0: Single generation - fail fast if slow"""
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("TEST 0: Single Generation (fail-fast check)")
    print("=" * 60)

    # Load model
    print("Loading Gemma-3-1B-it with vLLM...")
    start = time.time()

    llm = LLM(
        model="google/gemma-3-1b-it",
        trust_remote_code=True,
        # A100 has 40/80GB, this model needs ~2GB
        gpu_memory_utilization=0.5,
        # Gemma 3 uses sliding window attention, set max length
        max_model_len=2048,
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Simple test with SHORT output
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=64,  # Short output for quick test
    )

    prompt = "What is 2 + 2? Answer:"

    print(f"\nPrompt: {prompt}")
    print("Generating 1 response...")

    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    gen_time = time.time() - start

    response = outputs[0].outputs[0].text
    print(f"\nResponse ({gen_time:.2f}s): {response[:100]}")

    # FAIL FAST CHECK
    if gen_time > SINGLE_GEN_TIMEOUT:
        print(f"\n*** FAIL: Single generation took {gen_time:.2f}s (>{SINGLE_GEN_TIMEOUT}s) ***")
        print("*** vLLM is not working correctly. Aborting. ***")
        return None, None, gen_time

    print(f"\n[OK] Single generation in {gen_time:.2f}s - proceeding with tests")
    return llm, sampling_params, gen_time


def test_small_batch(llm, sampling_params, n=4):
    """Test small batch to verify scaling"""
    print(f"\n{'=' * 60}")
    print(f"TEST: Batch of {n} generations")
    print("=" * 60)

    prompts = [f"What is {i} + {i}? Answer:" for i in range(n)]

    print(f"Generating {n} responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start

    print(f"Generated {n} responses in {gen_time:.2f}s ({n/gen_time:.1f} gen/s)")

    # Show first response
    print(f"Sample: {outputs[0].outputs[0].text[:50]}")

    return gen_time


def test_full_batch(llm, sampling_params):
    """Full batch simulation (8 questions x 16 candidates = 128 gens)"""
    print("\n" + "=" * 60)
    print("TEST: Full Batch (8 x 16 = 128 generations)")
    print("=" * 60)

    questions = [
        "What is 15 + 27?",
        "Janet has 5 apples. She buys 3 more. How many?",
        "A train travels 60 miles in 1 hour. How far in 3 hours?",
        "Tom has $20. He spends $7. How much left?",
        "24 students, 1/3 are girls. How many boys?",
        "Rectangle: length 8, width 5. Area?",
        "3 pencils cost $1.50. How much for 10?",
        "48 cookies into 6 boxes. How many per box?",
    ]

    # 8 questions x 16 candidates each
    prompts = []
    for q in questions:
        for _ in range(16):
            prompts.append(f"Solve: {q} Answer:")

    print(f"Generating {len(prompts)} total responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start

    print(f"Generated {len(outputs)} responses in {gen_time:.2f}s")
    print(f"Rate: {len(outputs)/gen_time:.1f} generations/sec")

    return gen_time


def main():
    full_mode = "--full" in sys.argv

    print("=" * 60)
    print("VLLM + GEMMA-3-1B-IT TEST")
    print(f"Mode: {'FULL' if full_mode else 'QUICK (use --full for complete benchmark)'}")
    print("=" * 60)
    print()

    try:
        # Test 0: Single generation (fail-fast)
        llm, sampling_params, time_1 = test_single_generation()

        if llm is None:
            print("\nAborting due to slow single generation.")
            return 1

        # Test 1: Batch of 2
        time_2 = test_small_batch(llm, sampling_params, n=2)

        # Test 2: Batch of 4
        time_4 = test_small_batch(llm, sampling_params, n=4)

        # Quick summary
        print("\n" + "=" * 60)
        print("QUICK RESULTS")
        print("=" * 60)
        print(f"1 generation:  {time_1:.2f}s")
        print(f"2 generations: {time_2:.2f}s ({2/time_2:.1f} gen/s)")
        print(f"4 generations: {time_4:.2f}s ({4/time_4:.1f} gen/s)")

        if time_4 > 10:
            print("\n*** WARNING: 4 generations took >10s - vLLM may not be optimal ***")

        # Full mode: continue with larger tests
        if full_mode:
            time_16 = test_small_batch(llm, sampling_params, n=16)
            time_128 = test_full_batch(llm, sampling_params)

            print("\n" + "=" * 60)
            print("FULL RESULTS")
            print("=" * 60)
            print(f"16 generations:  {time_16:.2f}s  ({16/time_16:.1f} gen/s)")
            print(f"128 generations: {time_128:.2f}s ({128/time_128:.1f} gen/s)")
            print()
            print("Comparison with HuggingFace (estimated):")
            print(f"  HF:   ~30s for 128 generations")
            print(f"  vLLM: {time_128:.2f}s for 128 generations")
            print(f"  Speedup: ~{30/time_128:.1f}x")

            # Save results
            results = {
                "model": "google/gemma-3-1b-it",
                "backend": "vllm",
                "time_1_gen": time_1,
                "time_16_gens": time_16,
                "time_128_gens": time_128,
                "rate_gens_per_sec": 128/time_128,
                "estimated_speedup_vs_hf": 30/time_128,
            }
            with open("vllm_benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to vllm_benchmark_results.json")
        else:
            print("\nRun with --full for complete benchmark (16 and 128 generations)")

        print("=" * 60)

    except ImportError as e:
        print(f"ERROR: vLLM not installed. Run setup.sh first.")
        print(f"Details: {e}")
        return 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
