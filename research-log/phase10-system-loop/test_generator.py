#!/usr/bin/env python3
"""
Stage 1: Generator Qualification Test
Verifies that the generator produces correctly formatted, clean, and reasonable answers.
"""
import json
import sys
import os
from pathlib import Path
from generator import HuggingFaceGenerator
from utils import is_correct, parse_gsm8k_answer

DATA_PATH = "../phase7-ouroboros/data/gsm8k/test.json"
OUTPUT_PATH = "results_stage1_gen.jsonl"

def main():
    print("=" * 60)
    print("STAGE 1: GENERATOR QUALIFICATION")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)[:20] # Test on first 20 examples

    # 2. Load Generator
    print("Loading Generator (Gemma-3-1B)...")
    try:
        # Use 'auto' device to support local MPS (Mac) or CUDA (Linux)
        generator = HuggingFaceGenerator(
            model_name="google/gemma-3-1b-it",
            device="auto",
            do_sample=True,
            temperature=0.7
        )
    except Exception as e:
        print(f"Failed to load generator: {e}")
        sys.exit(1)

    # 3. Run Generation Test
    questions = [item['question'] for item in data]
    ground_truths = [item['answer'] for item in data]
    
    print(f"Generating solutions for {len(questions)} problems (Batch Mode)...")
    
    # We test generate_batch specifically as it uses a different path than generate()
    try:
        candidates_batch = generator.generate_batch(questions, n=1)
    except Exception as e:
        print(f"FATAL: generate_batch failed: {e}")
        sys.exit(1)
    
    # 4. Analyze Results
    passed = 0
    valid_format = 0
    clean_output = 0
    
    debug_log = []

    for i, (q, cands) in enumerate(zip(questions, candidates_batch)):
        cand = cands[0]
        gt = ground_truths[i]
        _, gt_clean = parse_gsm8k_answer(gt)
        
        # Criteria 1: Format (Must contain ####)
        has_answer_marker = "####" in cand
        
        # Criteria 2: Cleanliness (No leaked chat tokens)
        is_clean = "<start_of_turn>" not in cand and "<end_of_turn>" not in cand
        
        # Criteria 3: Accuracy
        correct = is_correct(cand, gt_clean)
        
        if has_answer_marker: valid_format += 1
        if is_clean: clean_output += 1
        if correct: passed += 1
        
        status = "PASS" if correct else "FAIL"
        if not is_clean: status = "DIRTY"
        elif not has_answer_marker: status = "BAD_FMT"
        
        print(f"[{i+1:02d}] {status} | Correct: {correct} | Valid: {has_answer_marker} | Clean: {is_clean}")
        
        debug_log.append({
            "question": q,
            "output": cand,
            "ground_truth": gt_clean,
            "correct": correct,
            "clean": is_clean,
            "valid_format": has_answer_marker
        })

    # Save debug log
    with open(OUTPUT_PATH, "w") as f:
        for entry in debug_log:
            f.write(json.dumps(entry) + "\n")
    print(f"\nDebug outputs saved to {OUTPUT_PATH}")

    # 5. Final Report
    accuracy = 100 * passed / len(data)
    format_rate = 100 * valid_format / len(data)
    clean_rate = 100 * clean_output / len(data)

    print("-" * 60)
    print(f"Accuracy:      {passed}/{len(data)} ({accuracy:.1f}%)")
    print(f"Valid Format:  {valid_format}/{len(data)} ({format_rate:.1f}%)")
    print(f"Clean Output:  {clean_output}/{len(data)} ({clean_rate:.1f}%)")
    print("-" * 60)

    # 6. Gate Logic
    success = True
    
    if accuracy < 25.0:
        print("FAIL: Accuracy < 25% (Model is too dumb or prompting is broken)")
        success = False
        
    if format_rate < 80.0:
        print("FAIL: Valid Format < 80% (Model not following '####' convention)")
        success = False
        
    if clean_rate < 100.0:
        print("FAIL: Clean Output < 100% (Leaked chat template tokens)")
        success = False
        
    if success:
        print("✅ STAGE 1 PASSED")
        sys.exit(0)
    else:
        print("❌ STAGE 1 FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
