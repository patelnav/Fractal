#!/usr/bin/env python3
import time
import sys
import torch
from generator import HuggingFaceGenerator

def benchmark():
    print("Initializing Generator...")
    try:
        gen = HuggingFaceGenerator(
            model_name="google/gemma-3-1b-it",
            device="cuda",
            do_sample=True
        )
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)
    
    questions = [f"Question {i}: What is {i} + {i}?" for i in range(100)]
    
    print(f"Starting Generation Benchmark on {len(questions)} prompts x 5 candidates...")
    start_time = time.time()
    
    try:
        results = gen.generate_batch(questions, n=5)
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Generated {len(results)} sets of 5 candidates in {total_time:.2f} seconds.")
    
    seq_per_sec = (len(questions) * 5) / total_time
    print(f"Throughput: {seq_per_sec:.2f} seq/sec")

    if total_time < 15:
        print("PASS: Speed is sufficient (< 15s).")
    else:
        print("FAIL: Speed is too slow (> 15s).")

if __name__ == "__main__":
    benchmark()
