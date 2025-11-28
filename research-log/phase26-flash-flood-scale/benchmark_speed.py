import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Or gpt2-xl
SEQ_LEN = 2048  # Target tokens
PARALLEL_STEPS = 10  # Number of refinement steps for Flash Flood

def benchmark_ar(model, tokenizer, seq_len, device):
    print(f"Benchmarking Autoregressive (Target: {seq_len} tokens)...")
    
    input_ids = tokenizer("The quick brown fox", return_tensors="pt").input_ids.to(device)
    start_len = input_ids.shape[1]
    target_len = start_len + seq_len
    
    # Warmup
    model(input_ids)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Standard AR generation loop (without kv-cache optimization for worst case? 
    # No, use standard optimized generation)
    # Actually, standard HF generate uses KV cache.
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=seq_len, 
            min_new_tokens=seq_len,
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_sec = seq_len / total_time
    print(f"AR Time: {total_time:.2f}s | TPS: {tokens_per_sec:.2f}")
    return total_time

def benchmark_flash_flood(model, seq_len, steps, device):
    print(f"Benchmarking Flash Flood (Target: {seq_len} tokens, Steps: {steps})...")
    
    # In FF, we process the WHOLE sequence in parallel.
    # Batch size = 1, Seq Len = target_len
    # We run 'steps' forward passes.
    
    # Mock input: Full sequence of noise/tokens
    # We are measuring the COST of the forward pass.
    input_ids = torch.randint(0, 1000, (1, seq_len)).to(device)
    
    # Warmup
    model(input_ids)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(steps):
            # Full parallel pass
            # We disable KV cache because we re-compute full attention every time (or use specific attention masks)
            # For Flash Flood, we typically re-process.
            _ = model(input_ids, use_cache=False)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    # "Effective" TPS = Total Tokens / Total Time
    tokens_per_sec = seq_len / total_time
    print(f"FF Time: {total_time:.2f}s | TPS: {tokens_per_sec:.2f}")
    return total_time

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).half() # FP16
    except:
        print(f"Could not load {MODEL_NAME}, falling back to gpt2")
        MODEL_NAME_FALLBACK = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FALLBACK)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_FALLBACK).to(device)

    model.eval()
    
    # Benchmark
    lengths = [128, 512, 1024, 2048]
    ar_times = []
    ff_times = []
    
    for l in lengths:
        print(f"\n--- Length {l} ---")
        t_ar = benchmark_ar(model, tokenizer, l, device)
        t_ff = benchmark_flash_flood(model, l, PARALLEL_STEPS, device)
        
        ar_times.append(t_ar)
        ff_times.append(t_ff)
        
        speedup = t_ar / t_ff
        print(f"Speedup: {speedup:.2f}x")

    # Save Results
    with open("research-log/phase26-flash-flood-scale/results_speed.txt", "w") as f:
        f.write(f"Lengths: {lengths}\n")
        f.write(f"AR_Times: {ar_times}\n")
        f.write(f"FF_Times: {ff_times}\n")
        f.write(f"Speedups: {[a/b for a,b in zip(ar_times, ff_times)]}\n")

if __name__ == "__main__":
    main()
