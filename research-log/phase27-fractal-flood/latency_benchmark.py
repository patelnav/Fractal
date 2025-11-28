import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
# We use GPT2 as proxy for compute cost
MODEL_NAME = "gpt2" 
PARALLEL_STEPS = 10
SKETCH_RATIO = 0.1  # Sketch is 10% of total length

def benchmark_ar(model, target_len, device):
    # Standard AR generation
    # Input: 10 tokens
    # Output: target_len
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        # Simulate AR loop: 1 forward pass per token
        # We use a mock loop to avoid overhead of sampling logic, just measuring Model Forward
        # But HF generate is optimized. Let's use generate for realism.
        model.generate(input_ids, max_new_tokens=target_len, min_new_tokens=target_len, use_cache=True)
        
    torch.cuda.synchronize()
    return time.time() - start

def benchmark_fractal_flood(model, target_len, device):
    # Stage 1: Sketch (AR)
    sketch_len = int(target_len * SKETCH_RATIO)
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    
    torch.cuda.synchronize()
    start = time.time()
    
    # 1. AR Sketch
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=sketch_len, min_new_tokens=sketch_len, use_cache=True)
        
    # 2. Flash Flood Expansion
    # We assume we expand from Sketch -> Full
    # Input length = target_len (full context)
    # 10 Parallel Steps
    full_input = torch.randint(0, 1000, (1, target_len)).to(device)
    
    with torch.no_grad():
        for _ in range(PARALLEL_STEPS):
             _ = model(full_input, use_cache=False)
             
    torch.cuda.synchronize()
    return time.time() - start

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    except:
        print("Fallback to cpu")
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    model.eval()
    
    lengths = [100, 500, 1000, 2000, 4000]
    ar_times = []
    ff_times = []
    
    print(f"{ 'Length':<10} | { 'AR (s)':<10} | { 'FF (s)':<10} | { 'Speedup':<10}")
    print("""---------------------------------------------------""")
    
    for l in lengths:
        t_ar = benchmark_ar(model, l, device)
        t_ff = benchmark_fractal_flood(model, l, device)
        speedup = t_ar / t_ff
        
        ar_times.append(t_ar)
        ff_times.append(t_ff)
        
        print(f"{l:<10} | {t_ar:<10.2f} | {t_ff:<10.2f} | {speedup:<10.2f}x")

    # Save
    with open("research-log/phase27-fractal-flood/results_latency.txt", "w") as f:
        f.write(f"Lengths: {lengths}\n")
        f.write(f"AR: {ar_times}\n")
        f.write(f"FF: {ff_times}\n")

if __name__ == "__main__":
    main()
