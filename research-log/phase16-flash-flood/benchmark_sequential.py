
import sys
import time
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))

from generate_hybrid import load_hybrid_system, render_root, DEVICE
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

def benchmark_sequential(num_roots=20):
    """
    Benchmark the existing sequential generation pipeline.
    """
    # Load system
    manager, fractal_model, tokenizer, config = load_hybrid_system()
    
    print(f"\nStarting Sequential Benchmark...")
    print(f"Target: Generate {num_roots} roots and render them.")
    
    # 1. Generate Roots (Manager)
    start_time = time.time()
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated_roots = manager.generate(
            start_idx,
            max_new_tokens=num_roots,
            temperature=0.8,
            top_k=50
        )
    root_seq = generated_roots[0].tolist()[1:]
    manager_time = time.time() - start_time
    print(f"Manager Generation Time: {manager_time:.4f}s")
    
    # 2. Render Roots (Fractal Engine - Sequential Loop)
    render_start_time = time.time()
    total_chars = 0
    
    print("Rendering...")
    for i, root_id in enumerate(root_seq):
        if root_id >= len(tokenizer.root_vocab):
            continue
            
        text, stats = render_root(
            fractal_model, tokenizer, config, root_id, DEVICE
        )
        total_chars += len(text)
        # print(f"\rRoot {i+1}/{len(root_seq)}: {len(text)} chars", end="")
        
    render_time = time.time() - render_start_time
    total_time = time.time() - start_time
    
    print(f"\n\nResults:")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Manager Time: {manager_time:.4f}s")
    print(f"  Render Time: {render_time:.4f}s")
    print(f"  Total Chars: {total_chars}")
    print(f"  Chars/Sec (End-to-End): {total_chars / total_time:.2f}")
    print(f"  Chars/Sec (Render Only): {total_chars / render_time:.2f}")
    
    # Approximate tokens (assuming ~4 chars per token)
    tokens_approx = total_chars / 4
    print(f"  Tokens/Sec (Approx @ 4chars/tok): {tokens_approx / total_time:.2f}")

if __name__ == "__main__":
    benchmark_sequential(num_roots=50)
