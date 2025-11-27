
import sys
import torch
import random
import time
from pathlib import Path
from typing import List

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase6-hybrid"))

from generate_hybrid import load_hybrid_system, DEVICE
from run_fractal_engine import FractalModelConfig
from train_manager import ManagerConfig

def run_drift_experiment(num_trials=20):
    """
    Demonstrate "Butterfly Effect" in standard AR generation.
    Runs multiple trials to quantify how much the future drifts when one token changes.
    """
    print("\n" + "=" * 60)
    print(f"BASELINE DRIFT EXPERIMENT (Standard AR, N={num_trials})")
    print("Hypothesis: Changing token T changes tokens T+1...N")
    print("=" * 60)

    # Load Manager
    manager, _, tokenizer, config = load_hybrid_system()
    
    match_rates = []
    
    print(f"Running {num_trials} trials...")
    
    for i in range(num_trials):
        seed = 42 + i
        torch.manual_seed(seed)
        
        # 1. Generate Baseline
        start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            seq_A = manager.generate(start_idx, max_new_tokens=20, temperature=0.8, top_k=50)
        tokens_A = seq_A[0].tolist()
        
        # 2. Intervene at Step 5
        intervention_idx = 5
        prefix = seq_A[:, :intervention_idx+1]
        
        original_token = tokens_A[intervention_idx+1]
        # Ensure we pick a different token
        new_token = (original_token + 1) % config.num_roots
        
        forced_step = torch.tensor([[new_token]], dtype=torch.long, device=DEVICE)
        prefix_forced = torch.cat([prefix, forced_step], dim=1)
        
        # 3. Regenerate
        torch.manual_seed(seed) # Reset seed for fair comparison of logic
        remaining_steps = 20 - (intervention_idx + 1)
        
        with torch.no_grad():
            seq_B = manager.generate(prefix_forced, max_new_tokens=remaining_steps, temperature=0.8, top_k=50)
            
        tokens_B = seq_B[0].tolist()
        
        # 4. Compare Suffixes
        suffix_start = intervention_idx + 2
        suffix_A = tokens_A[suffix_start:]
        suffix_B = tokens_B[suffix_start:]
        
        # Exact match count at same positions
        # (Simple Hamming distance check effectively)
        matches = sum(a == b for a, b in zip(suffix_A, suffix_B))
        if len(suffix_A) > 0:
            rate = matches / len(suffix_A) * 100.0
        else:
            rate = 100.0
            
        match_rates.append(rate)
        # print(f"  Trial {i+1}: Match Rate = {rate:.1f}%")

    # Stats
    avg_match = sum(match_rates) / len(match_rates)
    import math
    variance = sum((x - avg_match) ** 2 for x in match_rates) / len(match_rates)
    std_dev = math.sqrt(variance)
    
    print("\n" + "-" * 30)
    print(f"RESULTS (N={num_trials})")
    print("-" * 30)
    print(f"Mean Suffix Match Rate: {avg_match:.2f}%")
    print(f"Std Dev:                {std_dev:.2f}%")
    print("-" * 30)
    
    if avg_match < 10.0:
        print("CONCLUSION: High Drift confirmed. Future timelines diverge significantly.")
    else:
        print("CONCLUSION: Low Drift observed.")

if __name__ == "__main__":
    run_drift_experiment(num_trials=20)
