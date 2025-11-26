import json
import numpy as np
from collections import Counter

def load_results(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_failures(data):
    # Metrics counters
    total = len(data)
    baseline_correct_count = 0
    ouroboros_correct_count = 0
    wins = [] # (baseline_wrong, ouroboros_correct)
    
    win_types = Counter()
    
    print(f"Analyzing {total} problems...")
    
    for item in data:
        candidates = item['candidates']
        energies = item['energies']
        candidate_correct = item['candidate_correct']
        
        # Indices
        baseline_idx = 0
        ouroboros_idx = np.argmin(energies)
        
        # Correctness
        base_ok = candidate_correct[baseline_idx]
        our_ok = candidate_correct[ouroboros_idx]
        
        if base_ok: baseline_correct_count += 1
        if our_ok: ouroboros_correct_count += 1
        
        # Success Case: Baseline was wrong, Ouroboros picked a right one
        if not base_ok and our_ok:
            baseline_candidate = candidates[baseline_idx]
            ouroboros_candidate = candidates[ouroboros_idx]
            baseline_energy = energies[baseline_idx]
            ouroboros_energy = energies[ouroboros_idx]
            
            is_penalty_win = baseline_energy >= 99.0
            
            win_info = {
                'question': item['question'],
                'baseline_candidate': baseline_candidate,
                'ouroboros_candidate': ouroboros_candidate,
                'baseline_energy': baseline_energy,
                'ouroboros_energy': ouroboros_energy,
                'energy_gap': baseline_energy - ouroboros_energy,
                'type': 'Penalty Win' if is_penalty_win else 'Energy Win'
            }
            wins.append(win_info)
            win_types[win_info['type']] += 1

    print(f"\nBaseline Accuracy: {baseline_correct_count/total:.1%}")
    print(f"Ouroboros Accuracy: {ouroboros_correct_count/total:.1%}")
    print(f"Net Lift: {(ouroboros_correct_count - baseline_correct_count)/total:.1%}")
    
    print(f"\nFound {len(wins)} wins (Baseline Wrong -> Ouroboros Right)")
    print(f"Win Breakdown:")
    for k, v in win_types.most_common():
        print(f"- {k}: {v} ({v/len(wins):.1%})")
        
    print("\n--- Example 'Energy Wins' (Real Reasoning) ---")
    energy_wins = [w for w in wins if w['type'] == 'Energy Win']
    energy_wins.sort(key=lambda x: x['energy_gap'], reverse=True)
    
    if not energy_wins:
        print("None! All wins were due to the formatting penalty.")
    else:
        for i, case in enumerate(energy_wins[:3]):
            print(f"\nExample #{i+1} (Gap: {case['energy_gap']:.4f}):")
            print(f"Question: {case['question'][:100]}...")
            print(f"REJECTED WRONG (Energy {case['baseline_energy']:.4f}):\n{case['baseline_candidate'][:200]}...")
            print(f"\nSELECTED CORRECT (Energy {case['ouroboros_energy']:.4f}):\n{case['ouroboros_candidate'][:200]}...")

if __name__ == "__main__":
    data = load_results("results/gsm8k_eval/results_incremental.jsonl")
    analyze_failures(data)
