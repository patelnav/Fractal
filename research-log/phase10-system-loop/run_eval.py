#!/usr/bin/env python3
import json
import torch
import os
import sys
import time
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from generator import HuggingFaceGenerator
from model import OuroborosModel, OuroborosConfig
from tokenizer import OuroborosTokenizer
from utils import is_correct, parse_gsm8k_answer

# Configuration
DATA_PATH = "../phase7-ouroboros/data/gsm8k/test.json"
CHECKPOINT_PATH = "../phase9-hardening/checkpoints/ckpt_hardened.pt"
OUTPUT_PATH = "results_phase10.jsonl"

# Hyperparameters
BATCH_SIZE = 8       # Number of problems to process in parallel
N_CANDIDATES = 16    # Candidates per problem (System 2 search width)
MAX_SEQ_LEN = 1024   # Truncation limit (not used directly, see inner logic)
LIMIT = None           # Limit number of problems for testing (None for full run)

def load_data(path):
    print(f"Loading data from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    if LIMIT:
        print(f"limiting to {LIMIT} problems for testing")
        return data[:LIMIT]
    return data

def main():
    print("=" * 60)
    print("PHASE 10: FULL SYSTEM EVALUATION")
    print("=" * 60)

    # Determine Device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    # 1. Load Generator
    print("Loading Generator (Gemma-3-1B)...")
    try:
        # Note: HuggingFaceGenerator handles "auto" well, but we can pass specific device if needed.
        # For consistency, we pass the detected device (or "auto" if we want HF to decide).
        # But "mps" support in HF sometimes needs explicit "mps".
        generator = HuggingFaceGenerator(
            model_name="google/gemma-3-1b-it",
            device=DEVICE, 
            do_sample=True,
            temperature=0.7
        )
    except Exception as e:
        print(f"Error loading generator: {e}")
        sys.exit(1)
    
    # 2. Load Verifier
    print("Loading Verifier (Ouroboros Hardened)...")
    config = OuroborosConfig() 
    verifier = OuroborosModel(config)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            if "model" in checkpoint:
                verifier.load_state_dict(checkpoint["model"])
            elif "model_state_dict" in checkpoint:
                verifier.load_state_dict(checkpoint["model_state_dict"])
            else:
                verifier.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    else:
        print(f"CRITICAL: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    verifier.to(DEVICE)
    verifier.eval()
    
    tokenizer = OuroborosTokenizer()
    
    # 3. Load Data
    dataset = load_data(DATA_PATH)
    print(f"Evaluating on {len(dataset)} test problems.")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Stats tracking
    total = 0
    correct_greedy = 0  # First candidate correct?
    correct_ouroboros = 0 # Selected candidate correct?
    correct_oracle = 0    # Any candidate correct? (Upper bound)
    
    # 4. Evaluation Loop
    start_time = time.time()
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Evaluating"):
        batch = dataset[i:i+BATCH_SIZE]
        questions = [item['question'] for item in batch]
        ground_truths = [item['answer'] for item in batch]
        
        batch_results = [] # New list for current batch results
        
        # A. Generate Candidates
        try:
            # shape: [batch_size, n_candidates]
            candidates_batch = generator.generate_batch(questions, n=N_CANDIDATES)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
            
        # Prepare verification batch
        context_ids_list = []
        target_ids_list = []
        mapping = [] # (batch_idx, cand_idx) map for flattened list
        
        # Prepare input for Verifier
        for b_idx, (q, cands) in enumerate(zip(questions, candidates_batch)):
            q_tokens = tokenizer.encode(f"{tokenizer.config.MATH_START}{q}")
            # Truncate context
            if len(q_tokens) > 256: q_tokens = q_tokens[:256]
            
            for c_idx, cand in enumerate(cands):
                s_text = f"{tokenizer.config.STEP_SEP}{cand}{tokenizer.config.MATH_END}"
                s_tokens = tokenizer.encode(s_text)
                # Truncate target
                if len(s_tokens) > 256: s_tokens = s_tokens[:256]
                
                context_ids_list.append(torch.tensor(q_tokens))
                target_ids_list.append(torch.tensor(s_tokens))
                mapping.append((b_idx, c_idx))
        
        # B. Verify Candidates
        if not context_ids_list: continue
        
        energies_flat = []
        try:
            ctx_padded = pad_sequence(context_ids_list, batch_first=True, padding_value=50256).to(DEVICE)
            tgt_padded = pad_sequence(target_ids_list, batch_first=True, padding_value=50256).to(DEVICE)
            
            # Process in sub-batches if too large for GPU memory
            # 8 * 16 = 128 sequences. Should fit on A100 easily.
            with torch.no_grad():
                energies = verifier.get_energy(ctx_padded, tgt_padded)
                energies_flat = energies.cpu().numpy()
                
        except Exception as e:
            print(f"Verification failed: {e}")
            # Fallback: random scores
            energies_flat = np.random.rand(len(context_ids_list))

        # Reconstruct structure and evaluate
        # energies_structured[b_idx] = [(cand, score), ...]
        energies_structured = [[] for _ in range(len(batch))]
        for idx, (b_idx, c_idx) in enumerate(mapping):
            score = energies_flat[idx]
            cand = candidates_batch[b_idx][c_idx]
            energies_structured[b_idx].append((cand, score))
            
        # C. Metrics
        for b_idx in range(len(batch)):
            gt_full = ground_truths[b_idx]
            _, gt_clean = parse_gsm8k_answer(gt_full)
            
            cands_with_scores = energies_structured[b_idx]
            # cands_with_scores is list of (cand_str, score_float)
            
            # 1. Greedy Accuracy (Candidate 0 - assuming generator sorts by logprob? 
            # Actually HF generator sample returns random order if do_sample=True.
            # So "Greedy" here is just "Random Sample 1".
            base_cand = cands_with_scores[0][0]
            is_base_correct = is_correct(base_cand, gt_clean)
            
            # 2. Oracle Accuracy (Upper Bound)
            is_any_correct = any(is_correct(c, gt_clean) for c, _ in cands_with_scores)
            
            # 3. Ouroboros Selection
            # Sort by energy (ascending) -> Lowest energy is best
            best_cand, best_score = sorted(cands_with_scores, key=lambda x: x[1])[0]
            is_ouro_correct = is_correct(best_cand, gt_clean)
            
            batch_results.append({
                "question": questions[b_idx],
                "ground_truth": gt_clean,
                "baseline_correct": is_base_correct,
                "ouroboros_correct": is_ouro_correct,
                "oracle_correct": is_any_correct,
                "selected_energy": float(best_score),
                "selected_answer": best_cand
            })
            
        # Write batch results incrementally
        with open(OUTPUT_PATH, 'a') as f:
            for r in batch_results:
                f.write(json.dumps(r) + "\n")

        # Update overall stats AFTER writing batch results
        for r in batch_results:
            total += 1
            if r["baseline_correct"]: correct_greedy += 1
            if r["ouroboros_correct"]: correct_ouroboros += 1
            if r["oracle_correct"]: correct_oracle += 1

    # 5. Final Report
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Problems: {total}")
    print(f"Time Taken:     {elapsed:.2f}s ({total/elapsed:.2f} prob/s)")
    print("-" * 60)
    print(f"Baseline (Pass@1):   {correct_greedy}/{total} ({100*correct_greedy/total:.2f}%)")
    print(f"Oracle (Pass@N):     {correct_oracle}/{total} ({100*correct_oracle/total:.2f}%)")
    print(f"Ouroboros (Verifier): {correct_ouroboros}/{total} ({100*correct_ouroboros/total:.2f}%)")
    print("=" * 60)
    
    # Do not save results here, as they are saved incrementally
    print(f"Detailed results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()