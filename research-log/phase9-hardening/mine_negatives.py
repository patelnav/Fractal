#!/usr/bin/env python3
import json
import torch
import os
import sys
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from generator import HuggingFaceGenerator
from model import OuroborosModel, OuroborosConfig
from tokenizer import OuroborosTokenizer
from utils import is_correct, parse_gsm8k_answer

# Configuration
DATA_PATH = "../phase7-ouroboros/data/gsm8k/train.json"
CHECKPOINT_PATH = "../phase7-ouroboros/checkpoints/ckpt.pt"
OUTPUT_PATH = "data/hard_negatives.jsonl"
BATCH_SIZE = 64  # Increased from 8 to saturate GPU
N_CANDIDATES = 5
ENERGY_THRESHOLD = 0.5 

def load_gsm8k(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def main():
    print("Initializing Mining Operation...")
    
    # 1. Load Generator
    print("Loading Generator...")
    try:
        generator = HuggingFaceGenerator(
            model_name="google/gemma-3-1b-it",
            device="cuda",
            do_sample=True
        )
    except Exception as e:
        print(f"Error loading generator: {e}")
        sys.exit(1)
    
    # 2. Load Verifier
    print("Loading Verifier...")
    config = OuroborosConfig() 
    verifier = OuroborosModel(config)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda", weights_only=False)
            if "model" in checkpoint:
                verifier.load_state_dict(checkpoint["model"])
            elif "model_state_dict" in checkpoint:
                verifier.load_state_dict(checkpoint["model_state_dict"])
            else:
                verifier.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights (Mining will be ineffective).")

    verifier.to("cuda")
    verifier.eval()
    
    tokenizer = OuroborosTokenizer()
    
    # 3. Load Data
    dataset = load_gsm8k(DATA_PATH)
    if not dataset:
        print("No data found. Exiting.")
        sys.exit(1)

    print(f"Found {len(dataset)} problems.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Clear existing file if starting fresh
    if os.path.exists(OUTPUT_PATH):
        print(f"Clearing existing output file: {OUTPUT_PATH}")
        os.remove(OUTPUT_PATH)
    
    hard_negatives_count = 0
    
    # 4. Mining Loop
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Mining"):
        t0 = time.time()
        batch = dataset[i:i+BATCH_SIZE]
        questions = [item['question'] for item in batch]
        ground_truths = [item['answer'] for item in batch]
        
        # Generate candidates
        t_gen_start = time.time()
        try:
            candidates_batch = generator.generate_batch(questions, n=N_CANDIDATES)
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
        t_gen_end = time.time()
            
        # Process candidates
        context_ids_list = []
        target_ids_list = []
        metadata_list = [] # stores (question, candidate, gt)
        
        for q_idx, (question, candidates) in enumerate(zip(questions, candidates_batch)):
            gt_full = ground_truths[q_idx]
            _, gt_answer = parse_gsm8k_answer(gt_full)
            
            q_tokens = tokenizer.encode(f"{tokenizer.config.MATH_START}{question}")
            # Truncate context to leave room for target
            # Max seq len = 512. Let's say max context 256, max target 256.
            if len(q_tokens) > 256:
                q_tokens = q_tokens[:256]
            
            for cand in candidates:
                if not is_correct(cand, gt_answer):
                    s_text = f"{tokenizer.config.STEP_SEP}{cand}{tokenizer.config.MATH_END}"
                    s_tokens = tokenizer.encode(s_text)
                    # Truncate target
                    if len(s_tokens) > 256:
                        s_tokens = s_tokens[:256]
                    
                    context_ids_list.append(torch.tensor(q_tokens))
                    target_ids_list.append(torch.tensor(s_tokens))
                    metadata_list.append((question, cand, gt_full))

        if not context_ids_list:
            continue
            
        # Batch verify
        t_ver_start = time.time()
        try:
            ctx_padded = pad_sequence(context_ids_list, batch_first=True, padding_value=50256).to("cuda")
            tgt_padded = pad_sequence(target_ids_list, batch_first=True, padding_value=50256).to("cuda")
            
            # Ensure total length <= 512 (model limit)
            # Although we truncated individually, padding might align them.
            # OuroborosModel concatenates them.
            # If ctx + tgt > 512, we might still have issues if we are not careful.
            # But 256+256=512, so should be safe.
            
            with torch.no_grad():
                energies = verifier.get_energy(ctx_padded, tgt_padded)
            
            for idx, energy in enumerate(energies):
                energy_val = energy.item()
                if energy_val < ENERGY_THRESHOLD:
                    question, wrong_sol, gt = metadata_list[idx]
                    
                    hard_negative = {
                        "question": question,
                        "wrong_solution": wrong_sol,
                        "energy_score": energy_val,
                        "ground_truth": gt
                    }
                    
                    with open(OUTPUT_PATH, 'a') as f:
                        f.write(json.dumps(hard_negative) + "\n")
                    
                    hard_negatives_count += 1
                    
        except Exception as e:
            print(f"Verification failed: {e}")
            continue
        t_ver_end = time.time()
        
        # Optional debug timing (enable if needed)
        # print(f"Gen: {t_gen_end - t_gen_start:.2f}s, Ver: {t_ver_end - t_ver_start:.2f}s, Total: {time.time() - t0:.2f}s")

    print(f"Mining complete. Found {hard_negatives_count} hard negatives.")

if __name__ == "__main__":
    main()
