import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import signal
import json
import os
import numpy as np
from tqdm import tqdm
import random

# Fix CUDA + multiprocessing deadlock - must use spawn not fork
multiprocessing.set_start_method('spawn', force=True)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "research-log/phase15-rl/checkpoints"
MAX_LEN = 512
GROUP_SIZE = 4 # G - samples per prompt (higher = better advantage estimation)
MINI_BATCH_SIZE = 2 # Prompts per step (Total batch = 2 * 4 = 8 sequences)
GRAD_ACCUM = 16 # Accumulate to match effective batch size
LR = 1e-6
KL_COEFF = 0.05
EPOCHS = 5 # Train for longer

# Execution Helper (Copied from execute_mbpp.py for speed/import simplicity)
def handler(signum, frame):
    raise TimeoutError("Timeout")

def check_code(args):
    code, test, entry_point = args
    # Basic execution wrapper
    header = "import math\nimport heapq\nimport itertools\nimport re\nimport collections\nfrom typing import *\n\n"
    full_script = header + code + "\n\n" + test
    
    # Run in separate process logic is handled by Pool, but we need safety inside
    try:
        # Signal doesn't work well inside Pool worker on some platforms, 
        # but on Linux Lambda it should be fine.
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(2) # 2 second timeout
        
        exec_globals = {}
        exec(full_script, exec_globals)
        
        signal.alarm(0) # Disable alarm
        return 1.0 # Pass
    except:
        signal.alarm(0)
        return 0.0 # Fail

class MBPPPrompts(Dataset):
    def __init__(self):
        # Load MBPP train split directly from HuggingFace
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        self.data = []
        for item in ds:
            self.data.append({
                "prompt": item['prompt'],
                "tests": item['test_list'],
                "task_id": item['task_id']
            })
        print(f"Loaded {len(self.data)} prompts from MBPP train.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_grpo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"  # Required for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    
    # Ref Model for KL (Frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    dataset = MBPPPrompts()
    loader = DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
    
    print("Starting GRPO Training...")

    # Use spawn context for Pool (avoids CUDA fork deadlock)
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=32)
    
    step = 0
    for epoch in range(EPOCHS):
        for batch in tqdm(loader):
            prompts = [x['prompt'] for x in batch]
            tests_list = [x['tests'] for x in batch] # List of lists
            
            # 1. Generation (Sampling)
            # Format prompts
            input_texts = []
            for p in prompts:
                fmt = f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to solve the following problem:\n{p}\n\nYour code must pass these tests:\n{tests_list[0][0] if tests_list[0] else ''}\n<|im_end|>\n<|im_start|>assistant\n```python\n"
                # Repeat G times
                for _ in range(GROUP_SIZE):
                    input_texts.append(fmt)
            
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 2. Reward Calculation (Execution)
            # Extract code
            generated_codes = []
            full_seqs = [] # Store for update
            
            for i, seq in enumerate(outputs):
                # Decode only new tokens
                prompt_len = inputs.input_ids.shape[1]
                gen_tokens = seq[prompt_len:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if "```" in text:
                    text = text.split("```")[0]
                generated_codes.append(text)
                full_seqs.append(seq)
            
            # Execute in parallel
            exec_args = []
            for i in range(len(generated_codes)):
                # Map back to prompt index: i // GROUP_SIZE
                prompt_idx = i // GROUP_SIZE
                test_str = "\n".join(tests_list[prompt_idx])
                exec_args.append((generated_codes[i], test_str, ""))
            
            # Run execution
            # We use starmap with timeout handling implicitly? 
            # Standard pool doesn't support timeout per item easily.
            # For speed, we trust the code won't loop forever or we rely on simple checks.
            # Let's assume MBPP code is short.
            rewards = pool.map(check_code, exec_args)
            rewards = torch.tensor(rewards, device="cuda", dtype=torch.float)
            
            # 3. GRPO Update
            # Reshape rewards to [B, G] where B = actual batch size (may be smaller at end)
            actual_batch = len(batch)
            rewards = rewards.view(actual_batch, GROUP_SIZE)
            
            # Advantage: Normalize per group
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards - mean_r) / std_r # [B, G]
            
            # Flatten for batch processing
            advantages = advantages.view(-1) # [B*G]
            
            # Forward Pass to get LogProbs
            # We need to run the model on the *generated sequences*
            # Pad outputs to same length
            max_len = max([len(x) for x in full_seqs])
            padded_seqs = torch.full((len(full_seqs), max_len), tokenizer.pad_token_id, device="cuda")
            for i, seq in enumerate(full_seqs):
                padded_seqs[i, :len(seq)] = seq
            
            attention_mask = (padded_seqs != tokenizer.pad_token_id).long()
            
            # Clear cache before forward passes
            torch.cuda.empty_cache()

            # Current Policy LogProbs (with gradient)
            # use_cache=False is CRITICAL for gradient checkpointing to work
            outputs_policy = model(padded_seqs, attention_mask=attention_mask, use_cache=False)
            logits = outputs_policy.logits
            # Shift logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = padded_seqs[..., 1:].contiguous()
            
            # Gather log_probs of selected tokens
            log_probs = -nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
            log_probs = log_probs.view(padded_seqs.size(0), -1) # [B*G, L-1]
            
            # Ref Policy LogProbs
            with torch.no_grad():
                ref_logits = ref_model(padded_seqs, attention_mask=attention_mask).logits
                shift_ref_logits = ref_logits[..., :-1, :].contiguous()
                ref_log_probs = -nn.functional.cross_entropy(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_labels.view(-1), reduction='none')
                ref_log_probs = ref_log_probs.view(padded_seqs.size(0), -1)
                
            # Mask out prompt tokens (we only train on generation)
            # We need the prompt length for each sample. It was constant `inputs.input_ids.shape[1]`?
            # Yes, input_texts were padded.
            prompt_len = inputs.input_ids.shape[1]
            # Create mask for generation
            gen_mask = torch.zeros_like(log_probs)
            gen_mask[:, prompt_len-1:] = 1.0 
            # Apply attention mask as well (ignore padding)
            gen_mask = gen_mask * attention_mask[:, 1:]
            
            # KL Divergence (Token-level)
            # ratio = log(pi/ref) = log_pi - log_ref
            ratio = log_probs - ref_log_probs
            kl = torch.exp(ratio) - 1 - ratio # approximate KL? Or just use ratio directly.
            # PPO usually uses ratio. GRPO uses KL penalty in reward or loss.
            # DeepSeek GRPO formula: Loss = -mean(Adv * log_pi) + beta * KL
            
            token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1 # Inverse KL?
            # Simple KL: log_pi - log_ref. 
            # Let's use standard approx: kld = (log_pi - log_ref).exp() * (log_pi - log_ref) ? No.
            # Just use the diff: log_pi - log_ref
            
            # GRPO Objective
            # We want to maximize Adv * LogProb
            # Loss = - (Adv * LogProb).
            # Note: Adv is scalar per sample. LogProb is per token.
            # Broadcast Adv to tokens
            adv_expanded = advantages.view(-1, 1).expand_as(log_probs)
            
            pg_loss = -(adv_expanded * log_probs * gen_mask).sum() / gen_mask.sum()
            
            # KL Penalty
            kl_loss = ((log_probs - ref_log_probs) * gen_mask).sum() / gen_mask.sum()
            
            loss = pg_loss + KL_COEFF * kl_loss
            
            loss.backward()
            
            # Gradient Accumulation
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            step += 1
            
            if step % 5 == 0:
                print(f"E{epoch+1} Step {step}: Loss={loss.item():.4f}, Reward={rewards.mean().item():.2f}, KL={kl_loss.item():.4f}")
                
        # Save epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"grpo_e{epoch+1}.pt"))

if __name__ == "__main__":
    train_grpo()