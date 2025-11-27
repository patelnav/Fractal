from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import multiprocessing
from tqdm import tqdm

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CHECKPOINT_DIR = "research-log/phase15-rl/checkpoints"
OUTPUT_FILE = "research-log/phase15-rl/data/eval_results.json"
TEST_SAMPLES = 1 # Pass@1 (Greedy or Temperature?)
# For Pass@1, we usually use Greedy (Temp=0) or Temp=0.2.
# Let's use Greedy to see if the "Best" output has improved.
TEMP = 0.0

def evaluate_model(model_path, name):
    print(f"Evaluating {name}...")
    # Note: loading LoRA adapters into vLLM is possible but we saved full finetune?
    # train_grpo.py saves `model.state_dict()`.
    # We need to merge or load carefully.
    # Since we saved state_dict of a CausalLM, we can load it into HuggingFace model, 
    # save it to a temp dir, and load vLLM from there.
    
    # Or just use Hugging Face for generation (slow but fine for evaluation of 500 probs * 1 sample).
    # Let's use Hugging Face to avoid weight format issues.
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    
    if model_path:
        print(f"Loading checkpoint from {model_path}...")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # Load Test Data
    dataset = load_dataset("mbpp", "sanitized", split="test")
    
    results = []
    passed_count = 0
    
    print(f"Generating solutions for {len(dataset)} problems...")
    for item in tqdm(dataset):
        prompt = item['prompt']
        tests = item['test_list']
        task_id = item['task_id']
        
        fmt_prompt = f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to solve the following problem:\n{prompt}\n\nYour code must pass these tests:\n{tests[0] if tests else ''}\n<|im_end|>\n<|im_start|>assistant\n```python\n"
        
        inputs = tokenizer(fmt_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy
                pad_token_id=tokenizer.pad_token_id
            )
            
        code = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if "```" in code:
            code = code.split("```")[0]
            
        # Execute
        # We use the simple exec logic
        header = "import math\nimport heapq\nimport itertools\nimport re\nimport collections\nfrom typing import *\n\n"
        full_script = header + code + "\n\n" + "\n".join(tests)
        
        is_pass = False
        try:
            exec_globals = {}
            exec(full_script, exec_globals)
            is_pass = True
            passed_count += 1
        except:
            is_pass = False
            
        results.append({
            "task_id": task_id,
            "pass": is_pass
        })
        
    pass_rate = passed_count / len(dataset)
    print(f"{name} Pass@1: {pass_rate*100:.2f}%")
    return pass_rate

def run_eval():
    # 1. Evaluate Baseline (if needed, or just use known number ~60%)
    # baseline = evaluate_model(None, "Baseline")
    
    # 2. Evaluate Latest Checkpoint
    # Find latest epoch
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Sort by epoch
    checkpoints.sort(key=lambda x: int(x.split('_e')[1].split('.')[0]))
    latest = checkpoints[-1]
    
    print(f"Found checkpoints: {checkpoints}. Testing {latest}...")
    
    grpo_acc = evaluate_model(os.path.join(CHECKPOINT_DIR, latest), "GRPO_Model")
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"model": latest, "pass_at_1": grpo_acc}, f)

if __name__ == "__main__":
    run_eval()
