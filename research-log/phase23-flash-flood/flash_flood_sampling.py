import os
import json
import argparse
import re
import time
import numpy as np
from tqdm import tqdm
from qwen_interface import QwenInterface
from humaneval_harness import load_humaneval_dataset, check_correctness

OUTPUT_FILE = "research-log/phase23-flash-flood/results_flood.jsonl"
SAMPLES_N = 50
TEMP_DIVERSITY = 0.8

def extract_signature_and_docstring(prompt: str) -> tuple[str, str]:
    match = re.search(r'("""|\'\'\')', prompt)
    if match:
        split_idx = match.start()
        signature = prompt[:split_idx].strip()
        docstring = prompt[split_idx:].strip()
        return signature, docstring
    return prompt.strip(), "No docstring provided."

def pass_at_k(n, c, k):
    """
    Unbiased estimator for pass@k.
    n: total samples
    c: correct samples
    k: k
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def run_baseline_sampling(model: QwenInterface, problems: list):
    print(f"Running Baseline Sampling (N={SAMPLES_N}, T={TEMP_DIVERSITY})...")
    
    prompts = [p['prompt'] for p in problems]
    
    # vLLM handles batching efficiently.
    # We generate N samples per prompt.
    
    # Note: QwenInterface needs to handle 'system' prompt for raw completion?
    # Actually, for Baseline (direct code), we usually use a Chat template:
    # System: You are a coder. User: <prompt>.
    # But HumanEval prompt is a raw completion prefix.
    # Qwen-2.5-Coder-Instruct expects Chat format.
    # So we wrap the prompt in User message: "Complete this function:\n{prompt}"
    
    chat_prompts = []
    for p in problems:
        chat_prompts.append(f"""<|im_start|>system
You are an expert Python developer. Complete the function below. Return ONLY the function body.
<|im_end|>
<|im_start|>user
{p['prompt']}
<|im_end|>
<|im_start|>assistant
""")

    # Batch Generate
    batch_results = model.batch_generate(chat_prompts, max_tokens=512, temperature=TEMP_DIVERSITY, n=SAMPLES_N)
    
    results = []
    for i, (texts, usage) in enumerate(batch_results):
        problem = problems[i]
        passed_count = 0
        
        for code in texts:
            # Strip markdown
            clean_code = code.replace("```python", "").replace("```", "")
            res = check_correctness(problem, clean_code)
            if res['passed']:
                passed_count += 1
                
        # Calculate Metrics
        p1 = pass_at_k(SAMPLES_N, passed_count, 1)
        p10 = pass_at_k(SAMPLES_N, passed_count, 10)
        p50 = pass_at_k(SAMPLES_N, passed_count, 50)
        
        result = {
            "task_id": problem['task_id'],
            "method": "baseline_sampling",
            "n": SAMPLES_N,
            "passed": passed_count,
            "pass@1": p1,
            "pass@10": p10,
            "pass@50": p50,
            "usage": usage
        }
        results.append(result)
        
    return results

def run_fractal_sampling(model: QwenInterface, problems: list):
    print(f"Running Fractal Sampling (N={SAMPLES_N} Sketches, T={TEMP_DIVERSITY})...")
    
    results = []
    
    # Step 1: Generate N Sketches per problem
    sketch_prompts = []
    for p in problems:
        sig, doc = extract_signature_and_docstring(p['prompt'])
        sketch_prompts.append(f"""<|im_start|>system
You are an expert Python architect. Create a step-by-step implementation plan (pseudocode) for the function.
<|im_end|>
<|im_start|>user
Function Signature:
{sig}

Docstring:
{doc}

Please provide the plan.
<|im_end|>
<|im_start|>assistant
""")

    # Generate Sketches (High Temp)
    sketch_batches = model.batch_generate(sketch_prompts, max_tokens=512, temperature=TEMP_DIVERSITY, n=SAMPLES_N)
    
    # Step 2: Generate Code for EACH Sketch
    # We need to flatten this for batching
    # Total requests = Problems * N
    
    code_prompts = []
    map_to_problem = [] # index -> problem_index
    
    for prob_idx, (sketches, _) in enumerate(sketch_batches):
        p = problems[prob_idx]
        sig, doc = extract_signature_and_docstring(p['prompt'])
        
        for sketch in sketches:
            guide_prompt = f"""<|im_start|>system
You are an expert Python developer. Implement the full function body based on the provided plan.
<|im_end|>
<|im_start|>user
Function Signature:
{sig}

Docstring:
{doc}

Implementation Plan:
{sketch}

Write the complete Python code for the function body. Follow the plan exactly.
<|im_end|>
<|im_start|>assistant
"""
            code_prompts.append(guide_prompt)
            map_to_problem.append(prob_idx)
            
    # Generate Code (Low Temp - rely on Sketch for diversity)
    # Batch size might be huge (15 * 50 = 750). vLLM handles this fine usually.
    # If OOM, we might need to chunk.
    
    # Let's process in chunks of 100
    CHUNK_SIZE = 100
    all_texts = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    for i in range(0, len(code_prompts), CHUNK_SIZE):
        chunk = code_prompts[i:i+CHUNK_SIZE]
        print(f"  Processing Code Chunk {i}-{i+len(chunk)}...")
        chunk_res = model.batch_generate(chunk, max_tokens=1024, temperature=0.2, n=1)
        
        for texts, usage in chunk_res:
            all_texts.append(texts[0]) # n=1
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            
    # Aggregate Results
    # Group by problem
    problem_correct = [0] * len(problems)
    
    for i, code in enumerate(all_texts):
        prob_idx = map_to_problem[i]
        clean_code = code.replace("```python", "").replace("```", "")
        res = check_correctness(problems[prob_idx], clean_code)
        if res['passed']:
            problem_correct[prob_idx] += 1
            
    for i, p in enumerate(problems):
        c = problem_correct[i]
        p1 = pass_at_k(SAMPLES_N, c, 1)
        p10 = pass_at_k(SAMPLES_N, c, 10)
        p50 = pass_at_k(SAMPLES_N, c, 50)
        
        result = {
            "task_id": p['task_id'],
            "method": "fractal_sampling",
            "n": SAMPLES_N,
            "passed": c,
            "pass@1": p1,
            "pass@10": p10,
            "pass@50": p50,
            "usage": "aggregated" # Simplified
        }
        results.append(result)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--mode", choices=["baseline", "fractal", "both"], default="both")
    args = parser.parse_args()
    
    model = QwenInterface()
    problems = load_humaneval_dataset()
    
    if args.representative:
        target_ids = [
            'HumanEval/0', 'HumanEval/1', 'HumanEval/10', 'HumanEval/11', 'HumanEval/12', 
            'HumanEval/26', 'HumanEval/29', 'HumanEval/32', 'HumanEval/33', 'HumanEval/37', 
            'HumanEval/39', 'HumanEval/40', 'HumanEval/43', 'HumanEval/46', 'HumanEval/129'
        ]
        problems = [p for p in problems if p['task_id'] in target_ids]
        
    all_results = []
    
    if args.mode in ["baseline", "both"]:
        res_b = run_baseline_sampling(model, problems)
        all_results.extend(res_b)
        
    if args.mode in ["fractal", "both"]:
        res_f = run_fractal_sampling(model, problems)
        all_results.extend(res_f)
        
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
            
    print("Done. Saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
