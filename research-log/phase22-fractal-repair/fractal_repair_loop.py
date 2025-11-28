import os
import json
import argparse
import re
import time
from tqdm import tqdm
from qwen_interface import QwenInterface
from humaneval_harness import load_humaneval_dataset, check_correctness

OUTPUT_FILE = "research-log/phase22-fractal-repair/results_repair.jsonl"
MAX_RETRIES = 5

def extract_signature_and_docstring(prompt: str) -> tuple[str, str]:
    match = re.search(r'("""|\'\'\')', prompt)
    if match:
        split_idx = match.start()
        signature = prompt[:split_idx].strip()
        docstring = prompt[split_idx:].strip()
        return signature, docstring
    return prompt.strip(), "No docstring provided."

def generate_guided_code(model: QwenInterface, signature: str, docstring: str, sketch: str) -> tuple[str, dict]:
    guide_prompt = f"""<|im_start|>system
You are an expert Python developer. Implement the full function body based on the provided plan.
<|im_end|>
<|im_start|>user
Function Signature:
{signature}

Docstring:
{docstring}

Implementation Plan:
{sketch}

Write the complete Python code for the function body. Follow the plan exactly.
<|im_end|>
<|im_start|>assistant
"""
    code_body, usage = model._generate(guide_prompt, max_tokens=1024, temperature=0.2)
    code_body = code_body.replace("```python", "").replace("```", "")
    return code_body, usage

def solve_with_repair(model: QwenInterface, problem: dict) -> dict:
    start_time = time.time()
    full_prompt = problem['prompt']
    signature, docstring = extract_signature_and_docstring(full_prompt)
    
    trajectory = []
    final_status = "failed"
    
    # --- Attempt 0 (Baseline) ---
    sketch, usage_s = model.generate_sketch(signature, docstring)
    code, usage_c = generate_guided_code(model, signature, docstring, sketch)
    
    # Execute
    exec_result = check_correctness(problem, code)
    
    step_0_data = {
        "step": 0,
        "sketch": sketch,
        "code": code,
        "passed": exec_result['passed'],
        "error": exec_result.get('error'),
        "usage": {
            "prompt_tokens": usage_s["prompt_tokens"] + usage_c["prompt_tokens"],
            "completion_tokens": usage_s["completion_tokens"] + usage_c["completion_tokens"]
        }
    }
    trajectory.append(step_0_data)
    
    current_sketch = sketch
    current_code = code
    current_error = exec_result.get('error')
    
    if exec_result['passed']:
        final_status = "passed_at_0"
    else:
        # --- Repair Loop ---
        for i in range(1, MAX_RETRIES + 1):
            # 1. Repair Sketch
            new_sketch, usage_r = model.repair_sketch(signature, docstring, current_sketch, current_code, current_error)
            
            # Check for No-Op (simple string equality for now, maybe fuzzy later)
            if new_sketch.strip() == current_sketch.strip():
                # Break loop if model refuses to change plan
                # Or maybe increase temp? For now, just log and continue (maybe code gen varies)
                pass

            # 2. Regenerate Code
            new_code, usage_g = generate_guided_code(model, signature, docstring, new_sketch)
            
            # 3. Execute
            new_exec_result = check_correctness(problem, new_code)
            
            step_data = {
                "step": i,
                "sketch": new_sketch,
                "code": new_code,
                "passed": new_exec_result['passed'],
                "error": new_exec_result.get('error'),
                "usage": {
                    "prompt_tokens": usage_r["prompt_tokens"] + usage_g["prompt_tokens"],
                    "completion_tokens": usage_r["completion_tokens"] + usage_g["completion_tokens"]
                }
            }
            trajectory.append(step_data)
            
            if new_exec_result['passed']:
                final_status = f"passed_at_{i}"
                break
                
            # Update for next iteration
            current_sketch = new_sketch
            current_code = new_code
            current_error = new_exec_result.get('error')
            
    end_time = time.time()
    
    return {
        "task_id": problem['task_id'],
        "final_status": final_status,
        "total_latency": end_time - start_time,
        "trajectory": trajectory
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true", help="Run representative set")
    parser.add_argument("--limit", type=int, default=None)
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
        
    if args.limit:
        problems = problems[:args.limit]
        
    # Clear output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    print(f"Starting Repair Loop Evaluation on {len(problems)} problems...")
    
    results = []
    for problem in tqdm(problems):
        res = solve_with_repair(model, problem)
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(res) + "\n")
        results.append(res)
        
    # Quick Summary
    passed = len([r for r in results if "passed" in r['final_status']])
    print(f"Final Pass Rate: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
