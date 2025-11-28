import os
import json
import argparse
import re
import time
from tqdm import tqdm
from qwen_interface import QwenInterface
from humaneval_harness import load_humaneval_dataset, check_correctness

OUTPUT_FILE = "research-log/phase21-fractal-coder-v2/results_v3_guided.jsonl"

def extract_signature_and_docstring(prompt: str) -> tuple[str, str]:
    match = re.search(r'("""|\'\'\')', prompt)
    if match:
        split_idx = match.start()
        signature = prompt[:split_idx].strip()
        docstring = prompt[split_idx:].strip()
        return signature, docstring
    return prompt.strip(), "No docstring provided."

def solve_problem_guided(model: QwenInterface, problem: dict) -> dict:
    start_time = time.time()
    full_prompt = problem['prompt']
    signature, docstring = extract_signature_and_docstring(full_prompt)
    
    # 1. Generate Sketch
    sketch, sketch_usage = model.generate_sketch(signature, docstring)
    
    # 2. Render Full Body Guided by Sketch
    # We manually construct the prompt here to ensure the sketch is used as context
    
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
    code_body, code_usage = model._generate(guide_prompt, max_tokens=1024, temperature=0.2)
    
    code_body = code_body.replace("```python", "").replace("```", "")
    
    total_usage = {
        "prompt_tokens": sketch_usage["prompt_tokens"] + code_usage["prompt_tokens"],
        "completion_tokens": sketch_usage["completion_tokens"] + code_usage["completion_tokens"]
    }
    
    end_time = time.time()
    
    return {
        "sketch": sketch,
        "generated_code": code_body,
        "method": "fractal_v3_guided",
        "latency": end_time - start_time,
        "usage": total_usage
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true")
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
        
    for problem in tqdm(problems):
        result = solve_problem_guided(model, problem)
        exec_result = check_correctness(problem, result['generated_code'])
        result.update(exec_result)
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
