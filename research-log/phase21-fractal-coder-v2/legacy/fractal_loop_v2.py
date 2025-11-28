import os
import json
import argparse
import re
import time
from tqdm import tqdm
from qwen_interface import QwenInterface
from humaneval_harness import load_humaneval_dataset, check_correctness

OUTPUT_FILE = "research-log/phase21-fractal-coder-v2/results_v2.jsonl"

def parse_sketch(sketch: str) -> list[str]:
    """
    Parses the sketch into a list of steps (comments).
    Assumes the model outputs a list of comments starting with # or numbered list.
    Filters out conversational filler.
    """
    lines = sketch.split('\n')
    steps = []
    for line in lines:
        line = line.strip()
        
        # 1. Skip empty lines
        if not line:
            continue
            
        # 2. Skip likely conversational filler
        lower_line = line.lower()
        if lower_line.startswith("here is") or lower_line.startswith("sure") or lower_line.startswith("certainly"):
            continue
            
        # 3. Keep lines that look like plan steps
        # - Start with #
        # - Start with digits (1., 2))
        # - Start with - or * (bullets)
        if (line.startswith('#') or 
            (len(line) > 0 and line[0].isdigit()) or 
            line.startswith('-') or 
            line.startswith('*')):
             steps.append(line)
             
    return steps

def extract_signature_and_docstring(prompt: str) -> tuple[str, str]:
    """
    Separates the function signature and docstring from the HumanEval prompt.
    """
    # Simple heuristic: Split by the first docstring quote
    # HumanEval prompts usually start with def ...
    
    # Try to find the start of the docstring
    match = re.search(r'(\"\"\"|\'\'\')', prompt)
    if match:
        split_idx = match.start()
        signature = prompt[:split_idx].strip()
        # The rest is docstring (including the opening quotes)
        docstring = prompt[split_idx:].strip()
        return signature, docstring
    
    # Fallback: Treat entire prompt as signature if no docstring found (rare in HumanEval)
    return prompt.strip(), "No docstring provided."

def solve_problem_fractal(model: QwenInterface, problem: dict) -> dict:
    start_time = time.time()
    full_prompt = problem['prompt']
    signature, docstring = extract_signature_and_docstring(full_prompt)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # 1. Generate Sketch
    sketch, usage = model.generate_sketch(signature, docstring)
    total_prompt_tokens += usage.get("prompt_tokens", 0)
    total_completion_tokens += usage.get("completion_tokens", 0)
    
    steps = parse_sketch(sketch)
    
    if not steps:
        steps = ["# Implement the function logic"]
        
    # 2. Render (Sketch-then-Fill)
    code_body = ""
    
    for step in steps:
        context = full_prompt + "\n" + code_body
        
        segment, usage = model.render_step(signature, docstring, sketch, step, context_so_far=code_body)
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        
        # Clean markdown
        segment = segment.replace("```python", "").replace("```", "")
        
        lines = segment.split('\n')
        clean_lines = []
        
        # Normalize step for comparison
        step_norm = step.lstrip('# ').strip().lower()
        
        for line in lines:
            if not line.strip(): continue
            
            # Filter duplicate comments
            line_norm = line.lstrip('# ').strip().lower()
            if line_norm == step_norm:
                continue
            
            # Filter step counters
            if 0 < len(line_norm) < 5 and (line_norm[0].isdigit() or line_norm.startswith("step")):
                 continue

            clean_lines.append(line)
            
        # --- Dynamic Indentation V2 ---
        # Determine the required indent based on the *last line* of the current code_body
        
        # Defaults
        last_indent = 0
        force_indent = False
        
        if code_body:
            body_lines = code_body.rstrip().split('\n')
            if body_lines:
                last_line = body_lines[-1]
                last_indent = len(last_line) - len(last_line.lstrip())
                
                if last_line.strip().endswith(':'):
                    target_indent = last_indent + 4
                    force_indent = True
                else:
                    target_indent = last_indent
                    force_indent = False
        else:
            # Initial state: Inside function def, so base indent is 4
            target_indent = 4
            force_indent = False
        
        if not clean_lines:
            # Just append comment
            indent_str = " " * target_indent
            code_body += f"\n{indent_str}{step}\n"
            continue
            
        # Analyze Model Output
        first_line = clean_lines[0]
        model_indent = len(first_line) - len(first_line.lstrip())
        
        final_lines = []
        shift_needed = 0
        
        if model_indent == 0:
            # Case A: Model is flat. Assume it matches target.
            shift_needed = target_indent
        else:
            # Case B: Model is indented.
            if force_indent and model_indent < target_indent:
                # We MUST be deeper (opened a block), but model didn't indent enough.
                # Force it to target.
                shift_needed = target_indent - model_indent
            else:
                # Trust model (could be dedent, or correct indent)
                shift_needed = 0
                
        # Apply Shift
        for line in clean_lines:
            if shift_needed > 0:
                if model_indent == 0:
                    final_lines.append((" " * shift_needed) + line)
                else:
                    final_lines.append((" " * shift_needed) + line)
            else:
                final_lines.append(line)

        # Re-calculate block indent for the comment
        if final_lines:
            block_indent = final_lines[0][:len(final_lines[0]) - len(final_lines[0].lstrip())]
        else:
            block_indent = " " * target_indent
            
        code_body += f"\n{block_indent}{step}\n"
        code_body += "\n".join(final_lines) + "\n"
            
    end_time = time.time()
    
    return {
        "sketch": sketch,
        "steps": steps,
        "generated_code": code_body,
        "method": "fractal_sketch_fill",
        "latency": end_time - start_time,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens
    }

def solve_problem_baseline(model: QwenInterface, problem: dict) -> dict:
    start_time = time.time()
    full_prompt = problem['prompt']
    signature, docstring = extract_signature_and_docstring(full_prompt)
    
    # Generate full body in one go
    code, usage = model.render_full_body(signature, docstring, "Implement the function.")
    
    # Clean markdown
    code = code.replace("```python", "").replace("```", "")
    
    end_time = time.time()
    
    return {
        "generated_code": code,
        "method": "baseline",
        "latency": end_time - start_time,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--mode", type=str, default="fractal", choices=["fractal", "baseline"])
    parser.add_argument("--representative", action="store_true", help="Run only representative structural problems")
    args = parser.parse_args()

    # Initialize Model
    try:
        model = QwenInterface()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Load Data
    problems = load_humaneval_dataset()
    
    if args.representative:
        target_ids = [
            'HumanEval/0', 'HumanEval/1', 'HumanEval/10', 'HumanEval/11', 'HumanEval/12', 
            'HumanEval/26', 'HumanEval/29', 'HumanEval/32', 'HumanEval/33', 'HumanEval/37', 
            'HumanEval/39', 'HumanEval/40', 'HumanEval/43', 'HumanEval/46', 'HumanEval/129'
        ]
        problems = [p for p in problems if p['task_id'] in target_ids]
        print(f"Filtered to {len(problems)} representative structural problems.")
        
    if args.limit:
        problems = problems[:args.limit]
    
    print(f"Loaded {len(problems)} problems. Running mode: {args.mode}")
    
    results = []
    
    for problem in tqdm(problems):
        if args.mode == "fractal":
            result_data = solve_problem_fractal(model, problem)
        else:
            result_data = solve_problem_baseline(model, problem)
            
        # Execute
        # The 'generated_code' might need to be combined with 'prompt' depending on how check_correctness works.
        # check_correctness handles "def ..." check.
        # Our fractal generation produces a body (indented). 
        # Our baseline produces whatever Qwen outputs (likely full function or body).
        
        exec_result = check_correctness(problem, result_data['generated_code'])
        
        result_data.update(exec_result)
        results.append(result_data)
        
        # Save incrementally
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(result_data) + "\n")
            
    pass_rate = sum(1 for r in results if r['passed']) / len(results)
    print(f"Final Pass Rate: {pass_rate:.2%}")

if __name__ == "__main__":
    main()
