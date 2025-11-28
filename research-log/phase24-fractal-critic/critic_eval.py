import os
import json
import argparse
import random
from tqdm import tqdm
from qwen_interface import QwenInterface

INPUT_FILE = "research-log/phase23-flash-flood/results_flood.jsonl"
OUTPUT_FILE = "research-log/phase24-fractal-critic/results_critic.jsonl"

def load_candidates():
    """
    Loads candidates from Phase 23 results.
    Reconstructs the pool: We have aggregate stats, but not the raw code in the jsonl?
    Ah, Phase 23 script saved aggregate stats, NOT the raw code samples!
    
Mistake in Phase 23 implementation: We didn't save the 100 generated codes per problem.
    We only saved the stats.
    
    We need to RE-GENERATE the pool. 
    Or just run the generation loop again here.
    
    Let's regenerate a smaller pool (N=10) for this Critic experiment to save time.
    """
    pass

def run_critic_experiment(model: QwenInterface, representative: bool = True):
    from humaneval_harness import load_humaneval_dataset, check_correctness
    
    problems = load_humaneval_dataset()
    if representative:
        target_ids = [
            'HumanEval/0', 'HumanEval/1', 'HumanEval/10', 'HumanEval/11', 'HumanEval/12', 
            'HumanEval/26', 'HumanEval/29', 'HumanEval/32', 'HumanEval/33', 'HumanEval/37', 
            'HumanEval/39', 'HumanEval/40', 'HumanEval/43', 'HumanEval/46', 'HumanEval/129'
        ]
        problems = [p for p in problems if p['task_id'] in target_ids]
        
    results = []
    
    print("Generating Pool and Scoring...")
    
    for problem in tqdm(problems):
        task_id = problem['task_id']
        
        # 1. Generate Pool (N=10 Baseline)
        # We use Baseline Sampling because it had higher diversity/quality in Phase 23
        prompts = [f"""<|im_start|>system
You are an expert Python developer. Complete the function below. Return ONLY the function body.
<|im_end|>
<|im_start|>user
{problem['prompt']}
<|im_end|>
<|im_start|>assistant
"""]
        
        # Generate 10 samples
        batch_res = model.batch_generate(prompts, max_tokens=512, temperature=0.8, n=10)
        candidates = batch_res[0][0] # list of strings
        
        # 2. Ground Truth
        labeled_pool = []
        for code in candidates:
            clean_code = code.replace("```python", "").replace("```", "")
            exec_res = check_correctness(problem, clean_code)
            labeled_pool.append({
                "code": clean_code,
                "passed": exec_res['passed']
            })
            
        # 3. Critic Scoring
        # We score ALL candidates with BOTH methods
        
        # Extract Sig/Doc
        import re
        match = re.search(r'("""|""")', problem['prompt'])
        if match:
            split_idx = match.start()
            signature = problem['prompt'][:split_idx].strip()
            docstring = problem['prompt'][split_idx:].strip()
        else:
            signature = problem['prompt']
            docstring = ""
            
        best_baseline_score = -1
        best_baseline_candidate = None
        
        best_fractal_score = -1
        best_fractal_candidate = None
        
        for cand in labeled_pool:
            # Score Baseline
            s_base = model.score_solution(signature, docstring, cand['code'], method="baseline")
            cand['score_baseline'] = s_base
            
            if s_base > best_baseline_score:
                best_baseline_score = s_base
                best_baseline_candidate = cand
            
            # Score Fractal
            s_frac = model.score_solution(signature, docstring, cand['code'], method="fractal")
            cand['score_fractal'] = s_frac
            
            if s_frac > best_fractal_score:
                best_fractal_score = s_frac
                best_fractal_candidate = cand
                
        # 4. Evaluate Ranking
        # Did the best-scored candidate pass?
        
        # Tie-breaking: If multiple max scores, pick first (random).
        # Or average passed rate of top tier.
        
        # Baseline Selection Result
        baseline_success = False
        if best_baseline_candidate:
            baseline_success = best_baseline_candidate['passed']
            
        # Fractal Selection Result
        fractal_success = False
        if best_fractal_candidate:
            fractal_success = best_fractal_candidate['passed']
            
        # Any correct in pool?
        any_correct = any(c['passed'] for c in labeled_pool)
        
        res = {
            "task_id": task_id,
            "pool_size": 10,
            "any_correct": any_correct,
            "baseline_selected_correct": baseline_success,
            "fractal_selected_correct": fractal_success,
            "pool_details": labeled_pool
        }
        results.append(res)
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(res) + "\n")
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true")
    args = parser.parse_args()
    
    model = QwenInterface()
    run_critic_experiment(model, args.representative)

if __name__ == "__main__":
    main()
