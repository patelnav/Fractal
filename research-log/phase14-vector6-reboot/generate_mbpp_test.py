from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import os
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_test_generations.jsonl"
NUM_SAMPLES_PER_PROBLEM = 50
MAX_NEW_TOKENS = 512
TEMP = 0.8

def format_prompt(problem_desc, tests):
    tests_str = "\n".join(tests)
    prompt = f"""<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a Python function to solve the following problem:
{problem_desc}

Your code must pass these tests:
{tests_str}
<|im_end|>
<|im_start|>assistant
```python
"""
    return prompt

def generate_test_solutions():
    print(f"Initializing vLLM with {MODEL_NAME}...")
    llm = LLM(model=MODEL_NAME, dtype="float16", trust_remote_code=True)
    
    print("Loading MBPP Dataset (Test Split)...")
    dataset = load_dataset("mbpp", "sanitized")
    # MBPP 'test' split is problems 375-874 (500 problems)
    test_probs = list(dataset['test'])
    
    print(f"Preparing Prompts for {len(test_probs)} test problems...")
    
    prompts = []
    meta_data = []
    
    for prob in test_probs:
        p_str = format_prompt(prob['prompt'], prob['test_list'])
        prompts.append(p_str)
        meta_data.append({
            "task_id": prob['task_id'],
            "prompt": prob['prompt'],
            "tests": prob['test_list']
        })

    sampling_params = SamplingParams(
        temperature=TEMP,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        n=NUM_SAMPLES_PER_PROBLEM,
        stop=["```", "<|im_end|>"]
    )
    
    print(f"Generating {len(prompts) * NUM_SAMPLES_PER_PROBLEM} total solutions...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"Writing results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, output in enumerate(outputs):
            meta = meta_data[i]
            task_id = meta['task_id']
            
            for sample in output.outputs:
                code = sample.text.strip()
                if "```" in code:
                    code = code.split("```")[0]
                
                entry = {
                    "task_id": task_id,
                    "prompt": meta['prompt'],
                    "tests": meta['tests'],
                    "code": code
                }
                f.write(json.dumps(entry) + "\n")
                
    print("Test Generation Complete.")

if __name__ == "__main__":
    generate_test_solutions()
