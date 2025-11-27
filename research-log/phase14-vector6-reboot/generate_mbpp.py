from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import os
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_generations.jsonl"
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

def generate_solutions():
    print(f"Initializing vLLM with {MODEL_NAME}...")
    # vLLM handles memory management and batching automatically
    llm = LLM(model=MODEL_NAME, dtype="float16", trust_remote_code=True)
    
    print("Loading MBPP Dataset...")
    dataset = load_dataset("mbpp", "sanitized")
    train_probs = list(dataset['train']) + list(dataset['validation'])
    
    print(f"Preparing Prompts for {len(train_probs)} problems...")
    
    # We want 50 samples per problem.
    # vLLM supports 'n' in SamplingParams to generate multiple outputs per prompt efficienty.
    prompts = []
    meta_data = [] # To keep track of task_id etc. 
    
    for prob in train_probs:
        p_str = format_prompt(prob['prompt'], prob['test_list'])
        prompts.append(p_str)
        meta_data.append({
            "task_id": prob['task_id'],
            "prompt": prob['prompt'],
            "tests": prob['test_list']
        })

    # Configure Sampling
    # n=NUM_SAMPLES_PER_PROBLEM means generate 50 outputs for each prompt
    sampling_params = SamplingParams(
        temperature=TEMP,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        n=NUM_SAMPLES_PER_PROBLEM,
        stop=["```", "<|im_end|>"] # Stop early
    )
    
    print(f"Generating {len(prompts) * NUM_SAMPLES_PER_PROBLEM} total solutions...")
    
    # Run vLLM
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"Writing results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for i, output in enumerate(outputs):
            meta = meta_data[i]
            task_id = meta['task_id']
            
            for sample in output.outputs:
                code = sample.text.strip()
                
                # Cleanup if needed (vLLM stop tokens handles most, but just in case)
                if "```" in code:
                    code = code.split("```")[0]
                
                entry = {
                    "task_id": task_id,
                    "prompt": meta['prompt'],
                    "tests": meta['tests'],
                    "code": code,
                    "full_text": output.prompt + sample.text
                }
                f.write(json.dumps(entry) + "\n")
                
    print("Generation Complete.")

if __name__ == "__main__":
    generate_solutions()