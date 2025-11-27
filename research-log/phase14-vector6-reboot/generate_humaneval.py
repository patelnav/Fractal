from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import os
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/humaneval_generations.jsonl"
NUM_SAMPLES_PER_PROBLEM = 50
MAX_NEW_TOKENS = 512
TEMP = 0.8

def format_prompt(problem_desc, entry_point):
    # HumanEval prompts usually include the signature.
    # The dataset provides 'prompt' which is the function signature + docstring.
    # Qwen expects a chat format.
    
    prompt_content = f"""<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Complete the following Python function:
{problem_desc}
<|im_end|>
<|im_start|>assistant
```python
"""
    return prompt_content

def generate_humaneval():
    print(f"Initializing vLLM with {MODEL_NAME}...")
    llm = LLM(model=MODEL_NAME, dtype="float16", trust_remote_code=True)
    
    print("Loading HumanEval Dataset...")
    dataset = load_dataset("openai_humaneval")
    problems = list(dataset['test'])
    
    print(f"Preparing Prompts for {len(problems)} problems...")
    
    prompts = []
    meta_data = []
    
    for prob in problems:
        # HumanEval 'prompt' field contains the function signature and docstring.
        p_str = format_prompt(prob['prompt'], prob['entry_point'])
        prompts.append(p_str)
        meta_data.append({
            "task_id": prob['task_id'],
            "prompt": prob['prompt'],
            "test": prob['test'],
            "entry_point": prob['entry_point']
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
                
                # HumanEval requires stitching the prompt + code usually, 
                # but since we prompt Qwen to "Complete", it might output just the body 
                # or the full function depending on how it feels.
                # Qwen-Coder usually outputs the full block if prompted with ```python
                
                entry = {
                    "task_id": task_id,
                    "prompt": meta['prompt'],
                    "test": meta['test'],
                    "entry_point": meta['entry_point'],
                    "code": code
                }
                f.write(json.dumps(entry) + "\n")
                
    print("HumanEval Generation Complete.")

if __name__ == "__main__":
    generate_humaneval()
