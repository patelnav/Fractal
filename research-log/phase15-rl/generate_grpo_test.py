import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CHECKPOINT_PATH = "research-log/phase15-rl/checkpoints/grpo_e5.pt" # Using Epoch 5
OUTPUT_FILE = "research-log/phase15-rl/data/grpo_test_generations.jsonl"
NUM_SAMPLES = 50
TEMP = 0.8

def generate_grpo_test():
    print(f"Loading GRPO Checkpoint from {CHECKPOINT_PATH}...")
    
    # vLLM cannot load a state_dict directly easily without saving to HF format first.
    # So we will convert the checkpoint to a HF directory first.
    
    temp_model_dir = "research-log/phase15-rl/checkpoints/grpo_hf_format"
    
    if not os.path.exists(temp_model_dir):
        print("Converting checkpoint to HF format...")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)
        print(f"Saved HF model to {temp_model_dir}")
    
    print("Initializing vLLM with GRPO Model...")
    llm = LLM(model=temp_model_dir, dtype="bfloat16", trust_remote_code=True)
    
    print("Loading MBPP Test Data...")
    dataset = load_dataset("mbpp", "sanitized", split="test")
    
    prompts = []
    meta_data = []
    
    for prob in dataset:
        tests_str = "\n".join(prob['test_list'])
        p_str = f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to solve the following problem:\n{prob['prompt']}\n\nYour code must pass these tests:\n{tests_str}\n<|im_end|>\n<|im_start|>assistant\n```python\n"
        
        prompts.append(p_str)
        meta_data.append({
            "task_id": prob['task_id'],
            "prompt": prob['prompt'],
            "tests": prob['test_list']
        })
        
    sampling_params = SamplingParams(
        temperature=TEMP,
        top_p=0.95,
        max_tokens=512,
        n=NUM_SAMPLES,
        stop=["```", "<|im_end|>"]
    )
    
    print(f"Generating {len(prompts) * NUM_SAMPLES} solutions...")
    outputs = llm.generate(prompts, sampling_params)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for i, output in enumerate(outputs):
            meta = meta_data[i]
            for sample in output.outputs:
                code = sample.text.strip()
                if "```" in code:
                    code = code.split("```")[0]
                
                entry = {
                    "task_id": meta['task_id'],
                    "prompt": meta['prompt'],
                    "tests": meta['tests'],
                    "code": code
                }
                f.write(json.dumps(entry) + "\n")
                
    print("Done.")

if __name__ == "__main__":
    generate_grpo_test()
