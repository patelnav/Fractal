from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

dataset = load_dataset("mbpp", "sanitized", split="test")
passed = 0

print(f"Evaluating Baseline on {len(dataset)} problems...")
for item in tqdm(dataset):
    prompt = item['prompt']
    tests = item['test_list']
    fmt_prompt = f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to solve the following problem:\n{prompt}\n\nYour code must pass these tests:\n{tests[0] if tests else ''}\n<|im_end|>\n<|im_start|>assistant\n```python\n"
    inputs = tokenizer(fmt_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    code = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    if "```" in code:
        code = code.split("```")[0]
    header = "import math\nimport heapq\nimport itertools\nimport re\nimport collections\nfrom typing import *\n\n"
    full_script = header + code + "\n\n" + "\n".join(tests)
    try:
        exec(full_script, {})
        passed += 1
    except:
        pass

print(f"Baseline Pass@1: {passed/len(dataset)*100:.2f}%")
