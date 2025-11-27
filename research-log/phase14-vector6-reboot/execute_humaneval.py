
import json
import multiprocessing
import time
import os
from tqdm import tqdm
import signal

# Configuration
INPUT_FILE = "research-log/phase14-vector6-reboot/data/humaneval_generations.jsonl"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/humaneval_labeled.jsonl"
TIMEOUT = 2.0

def handler(signum, frame):
    raise TimeoutError("Timeout")

def check_solution_safe(sample):
    import signal
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(TIMEOUT) + 1)
    
    # HumanEval Logic:
    # The 'code' usually completes the function.
    # The 'prompt' has the signature.
    # The 'test' has the assertions.
    # We need to combine: imports + prompt + code + test + check_function
    
    prompt = sample['prompt']
    code = sample['code']
    test = sample['test']
    entry_point = sample['entry_point']
    
    # Clean code: Sometimes models repeat the signature.
    # If code starts with "def ", we might not need the prompt?
    # But the prompt has imports sometimes.
    # Safe bet: standard imports + prompt + code + test + check(entry_point)
    
    header = "import math\nimport re\nimport collections\nfrom typing import *\nimport heapq\nimport itertools\n\n"
    
    # We need to be careful. If Qwen repeats the signature, appending prompt + code causes SyntaxError.
    # Simple heuristic: If code contains "def {entry_point}", use only code. Else use prompt + code.
    
    if f"def {entry_point}" in code:
        full_code = header + code
    else:
        full_code = header + prompt + code
        
    # HumanEval tests usually call the function directly or use check(candidate).
    # The 'test' field in HuggingFace dataset usually contains a function `check(candidate)` 
    # and then calls `check(entry_point_function)`.
    
    full_script = full_code + "\n\n" + test + f"\n\ncheck({entry_point})"
    
    try:
        exec_globals = {}
        exec(full_script, exec_globals)
        sample['status'] = 'passed'
        sample['error'] = None
    except TimeoutError:
        sample['status'] = 'timeout'
        sample['error'] = 'Execution Timed Out'
    except Exception as e:
        sample['status'] = 'failed'
        sample['error'] = str(e)
    finally:
        signal.alarm(0)
        
    return sample

def process_file():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    print(f"Labeling HumanEval data from {INPUT_FILE}...")
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"Loaded {len(data)} samples. Running tests...")
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(check_solution_safe, data), total=len(data)))
        
    print(f"Writing {len(results)} labeled samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    passed = sum(1 for x in results if x['status'] == 'passed')
    print(f"Pass Rate: {passed}/{len(results)} ({passed/len(results)*100:.2f}%)")

if __name__ == "__main__":
    process_file()
