
import json
import multiprocessing
import time
import os
from tqdm import tqdm
import signal

# Configuration
INPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_test_generations.jsonl"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_test_labeled.jsonl"
TIMEOUT = 2.0 

def handler(signum, frame):
    raise TimeoutError("Timeout")

def check_solution_safe(sample):
    import signal
    
    # Register signal handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(TIMEOUT) + 1)
    
    code = sample['code']
    tests = sample['tests']
    
    header = "import math\nimport heapq\nimport itertools\nimport re\nimport collections\nfrom typing import *\n\n"
    full_script = header + code + "\n\n" + "\n".join(tests)
    
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

    print(f"Labeling Test data from {INPUT_FILE}...")
    
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
