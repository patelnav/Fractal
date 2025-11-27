
import json
import multiprocessing
import time
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_generations.jsonl"
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/mbpp_labeled.jsonl"
TIMEOUT = 2.0 # Seconds

def check_solution(sample):
    """
    Executes the code and tests in a separate process.
    Returns the modified sample with 'status' field.
    """
    code = sample['code']
    tests = sample['tests'] # List of assert statements
    
    # Construct the execution script
    # We need to allow standard imports usually required by MBPP (math, etc)
    # For safety, ideally we restrict, but for this experiment we trust the LLM output mostly.
    
    # Common imports needed by MBPP
    header = "import math\nimport heapq\nimport itertools\nimport re\nimport collections\nfrom typing import *\n\n"
    
    full_script = header + code + "\n\n" + "\n".join(tests)
    
    # We run this in a worker process
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    
    p = multiprocessing.Process(target=_execute_script, args=(full_script, result_dict))
    p.start()
    p.join(TIMEOUT)
    
    if p.is_alive():
        p.terminate()
        p.join()
        status = "timeout"
        error = "Execution timed out"
    else:
        if "error" in result_dict:
            status = "failed"
            error = result_dict["error"]
        else:
            status = "passed"
            error = None
            
    sample['status'] = status
    sample['error'] = error
    return sample

def _execute_script(script, result_dict):
    try:
        # We use a limited scope, but MBPP relies on global function definitions.
        # exec() in a fresh dict works best.
        exec_globals = {}
        exec(script, exec_globals)
        result_dict["success"] = True
    except Exception as e:
        result_dict["error"] = str(e)

def process_file():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    print(f"Labeling data from {INPUT_FILE}...")
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    print(f"Loaded {len(data)} samples. Running tests...")
    
    # Use Pool for parallel execution (CPU bound-ish due to overhead)
    # But multiprocessing inside multiprocessing (check_solution uses Process) is tricky.
    # Actually, check_solution spawns a Process. We should run check_solution sequentially or 
    # manage the pool carefully. 
    # Since we have 24k samples, serial execution with 2s timeout worst case = 13 hours.
    # We NEED parallel execution.
    # Better approach: Use Pool, but the worker function `check_solution_worker` 
    # just runs exec(). The Pool handles the timeout via `get(timeout)`.
    
    # Let's rewrite the parallel logic below.
    
    labeled_data = []
    
    # CPU count
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Map data to results
        # We use imap_unordered for progress bar
        results = list(tqdm(pool.imap(check_solution_safe, data), total=len(data)))
        
    print(f"Writing {len(results)} labeled samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
            
    # Stats
    passed = sum(1 for x in results if x['status'] == 'passed')
    print(f"Pass Rate: {passed}/{len(results)} ({passed/len(results)*100:.2f}%)")

def check_solution_safe(sample):
    """
    Wrapper to run check_solution logic properly in a pool.
    Since we can't spawn processes from daemonic processes (Pool workers),
    we must rely on the Pool itself or use a different strategy.
    
    Actually, standard 'exec' is blocking. To enforce timeout in a Pool worker, 
    we can use signal.alarm (Unix only) or just risk it?
    MBPP code rarely infinite loops, but it happens.
    
    Robust solution for Pool:
    Worker runs code. If it hangs, the whole Pool might stall.
    
    Alternative: Don't use Pool. Use a loop that spawns Process for each item (slow).
    
    Compromise: Use a custom timeout function using signals (works on Linux/Mac).
    """
    try:
        return run_with_timeout(sample)
    except Exception as e:
        sample['status'] = 'error'
        sample['error'] = str(e)
        return sample

def handler(signum, frame):
    raise TimeoutError("Timeout")

def run_with_timeout(sample):
    import signal
    
    # Register signal handler for timeout
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(TIMEOUT) + 1) # Set alarm
    
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
        signal.alarm(0) # Disable alarm
        
    return sample

if __name__ == "__main__":
    process_file()
