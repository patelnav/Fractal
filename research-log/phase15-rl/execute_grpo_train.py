
import json
import multiprocessing
import os
from tqdm import tqdm
import signal

INPUT_FILE = "research-log/phase15-rl/data/grpo_train_generations.jsonl"
OUTPUT_FILE = "research-log/phase15-rl/data/grpo_train_labeled.jsonl"
TIMEOUT = 2.0

def handler(signum, frame):
    raise TimeoutError("Timeout")

def check_solution_safe(sample):
    import signal
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
    except:
        sample['status'] = 'failed'
    finally:
        signal.alarm(0)

    return sample

def process_file():
    if not os.path.exists(INPUT_FILE):
        print("Input not found")
        return

    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Labeling {len(data)} samples...")

    with multiprocessing.Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(check_solution_safe, data), total=len(data)))

    with open(OUTPUT_FILE, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    passed = sum(1 for x in results if x['status'] == 'passed')
    print(f"Pass Rate: {passed/len(results)*100:.2f}%")

if __name__ == "__main__":
    process_file()
