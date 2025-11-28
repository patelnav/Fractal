import signal
import ast
from typing import Dict, Any, Optional

# Timeout for execution in seconds
TIMEOUT = 3.0

def handler(signum, frame):
    raise TimeoutError("Execution Timed Out")

def load_humaneval_dataset(path: str = "HumanEval.jsonl"):
    """
    Loads HumanEval dataset from a JSONL file or tries to load from HuggingFace if file not found.
    """
    import json
    import os
    
    problems = []
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        return problems
    
    try:
        from datasets import load_dataset
        print("Loading HumanEval from HuggingFace...")
        ds = load_dataset("openai_humaneval", split="test")
        return [item for item in ds]
    except ImportError:
        print("Error: 'datasets' library not found and local file missing.")
        return []

def check_correctness(problem: Dict[str, Any], completion: str, timeout: float = 3.0) -> Dict[str, Any]:
    """
    Evaluates the functional correctness of a generated completion.
    """
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout) + 1)
    
    prompt = problem['prompt']
    test = problem['test']
    entry_point = problem['entry_point']
    
    # Construct the full script
    # We assume completion contains the function body or full function.
    # We need to handle imports. HumanEval prompts usually include imports if needed, 
    # but adding standard ones is safer.
    
    header = "import math\nimport re\nimport collections\nfrom typing import *\nimport heapq\nimport itertools\n\n"
    
    # Simple heuristic to avoid double definition
    if f"def {entry_point}" in completion:
        full_code = header + completion
    else:
        full_code = header + prompt + completion
        
    full_script = full_code + "\n\n" + test + f"\n\ncheck({entry_point})"
    
    result = {
        "task_id": problem.get('task_id', 'unknown'),
        "passed": False,
        "error": None,
        "code": completion
    }
    
    import traceback

    try:
        # Create a new global namespace for execution
        exec_globals = {}
        exec(full_script, exec_globals)
        result['passed'] = True
    except TimeoutError:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        
    return result

if __name__ == "__main__":
    # Test with a dummy problem
    dummy_problem = {
        "task_id": "test/0",
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(func):\n    assert func(1, 2) == 3\n    assert func(-1, 1) == 0"
    }
    
    dummy_completion = "    return a + b"
    print("Testing correct solution...")
    res = check_correctness(dummy_problem, dummy_completion)
    print(res)
    
    dummy_completion_wrong = "    return a - b"
    print("Testing incorrect solution...")
    res = check_correctness(dummy_problem, dummy_completion_wrong)
    print(res)
