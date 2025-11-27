
import sys
import random
import torch
from pathlib import Path
from typing import List, Tuple

# Add path for existing synthetic data logic
sys.path.insert(0, str(Path(__file__).parent.parent / "phase18-fractal-coder"))

# Re-implement execute_roots to ensure independence
def execute_roots(roots: List[int]) -> int:
    val = 0
    for r in roots:
        if 0 <= r <= 8:
            val += (r + 1)
        elif 10 <= r <= 18:
            val -= (r - 10 + 1)
        elif 20 <= r <= 22:
            val *= (r - 20 + 2)
    return val

class SyntheticCriticDatasetFull:
    def __init__(self, size=10000, seq_len=6):
        self.size = size
        self.seq_len = seq_len
        self.samples = []
        self.generate_data()
        
    def generate_data(self):
        """
        Generate (BuggyRoots, Error) -> (FaultyIndex, CorrectRootID)
        """
        valid_roots = list(range(0, 9)) + list(range(10, 19)) + list(range(20, 23))
        
        for _ in range(self.size):
            # 1. Ground Truth
            length = random.randint(3, self.seq_len)
            gt_roots = [random.choice(valid_roots) for _ in range(length)]
            target_val = execute_roots(gt_roots)
            
            # 2. Introduce Bug
            faulty_idx = random.randint(0, length - 1)
            correct_root = gt_roots[faulty_idx]
            
            while True:
                buggy_root = random.choice(valid_roots)
                if buggy_root != correct_root:
                    break
            
            buggy_roots = list(gt_roots)
            buggy_roots[faulty_idx] = buggy_root
            
            # 3. Execution Feedback
            current_val = execute_roots(buggy_roots)
            error = target_val - current_val
            
            # 4. Prepare Sample
            # Pad inputs
            padded_roots = buggy_roots + [30] * (self.seq_len - length)
            
            self.samples.append({
                'roots': padded_roots,
                'error': error,
                'label_idx': faulty_idx,
                'label_root': correct_root
            })

    def __getitem__(self, idx):
        return self.samples[idx]
        
    def __len__(self):
        return len(self.samples)
