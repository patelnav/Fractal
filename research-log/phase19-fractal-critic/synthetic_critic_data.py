
import sys
import random
import torch
from pathlib import Path
from typing import List, Tuple

# Add path for existing synthetic data logic
sys.path.insert(0, str(Path(__file__).parent.parent / "phase18-fractal-coder"))

from synthetic_data import OPS

# Re-implementing basic execution logic here to avoid circular imports or complexity
# Root Mapping (Same as Phase 18)
# 0-8: ADD 1..9
# 10-18: SUB 1..9
# 20-22: MUL 2..4
# Total roots: 30

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

class SyntheticCriticDataset:
    def __init__(self, size=10000, seq_len=6):
        self.size = size
        self.seq_len = seq_len
        self.samples = []
        self.generate_data()
        
    def generate_data(self):
        """
        Generate (BuggyRoots, Error, FaultyIndex) triplets.
        """
        # Valid Roots (Same as Phase 18)
        # We have holes in IDs (9, 19, 23-29 are unused), but that's fine.
        valid_roots = list(range(0, 9)) + list(range(10, 19)) + list(range(20, 23))
        
        for _ in range(self.size):
            # 1. Create a Ground Truth Program
            length = random.randint(3, self.seq_len)
            gt_roots = [random.choice(valid_roots) for _ in range(length)]
            
            # 2. Execute to get Target
            target_val = execute_roots(gt_roots)
            
            # 3. Create a Bug (Perturb one root)
            faulty_idx = random.randint(0, length - 1)
            original_root = gt_roots[faulty_idx]
            
            # Pick a DIFFERENT root
            while True:
                buggy_root = random.choice(valid_roots)
                if buggy_root != original_root:
                    break
            
            buggy_roots = list(gt_roots)
            buggy_roots[faulty_idx] = buggy_root
            
            # 4. Execute Buggy Program
            current_val = execute_roots(buggy_roots)
            
            # 5. Calculate Signal
            # We provide the Error = Target - Current
            # (Or Current and Target separately)
            # Let's provide Error.
            error = target_val - current_val
            
            # Store
            # Inputs: BuggyRoots, Error
            # Output: FaultyIndex
            
            # Pad roots to seq_len with special token (e.g., 30)
            # Config has 30 roots (0-29). Let's use 30 as PAD.
            padded_roots = buggy_roots + [30] * (self.seq_len - length)
            
            self.samples.append({
                'roots': padded_roots,
                'error': error,
                'label_idx': faulty_idx,
                'length': length
            })

    def __getitem__(self, idx):
        return self.samples[idx]
        
    def __len__(self):
        return len(self.samples)
