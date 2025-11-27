
import random
import torch
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# Synthetic "Math Code" Domain
# ============================================================================
# A "Program" is a sequence of operations that starts at 0 and reaches a Target.
# Roots: Operations [ADD, SUB, MUL] (plus args implicitly)
# Chunks: String literals [" + 5", " * 2"]

OPS = ['ADD', 'SUB', 'MUL']
# For simplicity in this tiny demo, Roots map 1:1 to specific operations on specific small integers
# Root 0: ADD 1
# Root 1: ADD 2
# ...
# Root 10: SUB 1
# ...
# Root 20: MUL 2

@dataclass
class CodeConfig:
    num_roots: int = 30  # 10 ADDs, 10 SUBs, 10 MULs
    num_chunks: int = 60 # ASCII chars + some specialized tokens
    expansion_size: int = 4 # Root -> 4 chars (e.g., " + 5")
    
class SyntheticCodeDataset:
    def __init__(self, size=10000):
        self.size = size
        self.samples = []
        self.generate_data()
        
    def generate_data(self):
        """
        Generate valid (RootSequence, Text) pairs.
        """
        for _ in range(self.size):
            # Generate a random program of length 3-6
            length = random.randint(3, 6)
            roots = []
            text_parts = []
            
            current_val = 0
            
            for _ in range(length):
                # Pick op type
                op_type = random.choice(OPS)
                
                if op_type == 'ADD':
                    val = random.randint(1, 9)
                    root_id = val - 1 # 0..8
                    text = f"+{val}"
                elif op_type == 'SUB':
                    val = random.randint(1, 9)
                    root_id = 10 + (val - 1) # 10..18
                    text = f"-{val}"
                elif op_type == 'MUL':
                    val = random.randint(2, 4)
                    root_id = 20 + (val - 2) # 20..22
                    text = f"*{val}"
                
                # Pad text to expansion size (4)
                while len(text) < 4:
                    text += " "
                
                roots.append(root_id)
                text_parts.append(text)
            
            self.samples.append((roots, "".join(text_parts)))

    def __getitem__(self, idx):
        return self.samples[idx]
        
    def __len__(self):
        return len(self.samples)

# Simple Char Tokenizer for the "Chunks"
CHARS = "+-*0123456789 "
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

def encode_text(text: str) -> List[int]:
    return [CHAR_TO_ID[c] for c in text]

def decode_text(ids: List[int]) -> str:
    return "".join([ID_TO_CHAR.get(i, '?') for i in ids])
