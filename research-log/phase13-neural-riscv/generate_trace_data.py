
import json
import torch
import os
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "research-log/phase13-neural-riscv/data"
TRAIN_BITS = 4

def to_bin(n, width):
    return [int(c) for c in format(n, f'0{width}b')[::-1]] # LSB First

def generate_trace_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating Imitation Learning Traces (Shift-and-Add)...")
    
    data = []
    max_val = 2**TRAIN_BITS
    
    # Op Codes: 0=NOOP, 1=ADD, 2=SHIFT
    # Registers: 0=A, 1=B, 2=Acc, 3=Zero
    
    for a in tqdm(range(max_val)):
        for b in range(max_val):
            
            # Trace generation logic
            # We mimic the exact steps the Controller should take.
            # We have 16 bits (max_len=16).
            # Algorithm:
            # For i in 0..15:
            #   if B & (1<<i):
            #      ADD R2, R0  (Acc += A)
            #   SHIFT R0       (A <<= 1)
            
            trace = []
            
            # Initial state: R0=A, R1=B, R2=0, R3=0
            curr_a = a
            
            # We iterate 16 times (fixed width)
            for i in range(16):
                b_bit = (b >> i) & 1
                
                if b_bit:
                    # Action: ADD R2, R0 -> R2
                    step = {
                        "op": 1, # ADD
                        "s1": 2, # R2
                        "s2": 0, # R0
                        "dest": 2 # R2
                    }
                    trace.append(step)
                else:
                    # Action: NOOP (or just don't ADD)
                    # We still need to fill the time step?
                    # The controller has fixed 32 steps?
                    # Or we can output NOOP.
                    step = {
                        "op": 0, # NOOP
                        "s1": 2, 
                        "s2": 0,
                        "dest": 2
                    }
                    trace.append(step)
                    
                # Action: SHIFT R0 -> R0
                step = {
                    "op": 2, # SHIFT
                    "s1": 0, # R0
                    "s2": 0, # Ignored
                    "dest": 0 # R0
                }
                trace.append(step)
                
            # Pad to 32 steps (if loop len < 16, but here it is exactly 16*2=32)
            
            entry = {
                "a": a,
                "b": b,
                "c": a * b,
                "trace": trace
            }
            data.append(entry)
            
    save_path = os.path.join(OUTPUT_DIR, "train_traces.jsonl")
    print(f"Saving {len(data)} traces to {save_path}...")
    
    with open(save_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    generate_trace_dataset()
