
import json
import random
import os
from tqdm import tqdm

# Configuration
TRAIN_BITS = 6     # 6-bit * 6-bit = 12-bit result
TEST_BITS = 8      # 8-bit * 8-bit = 16-bit result (Extrapolation)
OUTPUT_DIR = "data"
NUM_TEST_SAMPLES = 2000

def to_bin(n, width):
    return format(n, f'0{width}b')

def generate_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {TRAIN_BITS}-bit Multiplication dataset (Training)...")
    
    train_data = []
    max_val_train = 2**TRAIN_BITS
    
    # Exhaustive search for small bits to ensure we cover all corner cases
    for a in tqdm(range(max_val_train)):
        for b in range(max_val_train):
            c = a * b
            # Pad to 16 bits to accommodate the Test outputs (8*8=16)
            width = 16 
            
            entry = {
                "a_dec": a,
                "b_dec": b,
                "c_dec": c,
                "a_bin": to_bin(a, width),
                "b_bin": to_bin(b, width),
                "c_bin": to_bin(c, width),
                "split": "train"
            }
            train_data.append(entry)
            
    print(f"Generated {len(train_data)} training samples.")
    
    print(f"Generating {TEST_BITS}-bit Extrapolation dataset (Testing)...")
    test_data = []
    max_val_test = 2**TEST_BITS
    
    count = 0
    with tqdm(total=NUM_TEST_SAMPLES) as pbar:
        while count < NUM_TEST_SAMPLES:
            a = random.randint(0, max_val_test - 1)
            b = random.randint(0, max_val_test - 1)
            
            # Ensure extrapolation: At least one number > max_train
            if a < max_val_train and b < max_val_train:
                continue
                
            c = a * b
            width = 16
            
            entry = {
                "a_dec": a,
                "b_dec": b,
                "c_dec": c,
                "a_bin": to_bin(a, width),
                "b_bin": to_bin(b, width),
                "c_bin": to_bin(c, width),
                "split": "test_extrapolation"
            }
            test_data.append(entry)
            count += 1
            pbar.update(1)
            
    # Save
    train_path = os.path.join(OUTPUT_DIR, "train_mult.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_mult_extrapolate.jsonl")
    
    print(f"Saving to {train_path}...")
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saving to {test_path}...")
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    generate_data()
