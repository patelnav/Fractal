
import json
import random
import os
from tqdm import tqdm

# Configuration
TRAIN_BITS = 8
TEST_BITS = 12
OUTPUT_DIR = "data"
NUM_TEST_SAMPLES = 10000

def to_bin(n, width):
    return format(n, f'0{width}b')

def generate_data():
    print(f"Generating {TRAIN_BITS}-bit addition dataset (Training)...")
    
    train_data = []
    # Generate ALL pairs for 8-bit to ensure perfect coverage of the "easy" space
    # 256 * 256 = 65,536 samples. Small enough.
    max_val_train = 2**TRAIN_BITS
    
    for a in tqdm(range(max_val_train)):
        for b in range(max_val_train):
            c = a + b
            # We pad inputs to the TEST width (12) so the model architecture 
            # can handle the larger test inputs later without reshaping.
            # Actually, let's pad to 16 bits to be safe and standard.
            width = 16
            
            entry = {
                "a_dec": a,
                "b_dec": b,
                "c_dec": c,
                "a_bin": to_bin(a, width),
                "b_bin": to_bin(b, width),
                "c_bin": to_bin(c, width), # Output might need more bits, but 16 is safe for 12+12
                "split": "train"
            }
            train_data.append(entry)
            
    print(f"Generated {len(train_data)} training samples.")
    
    print(f"Generating {TEST_BITS}-bit extrapolation dataset (Testing)...")
    test_data = []
    max_val_test = 2**TEST_BITS
    
    count = 0
    with tqdm(total=NUM_TEST_SAMPLES) as pbar:
        while count < NUM_TEST_SAMPLES:
            a = random.randint(0, max_val_test - 1)
            b = random.randint(0, max_val_test - 1)
            
            # Crucial: Ensure we are strictly OUTSIDE the training distribution
            # i.e., at least one number must be > 255
            if a < max_val_train and b < max_val_train:
                continue
                
            c = a + b
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_8bit.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_12bit_extrapolate.jsonl")

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
