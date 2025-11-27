
import torch
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# Import Phase 11 and 12
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from train_logic import RecurrentDataset # Reuse dataset loader
from model_logic import BitConfig, BitTokenizer
from model_fractal_mult import FractalMultiplier

# Configuration
TEST_FILE = "data/test_mult_extrapolate.jsonl"
ADDER_CHECKPOINT = "../phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt" 
# Note: Make sure to pick the best checkpoint from Phase 11. ckpt_e10 was near 100%.

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def test_composition():
    print("PHASE 12: ZERO-SHOT FRACTAL MULTIPLIER")
    print("Hypothesis: A perfect Adder + Hard-coded Shift-loop = Perfect Multiplier (No Training)")
    
    tokenizer = BitTokenizer()
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    
    # Load the Composite Model
    model = FractalMultiplier(config, adder_checkpoint=ADDER_CHECKPOINT).to(DEVICE)
    model.eval()
    
    # Load Data
    # Reuse RecurrentDataset but pointing to Mult data
    # Mult data has same format: a_bin, b_bin, c_bin
    # We need enough length. 8-bit * 8-bit = 16-bit.
    # Shift-and-add requires the accumulator to hold the full result.
    # Max shift is 8. Max length is 16.
    test_dataset = RecurrentDataset(TEST_FILE, tokenizer, max_len=16)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(test_dataset)} samples (Extrapolation 8-bit x 8-bit)...")
    
    with torch.no_grad():
        for a, b, c in tqdm(test_loader):
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            
            # Run the recursive loop
            # Output is already indices (Argmaxed inside)
            preds = model(a, b) 
            
            # Check accuracy
            row_matches = (preds == c).all(dim=1)
            correct += row_matches.sum().item()
            total += a.size(0)
            
    acc = correct / total
    print(f"\nFinal Zero-Shot Accuracy: {acc*100:.2f}%")
    
    if acc > 0.9:
        print("VERDICT: SUCCESS! Logic is Fractal.")
    else:
        print("VERDICT: FAIL. Errors accumulated in the loop.")

if __name__ == "__main__":
    test_composition()
