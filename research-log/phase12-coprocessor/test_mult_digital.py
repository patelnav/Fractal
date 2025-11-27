
import torch
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from train_logic import RecurrentDataset
from model_logic import BitConfig, BitTokenizer
from model_fractal_mult import FractalMultiplier

# Configuration
# Test on 8-bit x 8-bit (Extrapolation from the 6-bit training logic)
TEST_FILE = "research-log/phase12-coprocessor/data/test_mult_extrapolate.jsonl"
# Ensure we point to the GRU-Gated checkpoint (Phase 11.1)
ADDER_CHECKPOINT = "research-log/phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def test_digital_composition():
    print("PHASE 12: ZERO-SHOT DIGITAL RESTORATION")
    
    tokenizer = BitTokenizer()
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    
    model = FractalMultiplier(config, adder_checkpoint=ADDER_CHECKPOINT).to(DEVICE)
    model.eval()
    
    # 8-bit mult produces 16-bit result. Max len 16 is tight but fits.
    test_dataset = RecurrentDataset(TEST_FILE, tokenizer, max_len=16)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for a, b, c in tqdm(test_loader):
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            
            logits = model(a, b)
            preds = torch.argmax(logits, dim=-1)
            
            row_matches = (preds == c).all(dim=1)
            correct += row_matches.sum().item()
            total += a.size(0)
            
    acc = correct / total
    print(f"\nFinal Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    test_digital_composition()
