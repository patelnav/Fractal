
import torch
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from model_logic import FractalRecurrentALU, BitConfig, BitTokenizer
from train_logic import RecurrentDataset # For data loading utils

# Configuration
ADDER_CHECKPOINT = "research-log/phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    print("Loading Adder...")
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    model = FractalRecurrentALU(config).to(DEVICE)
    state_dict = torch.load(ADDER_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_embedding(model, val_int, width=16):
    tokenizer = BitTokenizer()
    bin_str = format(val_int, f'0{width}b')
    # encode_bits returns [L] tensor
    seq = tokenizer.encode_bits(bin_str, width).unsqueeze(0).to(DEVICE) # [1, L]
    with torch.no_grad():
        emb = model.bit_emb(seq) # [1, L, D]
    return seq, emb

def shift_embedding(model, emb, shift_amount):
    """
    Re-implementation of the logic from model_fractal_mult.py to test it in isolation.
    """
    batch_size, seq_len, dim = emb.size()
    if shift_amount == 0:
        return emb
        
    # Get embedding of '0' token explicitly
    zero_idx = torch.tensor(0, device=DEVICE)
    zero_vec = model.bit_emb(zero_idx).view(1, 1, -1) # [1, 1, D]
    zeros = zero_vec.expand(batch_size, shift_amount, -1)
    
    # Concat at start (LSB side)
    shifted = torch.cat([zeros, emb], dim=1)
    return shifted[:, :seq_len, :]

def test_shift_logic(model):
    print("\n=== TEST 1: SHIFT LOGIC ===")
    # Compare Shift(Emb(A)) vs Emb(A << 1)
    
    vals = [1, 3, 5, 15]
    shifts = [1, 2, 3]
    
    passed = True
    for v in vals:
        for s in shifts:
            target_val = v << s
            _, emb_target = get_embedding(model, target_val)
            _, emb_source = get_embedding(model, v)
            
            emb_shifted = shift_embedding(model, emb_source, s)
            
            # Compare embeddings
            # They might not be identical due to padding logic, but let's check norm diff
            diff = torch.norm(emb_shifted - emb_target).item()
            
            if diff < 1e-5:
                print(f"[PASS] {v} << {s} (Diff: {diff:.6f})")
            else:
                print(f"[FAIL] {v} << {s} (Diff: {diff:.6f})")
                print(f"  Target: {target_val}")
                passed = False
    return passed

def test_adder_zero(model):
    print("\n=== TEST 2: ADDER ZERO HANDLING ===")
    # Check A + 0 == A
    
    vals = [5, 10, 123]
    passed = True
    
    for v in vals:
        # Inputs
        seq_a, emb_a = get_embedding(model, v)
        seq_0, emb_0 = get_embedding(model, 0)
        
        # Run Adder (Embeddings)
        logits, _ = model.forward_embeddings(emb_a, emb_0)
        preds = torch.argmax(logits, dim=-1) # [1, L]
        
        # Check against sequence A
        if torch.equal(preds, seq_a):
            print(f"[PASS] {v} + 0 == {v}")
        else:
            print(f"[FAIL] {v} + 0 != {v}")
            print(f"  Pred: {preds.tolist()}")
            print(f"  Gold: {seq_a.tolist()}")
            passed = False
            
    return passed

def test_accumulation_step(model):
    print("\n=== TEST 3: ACCUMULATION (A + (A<<1)) ===")
    # Check logic 3 + 6 = 9
    
    val = 3
    shifted_val = val << 1 # 6
    target = val + shifted_val # 9
    
    _, emb_a = get_embedding(model, val)
    emb_shifted = shift_embedding(model, emb_a, 1)
    
    logits, _ = model.forward_embeddings(emb_a, emb_shifted)
    preds = torch.argmax(logits, dim=-1)
    
    seq_target, _ = get_embedding(model, target)
    
    if torch.equal(preds, seq_target):
        print(f"[PASS] {val} + {shifted_val} == {target}")
        return True
    else:
        print(f"[FAIL] {val} + {shifted_val} != {target}")
        print(f"  Pred: {preds.tolist()}")
        print(f"  Gold: {seq_target.tolist()}")
        return False

def test_full_multiplication(model):
    print("\n=== TEST 4: FULL 4-BIT MULTIPLICATION (EXHAUSTIVE) ===")
    # Replicate the loop from model_fractal_mult.py MANUALLY here
    
    correct = 0
    total = 0
    
    for a in tqdm(range(16)):
        for b in range(16):
            c_target = a * b
            
            # Prepare Inputs
            seq_a, emb_a = get_embedding(model, a)
            seq_b, _ = get_embedding(model, b) # Need sequence for bit access
            
            # Init Accumulator (Emb(0))
            _, accumulator_emb = get_embedding(model, 0)
            
            # Loop (4 bits for 4-bit numbers? No, sequence length is 16)
            # We iterate over the full sequence length
            width = 16
            
            # Get bits of B (LSB first)
            # seq_b is [1, 16] tensor of 0/1
            b_bits = seq_b[0] 
            
            for i in range(width):
                bit = b_bits[i].item()
                
                # Prepare term
                shifted_a_emb = shift_embedding(model, emb_a, i)
                _, zero_emb_seq = get_embedding(model, 0)
                
                # Selection (Hard Python Logic)
                if bit == 1:
                    term_emb = shifted_a_emb
                else:
                    term_emb = zero_emb_seq
                    
                # Adder
                logits, _ = model.forward_embeddings(accumulator_emb, term_emb)
                
                # DIGITAL RESTORATION (Argmax + Re-Embed)
                preds = torch.argmax(logits, dim=-1)
                
                # Re-embed for next step
                # We can cheat and use get_embedding(int(preds))? 
                # No, that's not neural. We must use embedding lookup.
                accumulator_emb = model.bit_emb(preds)
                
            # Final Result
            res_val = 0
            # Decode binary [LSB...MSB] to int
            # seq is [1, 16]
            final_bits = preds[0].tolist()
            for idx, bit in enumerate(final_bits):
                if bit == 1:
                    res_val += (1 << idx)
            
            if res_val == c_target:
                correct += 1
            else:
                if total < 5: # Print first few failures
                    print(f"[FAIL] {a} * {b} = {c_target}, Got {res_val}")
            total += 1
            
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    return correct == total

if __name__ == "__main__":
    model = load_model()
    
    t1 = test_shift_logic(model)
    t2 = test_adder_zero(model)
    t3 = test_accumulation_step(model)
    t4 = test_full_multiplication(model)
    
    if t1 and t2 and t3 and t4:
        print("\nALL SYSTEM CHECKS PASSED. The Adder is composable.")
    else:
        print("\nSYSTEM CHECKS FAILED. Debug the components.")
