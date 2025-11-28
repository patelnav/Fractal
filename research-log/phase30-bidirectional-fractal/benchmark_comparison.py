import torch
import torch.nn.functional as F
from fractal_data import FractalMathDataset
from model import FractalTransformer, Config
import matplotlib.pyplot as plt
import sys

# Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ds = FractalMathDataset(1, 4) # Dummy
stoi = ds.stoi
itos = ds.itos

def load_model(path, causal):
    config = Config()
    config.vocab_size = len(ds.vocab)
    config.block_size = 128
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.causal = causal
    
    model = FractalTransformer(config)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(DEVICE)
    model.eval()
    return model

def decode(ids):
    return "".join([itos[i.item()] for i in ids if i.item() not in [stoi["<PAD>"], stoi["<BOS>"], stoi["<EOS>"]]])

def get_accuracy(pred, target, pad_id):
    # Simple token match rate excluding PAD
    mask = (target != pad_id)
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()

def run_flash_flood(model, input_ids, target_ids, max_steps=5):
    # input_ids: (1, T)
    current_x = input_ids.clone()
    pad_token_id = stoi["<PAD>"]
    is_pad = (current_x == pad_token_id)
    
    accuracies = []
    
    # Initial acc
    accuracies.append(get_accuracy(current_x, target_ids, pad_token_id))
    
    print(f"  Init: {decode(current_x[0])}")
    
    for step in range(max_steps):
        with torch.no_grad():
            logits, _ = model(current_x)
            
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        
        # Lock PADs
        pred_ids[is_pad] = pad_token_id
        
        # Update state
        current_x = pred_ids
        
        acc = get_accuracy(current_x, target_ids, pad_token_id)
        accuracies.append(acc)
        print(f"  Step {step+1}: {decode(current_x[0])} (Acc: {acc:.2f})")
        
    return accuracies, current_x

def run_benchmark():
    bidir_model = load_model("checkpoints/fractal_bidirectional.pt", causal=False)
    causal_model = load_model("checkpoints/fractal_causal.pt", causal=True)
    
    test_cases = [
        # (Expression, MaskIndices)
        # Case 1: (+ 5 (* 2 3))=11. Indices from previous run: [3, 6, 11, 12] -> Masking '5', '2', '1', '1'
        # Text: (+5(*23))=11 -> 012345678901
        # Tokens (shifted +1 for BOS): 
        # +:1, 5:2, (:3, *:4, 2:5, 3:6, ):7, ):8, =:9, 1:10, 1:11
        # Wait, previous run mask indices were: [3, 6, 11, 12]. 
        # 3 -> '5'. 6 -> '2'. 11 -> '1'. 12 -> '1'.
        ("((+ 5 (* 2 3))=11", [3, 6, 11, 12]), 
        
        # Case 2: (* 2 (+ 3 4))=14
        # Text: (*2(+34))=14
        # 012345678901
        # Mask '2'(idx 2), '3'(idx 5), '14'(idx 10,11)
        # Shifted: [3, 6, 11, 12] (Coincidentally similar structure)
        ("(* 2 (+ 3 4))=14", [3, 6, 11, 12]),
        
        # Case 3: (+ (* 2 2) (* 3 3))=13
        # Text: (+(*22)(*33))=13
        # 0123456789012345
        # Mask '2'(idx 4), '3'(idx 9), '13'(idx 14,15)
        # Shifted: [5, 10, 15, 16]
        ("(+ (* 2 2) (* 3 3))=13", [5, 10, 15, 16])
    ]
    
    results = {"Causal": [], "Bidirectional": []}
    
    print("--- BENCHMARK START ---")
    
    for i, (text_raw, mask_indices) in enumerate(test_cases):
        print(f"\nTestCase {i+1}: {text_raw}")
        
        # Prepare Tensor
        tokens = [stoi["<BOS>"]]
        for c in text_raw:
            if c == ' ': continue
            tokens.append(stoi[c])
        tokens.append(stoi["<EOS>"])
        
        # Pad
        L = len(tokens)
        tokens_padded = tokens + [stoi["<PAD>"]] * (128 - L)
        target_x = torch.tensor(tokens_padded).unsqueeze(0).to(DEVICE)
        
        # Mask
        noisy_x = target_x.clone()
        for idx in mask_indices:
            if idx < L:
                noisy_x[0, idx] = stoi["<MASK>"]
        
        print(f"Target: {decode(target_x[0])}")
        
        # Run Causal
        print(">> Causal Model (Jacobi):")
        acc_c, _ = run_flash_flood(causal_model, noisy_x, target_x)
        results["Causal"].append(acc_c)
        
        # Run Bidirectional
        print(">> Bidirectional Model (Flash Flood):")
        acc_b, _ = run_flash_flood(bidir_model, noisy_x, target_x)
        results["Bidirectional"].append(acc_b)

    # Average results
    avg_causal = [sum(x)/len(x) for x in zip(*results["Causal"])]
    avg_bidir = [sum(x)/len(x) for x in zip(*results["Bidirectional"])]
    
    print("\n--- SUMMARY STATS ---")
    print("Step | Causal Acc | Bidir Acc")
    for s in range(len(avg_causal)):
        print(f"{s:4d} | {avg_causal[s]:.4f}     | {avg_bidir[s]:.4f}")
        
if __name__ == "__main__":
    run_benchmark()
