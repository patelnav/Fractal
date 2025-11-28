import torch
import torch.nn.functional as F
from fractal_data import FractalMathDataset
from model import FractalTransformer, Config
import random

def load_model(path, vocab_size):
    config = Config()
    config.vocab_size = vocab_size
    config.block_size = 128
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.causal = False
    
    model = FractalTransformer(config)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def flash_flood_sample(model, input_ids, mask_token_id, max_steps=10):
    # input_ids: (1, T)
    current_x = input_ids.clone()
    pad_token_id = stoi["<PAD>"]
    
    # Identify padding positions (assuming they are at the end)
    # We can just check where input is PAD. 
    # Note: The input 'noisy_x' has PADs at the end.
    is_pad = (current_x == pad_token_id)
    
    print(f"Init: {decode(current_x[0])}")
    
    for step in range(max_steps):
        with torch.no_grad():
            logits, _ = model(current_x) # (1, T, V)
            
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1) # (1, T)
        
        # Force PADs to remain PADs
        pred_ids[is_pad] = pad_token_id
        
        # Calculate diff only on non-pad tokens
        diff_mask = ~is_pad
        diff = ((pred_ids != current_x) & diff_mask).sum().item()
        
        print(f"Step {step+1}: {decode(pred_ids[0])} (Diff: {diff})")
        
        current_x = pred_ids
        
        if diff == 0:
            print("Converged!")
            break
            
    return current_x

ds = FractalMathDataset(1, 4) # Dummy for vocab
stoi = ds.stoi
itos = ds.itos

def decode(ids):
    return "".join([itos[i.item()] for i in ids if i.item() not in [stoi["<PAD>"], stoi["<BOS>"], stoi["<EOS>"]]])

def run_test():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model("checkpoints/fractal_bidirectional.pt", len(ds.vocab)).to(DEVICE)
    
    # specific test case: (+ 5 (* 2 3))=11
    # Tokens: (, +, 5, (, *, 2, 3, ), ), =, 1, 1
    # Let's construct it manually to be sure
    text = "(+ 5 (* 2 3))=11"
    tokens = [stoi["<BOS>"]]
    for c in text:
        if c == ' ': continue
        tokens.append(stoi[c])
    tokens.append(stoi["<EOS>"])
    
    # Pad to 128
    L = len(tokens)
    tokens += [stoi["<PAD>"]] * (128 - L)
    
    clean_x = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    print(f"Target: {decode(clean_x[0])}")
    
    # Masking strategy:
    # Mask 50% of the CONTENT (indices 1 to L-2, excluding BOS/EOS)
    # We want to keep some 'islands'
    
    noisy_x = clean_x.clone()
    
    # Let's mask specific parts to test "Islands"
    # Target: (+ 5 (* 2 3))=11
    # Mask:   (+ _ (* _ 3))=__
    # We remove '5', '2', and the result '11'.
    # Indices in "text":
    # ( +   5   ( *   2   3 ) ) = 1 1
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    # Tokens indices will be +1 for BOS
    
    # Text indices to mask:
    # 5 is at index 3 (in text space with spaces removed? wait)
    # My manual tokenizer above removes spaces.
    # Text: (+5(*23))=11
    # 0123456789012345
    # (:0, +:1, 5:2, (:3, *:4, 2:5, 3:6, ):7, ):8, =:9, 1:10, 1:11
    # Mask '5' (idx 2), '2' (idx 5), '11' (idx 10, 11)
    # Shift +1 for BOS
    
    mask_indices = [2+1, 5+1, 10+1, 11+1] 
    
    print(f"Masking indices: {mask_indices}")
    for idx in mask_indices:
        noisy_x[0, idx] = stoi["<MASK>"]
        
    print("\n--- Running Flash Flood ---")
    flash_flood_sample(model, noisy_x, stoi["<MASK>"])

if __name__ == "__main__":
    run_test()
