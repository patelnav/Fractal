import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fractal_data import FractalMathDataset
from model import FractalTransformer, Config
import os

def train_causal():
    # Hyperparams
    BATCH_SIZE = 64
    MAX_ITERS = 1000
    LEARNING_RATE = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Data
    ds = FractalMathDataset(num_samples=20000, max_depth=6)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    config = Config()
    config.vocab_size = len(ds.vocab)
    config.block_size = 128 
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.causal = True # Causal!
    
    model = FractalTransformer(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    pad_token_id = ds.stoi["<PAD>"]

    model.train()
    iter_num = 0
    while iter_num < MAX_ITERS:
        for batch in dl:
            if iter_num >= MAX_ITERS:
                break
                
            x = batch.to(DEVICE) # (B, T)
            
            # Standard Causal Training (Next Token Prediction)
            # Input: x[:, :-1]
            # Target: x[:, 1:]
            
            inp = x[:, :-1]
            tgt = x[:, 1:]
            
            # Ignore PAD in loss
            # But we need to ensure targets have -1 where appropriate
            tgt_clone = tgt.clone()
            tgt_clone[tgt == pad_token_id] = -1
            
            optimizer.zero_grad()
            logits, loss = model(inp, tgt_clone)
            loss.backward()
            optimizer.step()
            
            if iter_num % 100 == 0:
                print(f"Iter {iter_num}: Loss {loss.item():.4f}")
            
            iter_num += 1
            
    # Save
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/fractal_causal.pt")
    print("Saved Causal model.")

if __name__ == "__main__":
    train_causal()
