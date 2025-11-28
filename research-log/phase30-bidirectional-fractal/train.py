import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fractal_data import FractalMathDataset
from model import FractalTransformer, Config
import os

def train():
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
    config.block_size = 128 # ample for small expressions
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.causal = False # Bidirectional!
    
    model = FractalTransformer(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    mask_token_id = ds.stoi["<MASK>"]
    pad_token_id = ds.stoi["<PAD>"]

    model.train()
    iter_num = 0
    while iter_num < MAX_ITERS:
        for batch in dl:
            if iter_num >= MAX_ITERS:
                break
                
            x = batch.to(DEVICE) # (B, T)
            B, T = x.size()
            
            # Create Mask
            # Random mask ratio per sample or per batch? Let's do per batch for simplicity or vectorized per sample
            # ratio ~ Uniform(0.1, 0.9)
            ratio = torch.rand(B, device=DEVICE) * 0.8 + 0.1
            ratio = ratio.unsqueeze(1) # (B, 1)
            
            # Probability matrix
            prob = torch.rand(B, T, device=DEVICE)
            mask = prob < ratio
            
            # Ensure we don't mask PAD, BOS, EOS generally, but maybe it's okay?
            # Let's protect BOS/EOS/PAD
            special_mask = (x == ds.stoi["<BOS>"]) | (x == ds.stoi["<EOS>"]) | (x == ds.stoi["<PAD>"])
            mask = mask & (~special_mask)
            
            input_ids = x.clone()
            input_ids[mask] = mask_token_id
            
            # Targets: We want to predict everything? Or just masked?
            # If we predict everything, the model learns identity for unmasked.
            # "Refines all positions jointly" suggests we should output everything.
            targets = x.clone()
            targets[x == pad_token_id] = -1 # Ignore PAD in loss
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()
            
            if iter_num % 100 == 0:
                print(f"Iter {iter_num}: Loss {loss.item():.4f}")
            
            iter_num += 1
            
    # Save
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/fractal_bidirectional.pt")
    print("Saved model.")

if __name__ == "__main__":
    train()
