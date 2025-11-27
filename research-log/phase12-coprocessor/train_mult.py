
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from train_logic import RecurrentDataset
from model_logic import BitConfig, BitTokenizer
from model_fractal_mult import FractalMultiplier

# Configuration
TRAIN_FILE = "data/train_mult.jsonl"
TEST_FILE = "data/test_mult_extrapolate.jsonl"
CHECKPOINT_DIR = "checkpoints"
ADDER_CHECKPOINT = "../phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt"

BATCH_SIZE = 128
EPOCHS = 20
LR = 5e-4 

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train_mult():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = BitTokenizer()
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    
    # Initialize with Pre-trained Adder
    model = FractalMultiplier(config, adder_checkpoint=ADDER_CHECKPOINT).to(DEVICE)
    
    # We want to train the Adder weights (fine-tune) and the new out_proj
    # The Loop logic is fixed (Shift-and-Add), but the representation is learned.
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Train on 6x6=12 bits. Pad to 16.
    train_dataset = RecurrentDataset(TRAIN_FILE, tokenizer, max_len=16)
    test_dataset = RecurrentDataset(TEST_FILE, tokenizer, max_len=16)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Starting Differentiable Multiplier Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for a, b, c in pbar:
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(a, b) # [B, L, 2]
            
            loss = criterion(logits.view(-1, 2), c.view(-1))
            loss.backward()
            
            # Gradient Clipping is crucial for Recurrent models
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                row_matches = (preds == c).all(dim=1)
                train_correct += row_matches.sum().item()
                train_total += a.size(0)
                
            pbar.set_postfix({"loss": loss.item(), "acc": train_correct/(train_total+1e-8)})
            
        # Extrapolation Test
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for a, b, c in test_loader:
                a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
                
                logits = model(a, b)
                preds = torch.argmax(logits, dim=-1)
                row_matches = (preds == c).all(dim=1)
                val_correct += row_matches.sum().item()
                val_total += a.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc (6-bit) = {train_correct/train_total:.4f}, Test Acc (8-bit Extrap) = {val_acc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_e{epoch+1}.pt"))

if __name__ == "__main__":
    train_mult()
