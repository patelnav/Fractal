
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
from model_riscv import NeuralCPU

# Configuration
TRAIN_FILE = "research-log/phase13-neural-riscv/data/train_riscv.jsonl"
TEST_FILE = "research-log/phase13-neural-riscv/data/test_riscv_extrapolate.jsonl"
CHECKPOINT_DIR = "research-log/phase13-neural-riscv/checkpoints"
# Using the proven Adder
ADDER_CHECKPOINT = "research-log/phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt"

BATCH_SIZE = 64 # Smaller batch for RL-like training
EPOCHS = 50
LR = 1e-4 # Low LR for stability

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train_riscv():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = BitTokenizer()
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    
    model = NeuralCPU(config, adder_checkpoint=ADDER_CHECKPOINT).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Train on 4-bit * 4-bit = 8-bit result. Pad to 16.
    train_dataset = RecurrentDataset(TRAIN_FILE, tokenizer, max_len=16)
    test_dataset = RecurrentDataset(TEST_FILE, tokenizer, max_len=16)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Starting Neural RISC-V Training...")
    print("Goal: Learn the Multiplication Algorithm (Shift-and-Add) from scratch.")
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for a, b, c in pbar:
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            
            optimizer.zero_grad()
            # Max steps = 16 (enough for 8 shifts + adds)
            logits, trace = model(a, b, max_steps=16) 
            
            loss = criterion(logits.view(-1, 2), c.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                row_matches = (preds == c).all(dim=1)
                train_correct += row_matches.sum().item()
                train_total += a.size(0)
                
            pbar.set_postfix({"loss": loss.item(), "acc": train_correct/(train_total+1e-8)})
            
            # Optional: Print a trace from the first batch
            if epoch % 5 == 0 and pbar.n == 0:
                # Print trace of first sample
                # Ops: 0=NOOP, 1=ADD, 2=SHIFT
                pass # Implement pretty print later
                
        # Extrapolation Test
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for a, b, c in test_loader:
                a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
                
                logits, _ = model(a, b, max_steps=16)
                preds = torch.argmax(logits, dim=-1)
                row_matches = (preds == c).all(dim=1)
                val_correct += row_matches.sum().item()
                val_total += a.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc = {train_correct/train_total:.4f}, Test Acc = {val_acc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_e{epoch+1}.pt"))

if __name__ == "__main__":
    train_riscv()
