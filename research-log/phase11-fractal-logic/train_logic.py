
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

from model_logic import FractalRecurrentALU, BitConfig, BitTokenizer

# Configuration
TRAIN_FILE = "data/train_8bit.jsonl"
TEST_FILE = "data/test_12bit_extrapolate.jsonl"
CHECKPOINT_DIR = "checkpoints_recurrent"
BATCH_SIZE = 512
EPOCHS = 10 
LR = 1e-3
MAX_LEN_TRAIN = 16 # Pad to 16 for training
MAX_LEN_TEST = 24  # Pad to 24 for testing (Extrapolation!)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

class RecurrentDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=16):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"Loading {path}...")
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenizer expects raw bin string, reverses it (LSB first), pads to max_len
        # Note: Data was generated with 16-bit padding already, but stripped of '0b'
        # item['a_bin'] is "0000...101" (MSB...LSB)
        
        a_seq = self.tokenizer.encode_bits(item['a_bin'], self.max_len)
        b_seq = self.tokenizer.encode_bits(item['b_bin'], self.max_len)
        c_seq = self.tokenizer.encode_bits(item['c_bin'], self.max_len)
        
        return a_seq, b_seq, c_seq

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = BitTokenizer()
    # Small, deep-ish config for the recurrent core
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0) 
    model = FractalRecurrentALU(config).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Train on 16-bit sequences (contains 8-bit numbers)
    train_dataset = RecurrentDataset(TRAIN_FILE, tokenizer, max_len=16)
    # Test on 24-bit sequences (contains 12-bit numbers). 
    # The model must run for 24 steps now!
    test_dataset = RecurrentDataset(TEST_FILE, tokenizer, max_len=24)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for a, b, c in pbar:
            a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(a, b) # [Batch, SeqLen, 2], discard soft_bits

            loss = criterion(logits.view(-1, 2), c.view(-1))
            loss.backward()
            optimizer.step()
            
            # Accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                row_matches = (preds == c).all(dim=1)
                train_correct += row_matches.sum().item()
                train_total += a.size(0)
                
            pbar.set_postfix({"loss": loss.item(), "acc": train_correct/(train_total+1e-8)})
            
        # Validation (Extrapolation)
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for a, b, c in test_loader:
                a, b, c = a.to(DEVICE), b.to(DEVICE), c.to(DEVICE)
                
                logits, _ = model(a, b)
                preds = torch.argmax(logits, dim=-1)
                
                row_matches = (preds == c).all(dim=1)
                val_correct += row_matches.sum().item()
                val_total += a.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc (8-bit) = {train_correct/train_total:.4f}, Test Acc (12-bit Extrapolation) = {val_acc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_e{epoch+1}.pt"))

if __name__ == "__main__":
    train()
