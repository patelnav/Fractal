
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from model_logic import BitConfig, BitTokenizer
from model_riscv import NeuralCPU

# Configuration
TRAIN_FILE = "research-log/phase13-neural-riscv/data/train_traces.jsonl"
CHECKPOINT_DIR = "research-log/phase13-neural-riscv/checkpoints_curriculum"
ADDER_CHECKPOINT = "research-log/phase11-fractal-logic/checkpoints_recurrent/ckpt_e10.pt"

BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class TraceDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=16):
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len # Bit width
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Inputs
        a_bin = format(item['a'], f'0{self.max_len}b')
        b_bin = format(item['b'], f'0{self.max_len}b')
        
        a_seq = self.tokenizer.encode_bits(a_bin, self.max_len)
        b_seq = self.tokenizer.encode_bits(b_bin, self.max_len)
        
        # Trace Labels
        # List of {op, s1, s2, dest}
        trace = item['trace']
        
        ops = [t['op'] for t in trace]
        s1s = [t['s1'] for t in trace]
        s2s = [t['s2'] for t in trace]
        dests = [t['dest'] for t in trace]
        
        return (a_seq, b_seq, 
                torch.tensor(ops), torch.tensor(s1s), torch.tensor(s2s), torch.tensor(dests))

def train_curriculum():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    tokenizer = BitTokenizer()
    config = BitConfig(vocab_size=2, dim=128, depth=2, heads=4, dropout=0.0)
    
    model = NeuralCPU(config, adder_checkpoint=ADDER_CHECKPOINT).to(DEVICE)
    
    # We only train the Controller (Adder is frozen in init)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # Losses
    crit_op = nn.CrossEntropyLoss()
    crit_reg = nn.CrossEntropyLoss()
    
    dataset = TraceDataset(TRAIN_FILE, tokenizer, max_len=16)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Starting Curriculum Training (Imitation Learning) on {len(dataset)} samples...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_ops = 0
        total_steps = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for a, b, t_op, t_s1, t_s2, t_dest in pbar:
            a, b = a.to(DEVICE), b.to(DEVICE)
            t_op, t_s1, t_s2, t_dest = t_op.to(DEVICE), t_s1.to(DEVICE), t_s2.to(DEVICE), t_dest.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (returns list of logits)
            all_logits, _ = model(a, b, max_steps=32)
            
            # Stack logits
            # all_logits is list of dicts. Need dict of tensors [B, T, Classes]
            l_op = torch.stack([step['op'] for step in all_logits], dim=1) # [B, T, 3]
            l_s1 = torch.stack([step['s1'] for step in all_logits], dim=1) # [B, T, 4]
            l_s2 = torch.stack([step['s2'] for step in all_logits], dim=1)
            l_d  = torch.stack([step['dest'] for step in all_logits], dim=1)
            
            # Reshape for Loss: [B*T, C] vs [B*T]
            loss_op = crit_op(l_op.view(-1, 3), t_op.view(-1))
            loss_s1 = crit_reg(l_s1.view(-1, 4), t_s1.view(-1))
            loss_s2 = crit_reg(l_s2.view(-1, 4), t_s2.view(-1))
            loss_d  = crit_reg(l_d.view(-1, 4),  t_dest.view(-1))
            
            loss = loss_op + loss_s1 + loss_s2 + loss_d
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            preds = torch.argmax(l_op, dim=-1)
            correct_ops += (preds == t_op).sum().item()
            total_steps += t_op.numel()
            
            pbar.set_postfix({"loss": loss.item(), "op_acc": correct_ops/total_steps})
            
        # Save
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_e{epoch+1}.pt"))
        print(f"Epoch {epoch+1}: Op Acc = {correct_ops/total_steps:.4f}")

if __name__ == "__main__":
    train_curriculum()
