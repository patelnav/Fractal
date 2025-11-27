
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import random

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
HARD_FILE = "research-log/phase14-vector6-reboot/data/hard_negatives.jsonl"
ALL_FILE = "research-log/phase14-vector6-reboot/data/mbpp_labeled.jsonl"
BASE_CHECKPOINT = "research-log/phase14-vector6-reboot/checkpoints_critic/critic_e3.pt"
OUTPUT_DIR = "research-log/phase14-vector6-reboot/checkpoints_hardening"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 2
LR = 5e-6 # Lower LR for fine-tuning

class CodeVerifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.config = self.backbone.config
        self.hidden_size = self.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        ).to("cuda").to(torch.bfloat16)
                
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_indices = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        pooled_output = last_hidden_state[torch.arange(batch_size), last_indices]
        logits = self.score_head(pooled_output)
        return logits

class MixedDataset(Dataset):
    def __init__(self, hard_path, all_path, tokenizer, max_len=512):
        self.data = []
        print(f"Loading Hard Samples from {hard_path}...")
        with open(hard_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        hard_count = len(self.data)
        print(f"Found {hard_count} hard samples.")
        
        print(f"Loading Easy Samples from {all_path}...")
        easy_candidates = []
        with open(all_path, 'r') as f:
            for line in f:
                easy_candidates.append(json.loads(line))
                
        # Sample same number of easy samples to balance
        random.seed(42)
        easy_samples = random.sample(easy_candidates, hard_count)
        self.data.extend(easy_samples)
        
        print(f"Total Training Data: {len(self.data)} (50% Hard / 50% Easy)")
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Check label
        label = 1.0 if item['status'] == 'passed' else 0.0
        
        text = f"{item['prompt']}\n\n# Solution:\n{item['code']}"
        enc = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_len, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

def train_hardening():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Initializing Critic from {BASE_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = CodeVerifier(MODEL_NAME)
    state_dict = torch.load(BASE_CHECKPOINT)
    model.load_state_dict(state_dict)
    
    dataset = MixedDataset(HARD_FILE, ALL_FILE, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Hardening...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Hardening Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            labels = batch["label"].to("cuda").view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Save
        save_path = os.path.join(OUTPUT_DIR, f"critic_hardened_e{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_hardening()
