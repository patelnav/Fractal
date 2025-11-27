
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "research-log/phase15-rl/data/grpo_test_labeled.jsonl"
# Using the MBPP-trained Critic from Phase 14
CRITIC_CHECKPOINT = "research-log/phase14-vector6-reboot/checkpoints_critic/critic_e3.pt" 
MAX_LEN = 512
BATCH_SIZE = 32

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

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['prompt']}\n\n# Solution:\n{item['code']}"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "idx": idx
        }

def evaluate_combined():
    print("Loading Data...")
    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    tasks = {}
    for item in data:
        tid = item['task_id']
        if tid not in tasks:
            tasks[tid] = []
        tasks[tid].append(item)
    
    print("Loading Critic...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = CodeVerifier(MODEL_NAME)
    state_dict = torch.load(CRITIC_CHECKPOINT)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Scoring Samples...")
    dataset = TestDataset(data, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            logits = model(input_ids, mask)
            scores = torch.sigmoid(logits).float().cpu().numpy().flatten()
            all_scores.extend(scores)
            
    for i, item in enumerate(data):
        item['critic_score'] = float(all_scores[i])
        
    print("Calculating Metrics...")
    
    pass_baseline = 0
    pass_critic = 0
    pass_oracle = 0
    total = 0
    
    for tid, samples in tasks.items():
        total += 1
        
        # Baseline (Random/Avg)
        passed = sum(1 for s in samples if s['status'] == 'passed')
        pass_baseline += (passed / len(samples))
        
        # Oracle
        if passed > 0:
            pass_oracle += 1
            
        # Critic
        samples.sort(key=lambda x: x['critic_score'], reverse=True)
        if samples[0]['status'] == 'passed':
            pass_critic += 1
            
    print("="*40)
    print("GRAND UNIFICATION RESULTS")
    print("="*40)
    print(f"Base Pass@1 (Random): {pass_baseline/total*100:.2f}%")
    print(f"Critic Pass@1 (Top-1): {pass_critic/total*100:.2f}%")
    print(f"Oracle Pass@N: {pass_oracle/total*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_combined()
