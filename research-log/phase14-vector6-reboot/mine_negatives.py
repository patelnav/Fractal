
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "research-log/phase14-vector6-reboot/data/mbpp_labeled.jsonl" # The training data
CRITIC_CHECKPOINT = "research-log/phase14-vector6-reboot/checkpoints_critic/critic_e3.pt" # Use best checkpoint
OUTPUT_FILE = "research-log/phase14-vector6-reboot/data/hard_negatives.jsonl"
BATCH_SIZE = 32
MAX_LEN = 512

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

class MiningDataset(Dataset):
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

def mine_hard_negatives():
    print(f"Loading Data from {DATA_FILE}...")
    data = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['status'] in ['passed', 'failed', 'timeout']:
                data.append(entry)
            
    print(f"Loaded {len(data)} samples.")
    
    print("Loading Critic...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = CodeVerifier(MODEL_NAME)
    state_dict = torch.load(CRITIC_CHECKPOINT)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Scoring Training Data...")
    dataset = MiningDataset(data, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            logits = model(input_ids, mask)
            scores = torch.sigmoid(logits).float().cpu().numpy().flatten()
            all_scores.extend(scores)
            
    # Analyze
    hard_negatives = []
    hard_positives = [] # Positives that got low scores (confusion) 
    
    for i, item in enumerate(data):
        score = float(all_scores[i])
        status = item['status']
        
        # Definition of Hard Negative: Actual=Fail, Predicted=High (Confidently Wrong)
        if status != 'passed' and score > 0.5:
            item['mining_type'] = 'hard_negative'
            item['score'] = score
            hard_negatives.append(item)
            
        # Definition of Hard Positive: Actual=Pass, Predicted=Low (Missed Opportunity)
        if status == 'passed' and score < 0.5:
            item['mining_type'] = 'hard_positive'
            item['score'] = score
            hard_positives.append(item)

    print("="*40)
    print("MINING RESULTS")
    print("="*40)
    print(f"Total Samples: {len(data)}")
    print(f"Hard Negatives (Fail but Score > 0.5): {len(hard_negatives)}")
    print(f"Hard Positives (Pass but Score < 0.5): {len(hard_positives)}")
    
    # Save for inspection/retraining
    # We want to construct a dataset that emphasizes these.
    # Strategy: Take ALL Hard Negatives + ALL Hard Positives + Random Sample of Easy cases to prevent forgetting.
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in hard_negatives:
            f.write(json.dumps(item) + "\n")
        for item in hard_positives:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(hard_negatives) + len(hard_positives)} mined samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    mine_hard_negatives()
