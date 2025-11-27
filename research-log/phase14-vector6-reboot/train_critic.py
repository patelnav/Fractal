
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "research-log/phase14-vector6-reboot/data/mbpp_labeled.jsonl"
CHECKPOINT_DIR = "research-log/phase14-vector6-reboot/checkpoints_critic"
MAX_LEN = 512
BATCH_SIZE = 8 # Fit on A100
EPOCHS = 3
LR = 2e-5

class CodeVerifier(nn.Module):
    """
    Qwen-based Code Verifier.
    Uses the CausalLM backbone but pools the last hidden state for classification.
    """
    def __init__(self, model_name):
        super().__init__()
        # Load base model
        # We discard the LM head, or just ignore it.
        # To save memory, we can load with output_hidden_states=True and not load the lm_head weights?
        # Easier to just load the full CausalLM and ignore the head.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, # Ampere efficiency
            device_map="cuda"
        )
        self.config = self.backbone.config
        
        # Classification Head
        # Input: Hidden size (e.g. 2048 for 1.5B?)
        self.hidden_size = self.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1) # Output: Logit for "Pass"
        ).to("cuda").to(torch.bfloat16)
        
        # Freeze backbone initially?
        # Let's finetune top 2 layers + head.
        # Identify layers. Qwen uses 'model.layers'
        total_layers = len(self.backbone.model.layers)
        freeze_layers = total_layers - 2
        
        print(f"Freezing bottom {freeze_layers} layers...")
        for param in self.backbone.model.embed_tokens.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.backbone.model.layers[i].parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        # Run backbone
        # We want the hidden state of the last token
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # [B, L, D]
        
        # Pooling: We take the representation of the LAST token (EOS or end of sequence)
        # Since we use left-padding or right-padding?
        # Usually right padding for training. We need to find the last non-pad token index.
        # Simple trick: Input is right-padded.
        # If we assume attention_mask is 1 for real tokens, we can find the index.
        # But Qwen/Llama usually handle this.
        
        # Let's just take the last token's embedding (at sequence_length - 1)
        # Actually, for CausalLM, the last token attends to everything.
        # So picking the last index is correct.
        
        # Find the last '1' in attention mask per row
        last_indices = attention_mask.sum(dim=1) - 1 # [B]
        
        # Gather
        batch_size = input_ids.size(0)
        
        # Correct gathering logic
        pooled_output = last_hidden_state[torch.arange(batch_size), last_indices] # [B, D]
        
        logits = self.score_head(pooled_output) # [B, 1]
        return logits

class MBPPDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        print(f"Loading {path}...")
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Only keep if status is 'passed' or 'failed' (ignore timeout/error?)
                # Actually, timeout is a Fail.
                if entry['status'] in ['passed', 'failed', 'timeout']:
                    label = 1.0 if entry['status'] == 'passed' else 0.0
                    self.data.append((entry['prompt'], entry['code'], label))
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Balance classes?
        # Usually Fail >> Pass.
        # Let's undersample Fails or use class weights.
        # For now, simple random sampling.
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, code, label = self.data[idx]
        
        # Format: <prompt> ... <code>
        text = f"{prompt}\n\n# Solution:\n{code}"
        
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
            "label": torch.tensor(label, dtype=torch.float) # BCEWithLogitsLoss needs float
        }

def train_critic():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Initializing Critic Model based on {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = CodeVerifier(MODEL_NAME)
    
    dataset = MBPPDataset(DATA_FILE, tokenizer, MAX_LEN)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            labels = batch["label"].to("cuda").view(-1, 1) # [B, 1]
            
            optimizer.zero_grad()
            
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to("cuda")
                mask = batch["attention_mask"].to("cuda")
                labels = batch["label"].to("cuda").view(-1, 1)
                
                logits = model(input_ids, mask)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"critic_e{epoch+1}.pt"))

if __name__ == "__main__":
    train_critic()
