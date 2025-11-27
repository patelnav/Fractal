
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

# Configuration - TRAIN DATA ONLY (not test!)
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "research-log/phase15-rl/data/grpo_train_labeled.jsonl"  # TRAIN data
CHECKPOINT_DIR = "research-log/phase15-rl/checkpoints_critic_clean"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

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

        total_layers = len(self.backbone.model.layers)
        freeze_layers = total_layers - 2

        print(f"Freezing bottom {freeze_layers} layers...")
        for param in self.backbone.model.embed_tokens.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.backbone.model.layers[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_indices = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        pooled_output = last_hidden_state[torch.arange(batch_size, device=input_ids.device), last_indices]
        logits = self.score_head(pooled_output)
        return logits

class MBPPDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        print(f"Loading {path}...")
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['status'] in ['passed', 'failed', 'timeout']:
                    label = 1.0 if entry['status'] == 'passed' else 0.0
                    self.data.append((entry['prompt'], entry['code'], label))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, code, label = self.data[idx]
        text = f"{prompt}\n\n# Solution:\n{code}"
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

def train_critic():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Training CLEAN Critic on TRAIN data: {DATA_FILE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CodeVerifier(MODEL_NAME)

    dataset = MBPPDataset(DATA_FILE, tokenizer, MAX_LEN)

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
            labels = batch["label"].to("cuda").view(-1, 1)

            optimizer.zero_grad()

            logits = model(input_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

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
