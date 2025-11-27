
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

from synthetic_critic_data import SyntheticCriticDataset
from fractal_critic import FractalCritic

def train_critic():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data
    dataset = SyntheticCriticDataset(size=10000, seq_len=8)
    print(f"Dataset: {len(dataset)} samples")
    
    # 2. Model
    model = FractalCritic(num_roots=31, embed_dim=64, max_len=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 3. Train Loop
    batch_size = 128
    
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle batching manually or via loader (manual for simplicity)
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset[j] for j in batch_indices]
            
            # Collate
            roots = torch.tensor([b['roots'] for b in batch], device=device)
            errors = torch.tensor([b['error'] for b in batch], device=device)
            labels = torch.tensor([b['label_idx'] for b in batch], device=device)
            
            # Forward
            logits = model(roots, errors)
            
            # Loss
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Acc {acc:.2f}%")
        
    # Save
    path = Path("research-log/phase19-fractal-critic/critic_model.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    train_critic()
