
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

from synthetic_critic_data_full import SyntheticCriticDatasetFull
from fractal_critic_full import FractalCriticFull

def train_critic_full():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data
    dataset = SyntheticCriticDatasetFull(size=10000, seq_len=8)
    
    # 2. Model
    model = FractalCriticFull(num_roots=31, embed_dim=64, max_len=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training Full Critic (Localization + Mutation)...")
    
    for epoch in range(10):
        total_loss = 0
        loc_acc = 0
        mut_acc = 0
        total = 0
        
        # Shuffle
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)
        
        batch_size = 128
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in indices[i:i+batch_size]]
            
            roots = torch.tensor([b['roots'] for b in batch], device=device)
            errors = torch.tensor([b['error'] for b in batch], device=device)
            idx_labels = torch.tensor([b['label_idx'] for b in batch], device=device)
            root_labels = torch.tensor([b['label_root'] for b in batch], device=device)
            
            # Forward
            loc_logits, mut_logits = model(roots, errors)
            
            # Loss (Multi-task)
            loss_loc = F.cross_entropy(loc_logits, idx_labels)
            loss_mut = F.cross_entropy(mut_logits, root_labels)
            loss = loss_loc + loss_mut
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Stats
            loc_preds = torch.argmax(loc_logits, dim=1)
            mut_preds = torch.argmax(mut_logits, dim=1)
            
            loc_acc += (loc_preds == idx_labels).sum().item()
            mut_acc += (mut_preds == root_labels).sum().item()
            total += len(batch)
            
        print(f"Epoch {epoch+1}: Loss {total_loss:.2f} | Loc Acc {loc_acc/total*100:.1f}% | Mut Acc {mut_acc/total*100:.1f}%")
        
    # Save
    path = Path("research-log/phase19.5-full-critic/critic_full.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    train_critic_full()
