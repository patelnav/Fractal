
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time

# Add path for Fractal Engine
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))

# Reuse the model architecture
from run_fractal_engine import FractalDiffusionModel, FractalModelConfig

from synthetic_data import SyntheticCodeDataset, CodeConfig, encode_text, CHARS

def train_fractal_coder():
    # 1. Setup Data
    dataset = SyntheticCodeDataset(size=5000)
    config = CodeConfig()
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Config: {config.num_roots} roots, {len(CHARS)} chars vocab")
    
    # 2. Setup Model
    # We map synthetic concepts to FractalModelConfig
    # Level 0: Root -> Chunks (Here: Root -> Chars directly for simplicity? 
    # Actually Plan says: Root -> Chunks -> Chars. 
    # But to save time, let's do 1-Level: Root -> Chars (Expansion Size 4)
    # The "Flash Flood" works for any number of levels.
    
    model_config = FractalModelConfig(
        num_chars=len(CHARS), 
        num_chunks=config.num_roots, # We treat Roots as Level 0, outputting Chars directly
        num_roots=config.num_roots, 
        pad_char_id=len(CHARS), # Pad ID
        pad_chunk_id=config.num_roots,
        max_char_len=4, # Expansion size
        expansion_size=1, # 1 Root -> 1 "Chunk" (which is 4 chars)
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        num_timesteps=50 # Faster training
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FractalDiffusionModel(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Training Model (Root -> Text mapping)...")
    
    # 3. Training Loop
    batch_size = 64
    
    for epoch in range(5):
        total_loss = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if not batch: break
            
            # Prepare Tensors
            # We are training Level 1: Condition=Root, Target=Chars
            
            # Flatten batch: (B * SeqLen)
            all_roots = []
            all_chars = []
            
            for roots, text in batch:
                for r_idx, r in enumerate(roots):
                    all_roots.append(r)
                    # Extract corresponding 4 chars
                    chunk_text = text[r_idx*4 : (r_idx+1)*4]
                    all_chars.append(encode_text(chunk_text))
            
            # Tensors
            cond = torch.tensor(all_roots, device=device)
            targets = torch.tensor(all_chars, device=device)
            
            # Add Noise
            t = torch.randint(0, model_config.num_timesteps, (len(cond),), device=device)
            
            # Simple discrete noise: replace with random token
            mask = torch.rand_like(targets.float()) < (t.float() / model_config.num_timesteps).unsqueeze(1)
            noise = torch.randint(0, model_config.num_chars, targets.shape, device=device)
            noisy_targets = torch.where(mask, noise, targets)
            
            # Forward
            # Model expects [Condition, Target]
            # cond needs offset. In this config, Chunks are the "output" of level 0. 
            # But we are using the model as [Root -> Chars]. 
            # So let's pretend Roots are Level 0 output (Chunks) and Chars are Level 1 output.
            # So we are training Level 1.
            
            # Offset for condition (Roots treated as Chunks)
            cond_input = (cond + model_config.chunk_offset).unsqueeze(1)
            target_input = noisy_targets # Chars are 0..N
            
            x = torch.cat([cond_input, target_input], dim=1)
            
            logits, _ = model(x, t, level=1) # Level 1: Chunk -> Chars
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss / (len(dataset)/batch_size):.4f}")

    # Save
    path = Path("research-log/phase18-fractal-coder/fractal_coder_model.pt")
    torch.save({
        'model': model.state_dict(),
        'config': model_config
    }, path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    train_fractal_coder()
