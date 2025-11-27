
import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalCritic(nn.Module):
    """
    Predicts which Root is faulty given the program and the execution error.
    """
    def __init__(self, num_roots=31, embed_dim=64, hidden_dim=128, max_len=10):
        super().__init__()
        self.num_roots = num_roots
        
        # Embeddings
        self.root_emb = nn.Embedding(num_roots, embed_dim)
        
        # Error Embedding: Simple MLP to project scalar error to vector
        # We handle error as a float.
        self.error_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer Encoder (Small)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output Head
        # Predicts a score for each position in the sequence
        self.head = nn.Linear(embed_dim, 1)
        
    def forward(self, roots, error):
        """
        roots: (B, L)
        error: (B,)
        """
        B, L = roots.shape
        
        # Embed roots
        x = self.root_emb(roots) # (B, L, E)
        
        # Embed error and broadcast/concat?
        # Strategy: Prepend Error as a [CLS]-like token or add to all?
        # Let's add to all.
        err_vec = self.error_proj(error.float().unsqueeze(1).unsqueeze(1)) # (B, 1, 1, E) -> proj -> (B, 1, E) ?
        # Error proj input (B, 1)
        err_vec = self.error_proj(error.float().view(B, 1)) # (B, E)
        
        # Add error info to every token
        x = x + err_vec.unsqueeze(1)
        
        # Transform
        # Mask padding? 
        # We assume 30 is PAD.
        src_key_padding_mask = (roots == 30)
        
        feat = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # (B, L, E)
        
        # Predict logits per position
        logits = self.head(feat).squeeze(-1) # (B, L)
        
        # Mask padding logits to -inf
        logits = logits.masked_fill(src_key_padding_mask, float('-inf'))
        
        return logits
