
import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalCriticFull(nn.Module):
    """
    Dual-Head Critic:
    1. Predicts Faulty Index (Localization).
    2. Predicts Correct Root (Mutation).
    """
    def __init__(self, num_roots=31, embed_dim=64, hidden_dim=128, max_len=10):
        super().__init__()
        self.num_roots = num_roots
        
        # Shared Encoder
        self.root_emb = nn.Embedding(num_roots, embed_dim)
        self.error_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Head 1: Localization (Pointer)
        self.loc_head = nn.Linear(embed_dim, 1)
        
        # Head 2: Mutation (Vocab)
        # Input: Global pooled features (average) + Error info
        # Or: Attend to the predicted faulty location? 
        # For simplicity (and since we don't have ground truth location at inference time for this head without recursive logic),
        # we will use the [Global Context] to predict "What is the missing operation?".
        # The error signal dominates this decision (e.g. Error=+10 -> likely need ADD 10).
        self.mut_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roots)
        )
        
    def forward(self, roots, error):
        """
        roots: (B, L)
        error: (B,)
        Returns: loc_logits (B, L), mut_logits (B, V)
        """
        B, L = roots.shape
        
        # Embed
        x = self.root_emb(roots) # (B, L, E)
        err_vec = self.error_proj(error.float().view(B, 1)) # (B, E)
        
        # Add error to tokens
        x = x + err_vec.unsqueeze(1)
        
        # Mask padding
        src_key_padding_mask = (roots == 30)
        
        # Transform
        feat = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # (B, L, E)
        
        # 1. Localization Head
        loc_logits = self.loc_head(feat).squeeze(-1) # (B, L)
        loc_logits = loc_logits.masked_fill(src_key_padding_mask, float('-inf'))
        
        # 2. Mutation Head
        # Global pooling (Mean of non-pad tokens)
        # Simple hack: just mean all, pad embeddings will dilute but it's okay for this scale.
        # Better: Weighted sum? 
        # Let's just take mean of feat.
        global_feat = feat.mean(dim=1) # (B, E)
        mut_logits = self.mut_head(global_feat) # (B, V)
        
        return loc_logits, mut_logits
