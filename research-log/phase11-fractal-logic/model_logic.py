import torch
import torch.nn as nn
import math

class BitConfig:
    def __init__(self, 
                 vocab_size=4, 
                 dim=128, 
                 depth=2, 
                 heads=4, 
                 max_len=64,
                 dropout=0.0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_len = max_len
        self.dropout = dropout

class FractalRecurrentALU(nn.Module):
    """
    A Recurrent Transformer with Gated Residuals (GRU-like).
    Improves gradient flow for deep recursion.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        self.bit_emb = nn.Embedding(config.vocab_size, config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        
        # Gating Mechanism (GRU Cell)
        # Input: Transformer Output (New info)
        # Hidden: Old State (Accumulator)
        self.gate = nn.GRUCell(config.dim, config.dim)
        
        self.start_state = nn.Parameter(torch.randn(1, config.dim)) # [1, D] for GRU
        self.head = nn.Linear(config.dim, 2)

    def forward(self, a_seq, b_seq):
        emb_a = self.bit_emb(a_seq)
        emb_b = self.bit_emb(b_seq)
        return self.forward_embeddings(emb_a, emb_b)

    def forward_embeddings(self, emb_a, emb_b):
        batch_size, seq_len, _ = emb_a.size()
        
        # Initial State [B, D]
        h_t = self.start_state.expand(batch_size, -1)
        
        outputs = [] 
        hidden_outputs = []
        
        for t in range(seq_len):
            in_a = emb_a[:, t:t+1, :]
            in_b = emb_b[:, t:t+1, :]
            
            # Transformer processes inputs + current state context
            # We view h_t as a token [B, 1, D]
            state_token = h_t.unsqueeze(1)
            
            # Context: [State, A, B]
            # The model sees the old state and the new inputs
            current_input = torch.cat([state_token, in_a, in_b], dim=1)
            
            # Run Core
            out_seq = self.core(current_input)
            
            # Extract the processed "Update Vector" (first token corresponding to state)
            update_vec = out_seq[:, 0, :] # [B, D]
            
            # Gated Update: h_t = GRU(update, h_t)
            # This allows the model to learn "Identity" easily (Keep state)
            h_next = self.gate(update_vec, h_t)
            
            h_t = h_next
            
            logits_t = self.head(h_t).unsqueeze(1) # [B, 1, 2]
            outputs.append(logits_t)
            hidden_outputs.append(h_t.unsqueeze(1))
            
        logits = torch.cat(outputs, dim=1)
        soft_bits = torch.cat(hidden_outputs, dim=1)
        soft_bits = self.out_proj(soft_bits)
        
        return logits, soft_bits

class BitTokenizer:
    def __init__(self):
        self.vocab = {'0': 0, '1': 1, 'SEP': 2, 'PAD': 3}
        
    def encode_bits(self, bits_str, max_len):
        ids = [int(c) for c in bits_str]
        ids = ids[::-1] 
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return torch.tensor(ids[:max_len])