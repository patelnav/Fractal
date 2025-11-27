import torch
import torch.nn as nn
import math

class BitConfig:
    def __init__(self, 
                 vocab_size=4, 
                 dim=128, 
                 depth=2, # Deeper per step? Or shallow per step? 2 is enough for a full adder.
                 heads=4, 
                 max_len=64,
                 dropout=0.0): # No dropout for logic
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_len = max_len
        self.dropout = dropout

class FractalRecurrentALU(nn.Module):
    """
    A Recurrent Transformer for Bitwise Logic.
    Processes bits from LSB to MSB sharing the EXACT same weights.
    Guarantees extrapolation if the logic is learned.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        # Embeddings for 0 and 1 (and Pad/Sep if needed, but mainly 0/1)
        self.bit_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # The Recurrent Core (The "Fractal Block")
        # Input: [Batch, 1, Dim] (Current Bit Embedding + Hidden State)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        
        # Carry/Hidden State
        # We learn a "Start" hidden state
        self.start_state = nn.Parameter(torch.randn(1, 1, config.dim))
        
        # Output Head (Predicts next bit 0/1)
        self.head = nn.Linear(config.dim, 2)
        
        # State Update Projector (Optional, or just use output of Transformer)
        # Transformer output is [B, 1, D]. We use this as next hidden state.

    def forward(self, a_seq, b_seq):
        """
        a_seq: [Batch, SeqLen] (Indices 0/1) - LSB first!
        b_seq: [Batch, SeqLen] (Indices 0/1) - LSB first!
        """
        batch_size, seq_len = a_seq.size()
        
        # Embed inputs
        # [B, L, D]
        emb_a = self.bit_emb(a_seq)
        emb_b = self.bit_emb(b_seq)
        
        # Initial State (Carry=0 effectively)
        # [B, 1, D]
        h_t = self.start_state.expand(batch_size, -1, -1)
        
        outputs = []
        
        for t in range(seq_len):
            # Input at step t: Concatenate or Add?
            # Let's Add: Input = Emb(A_t) + Emb(B_t) + Hidden_t
            # Or Concat? Concat is safer for distinct inputs.
            # Let's Add for now to keep dims same, but maybe project first?
            # Simple addition: The model has to learn to separate them.
            # Better: Input = Emb(A_t) + Emb(B_t) + Hidden_t
            
            # Slicing [B, 1, D]
            in_a = emb_a[:, t:t+1, :]
            in_b = emb_b[:, t:t+1, :]
            
            # Combine
            # To distinguish A from B, we might need separate embeddings or learnable position codes?
            # But A and B are symmetric for addition.
            # Let's just sum them. The hidden state holds the "Carry".
            current_input = in_a + in_b + h_t
            
            # Run Core
            # [B, 1, D]
            out_t = self.core(current_input)
            
            # Update Hidden State for next step
            h_t = out_t
            
            # Prediction
            logits_t = self.head(out_t) # [B, 1, 2]
            outputs.append(logits_t)
            
        # Stack outputs [B, L, 2]
        return torch.cat(outputs, dim=1)

class BitTokenizer:
    # Same as before, but we need helper to reverse string for LSB processing
    def __init__(self):
        self.vocab = {'0': 0, '1': 1, 'SEP': 2, 'PAD': 3}
        
    def encode_bits(self, bits_str, max_len):
        # Bits string "101" (Decimal 5).
        # We need LSB first: [1, 0, 1]
        # Pad with 0s to max_len
        ids = [int(c) for c in bits_str]
        # Reverse to get LSB at index 0? 
        # "101" -> MSB is left. 
        # If string is "100" (Decimal 4), LSB is right.
        # Standard string: MSB...LSB
        # We want LSB...MSB for recurrence (carry propagates up)
        ids = ids[::-1] 
        
        # Pad with 0s (which is ID 0)
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return torch.tensor(ids[:max_len])