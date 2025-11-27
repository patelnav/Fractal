import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase11-fractal-logic')))
from model_logic import FractalRecurrentALU, BitConfig

class FractalMultiplier(nn.Module):
    """
    A Zero-Shot Neural Multiplier.
    Composes a pre-trained Adder into a digital shift-and-add loop.
    """
    def __init__(self, config, adder_checkpoint=None):
        super().__init__()
        self.config = config
        
        self.adder = FractalRecurrentALU(config)
        
        if adder_checkpoint:
            print(f"Loading Adder from {adder_checkpoint}")
            state_dict = torch.load(adder_checkpoint, map_location='cpu')
            self.adder.load_state_dict(state_dict, strict=False)
            
        # CRITICAL FIX: Freeze the Adder.
        # We are not training. We are proving compositionality.
        for param in self.adder.parameters():
            param.requires_grad = False
            
    def shift_embedding(self, emb, shift_amount):
        batch_size, seq_len, dim = emb.size()
        if shift_amount == 0:
            return emb
            
        zero_vec = self.adder.bit_emb(torch.tensor(0, device=emb.device)).view(1, 1, -1)
        zeros = zero_vec.expand(batch_size, shift_amount, -1)
        
        shifted = torch.cat([zeros, emb], dim=1)
        return shifted[:, :seq_len, :]

    def forward(self, a_seq, b_seq):
        """
        Forward pass implementing Digital Shift-and-Add.
        """
        # 1. Embed Inputs
        emb_a = self.adder.bit_emb(a_seq) 
        
        # 2. Initialize Accumulator
        zeros_idx = torch.zeros_like(a_seq)
        accumulator_emb = self.adder.bit_emb(zeros_idx)
        
        # 3. Loop
        batch_size, seq_len = a_seq.size()
        
        for i in range(seq_len):
            b_bit = b_seq[:, i] 
            
            shifted_a = self.shift_embedding(emb_a, i)
            zeros_seq = self.adder.bit_emb(zeros_idx)
            
            # Masking (Digital Logic)
            mask = b_bit.view(-1, 1, 1).float()
            term_emb = shifted_a * mask + zeros_seq * (1 - mask)
            
            # Call Adder
            logits, _ = self.adder.forward_embeddings(accumulator_emb, term_emb)
            
            # DIGITAL RESTORATION (Hard Snap)
            # We use Argmax to simulate a perfect digital wire.
            # This prevents analog noise from accumulating.
            tokens = torch.argmax(logits, dim=-1) # [B, L]
            
            # Re-Embed cleanly
            accumulator_emb = self.adder.bit_emb(tokens)
            
        # Final Result
        return logits