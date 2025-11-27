
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class FlashFloodDecoder:
    """
    Implements the "Flash Flood" hierarchical parallel decoding strategy.
    Decouples planning (Roots) from rendering (Tokens) using massive parallelism.
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
    def expand_level_parallel(
        self, 
        condition_ids: torch.Tensor, 
        level: int, 
        k: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand a batch of condition IDs into target sequences in parallel.
        Uses Best-of-K rejection sampling vectorized over the batch.
        
        Args:
            condition_ids: (B,) tensor of condition tokens (Roots or Chunks)
            level: 0 for Root->Chunks, 1 for Chunk->Chars
            k: Number of candidates to generate per condition (Best-of-K)
            
        Returns:
            selected_targets: (B, target_seq_len)
            selected_energies: (B,)
        """
        B = condition_ids.size(0)
        
        # 1. Setup config based on level
        if level == 0:
            target_vocab = self.config.num_chunks
            seq_len = self.config.expansion_size
            cond_offset = self.config.root_offset
            target_offset = self.config.chunk_offset
        else:
            target_vocab = self.config.num_chars
            seq_len = self.config.max_char_len
            cond_offset = self.config.chunk_offset
            target_offset = 0
            
        # 2. Replicate inputs for Best-of-K
        # (B,) -> (B*K,)
        cond_expanded = condition_ids.repeat_interleave(k)
        
        # 3. Generate Candidates (Random Noise)
        # We use t=T-1 (pure noise) as input, but model is trained to denoise.
        # Actually, the current model is a "One-Step" diffusion (or close to it) for generation 
        # in the demo, but properly it's a diffusion model.
        # The 'generate_with_rejection' in Phase 6 uses:
        #   noise = rand(...)
        #   logits = model(noise, t=99)
        #   candidate = multinomial(logits)
        #   energy = model(candidate, t=0)
        
        # Step A: Propose Candidates
        # Noise: (B*k, seq_len)
        noise = torch.randint(0, target_vocab, (B * k, seq_len), device=self.device)
        
        # Input: [Condition, Noise]
        # cond_expanded: (B*k,) -> (B*k, 1)
        cond_input = (cond_expanded + cond_offset).unsqueeze(1)
        target_input = noise + target_offset
        x_t = torch.cat([cond_input, target_input], dim=1)
        
        # Timestep T-1 (Maximum noise)
        t_max = torch.tensor([self.config.num_timesteps - 1], dtype=torch.long, device=self.device)
        t_batch = t_max.repeat(B * k)
        
        with torch.no_grad():
            # We don't need energy here, just logits to sample likely candidates
            logits, _ = self.model(x_t, t_batch, level=level, return_energy=False)
            
        # Sample from logits
        # logits: (B*k, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)
        
        # We can take argmax or sample. Using Argmax for speed/stability in this demo.
        # (B*k, seq_len)
        candidates = torch.argmax(probs, dim=-1)
        
        # Step B: Verify Candidates (Energy Check)
        # Input: [Condition, Candidate]
        candidate_input = candidates + target_offset
        x_0 = torch.cat([cond_input, candidate_input], dim=1)
        
        # Timestep 0 (Clean data)
        t_0 = torch.zeros(B * k, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            _, energies = self.model(x_0, t_0, level=level, return_energy=True)
            
        # energies: (B*k,)
        
        # 4. Selection (Best-of-K)
        # Reshape to (B, k)
        energies_reshaped = energies.view(B, k)
        
        # Find index of min energy for each B
        best_indices = torch.argmin(energies_reshaped, dim=1) # (B,)
        
        # Gather the best candidates
        # candidates: (B*k, seq_len)
        # We need to select the row corresponding to (b * k + best_index[b])
        
        offsets = torch.arange(B, device=self.device) * k
        flat_indices = offsets + best_indices # (B,)
        
        selected_targets = candidates[flat_indices] # (B, seq_len)
        selected_energies = energies[flat_indices]  # (B,)
        
        return selected_targets, selected_energies

    def render_parallel(self, root_ids: torch.Tensor, k: int = 16):
        """
        Full hierarchical rendering: Roots -> Chunks -> Chars
        """
        B = root_ids.size(0)
        
        # === Level 0: Roots -> Chunks ===
        # chunks: (B, ExpansionSize)
        chunks, energy0 = self.expand_level_parallel(root_ids, level=0, k=k)
        
        # Flatten chunks to be the input for Level 1
        # (B * ExpansionSize,)
        chunk_ids_flat = chunks.reshape(-1)
        
        # === Level 1: Chunks -> Chars ===
        # chars: (B * ExpansionSize, MaxCharLen)
        chars, energy1 = self.expand_level_parallel(chunk_ids_flat, level=1, k=k)
        
        # Reshape back to (B, ExpansionSize, MaxCharLen)
        chars_reshaped = chars.view(B, self.config.expansion_size, self.config.max_char_len)
        
        return chars_reshaped
