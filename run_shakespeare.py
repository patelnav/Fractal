"""
Shakespeare Fractal Diffusion Training

Phase 2: Training discrete diffusion on real natural language (tinyshakespeare).

Architecture improvements:
- Rotary Positional Embeddings (RoPE) for scale-invariant position understanding
- Scaled model (4-6 layers, 128-256 embed dim)
- Energy calculation for hallucination detection (Chen's Lemma 7)
"""

import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random
from tqdm import tqdm

from fractal_shakespeare import ShakespeareHierarchy, HierarchyConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ShakespeareConfig:
    """Configuration for Shakespeare fractal diffusion."""
    # Model architecture
    vocab_size: int = 4163       # Will be overwritten by hierarchy
    n_embd: int = 256            # Increased from 64
    n_head: int = 8              # Increased from 4
    n_layer: int = 6             # Increased from 2
    block_size: int = 5          # 1 condition + 4 target tokens
    dropout: float = 0.1         # Light dropout for regularization
    bias: bool = False

    # Diffusion
    num_timesteps: int = 100

    # Training
    batch_size: int = 128
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    warmup_iters: int = 500

    # Paths
    checkpoint_dir: str = "checkpoints"


# =============================================================================
# Rotary Positional Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position as rotation in complex space, enabling:
    - Relative position awareness
    - Extrapolation to longer sequences
    - Scale-invariant "nth position" understanding
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for all positions
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Update the cos/sin cache for a given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for rotary embedding.

        Args:
            x: Input tensor (for device)
            seq_len: Sequence length

        Returns:
            cos, sin tensors of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to queries and keys.

    Args:
        q: (B, n_heads, T, head_dim)
        k: (B, n_heads, T, head_dim)
        cos: (T, head_dim)
        sin: (T, head_dim)

    Returns:
        Rotated q and k
    """
    # Expand cos/sin to match q/k shape
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# =============================================================================
# Bidirectional Self-Attention with RoPE
# =============================================================================

class RoPEBidirectionalAttention(nn.Module):
    """
    Bidirectional self-attention with Rotary Position Embeddings.

    Key differences from standard attention:
    - No causal mask (bidirectional)
    - Uses RoPE instead of learned position embeddings
    """

    def __init__(self, config: ShakespeareConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention (bidirectional - no causal mask)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False
        )

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config: ShakespeareConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ShakespeareConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RoPEBidirectionalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimeEmbeddingMLP(nn.Module):
    """MLP to project timestep into embedding space."""

    def __init__(self, config: ShakespeareConfig):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal(t)
        return self.mlp(t_emb)


# =============================================================================
# Shakespeare Diffusion Model
# =============================================================================

class ShakespeareDiffusionModel(nn.Module):
    """
    Fractal Diffusion Model for Shakespeare text.

    Uses RoPE for scale-invariant position encoding.
    Handles both Root->Chunk and Chunk->Char denoising with shared weights.
    """

    def __init__(self, config: ShakespeareConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings - using RoPE)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Time embedding
        self.time_mlp = TimeEmbeddingMLP(config)

        # Transformer blocks with RoPE
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Shakespeare Diffusion Model: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,      # (B, 5) = [condition, target1, target2, target3, target4]
        t: torch.Tensor,      # (B,) timestep
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: (B, 4, vocab_size) predictions for target positions only
        """
        B, T = x.size()

        # Token embeddings
        tok_emb = self.wte(x)  # (B, T, n_embd)

        # Time embeddings - add to all positions
        time_emb = self.time_mlp(t)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)  # (B, 1, n_embd)

        # Combine embeddings (position is handled by RoPE in attention)
        h = tok_emb + time_emb

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)

        # Only predict for target positions (indices 1-4)
        logits = self.lm_head(h[:, 1:, :])  # (B, 4, vocab_size)

        return logits

    def get_scores(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Get discrete score function (for energy calculation).

        The score is defined as: S_i = logit_i - mean(logits)

        Returns:
            scores: (B, 4, vocab_size) centered logits
        """
        logits = self.forward(x, t)
        scores = logits - logits.mean(dim=-1, keepdim=True)
        return scores


# =============================================================================
# Discrete Diffusion for Shakespeare
# =============================================================================

class ShakespeareDiffusion:
    """
    Discrete diffusion with vocabulary-aware noise.

    Handles the unified vocabulary (chars, chunks, roots) properly.
    """

    def __init__(self, config: ShakespeareConfig, hierarchy: ShakespeareHierarchy):
        self.num_timesteps = config.num_timesteps
        self.hierarchy = hierarchy

        # Noise schedule: linear from 0 to 0.9
        self.noise_schedule = torch.linspace(0, 0.9, config.num_timesteps)

    def add_noise(
        self,
        x: torch.Tensor,           # (B, 4) clean target tokens
        t: torch.Tensor,           # (B,) timesteps
        device: torch.device
    ) -> torch.Tensor:
        """
        Add noise by randomly replacing tokens.

        Respects vocabulary boundaries - only replaces with tokens from the same level.
        """
        B, L = x.size()

        # Get noise probabilities
        probs = self.noise_schedule[t.cpu()].to(device)
        probs = probs.unsqueeze(1).expand(B, L)

        # Determine which positions to corrupt
        mask = torch.rand(B, L, device=device) < probs

        # Determine token levels
        char_offset = 0
        chunk_offset = self.hierarchy.chunk_offset
        root_offset = self.hierarchy.root_offset

        is_char = x < chunk_offset
        is_chunk = (x >= chunk_offset) & (x < root_offset)
        is_root = x >= root_offset

        # Generate random replacements for each level
        random_chars = torch.randint(0, self.hierarchy.char_vocab_size, (B, L), device=device)
        random_chunks = torch.randint(0, self.hierarchy.chunk_vocab_size, (B, L), device=device) + chunk_offset
        random_roots = torch.randint(0, self.hierarchy.root_vocab_size, (B, L), device=device) + root_offset

        # Select appropriate random tokens
        random_tokens = torch.where(is_char, random_chars,
                        torch.where(is_chunk, random_chunks, random_roots))

        # Apply noise
        noisy_x = torch.where(mask, random_tokens, x)

        return noisy_x

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


# =============================================================================
# Energy Calculation (Chen's Lemma 7)
# =============================================================================

@torch.no_grad()
def compute_generation_energy(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    condition: int,
    target_sequence: List[int],
    device: torch.device,
    num_integration_steps: int = 50
) -> float:
    """
    Compute the generation energy: E[integral of S_i^2 dt]

    This measures how "difficult" it is for the model to generate the target
    sequence from the condition. Higher energy = more resistance = likely hallucination.

    Args:
        model: The diffusion model
        diffusion: The diffusion process
        condition: Condition token ID
        target_sequence: Target token IDs (4 tokens)
        device: Torch device
        num_integration_steps: Number of steps for numerical integration

    Returns:
        Total energy (sum of squared scores integrated over time)
    """
    model.eval()

    condition_tensor = torch.tensor([[condition]], device=device)
    targets = torch.tensor([target_sequence], device=device)

    total_energy = 0.0
    dt = 1.0 / num_integration_steps

    # Integrate from t=0 to t=T
    for step in range(num_integration_steps):
        t_val = int(step * diffusion.num_timesteps / num_integration_steps)
        t = torch.tensor([t_val], device=device)

        # Add noise at this timestep
        noisy_targets = diffusion.add_noise(targets, t, device)

        # Build input
        x = torch.cat([condition_tensor, noisy_targets], dim=1)

        # Get scores
        scores = model.get_scores(x, t)  # (1, 4, vocab_size)

        # Energy at this timestep: sum of squared scores
        # Focus on the score for the true target tokens
        target_scores = torch.gather(
            scores,
            dim=2,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)  # (1, 4)

        energy_t = (scores ** 2).sum().item()
        total_energy += energy_t * dt

    return total_energy


# =============================================================================
# Training Loop
# =============================================================================

def train_shakespeare(
    config: ShakespeareConfig,
    hierarchy: ShakespeareHierarchy,
    device: torch.device
):
    """Train the Shakespeare diffusion model."""

    print("=" * 70)
    print("SHAKESPEARE FRACTAL DIFFUSION")
    print("Phase 2: Natural Language Discrete Diffusion")
    print("=" * 70)

    # Update vocab size from hierarchy
    config.vocab_size = hierarchy.total_vocab_size
    print(f"\nVocabulary size: {config.vocab_size}")

    # Create model and diffusion
    model = ShakespeareDiffusionModel(config).to(device)
    diffusion = ShakespeareDiffusion(config, hierarchy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler with warmup
    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * (it + 1) / config.warmup_iters
        return config.learning_rate

    print(f"\nTraining for {config.max_iters} iterations...")
    print(f"Device: {device}")
    print("-" * 70)

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    model.train()
    start_time = time.time()
    best_loss = float('inf')

    pbar = tqdm(range(config.max_iters), desc="Training", unit="iter")
    for iter_num in pbar:
        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get mixed batch (both Root->Chunk and Chunk->Char)
        conditions, targets = hierarchy.get_mixed_batch(config.batch_size, device)

        # Sample timesteps and add noise
        t = diffusion.sample_timesteps(config.batch_size, device)
        noisy_targets = diffusion.add_noise(targets, t, device)

        # Build input sequence
        x = torch.cat([conditions.unsqueeze(1), noisy_targets], dim=1)

        # Forward pass
        logits = model(x, t)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        # Logging and evaluation
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            model.eval()
            with torch.no_grad():
                # Evaluate on held-out samples
                eval_loss = evaluate_loss(model, diffusion, hierarchy, device)

            model.train()

            elapsed = time.time() - start_time
            iters_per_sec = (iter_num + 1) / elapsed if elapsed > 0 else 0
            remaining = (config.max_iters - iter_num - 1) / iters_per_sec if iters_per_sec > 0 else 0

            tqdm.write(f"Iter {iter_num:5d} | Train Loss: {loss.item():.4f} | "
                       f"Eval Loss: {eval_loss:.4f} | LR: {lr:.2e} | "
                       f"ETA: {remaining/60:.1f}min")

            # Save best checkpoint
            if eval_loss < best_loss:
                best_loss = eval_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_loss': best_loss,
                }
                torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'best.pt'))
                tqdm.write(f"  -> New best model saved (loss: {best_loss:.4f})")

    print("-" * 70)
    print(f"Training complete! Best loss: {best_loss:.4f}")

    # Save final checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter_num': config.max_iters,
        'best_loss': best_loss,
    }
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'final.pt'))

    return model, diffusion


@torch.no_grad()
def evaluate_loss(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    hierarchy: ShakespeareHierarchy,
    device: torch.device,
    n_batches: int = 10
) -> float:
    """Evaluate average loss on random batches."""
    model.eval()
    total_loss = 0.0

    for _ in range(n_batches):
        conditions, targets = hierarchy.get_mixed_batch(64, device)
        t = diffusion.sample_timesteps(64, device)
        noisy_targets = diffusion.add_noise(targets, t, device)
        x = torch.cat([conditions.unsqueeze(1), noisy_targets], dim=1)

        logits = model(x, t)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            targets.reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / n_batches


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_from_condition(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    hierarchy: ShakespeareHierarchy,
    condition: int,
    device: torch.device,
    num_steps: int = 50
) -> List[int]:
    """Generate 4 tokens from a condition using reverse diffusion."""
    model.eval()

    # Determine output token range based on condition
    if condition >= hierarchy.root_offset:
        # Condition is a root -> output chunks
        start = hierarchy.chunk_offset
        end = hierarchy.chunk_offset + hierarchy.chunk_vocab_size
    elif condition >= hierarchy.chunk_offset:
        # Condition is a chunk -> output chars
        start = 0
        end = hierarchy.char_vocab_size
    else:
        # Condition is a char -> shouldn't happen in normal use
        start = 0
        end = hierarchy.char_vocab_size

    # Start with random tokens
    x = torch.randint(start, end, (1, 4), device=device)
    condition_tensor = torch.tensor([[condition]], device=device)

    # Reverse diffusion
    timesteps = list(range(diffusion.num_timesteps - 1, -1, -max(1, diffusion.num_timesteps // num_steps)))
    if 0 not in timesteps:
        timesteps.append(0)

    for t_val in timesteps:
        t = torch.tensor([t_val], device=device)
        inp = torch.cat([condition_tensor, x], dim=1)
        logits = model(inp, t)

        # Use argmax at low t, sample at high t
        if t_val < diffusion.num_timesteps // 4:
            x = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, 4)

    return x[0].tolist()


@torch.no_grad()
def generate_text_recursive(
    model: ShakespeareDiffusionModel,
    diffusion: ShakespeareDiffusion,
    hierarchy: ShakespeareHierarchy,
    root_id: int,
    device: torch.device
) -> str:
    """Recursively generate text from a root token."""

    # Root -> Chunks
    chunk_ids = generate_from_condition(model, diffusion, hierarchy, root_id, device)

    # Chunks -> Chars
    all_chars = []
    for chunk_id in chunk_ids:
        char_ids = generate_from_condition(model, diffusion, hierarchy, chunk_id, device)
        all_chars.extend(char_ids)

    # Decode to text
    return hierarchy.decode_chars(all_chars)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load or build hierarchy
    hierarchy_path = "data/shakespeare_hierarchy.pkl"
    if os.path.exists(hierarchy_path):
        hierarchy = ShakespeareHierarchy.load(hierarchy_path)
    else:
        print("Building hierarchy...")
        hierarchy = ShakespeareHierarchy(HierarchyConfig())
        hierarchy.build()
        hierarchy.save()

    # Create config
    config = ShakespeareConfig()

    # Train
    model, diffusion = train_shakespeare(config, hierarchy, device)

    # Demo: generate some text
    print("\n" + "=" * 70)
    print("GENERATION DEMO")
    print("=" * 70)

    # Pick some random roots and generate
    for i in range(5):
        root_id = random.choice(list(hierarchy.root_to_id.values())) + hierarchy.root_offset
        text = generate_text_recursive(model, diffusion, hierarchy, root_id, device)
        root_str = hierarchy.decode_root(root_id)
        print(f"\nRoot: '{root_str}'")
        print(f"Generated: '{text}'")
