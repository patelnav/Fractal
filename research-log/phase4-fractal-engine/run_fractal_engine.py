"""
Fractal Engine: Shared-Weight Hierarchical Diffusion with Energy Verification

Phase 4: The Integrated Fractal Engine

Key innovations:
1. SHARED WEIGHTS: Same transformer handles both Level 0 (Root→Chunks) and Level 1 (Chunk→Chars)
2. LEVEL EMBEDDINGS: Minimal adapter that tells model "you are plotting" vs "you are spelling"
3. ENERGY HEAD: Contrastive training for hallucination detection at BOTH levels
4. INTERLEAVED TRAINING: 50% Level 0 + 50% Level 1 per batch

This tests the hypothesis that intelligence is SCALE-INVARIANT:
the logic of "Plotting" is mathematically identical to the logic of "Spelling",
just rotated in the vector space.

Based on Chen's 2025 paper on diffusion in Boolean hypercubes.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from hierarchical_bpe import (
    HierarchicalBPE,
    HierarchicalBPEConfig,
    FractalDataset,
    load_fractal_dataset,
    build_fractal_dataset,
    print_fractal_stats
)
from fractal_loader import (
    FractalDataLoader,
    FractalBatch,
    get_level0_batch,
    get_level1_batch,
    create_attention_mask
)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class FractalModelConfig:
    """Configuration for the Fractal Diffusion Model."""
    # Architecture
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1

    # Vocabularies (set from dataset)
    num_chars: int = 65  # Character vocabulary
    num_chunks: int = 1024  # Level 1 tokens
    num_roots: int = 2048  # Level 0 tokens (chunks + merged roots)

    # Padding
    pad_char_id: int = 65
    pad_chunk_id: int = 1024

    # Sequence lengths
    max_char_len: int = 16  # Max chars per chunk (Level 1)
    expansion_size: int = 4  # Root -> N chunks (Level 0)

    # Diffusion
    num_timesteps: int = 100

    # Number of hierarchy levels
    num_levels: int = 2  # Level 0 and Level 1

    @property
    def total_vocab_size(self):
        """
        Total vocabulary for unified embedding:
        - Chars: 0..num_chars-1
        - Pad char: num_chars
        - Chunks: num_chars+1..num_chars+num_chunks
        - Pad chunk: num_chars+1+num_chunks
        - Roots: num_chars+2+num_chunks..
        """
        return self.num_chars + 1 + self.num_chunks + 1 + self.num_roots

    @property
    def chunk_offset(self):
        """Offset for chunk tokens in unified vocab."""
        return self.num_chars + 1

    @property
    def root_offset(self):
        """Offset for root tokens in unified vocab."""
        return self.num_chars + 1 + self.num_chunks + 1


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute cos/sin frequencies for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to queries/keys."""
    B, n_head, T, head_dim = x.shape
    x_reshape = x.view(B, n_head, T, head_dim // 2, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).view(B, n_head, T, head_dim)
    return out


# ============================================================================
# Transformer Components
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention with RoPE (bidirectional)."""

    def __init__(self, config: FractalModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor = None):
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: FractalModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: FractalModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# Fractal Diffusion Model
# ============================================================================

class FractalDiffusionModel(nn.Module):
    """
    Fractal Diffusion Model with Shared Weights and Level Embeddings.

    Handles both:
    - Level 0: Root -> Chunks (plotting)
    - Level 1: Chunk -> Chars (spelling)

    Uses:
    - Unified token embedding for all token types
    - Level embedding to condition on hierarchy level
    - Energy head for hallucination detection
    """

    def __init__(self, config: FractalModelConfig):
        super().__init__()
        self.config = config

        # Unified token embedding (all levels share embedding space)
        self.tok_emb = nn.Embedding(config.total_vocab_size, config.n_embd)

        # Level embedding: tells model "plotting" vs "spelling"
        self.level_emb = nn.Embedding(config.num_levels, config.n_embd)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        )

        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output heads for each level
        # Level 0: predicts chunks (including pad)
        self.head_level0 = nn.Linear(config.n_embd, config.num_chunks + 1, bias=False)
        # Level 1: predicts chars (including pad)
        self.head_level1 = nn.Linear(config.n_embd, config.num_chars + 1, bias=False)

        # Energy head: shared across levels for hallucination detection
        self.energy_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1)
        )

        # Precompute RoPE frequencies for max sequence length
        max_seq = max(config.expansion_size, config.max_char_len) + 1
        cos, sin = precompute_freqs_cis(config.n_embd // config.n_head, max_seq)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        # Initialize weights
        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.config.n_embd // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_emb(emb)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        level: int,
        attention_mask: torch.Tensor = None,
        return_energy: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a single hierarchy level.

        Args:
            x: (B, seq_len) token ids [condition, target_1, target_2, ...]
            t: (B,) timesteps
            level: 0 for Root->Chunks, 1 for Chunk->Chars
            attention_mask: optional attention mask
            return_energy: if True, also return energy prediction

        Returns:
            logits: (B, seq_len-1, vocab_size) predictions for target positions
            energy: (B,) energy scalar per sample (if return_energy=True)
        """
        B, T = x.shape

        # Token embeddings
        h = self.tok_emb(x)  # (B, T, n_embd)

        # Add level embedding (broadcast to all positions)
        level_tensor = torch.tensor([level], device=x.device)
        level_vec = self.level_emb(level_tensor)  # (1, n_embd)
        h = h + level_vec.unsqueeze(0)  # Broadcast add

        # Add time embedding
        t_emb = self.get_time_embedding(t)  # (B, n_embd)
        h = h + t_emb.unsqueeze(1)  # Broadcast to all positions

        # Transformer blocks
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin, None)

        h = self.ln_f(h)

        # Output logits for target positions (skip condition)
        target_h = h[:, 1:, :]  # (B, T-1, n_embd)

        if level == 0:
            logits = self.head_level0(target_h)  # (B, T-1, num_chunks+1)
        else:
            logits = self.head_level1(target_h)  # (B, T-1, num_chars+1)

        if return_energy:
            # Energy from target positions
            energy_per_pos = self.energy_head(target_h)  # (B, T-1, 1)
            energy = energy_per_pos.mean(dim=(1, 2))  # (B,) scalar
            return logits, energy

        return logits, None


# ============================================================================
# Discrete Diffusion
# ============================================================================

class DiscreteDiffusion:
    """Discrete diffusion with Poisson noise."""

    def __init__(self, config: FractalModelConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.noise_schedule = torch.linspace(0, 0.9, config.num_timesteps)

    def add_noise_level0(self, chunks: torch.Tensor, t: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """Add noise to chunk tokens (Level 0)."""
        B, L = chunks.shape
        noise_probs = self.noise_schedule[t.cpu()].to(device)
        noise_mask = torch.rand(B, L, device=device) < noise_probs.unsqueeze(1)
        random_chunks = torch.randint(0, self.config.num_chunks, (B, L), device=device)
        noisy = torch.where(noise_mask, random_chunks, chunks)
        return noisy

    def add_noise_level1(self, chars: torch.Tensor, t: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
        """Add noise to character tokens (Level 1)."""
        B, L = chars.shape
        noise_probs = self.noise_schedule[t.cpu()].to(device)
        noise_mask = torch.rand(B, L, device=device) < noise_probs.unsqueeze(1)
        random_chars = torch.randint(0, self.config.num_chars, (B, L), device=device)
        noisy = torch.where(noise_mask, random_chars, chars)
        return noisy


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 3e-4
    max_iters: int = 15000
    warmup_iters: int = 500
    eval_interval: int = 500
    eval_samples: int = 500
    save_interval: int = 2000
    grad_clip: float = 1.0
    lambda_energy: float = 1.0
    level0_ratio: float = 0.5  # Ratio of Level 0 in each batch


# ============================================================================
# Training Loop
# ============================================================================

def train(
    model_config: FractalModelConfig,
    train_config: TrainConfig,
    dataset: FractalDataset,
    save_dir: str = "checkpoints"
):
    """Train the Fractal Diffusion Model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                         if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = FractalDiffusionModel(model_config).to(device)
    diffusion = DiscreteDiffusion(model_config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # LR schedule
    def get_lr(it):
        if it < train_config.warmup_iters:
            return train_config.learning_rate * it / train_config.warmup_iters
        return train_config.learning_rate

    # Save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    best_loss = float('inf')
    start_time = time.time()

    print("\n" + "=" * 70)
    print("TRAINING FRACTAL ENGINE (Phase 4)")
    print(f"  Level 0 (Root->Chunks): {train_config.level0_ratio*100:.0f}% of batch")
    print(f"  Level 1 (Chunk->Chars): {(1-train_config.level0_ratio)*100:.0f}% of batch")
    print(f"  lambda_energy = {train_config.lambda_energy}")
    print("=" * 70)

    for iter_num in range(train_config.max_iters):
        model.train()

        # === LEVEL 0: Root -> Chunks ===
        level0_size = int(train_config.batch_size * train_config.level0_ratio)
        l0_cond, l0_targets, l0_lens, l0_wrong = get_level0_batch(
            dataset, level0_size, device
        )

        # Sample timesteps
        t0 = torch.randint(0, model_config.num_timesteps, (level0_size,), device=device)

        # Add noise to correct targets
        noisy_l0 = diffusion.add_noise_level0(l0_targets, t0, device)

        # Build input: [root_id (offset), noisy_chunks (offset)]
        x0_correct = torch.cat([
            (l0_cond + model_config.root_offset).unsqueeze(1),
            noisy_l0 + model_config.chunk_offset
        ], dim=1)

        # Forward with energy
        logits0_correct, energy0_correct = model(x0_correct, t0, level=0, return_energy=True)

        # Diffusion loss (Level 0)
        loss_diff0 = F.cross_entropy(
            logits0_correct.reshape(-1, logits0_correct.shape[-1]),
            l0_targets.reshape(-1),
            ignore_index=model_config.pad_chunk_id - model_config.chunk_offset
        )

        # Wrong pairs for Level 0
        noisy_l0_wrong = diffusion.add_noise_level0(l0_wrong, t0, device)
        x0_wrong = torch.cat([
            (l0_cond + model_config.root_offset).unsqueeze(1),
            noisy_l0_wrong + model_config.chunk_offset
        ], dim=1)
        _, energy0_wrong = model(x0_wrong, t0, level=0, return_energy=True)

        # Energy loss (Level 0)
        loss_energy0 = (
            F.mse_loss(energy0_correct, torch.zeros_like(energy0_correct)) +
            F.mse_loss(energy0_wrong, torch.ones_like(energy0_wrong))
        )

        # === LEVEL 1: Chunk -> Chars ===
        level1_size = train_config.batch_size - level0_size
        l1_cond, l1_targets, l1_lens, l1_wrong = get_level1_batch(
            dataset, level1_size, device
        )

        # Sample timesteps
        t1 = torch.randint(0, model_config.num_timesteps, (level1_size,), device=device)

        # Add noise to correct targets
        noisy_l1 = diffusion.add_noise_level1(l1_targets, t1, device)

        # Build input: [chunk_id (offset), noisy_chars]
        x1_correct = torch.cat([
            (l1_cond + model_config.chunk_offset).unsqueeze(1),
            noisy_l1  # chars are in range 0..num_chars-1, no offset needed
        ], dim=1)

        # Forward with energy
        logits1_correct, energy1_correct = model(x1_correct, t1, level=1, return_energy=True)

        # Diffusion loss (Level 1)
        loss_diff1 = F.cross_entropy(
            logits1_correct.reshape(-1, logits1_correct.shape[-1]),
            l1_targets.reshape(-1),
            ignore_index=model_config.pad_char_id
        )

        # Wrong pairs for Level 1
        noisy_l1_wrong = diffusion.add_noise_level1(l1_wrong, t1, device)
        x1_wrong = torch.cat([
            (l1_cond + model_config.chunk_offset).unsqueeze(1),
            noisy_l1_wrong
        ], dim=1)
        _, energy1_wrong = model(x1_wrong, t1, level=1, return_energy=True)

        # Energy loss (Level 1)
        loss_energy1 = (
            F.mse_loss(energy1_correct, torch.zeros_like(energy1_correct)) +
            F.mse_loss(energy1_wrong, torch.ones_like(energy1_wrong))
        )

        # === COMBINED LOSS ===
        loss_diff = (loss_diff0 + loss_diff1) / 2
        loss_energy = (loss_energy0 + loss_energy1) / 2
        loss = loss_diff + train_config.lambda_energy * loss_energy

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Update LR
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # Logging
        if iter_num % 100 == 0:
            elapsed = time.time() - start_time
            with torch.no_grad():
                det0 = (energy0_wrong > energy0_correct).float().mean().item() * 100
                det1 = (energy1_wrong > energy1_correct).float().mean().item() * 100
            print(f"Iter {iter_num:5d} | Loss: {loss.item():.4f} "
                  f"(D0:{loss_diff0.item():.3f} D1:{loss_diff1.item():.3f} "
                  f"E0:{loss_energy0.item():.3f} E1:{loss_energy1.item():.3f}) | "
                  f"Det: L0={det0:.0f}% L1={det1:.0f}% | {elapsed:.0f}s")

        # Evaluation
        if iter_num % train_config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_det0 = []
                eval_det1 = []

                for _ in range(train_config.eval_samples // train_config.batch_size):
                    # Level 0 eval
                    l0_c, l0_t, l0_l, l0_w = get_level0_batch(dataset, level0_size, device)
                    t0 = torch.randint(0, model_config.num_timesteps, (level0_size,), device=device)
                    noisy_l0 = diffusion.add_noise_level0(l0_t, t0, device)
                    x0_c = torch.cat([
                        (l0_c + model_config.root_offset).unsqueeze(1),
                        noisy_l0 + model_config.chunk_offset
                    ], dim=1)
                    _, e0_c = model(x0_c, t0, level=0, return_energy=True)

                    noisy_l0_w = diffusion.add_noise_level0(l0_w, t0, device)
                    x0_w = torch.cat([
                        (l0_c + model_config.root_offset).unsqueeze(1),
                        noisy_l0_w + model_config.chunk_offset
                    ], dim=1)
                    _, e0_w = model(x0_w, t0, level=0, return_energy=True)
                    eval_det0.append((e0_w > e0_c).float().mean().item())

                    # Level 1 eval
                    l1_c, l1_t, l1_l, l1_w = get_level1_batch(dataset, level1_size, device)
                    t1 = torch.randint(0, model_config.num_timesteps, (level1_size,), device=device)
                    noisy_l1 = diffusion.add_noise_level1(l1_t, t1, device)
                    x1_c = torch.cat([
                        (l1_c + model_config.chunk_offset).unsqueeze(1),
                        noisy_l1
                    ], dim=1)
                    _, e1_c = model(x1_c, t1, level=1, return_energy=True)

                    noisy_l1_w = diffusion.add_noise_level1(l1_w, t1, device)
                    x1_w = torch.cat([
                        (l1_c + model_config.chunk_offset).unsqueeze(1),
                        noisy_l1_w
                    ], dim=1)
                    _, e1_w = model(x1_w, t1, level=1, return_energy=True)
                    eval_det1.append((e1_w > e1_c).float().mean().item())

                avg_det0 = sum(eval_det0) / len(eval_det0) * 100
                avg_det1 = sum(eval_det1) / len(eval_det1) * 100

                print(f"  -> Eval Detection: Level0={avg_det0:.1f}% Level1={avg_det1:.1f}%")

                # Save best
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({
                        'model': model.state_dict(),
                        'config': model_config,
                        'iter': iter_num,
                        'loss': best_loss,
                        'det0': avg_det0,
                        'det1': avg_det1
                    }, save_path / "best_model.pt")
                    print(f"  -> New best! Saved.")

        # Periodic save
        if iter_num > 0 and iter_num % train_config.save_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'config': model_config,
                'iter': iter_num,
                'loss': loss.item()
            }, save_path / f"model_{iter_num}.pt")

    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': model_config,
        'iter': train_config.max_iters,
        'loss': loss.item()
    }, save_path / "final_model.pt")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 70)

    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Check for data
    data_path = Path("data/fractal_hierarchy.pkl")

    if not data_path.exists():
        print("Dataset not found. Building...")

        # Create data directory
        Path("data").mkdir(exist_ok=True)

        # Download Shakespeare if needed
        text_path = Path("data/tinyshakespeare.txt")
        if not text_path.exists():
            print("Downloading tinyshakespeare...")
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, str(text_path))

        config = HierarchicalBPEConfig(
            chunk_vocab_size=1024,
            max_chunk_len=16,
            min_chunk_freq=2,
            root_vocab_size=1024,
            root_expansion_size=4,
            min_root_freq=2
        )
        tokenizer, dataset = build_fractal_dataset(
            str(text_path), config, str(data_path)
        )
        print_fractal_stats(tokenizer, dataset)
    else:
        print(f"Loading dataset from {data_path}...")
        tokenizer, dataset, bpe_config = load_fractal_dataset(str(data_path))
        print(f"  Level 0 samples: {len(dataset.root_ids):,}")
        print(f"  Level 1 samples: {len(dataset.chunk_ids):,}")

    # Model config
    model_config = FractalModelConfig(
        num_chars=dataset.num_chars,
        num_chunks=dataset.num_chunks,
        num_roots=dataset.num_roots,
        pad_char_id=dataset.pad_char_id,
        pad_chunk_id=dataset.pad_chunk_id,
        max_char_len=dataset.max_chunk_len,
        expansion_size=dataset.expansion_size,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        num_timesteps=100
    )

    print(f"\nModel config:")
    print(f"  Num chars: {model_config.num_chars}")
    print(f"  Num chunks: {model_config.num_chunks}")
    print(f"  Num roots: {model_config.num_roots}")
    print(f"  Total vocab: {model_config.total_vocab_size}")

    # Train config
    train_config = TrainConfig(
        batch_size=128,
        learning_rate=3e-4,
        max_iters=15000,
        warmup_iters=500,
        eval_interval=500,
        save_interval=2000,
        lambda_energy=1.0,
        level0_ratio=0.5
    )

    # Train
    model = train(model_config, train_config, dataset)

    print("\nDone!")
