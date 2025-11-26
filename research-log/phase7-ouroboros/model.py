"""
Ouroboros Model: Energy-Based Reasoning Verification

A bidirectional transformer with an energy head that learns to distinguish
correct reasoning from incorrect reasoning through contrastive training.

Architecture:
- Bidirectional attention (no causal mask)
- RoPE positional embeddings
- Energy head: MLP that outputs scalar "energy" (0 = correct, 1 = wrong)
- Contrastive loss: MSE(energy_correct, 0) + MSE(energy_wrong, 1)

Based on Phase 4 Fractal Engine, simplified for reasoning verification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class OuroborosConfig:
    """Model configuration."""
    # Architecture
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    # Vocabulary (tiktoken GPT-2)
    vocab_size: int = 50257

    # Sequence length
    max_seq_len: int = 512

    # Diffusion (optional, for future work)
    num_timesteps: int = 100

    # Training
    energy_weight: float = 1.0  # Weight for energy loss


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

    def __init__(self, config: OuroborosConfig):
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

        # Scaled dot-product attention (bidirectional)
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
    """Feed-forward network with GELU activation."""

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: OuroborosConfig):
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
# Ouroboros Model
# ============================================================================

class OuroborosModel(nn.Module):
    """
    Ouroboros: Energy-Based Reasoning Verifier

    Takes context + target tokens and outputs:
    1. Energy scalar: low (0) for correct reasoning, high (1) for wrong reasoning
    2. Optional: token logits for diffusion/generation

    Training:
    - Contrastive loss on (correct, wrong) pairs
    - Energy(correct) -> 0, Energy(wrong) -> 1
    """

    def __init__(self, config: OuroborosConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Energy head: produces scalar "energy" per sequence
        self.energy_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1)
        )

        # Optional: LM head for generation/diffusion
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.max_seq_len
        )
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"OuroborosModel: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        context_lens: torch.Tensor = None,
        target_lens: torch.Tensor = None,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for energy prediction.

        Args:
            context_ids: (B, max_ctx_len) context token ids
            target_ids: (B, max_tgt_len) target token ids
            context_lens: (B,) actual context lengths (optional)
            target_lens: (B,) actual target lengths (optional)
            return_logits: if True, also return LM logits

        Returns:
            energy: (B,) energy scalar per sample
            logits: (B, seq_len, vocab_size) if return_logits=True
        """
        B = context_ids.shape[0]

        # Concatenate context and target
        x = torch.cat([context_ids, target_ids], dim=1)  # (B, ctx_len + tgt_len)
        T = x.shape[1]

        # Token embeddings
        h = self.tok_emb(x)  # (B, T, n_embd)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)

        h = self.ln_f(h)  # (B, T, n_embd)

        # Energy: average over target positions only
        ctx_len = context_ids.shape[1]
        target_h = h[:, ctx_len:, :]  # (B, tgt_len, n_embd)

        # Create mask for actual target positions (ignore padding)
        if target_lens is not None:
            mask = torch.arange(target_ids.shape[1], device=x.device)[None, :] < target_lens[:, None]
            mask = mask.unsqueeze(-1).float()  # (B, tgt_len, 1)
            # Masked mean
            energy_per_pos = self.energy_head(target_h)  # (B, tgt_len, 1)
            energy = (energy_per_pos * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            energy = energy.squeeze(-1)  # (B,)
        else:
            energy_per_pos = self.energy_head(target_h)  # (B, tgt_len, 1)
            energy = energy_per_pos.mean(dim=(1, 2))  # (B,)

        if return_logits:
            logits = self.lm_head(h)  # (B, T, vocab_size)
            return energy, logits

        return energy, None

    def get_energy(self, context_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Convenience method to get just the energy."""
        energy, _ = self.forward(context_ids, target_ids)
        return energy


def compute_contrastive_loss(
    model: OuroborosModel,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    context_lens: torch.Tensor = None,
    target_lens: torch.Tensor = None
) -> Tuple[torch.Tensor, dict]:
    """
    Compute contrastive energy loss.

    The model learns:
    - Energy(correct sample) -> 0
    - Energy(wrong sample) -> 1

    Args:
        model: OuroborosModel
        contexts: (B, ctx_len) context tokens
        targets: (B, tgt_len) target tokens
        labels: (B,) 1 for correct, 0 for wrong
        context_lens: (B,) actual context lengths
        target_lens: (B,) actual target lengths

    Returns:
        loss: scalar contrastive loss
        metrics: dict with detailed metrics
    """
    energy, _ = model(contexts, targets, context_lens, target_lens)

    # Target energy: 0 for correct (label=1), 1 for wrong (label=0)
    target_energy = 1.0 - labels.float()

    # MSE loss
    loss = F.mse_loss(energy, target_energy)

    # Metrics
    with torch.no_grad():
        correct_mask = labels == 1
        wrong_mask = labels == 0

        metrics = {
            'loss': loss.item(),
            'energy_correct': energy[correct_mask].mean().item() if correct_mask.any() else 0.0,
            'energy_wrong': energy[wrong_mask].mean().item() if wrong_mask.any() else 0.0,
            'energy_correct_std': energy[correct_mask].std().item() if correct_mask.sum() > 1 else 0.0,
            'energy_wrong_std': energy[wrong_mask].std().item() if wrong_mask.sum() > 1 else 0.0,
        }

        # Accuracy at threshold 0.5
        predictions = (energy > 0.5).long()
        expected = (labels == 0).long()  # wrong = 1, correct = 0 for predictions
        accuracy = (predictions == expected).float().mean().item()
        metrics['accuracy'] = accuracy

    return loss, metrics


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("OUROBOROS MODEL TEST")
    print("=" * 60)

    config = OuroborosConfig(
        n_layer=6,
        n_head=8,
        n_embd=256,
        vocab_size=50257,
        max_seq_len=512
    )

    model = OuroborosModel(config)

    # Test forward pass
    B = 4
    ctx_len = 64
    tgt_len = 128

    contexts = torch.randint(0, config.vocab_size, (B, ctx_len))
    targets = torch.randint(0, config.vocab_size, (B, tgt_len))
    labels = torch.tensor([1, 0, 1, 0])  # correct, wrong, correct, wrong

    energy, logits = model(contexts, targets, return_logits=True)

    print(f"\nInput shapes: contexts={contexts.shape}, targets={targets.shape}")
    print(f"Output energy shape: {energy.shape}")
    print(f"Energy values: {energy.tolist()}")

    # Test contrastive loss
    loss, metrics = compute_contrastive_loss(model, contexts, targets, labels)
    print(f"\nContrastive loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n--- Testing on CUDA ---")
        model = model.cuda()
        contexts = contexts.cuda()
        targets = targets.cuda()
        labels = labels.cuda()

        energy, _ = model(contexts, targets)
        print(f"CUDA energy: {energy.tolist()}")

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)
