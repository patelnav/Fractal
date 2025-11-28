"""
Universal Denoiser: Recurrent Bidirectional Transformer for unified generation/repair/editing.

Key differences from Phase 30:
- Recurrent: Single block applied K times (trade compute for quality)
- Noise embedding: Explicit sigma input for noise level
- Stage 1: No energy head (added in Stage 2)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BidirectionalSelfAttention(nn.Module):
    """Bidirectional attention - every token attends to every other token."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Bidirectional: no masking
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Single transformer block - can be applied recurrently."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NoiseEmbedding(nn.Module):
    """
    Embed continuous noise level sigma into n_embd dimensions.
    Uses sinusoidal encoding (like diffusion timesteps) + learned projection.
    """

    def __init__(self, n_embd, max_period=10000):
        super().__init__()
        self.n_embd = n_embd
        self.max_period = max_period
        # Project sinusoidal encoding to n_embd
        self.proj = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, sigma):
        """
        sigma: (B,) tensor of noise levels in [0, 1]
        Returns: (B, n_embd) noise embeddings
        """
        # Sinusoidal encoding
        half = self.n_embd // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=sigma.device) / half
        )
        # sigma is in [0, 1], scale to [0, 1000] for better frequency spread
        args = sigma[:, None] * 1000 * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.n_embd % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class UniversalDenoiser(nn.Module):
    """
    Recurrent Bidirectional Transformer for unified denoising.

    - Token embedding + Position embedding + Noise embedding
    - Single block applied K times (recurrent depth)
    - Output head predicts all positions
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # Position
        self.wne = NoiseEmbedding(config.n_embd)  # Noise level

        self.drop = nn.Dropout(config.dropout)

        # Recurrent blocks (n_layer blocks, each can be iterated)
        # For true recurrence: n_layer=1, K_iter>1
        # For hybrid: n_layer=2, K_iter=2
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, sigma, targets=None, K_iter=1):
        """
        idx: (B, T) token indices (possibly corrupted/masked)
        sigma: (B,) noise levels in [0, 1]
        targets: (B, T) ground truth tokens (optional, for loss)
        K_iter: number of times to iterate through blocks (recurrence)

        Returns: logits, loss (if targets provided)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        noise_emb = self.wne(sigma)  # (B, n_embd)

        # Combine: token + position + noise (broadcast noise to all positions)
        x = tok_emb + pos_emb + noise_emb.unsqueeze(1)
        x = self.drop(x)

        # Recurrent application of blocks
        for _ in range(K_iter):
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore PAD tokens (set to -1 in data)
            )

        return logits, loss

    def get_hidden(self, idx, sigma, K_iter=1):
        """Get hidden states (for future energy head)."""
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        noise_emb = self.wne(sigma)

        x = tok_emb + pos_emb + noise_emb.unsqueeze(1)
        x = self.drop(x)

        for _ in range(K_iter):
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        return x


class Config:
    """Model configuration."""
    n_layer = 2          # Number of blocks (for recurrence, use 1-2)
    n_head = 4           # Attention heads
    n_embd = 128         # Embedding dimension
    dropout = 0.0        # Dropout rate
    block_size = 128     # Max sequence length
    bias = False         # Use bias in linear layers
    vocab_size = 19      # Will be set by dataset


if __name__ == "__main__":
    # Quick test
    config = Config()
    config.vocab_size = 19
    model = UniversalDenoiser(config)

    # Test forward pass
    B, T = 4, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    sigma = torch.rand(B)
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, sigma, targets, K_iter=2)
    print(f"Input shape: {idx.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
