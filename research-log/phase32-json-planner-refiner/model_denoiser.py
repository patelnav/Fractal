"""
JSON Denoiser Model - adapted from Phase 31 Universal Denoiser.

Key adaptations for JSON:
1. JSON-specific vocabulary size
2. Structural token awareness (optional positional biases)
3. Energy head for validity scoring

Architecture:
- Bidirectional transformer (BERT-style attention)
- Noise level embedding (continuous sigma)
- Recurrent blocks for iterative refinement
- Energy head for scoring repair quality
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class JSONDenoiserConfig:
    """Configuration for JSON Denoiser model."""
    vocab_size: int = 105       # From JSONTokenizer
    n_layer: int = 4            # Number of transformer blocks
    n_head: int = 4             # Number of attention heads
    n_embd: int = 256           # Embedding dimension
    dropout: float = 0.1        # Dropout rate
    block_size: int = 256       # Maximum sequence length
    bias: bool = False          # Use bias in linear layers
    use_self_cond: bool = False # Self-conditioning (feed previous predictions back)


class BidirectionalSelfAttention(nn.Module):
    """
    Bidirectional self-attention - every token attends to every other token.

    Unlike causal attention in GPT, this allows the model to see the full
    context in both directions, which is essential for repair/denoising.
    """

    def __init__(self, config: JSONDenoiserConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Bidirectional attention (no mask)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: JSONDenoiserConfig):
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
    """Single transformer block with pre-norm residual connections."""

    def __init__(self, config: JSONDenoiserConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NoiseEmbedding(nn.Module):
    """
    Embed continuous noise level sigma into n_embd dimensions.

    Uses sinusoidal encoding (like diffusion timesteps) + learned projection.
    This tells the model how much noise/corruption to expect.
    """

    def __init__(self, n_embd: int, max_period: int = 10000):
        super().__init__()
        self.n_embd = n_embd
        self.max_period = max_period

        self.proj = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: (B,) noise levels in [0, 1]

        Returns:
            (B, n_embd) noise embeddings
        """
        half = self.n_embd // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=sigma.device) / half
        )
        # Scale sigma for better frequency spread
        args = sigma[:, None] * 1000 * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.n_embd % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class JSONDenoiser(nn.Module):
    """
    Bidirectional Transformer for JSON denoising/repair.

    Given a corrupted JSON token sequence and noise level sigma,
    predicts the clean token at each position.

    Input: (B, T) corrupted token IDs + (B,) sigma
    Output: (B, T, vocab_size) logits for clean tokens
    """

    def __init__(self, config: JSONDenoiserConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # Noise embedding
        self.wne = NoiseEmbedding(config.n_embd)

        # Self-conditioning: embedding for previous predictions
        self.use_self_cond = config.use_self_cond
        if self.use_self_cond:
            self.wte_prev = nn.Embedding(config.vocab_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head
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

    def forward(
        self,
        idx: torch.Tensor,
        sigma: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        prev_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: (B, T) corrupted token indices
            sigma: (B,) noise levels in [0, 1]
            targets: (B, T) ground truth clean tokens (optional, for loss)
            prev_logits: (B, T, vocab_size) previous iteration's logits (for self-conditioning)

        Returns:
            logits: (B, T, vocab_size) predictions
            loss: scalar cross-entropy loss (if targets provided)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # Positions
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        noise_emb = self.wne(sigma)  # (B, n_embd)

        # Combine embeddings: token + position + noise (broadcast noise to all positions)
        x = tok_emb + pos_emb + noise_emb.unsqueeze(1)

        # Self-conditioning
        if prev_logits is not None and self.use_self_cond:
            prev_pred = prev_logits.argmax(dim=-1)
            prev_emb = self.wte_prev(prev_pred)
            x = x + 0.5 * prev_emb

        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0  # Ignore PAD tokens (id=0)
            )

        return logits, loss

    def get_hidden(
        self,
        idx: torch.Tensor,
        sigma: torch.Tensor,
        prev_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get hidden states (for energy head).

        Args:
            idx: (B, T) token indices
            sigma: (B,) noise levels
            prev_logits: Previous iteration's logits

        Returns:
            hidden: (B, T, n_embd) hidden states
        """
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        noise_emb = self.wne(sigma)

        x = tok_emb + pos_emb + noise_emb.unsqueeze(1)

        if prev_logits is not None and self.use_self_cond:
            prev_pred = prev_logits.argmax(dim=-1)
            prev_emb = self.wte_prev(prev_pred)
            x = x + 0.5 * prev_emb

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x


class EnergyHead(nn.Module):
    """
    Energy head for validity scoring.

    Takes pooled hidden states and outputs a scalar energy score.
    Low energy = valid/clean JSON, High energy = invalid/corrupted.

    Used for:
    1. Ranking multiple repair candidates
    2. Rejecting obviously bad repairs
    3. Guiding beam search
    """

    def __init__(self, n_embd: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or n_embd // 2

        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (B, T, n_embd) hidden states
            mask: (B, T) boolean mask of valid (non-pad) positions

        Returns:
            energy: (B,) energy scores
        """
        # Pool over sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        energy = self.net(pooled).squeeze(-1)
        return energy


class JSONDenoiserWithEnergy(nn.Module):
    """
    JSON Denoiser with Energy Head for repair quality scoring.

    Two-stage training:
    - Stage 1: Train denoiser on (corrupted, clean) pairs
    - Stage 2: Freeze denoiser, train energy head on (valid, invalid) pairs
    """

    def __init__(
        self,
        config: JSONDenoiserConfig,
        pretrained_denoiser: Optional[JSONDenoiser] = None,
    ):
        super().__init__()
        self.config = config

        if pretrained_denoiser is not None:
            self.denoiser = pretrained_denoiser
        else:
            self.denoiser = JSONDenoiser(config)

        self.energy_head = EnergyHead(config.n_embd)

    def freeze_denoiser(self):
        """Freeze denoiser for energy head training."""
        for param in self.denoiser.parameters():
            param.requires_grad = False

    def unfreeze_denoiser(self):
        """Unfreeze for joint fine-tuning."""
        for param in self.denoiser.parameters():
            param.requires_grad = True

    def forward(
        self,
        idx: torch.Tensor,
        sigma: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_energy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional energy computation.

        Args:
            idx: (B, T) input tokens
            sigma: (B,) noise levels
            targets: (B, T) target tokens
            return_energy: Whether to compute energy

        Returns:
            logits: (B, T, vocab_size)
            loss: reconstruction loss
            energy: (B,) energy scores (if return_energy=True)
        """
        # Get hidden states
        hidden = self.denoiser.get_hidden(idx, sigma)

        # Logits
        logits = self.denoiser.lm_head(hidden)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )

        # Energy
        energy = None
        if return_energy:
            mask = idx != 0  # Non-PAD mask
            energy = self.energy_head(hidden, mask)

        return logits, loss, energy

    def compute_energy(
        self,
        idx: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy score only."""
        hidden = self.denoiser.get_hidden(idx, sigma)
        mask = idx != 0
        return self.energy_head(hidden, mask)

    def denoise(
        self,
        idx: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Get denoised predictions (logits)."""
        logits, _ = self.denoiser(idx, sigma)
        return logits


def test_model():
    """Test the JSON denoiser model."""
    from tokenizer_json import JSONTokenizer

    tokenizer = JSONTokenizer()

    config = JSONDenoiserConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        block_size=128,
    )

    print(f"=== JSONDenoiser Test ===")
    print(f"Config: vocab_size={config.vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")

    model = JSONDenoiser(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test forward pass
    B, T = 4, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    sigma = torch.rand(B) * 0.5  # Random noise levels
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, sigma, targets)
    print(f"\nForward pass:")
    print(f"  Input shape: {idx.shape}")
    print(f"  Sigma shape: {sigma.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test with real JSON
    json_text = '{"name": "Alice", "age": 30}'
    token_ids = tokenizer.tokenize(json_text)
    print(f"\nReal JSON test:")
    print(f"  JSON: {json_text}")
    print(f"  Tokens: {token_ids[:20]}...")

    # Pad to batch
    if len(token_ids) < T:
        token_ids = token_ids + [0] * (T - len(token_ids))
    idx = torch.tensor([token_ids[:T]], dtype=torch.long)
    sigma = torch.tensor([0.2])

    logits, _ = model(idx, sigma)
    pred = logits.argmax(dim=-1)
    decoded = tokenizer.detokenize(pred[0].tolist())
    print(f"  Predicted: {decoded[:50]}...")

    # Test with energy head
    print(f"\n=== JSONDenoiserWithEnergy Test ===")
    model_with_energy = JSONDenoiserWithEnergy(config)

    logits, loss, energy = model_with_energy(
        torch.randint(0, config.vocab_size, (B, T)),
        torch.rand(B) * 0.5,
        targets=torch.randint(0, config.vocab_size, (B, T)),
        return_energy=True,
    )
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Energy shape: {energy.shape}")
    print(f"  Energy values: {energy.tolist()}")


if __name__ == "__main__":
    test_model()
