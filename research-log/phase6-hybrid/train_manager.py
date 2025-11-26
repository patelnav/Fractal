"""
train_manager.py
Phase 6: The Manager Model

A tiny autoregressive GPT that learns to generate sequences of Root IDs.
This is the "plot generator" - it dreams up abstract sequences that the
Fractal Engine then renders into text.

Architecture:
- Input: Sequence of root IDs from Shakespeare
- Output: Next root ID prediction
- Model: ~1M param GPT

The Manager operates on "abstractions" (root tokens) while the Fractal Engine
handles the "details" (chunks -> chars). Together they form a complete
language model with explicit hierarchical structure.
"""

import math
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add phase4 to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4-fractal-engine"))

from hierarchical_bpe import HierarchicalBPE, load_fractal_dataset


# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

@dataclass
class ManagerConfig:
    """Configuration for the Manager GPT."""
    # Model architecture
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # Vocabulary (set from tokenizer)
    vocab_size: int = 2048  # Root vocabulary size

    # Training
    block_size: int = 64  # Context window for root sequences
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100


# Block size for sequence generation
BLOCK_SIZE = 64


# ============================================================================
# Model Components
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ManagerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ManagerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: ManagerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# Manager GPT
# ============================================================================

class ManagerGPT(nn.Module):
    """
    The Manager: A tiny GPT that generates sequences of root IDs.

    This model learns the "plot structure" of Shakespeare - which roots
    (high-level concepts like "KING:", "Enter", "thou") follow each other.
    """

    def __init__(self, config: ManagerConfig = None):
        super().__init__()

        if config is None:
            config = ManagerConfig()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        # Initialize
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Manager GPT: {n_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of root token indices

        Returns:
            logits: (B, T, vocab_size) next-token predictions
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence too long: {T} > {self.config.block_size}"

        # Token + position embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generate new root tokens autoregressively.

        Args:
            idx: (B, T) conditioning sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k tokens

        Returns:
            (B, T + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ============================================================================
# Data Loading
# ============================================================================

def load_root_sequence(tokenizer: HierarchicalBPE, text_path: str) -> List[int]:
    """Encode the full text as a sequence of root IDs."""
    text = Path(text_path).read_text()
    root_ids = tokenizer.encode_to_roots(text)
    return root_ids


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch of sequences."""
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])

    return x.to(device), y.to(device)


# ============================================================================
# Training
# ============================================================================

def train_manager(config: ManagerConfig = None):
    """Train the Manager GPT on root sequences."""

    if config is None:
        config = ManagerConfig()

    print(f"Using device: {DEVICE}")

    # Load tokenizer from Phase 4
    phase4_data = Path(__file__).parent.parent / "phase4-fractal-engine/data/fractal_hierarchy.pkl"
    if not phase4_data.exists():
        raise FileNotFoundError(f"Phase 4 data not found at {phase4_data}. Run Phase 4 first.")

    print(f"Loading tokenizer from {phase4_data}...")
    tokenizer, dataset, bpe_config = load_fractal_dataset(str(phase4_data))

    # Update config with actual vocabulary size
    config.vocab_size = dataset.num_roots
    print(f"  Root vocabulary size: {config.vocab_size}")

    # Load text and encode to roots
    text_path = Path(__file__).parent.parent / "phase4-fractal-engine/data/tinyshakespeare.txt"
    print(f"Encoding text to root IDs...")
    root_ids = load_root_sequence(tokenizer, str(text_path))
    print(f"  {len(root_ids):,} root tokens")

    # Train/val split
    data = torch.tensor(root_ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,}")

    # Create model
    model = ManagerGPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING MANAGER GPT")
    print("=" * 60)

    best_val_loss = float('inf')
    start_time = time.time()

    for iter_num in range(config.max_iters):
        model.train()

        # Get batch
        x, y = get_batch(train_data, config.block_size, config.batch_size, DEVICE)

        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if iter_num % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iter_num:5d} | Loss: {loss.item():.4f} | {elapsed:.0f}s")

        # Evaluation
        if iter_num % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(config.eval_iters):
                    x, y = get_batch(val_data, config.block_size, config.batch_size, DEVICE)
                    logits = model(x)
                    val_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    val_losses.append(val_loss.item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"  -> Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = Path(__file__).parent / "manager.pt"
                    torch.save({
                        'model': model.state_dict(),
                        'config': config,
                        'iter': iter_num,
                        'val_loss': best_val_loss
                    }, save_path)
                    print(f"  -> New best! Saved to {save_path}")

                # Generate sample
                if iter_num > 0:
                    print("\n  Sample generation:")
                    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                    generated = model.generate(start_idx, max_new_tokens=10, temperature=0.8)
                    root_seq = generated[0].tolist()

                    # Decode to text
                    text_parts = []
                    for rid in root_seq[:10]:
                        if rid < len(tokenizer.root_vocab):
                            text_parts.append(tokenizer.decode_root(rid))
                    print(f"    Roots: {root_seq[:10]}")
                    print(f"    Text:  \"{''.join(text_parts)[:80]}...\"")
                    print()

    # Final save
    save_path = Path(__file__).parent / "manager_final.pt"
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'iter': config.max_iters,
        'val_loss': best_val_loss
    }, save_path)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 60)

    return model, tokenizer


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = ManagerConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        block_size=64,
        batch_size=64,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=500,
        eval_iters=100
    )

    model, tokenizer = train_manager(config)

    # Generate a sample
    print("\n" + "=" * 60)
    print("GENERATION DEMO")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        generated = model.generate(start_idx, max_new_tokens=20, temperature=0.8)
        root_seq = generated[0].tolist()

        print(f"\nGenerated root sequence: {root_seq}")

        # Decode
        text_parts = []
        for rid in root_seq:
            if rid < len(tokenizer.root_vocab):
                text_parts.append(tokenizer.decode_root(rid))

        print(f"\nDecoded text:")
        print("".join(text_parts))
