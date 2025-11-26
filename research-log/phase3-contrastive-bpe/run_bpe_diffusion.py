"""
BPE Decompression via Discrete Diffusion with Contrastive Energy

Phase 3: Contrastive energy training for hallucination detection
- Input: Single BPE token
- Output: Character sequence (variable length, padded)
- Model: Bidirectional transformer with RoPE + Energy Head
- Diffusion: Discrete Poisson bit-flip noise
- Training: Dual loss (diffusion + contrastive energy)

Key insight: Instead of computing expensive Chen Score at inference,
train an Energy Head to predict energy directly using contrastive pairs.
- Correct pairs (BPE token -> its chars): target energy = 0
- Wrong pairs (BPE token -> random other chars): target energy = 1

Based on Chen's 2025 paper on diffusion in Boolean hypercubes.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from bpe_tokenizer import (
    load_decompression_dataset,
    build_decompression_dataset,
    BPEConfig,
    MinimalBPE,
    DecompressionDataset
)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    # Architecture
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1

    # Vocabulary (set from dataset)
    char_vocab_size: int = 65  # Characters
    bpe_vocab_size: int = 512  # BPE tokens
    pad_id: int = 65  # Padding token

    # Sequence
    max_seq_len: int = 17  # 1 condition + 16 max chars

    # Diffusion
    num_timesteps: int = 100

    @property
    def total_vocab_size(self):
        """Total vocab: chars + pad + BPE tokens."""
        return self.char_vocab_size + 1 + self.bpe_vocab_size  # +1 for pad


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
    # x: (B, n_head, T, head_dim)
    B, n_head, T, head_dim = x.shape

    # Reshape for rotation
    x_reshape = x.view(B, n_head, T, head_dim // 2, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]

    # Get cos/sin for this sequence length
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).view(B, n_head, T, head_dim)
    return out


# ============================================================================
# Transformer Components
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention with RoPE (bidirectional)."""

    def __init__(self, config: ModelConfig):
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

        # Compute Q, K, V
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Apply mask if provided (for padding)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ModelConfig):
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

    def __init__(self, config: ModelConfig):
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
# Diffusion Model
# ============================================================================

class DecompressionDiffusion(nn.Module):
    """
    Decompression model: BPE token -> character sequence.

    Input: [bpe_token, char1, char2, ..., charN, pad, pad, ...]
    Output: logits for positions 1..max_len (character predictions)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (shared across all token types)
        self.tok_emb = nn.Embedding(config.total_vocab_size, config.n_embd)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection (to character vocab only, not BPE)
        self.head = nn.Linear(config.n_embd, config.char_vocab_size + 1, bias=False)

        # Energy head: predicts scalar energy for hallucination detection
        # Trained contrastively: correct pairs -> 0, wrong pairs -> 1
        self.energy_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1)
        )

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.max_seq_len
        )
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

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                attention_mask: torch.Tensor = None,
                return_energy: bool = False):
        """
        Forward pass.

        Args:
            x: (B, seq_len) token ids [bpe_id, char1, char2, ...]
            t: (B,) timesteps
            attention_mask: (B, seq_len) - unused, kept for API compatibility
            return_energy: If True, also return energy prediction

        Returns:
            If return_energy=False:
                logits: (B, seq_len-1, char_vocab+1) predictions for char positions
            If return_energy=True:
                (logits, energy): logits + (B,) energy scalar per sample
        """
        B, T = x.shape

        # Token embeddings
        h = self.tok_emb(x)  # (B, T, n_embd)

        # Add time embedding
        t_emb = self.get_time_embedding(t)  # (B, n_embd)
        h = h + t_emb.unsqueeze(1)  # Broadcast to all positions

        # Transformer blocks (no attention mask - attend to all positions)
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin, None)

        h = self.ln_f(h)

        # Output logits for character positions (skip BPE condition)
        logits = self.head(h[:, 1:, :])  # (B, T-1, char_vocab+1)

        if return_energy:
            # Energy head: average over sequence positions, squeeze to scalar
            # Uses hidden states from character positions (skip BPE token)
            energy_per_pos = self.energy_head(h[:, 1:, :])  # (B, T-1, 1)
            energy = energy_per_pos.mean(dim=(1, 2))  # (B,) scalar per sample
            return logits, energy

        return logits

    def get_scores(self, x: torch.Tensor, t: torch.Tensor,
                   attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Get centered scores for energy calculation."""
        logits = self.forward(x, t, attention_mask)
        scores = logits - logits.mean(dim=-1, keepdim=True)
        return scores


# ============================================================================
# Discrete Diffusion
# ============================================================================

class DiscreteDiffusion:
    """Discrete diffusion with Poisson noise on character tokens."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps

        # Linear noise schedule
        self.noise_schedule = torch.linspace(0, 0.9, config.num_timesteps)

    def add_noise(self, chars: torch.Tensor, t: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
        """
        Add noise to character tokens.

        Args:
            chars: (B, L) character token ids
            t: (B,) timesteps

        Returns:
            noisy_chars: (B, L) noised character ids
        """
        B, L = chars.shape

        # Get noise probability for each sample
        noise_probs = self.noise_schedule[t.cpu()].to(device)  # (B,)

        # Create noise mask
        noise_mask = torch.rand(B, L, device=device) < noise_probs.unsqueeze(1)

        # Generate random character replacements (within char vocab only)
        random_chars = torch.randint(
            0, self.config.char_vocab_size,
            (B, L), device=device
        )

        # Apply noise
        noisy_chars = torch.where(noise_mask, random_chars, chars)

        return noisy_chars


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 3e-4
    max_iters: int = 10000
    warmup_iters: int = 500
    eval_interval: int = 500
    eval_samples: int = 1000
    save_interval: int = 2000
    grad_clip: float = 1.0
    # Contrastive energy training
    lambda_energy: float = 1.0  # Weight for energy loss


def get_batch(dataset: DecompressionDataset, batch_size: int,
              device: torch.device):
    """Get a random batch from dataset."""
    idx = torch.randint(len(dataset), (batch_size,))

    bpe_ids = dataset.bpe_token_ids[idx].to(device)  # (B,)
    char_seqs = dataset.char_sequences[idx].to(device)  # (B, max_len)
    seq_lens = dataset.sequence_lengths[idx].to(device)  # (B,)

    # Create attention mask (1 for real tokens, 0 for padding)
    # Shape: (B, 1 + max_len) for [bpe_token, chars...]
    max_len = char_seqs.shape[1]
    positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    char_mask = positions < seq_lens.unsqueeze(1)  # (B, max_len)

    # Add 1 for the BPE condition token
    attention_mask = torch.cat([
        torch.ones(batch_size, 1, device=device, dtype=torch.bool),
        char_mask
    ], dim=1)  # (B, 1 + max_len)

    return bpe_ids, char_seqs, seq_lens, attention_mask


def get_contrastive_batch(dataset: DecompressionDataset, batch_size: int,
                          device: torch.device):
    """
    Get a batch with both correct and wrong pairings for contrastive training.

    Returns:
        bpe_ids: (B,) BPE token ids
        correct_chars: (B, max_len) correct character sequences
        wrong_chars: (B, max_len) wrong character sequences (from different tokens)
        seq_lens: (B,) actual sequence lengths
        attention_mask: (B, 1 + max_len) attention mask
    """
    # Sample two sets of indices to create mismatched pairs
    idx_a = torch.randint(len(dataset), (batch_size,))
    idx_b = torch.randint(len(dataset), (batch_size,))

    # Ensure idx_b is different from idx_a for wrong pairs
    # Re-sample where they're the same
    same_mask = idx_a == idx_b
    while same_mask.any():
        idx_b[same_mask] = torch.randint(len(dataset), (same_mask.sum(),))
        same_mask = idx_a == idx_b

    bpe_ids = dataset.bpe_token_ids[idx_a].to(device)  # (B,)
    correct_chars = dataset.char_sequences[idx_a].to(device)  # (B, max_len)
    wrong_chars = dataset.char_sequences[idx_b].to(device)  # (B, max_len) - from different tokens
    seq_lens = dataset.sequence_lengths[idx_a].to(device)  # (B,)

    # Create attention mask
    max_len = correct_chars.shape[1]
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    char_mask = positions < seq_lens.unsqueeze(1)

    attention_mask = torch.cat([
        torch.ones(batch_size, 1, device=device, dtype=torch.bool),
        char_mask
    ], dim=1)

    return bpe_ids, correct_chars, wrong_chars, seq_lens, attention_mask


def train(model_config: ModelConfig, train_config: TrainConfig,
          dataset: DecompressionDataset, save_dir: str = "checkpoints"):
    """Train the decompression model with contrastive energy loss."""

    device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                         if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = DecompressionDiffusion(model_config).to(device)
    diffusion = DiscreteDiffusion(model_config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate schedule
    def get_lr(it):
        if it < train_config.warmup_iters:
            return train_config.learning_rate * it / train_config.warmup_iters
        return train_config.learning_rate

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # BPE offset for building input sequences
    # BPE tokens come after chars and pad in the unified vocab
    bpe_offset = model_config.char_vocab_size + 1

    best_loss = float('inf')
    start_time = time.time()

    print("\n" + "=" * 60)
    print("TRAINING START (Phase 3: Contrastive Energy)")
    print(f"  lambda_energy = {train_config.lambda_energy}")
    print("=" * 60)

    for iter_num in range(train_config.max_iters):
        model.train()

        # Get contrastive batch: correct and wrong pairings
        bpe_ids, correct_chars, wrong_chars, seq_lens, attention_mask = get_contrastive_batch(
            dataset, train_config.batch_size, device
        )

        # Sample random timesteps
        t = torch.randint(0, model_config.num_timesteps,
                         (train_config.batch_size,), device=device)

        # --- CORRECT PAIRS: BPE token -> its correct characters ---
        noisy_correct = diffusion.add_noise(correct_chars, t, device)
        x_correct = torch.cat([
            (bpe_ids + bpe_offset).unsqueeze(1),
            noisy_correct
        ], dim=1)

        # Forward pass with energy for correct pairs
        logits_correct, energy_correct = model(x_correct, t, attention_mask, return_energy=True)

        # Diffusion loss: only on correct pairs
        loss_diff = F.cross_entropy(
            logits_correct.reshape(-1, logits_correct.shape[-1]),
            correct_chars.reshape(-1),
            ignore_index=model_config.pad_id
        )

        # --- WRONG PAIRS: BPE token -> wrong characters ---
        noisy_wrong = diffusion.add_noise(wrong_chars, t, device)
        x_wrong = torch.cat([
            (bpe_ids + bpe_offset).unsqueeze(1),
            noisy_wrong
        ], dim=1)

        # Forward pass with energy for wrong pairs
        _, energy_wrong = model(x_wrong, t, attention_mask, return_energy=True)

        # --- ENERGY LOSS: correct -> 0, wrong -> 1 ---
        target_correct = torch.zeros_like(energy_correct)
        target_wrong = torch.ones_like(energy_wrong)

        loss_energy = (
            F.mse_loss(energy_correct, target_correct) +
            F.mse_loss(energy_wrong, target_wrong)
        )

        # Total loss
        loss = loss_diff + train_config.lambda_energy * loss_energy

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # Logging
        if iter_num % 100 == 0:
            elapsed = time.time() - start_time
            # Compute energy detection accuracy for this batch
            with torch.no_grad():
                energy_detection = (energy_wrong > energy_correct).float().mean().item() * 100
            print(f"Iter {iter_num:5d} | Loss: {loss.item():.4f} "
                  f"(Diff: {loss_diff.item():.4f}, Energy: {loss_energy.item():.4f}) | "
                  f"Det: {energy_detection:.0f}% | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # Evaluation
        if iter_num % train_config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                eval_diff_losses = []
                eval_energy_losses = []
                eval_detections = []

                for _ in range(train_config.eval_samples // train_config.batch_size):
                    # Use contrastive batch for eval too
                    bpe_ids, correct_chars, wrong_chars, seq_lens, attention_mask = get_contrastive_batch(
                        dataset, train_config.batch_size, device
                    )
                    t = torch.randint(0, model_config.num_timesteps,
                                     (train_config.batch_size,), device=device)

                    # Correct pairs
                    noisy_correct = diffusion.add_noise(correct_chars, t, device)
                    x_correct = torch.cat([
                        (bpe_ids + bpe_offset).unsqueeze(1),
                        noisy_correct
                    ], dim=1)
                    logits_correct, energy_correct = model(x_correct, t, attention_mask, return_energy=True)

                    # Wrong pairs
                    noisy_wrong = diffusion.add_noise(wrong_chars, t, device)
                    x_wrong = torch.cat([
                        (bpe_ids + bpe_offset).unsqueeze(1),
                        noisy_wrong
                    ], dim=1)
                    _, energy_wrong = model(x_wrong, t, attention_mask, return_energy=True)

                    # Diffusion loss
                    loss_diff_eval = F.cross_entropy(
                        logits_correct.reshape(-1, logits_correct.shape[-1]),
                        correct_chars.reshape(-1),
                        ignore_index=model_config.pad_id
                    )
                    eval_diff_losses.append(loss_diff_eval.item())

                    # Energy loss
                    loss_energy_eval = (
                        F.mse_loss(energy_correct, torch.zeros_like(energy_correct)) +
                        F.mse_loss(energy_wrong, torch.ones_like(energy_wrong))
                    )
                    eval_energy_losses.append(loss_energy_eval.item())

                    # Detection rate
                    detection = (energy_wrong > energy_correct).float().mean().item()
                    eval_detections.append(detection)

                avg_diff_loss = sum(eval_diff_losses) / len(eval_diff_losses)
                avg_energy_loss = sum(eval_energy_losses) / len(eval_energy_losses)
                avg_detection = sum(eval_detections) / len(eval_detections) * 100
                avg_eval_loss = avg_diff_loss + train_config.lambda_energy * avg_energy_loss

                print(f"  -> Eval: Diff={avg_diff_loss:.4f}, Energy={avg_energy_loss:.4f}, "
                      f"Detection={avg_detection:.1f}%")

                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    torch.save({
                        'model': model.state_dict(),
                        'config': model_config,
                        'iter': iter_num,
                        'loss': best_loss,
                        'detection_rate': avg_detection
                    }, save_path / "best_model.pt")
                    print(f"  -> New best! Saved to {save_path}/best_model.pt")

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

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Best eval loss: {best_loss:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 60)

    return model


# ============================================================================
# Generation
# ============================================================================

def generate_decompression(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    bpe_token_id: int,
    seq_len: int,
    device: torch.device,
    temperature: float = 1.0,
    num_steps: int = None  # If None, use full num_timesteps
) -> torch.Tensor:
    """
    Generate character sequence from BPE token via reverse diffusion.

    Args:
        model: Trained model
        diffusion: Diffusion process
        bpe_token_id: BPE token to decompress
        seq_len: Expected output length
        device: Device to use
        temperature: Sampling temperature
        num_steps: Number of diffusion steps (default: all timesteps)

    Returns:
        chars: (seq_len,) generated character ids
    """
    model.eval()
    config = model.config

    bpe_offset = config.char_vocab_size + 1

    # Use strided timesteps for faster generation
    total_T = config.num_timesteps
    if num_steps is None:
        num_steps = total_T
    step_size = max(1, total_T // num_steps)
    timesteps = list(range(total_T - 1, -1, -step_size))

    with torch.no_grad():
        # Start with random characters
        chars = torch.randint(0, config.char_vocab_size, (1, config.max_seq_len - 1),
                             device=device)

        # Create attention mask (all ones for generation)
        attention_mask = torch.ones(1, config.max_seq_len, device=device)

        # Reverse diffusion with strided timesteps
        for t_val in timesteps:
            t = torch.tensor([t_val], device=device)

            # Build input
            x = torch.cat([
                torch.tensor([[bpe_token_id + bpe_offset]], device=device),
                chars
            ], dim=1)

            # Get predictions
            logits = model(x, t, attention_mask)  # (1, max_len-1, vocab)

            # Sample or argmax based on timestep
            if t_val > total_T // 4:
                # Sample with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                chars = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
                chars = chars.view(1, -1)
            else:
                # Argmax for low noise
                chars = logits.argmax(dim=-1)

        # Return only the relevant length
        return chars[0, :seq_len]


# ============================================================================
# Energy Calculation (for hallucination detection)
# ============================================================================

def compute_generation_energy(
    model: DecompressionDiffusion,
    diffusion: DiscreteDiffusion,
    bpe_token_id: int,
    target_chars: torch.Tensor,
    seq_len: int,
    device: torch.device,
    num_integration_steps: int = 50
) -> float:
    """
    Compute generation energy for a (bpe_token, char_sequence) pair.

    Energy = integral of ||score||^2 over diffusion time

    This implements Chen's Lemma 7 energy bound.
    """
    model.eval()
    config = model.config
    bpe_offset = config.char_vocab_size + 1

    # Ensure target is right shape
    if target_chars.dim() == 1:
        target_chars = target_chars.unsqueeze(0)

    # Pad target if needed
    if target_chars.shape[1] < config.max_seq_len - 1:
        pad_len = config.max_seq_len - 1 - target_chars.shape[1]
        target_chars = F.pad(target_chars, (0, pad_len), value=config.pad_id)

    total_energy = 0.0
    dt = 1.0 / num_integration_steps

    with torch.no_grad():
        for step in range(num_integration_steps):
            t_val = int(step * config.num_timesteps / num_integration_steps)
            t = torch.tensor([t_val], device=device)

            # Add noise at this timestep
            noisy_chars = diffusion.add_noise(target_chars, t, device)

            # Build input
            x = torch.cat([
                torch.tensor([[bpe_token_id + bpe_offset]], device=device),
                noisy_chars
            ], dim=1)

            # Create mask for actual sequence length
            attention_mask = torch.ones(1, config.max_seq_len, device=device)

            # Get scores
            scores = model.get_scores(x, t, attention_mask)

            # Only compute energy on real positions (not padding)
            scores_real = scores[0, :seq_len, :]

            # Energy = sum of squared scores
            energy_t = (scores_real ** 2).sum().item()
            total_energy += energy_t * dt

    return total_energy


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for data
    data_path = Path("data/bpe_decompression.pkl")
    if not data_path.exists():
        print("Dataset not found. Building...")
        from bpe_tokenizer import build_decompression_dataset, BPEConfig, print_dataset_stats

        bpe_config = BPEConfig(vocab_size=512, max_token_len=16)
        tokenizer, dataset = build_decompression_dataset(
            text_path="data/tinyshakespeare.txt",
            bpe_config=bpe_config,
            save_path=str(data_path)
        )
        print_dataset_stats(tokenizer, dataset)
    else:
        print(f"Loading dataset from {data_path}...")
        tokenizer, dataset, bpe_config = load_decompression_dataset(str(data_path))
        print(f"  Loaded {len(dataset):,} samples")

    # Model config
    model_config = ModelConfig(
        char_vocab_size=dataset.num_chars,
        bpe_vocab_size=dataset.num_bpe_tokens,
        pad_id=dataset.pad_id,
        max_seq_len=dataset.max_len + 1,  # +1 for BPE condition
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
        num_timesteps=100
    )

    print(f"\nModel config:")
    print(f"  Char vocab: {model_config.char_vocab_size}")
    print(f"  BPE vocab: {model_config.bpe_vocab_size}")
    print(f"  Total vocab: {model_config.total_vocab_size}")
    print(f"  Max seq len: {model_config.max_seq_len}")

    # Train config
    train_config = TrainConfig(
        batch_size=128,
        learning_rate=3e-4,
        max_iters=10000,
        warmup_iters=500,
        eval_interval=500,
        save_interval=2000
    )

    # Train
    model = train(model_config, train_config, dataset)

    print("\nDone!")
