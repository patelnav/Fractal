"""
Fractal Discrete Diffusion: The Smallest Falsifiable Test

Testing the Universal Refinement Hypothesis:
Can a single set of neural weights learn denoising at multiple abstraction levels?

Based on: "Talagrand's Convolution Conjecture via Perturbed Reverse Heat" (Yuansi Chen, 2025)
Architecture inspired by: nanoGPT (Karpathy)

Experiment: 1-to-4 Recursive Expansion
- Level 0 (Root): Integers 0-9
- Level 1 (Chunks): Each root maps to 4 chunk tokens
- Level 2 (Fine): Each chunk maps to 4 fine tokens

Success criteria: >99% accuracy on both levels with SHARED weights.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, List
import random
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FractalConfig:
    # Vocab: 10 roots (0-9) + 10 chunks (A-J = 10-19) + 10 fine (20-29) + specials
    vocab_size: int = 32  # 0-9 roots, 10-19 chunks, 20-29 fine, 30 PAD, 31 NOISE
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 5  # 1 condition + 4 target tokens
    dropout: float = 0.0
    bias: bool = False

    # Diffusion
    num_timesteps: int = 100

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 100


# =============================================================================
# Synthetic Hierarchical Data Generator
# =============================================================================

class FractalDataset:
    """
    Generates the deterministic 3-level hierarchy:
    - Root (0-9) -> 4 Chunks (10-19)
    - Chunk (10-19) -> 4 Fine tokens (20-29)

    Each mapping is deterministic and unique.
    """

    # Token ranges
    ROOT_START = 0
    ROOT_END = 10      # 0-9
    CHUNK_START = 10
    CHUNK_END = 20     # 10-19 (A-J conceptually)
    FINE_START = 20
    FINE_END = 30      # 20-29
    PAD_TOKEN = 30
    NOISE_TOKEN = 31

    def __init__(self, seed: int = 42):
        random.seed(seed)
        torch.manual_seed(seed)

        # Create deterministic mappings
        # Root -> 4 Chunks (each root gets a unique pattern)
        self.root_to_chunks: Dict[int, List[int]] = {}
        chunk_patterns = self._generate_patterns(10, 4, self.CHUNK_START)
        for root in range(10):
            self.root_to_chunks[root] = chunk_patterns[root]

        # Chunk -> 4 Fine tokens (each chunk gets a unique pattern)
        self.chunk_to_fine: Dict[int, List[int]] = {}
        fine_patterns = self._generate_patterns(10, 4, self.FINE_START)
        for i, chunk in enumerate(range(self.CHUNK_START, self.CHUNK_END)):
            self.chunk_to_fine[chunk] = fine_patterns[i]

    def _generate_patterns(self, n_patterns: int, pattern_len: int, offset: int) -> List[List[int]]:
        """Generate n unique patterns of length pattern_len."""
        patterns = []
        for i in range(n_patterns):
            # Create deterministic but varied patterns
            # Use a mix of the available tokens (0-9 in the range)
            pattern = []
            for j in range(pattern_len):
                # Deterministic formula that creates variety
                token = offset + ((i * 3 + j * 7 + i * j) % 10)
                pattern.append(token)
            patterns.append(pattern)
        return patterns

    def get_level0_to_level1_sample(self) -> Tuple[int, List[int]]:
        """Sample: Root -> 4 Chunks"""
        root = random.randint(0, 9)
        chunks = self.root_to_chunks[root]
        return root, chunks

    def get_level1_to_level2_sample(self) -> Tuple[int, List[int]]:
        """Sample: Chunk -> 4 Fine tokens"""
        chunk = random.randint(self.CHUNK_START, self.CHUNK_END - 1)
        fine = self.chunk_to_fine[chunk]
        return chunk, fine

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a mixed batch of Level 0->1 and Level 1->2 samples.

        Returns:
            conditions: (B,) tensor of condition tokens
            targets: (B, 4) tensor of target token sequences
        """
        conditions = []
        targets = []

        for _ in range(batch_size):
            if random.random() < 0.5:
                cond, tgt = self.get_level0_to_level1_sample()
            else:
                cond, tgt = self.get_level1_to_level2_sample()
            conditions.append(cond)
            targets.append(tgt)

        return (
            torch.tensor(conditions, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device)
        )

    def get_full_expansion(self, root: int) -> List[int]:
        """Get the full 16-token expansion of a root (for testing)."""
        chunks = self.root_to_chunks[root]
        fine_tokens = []
        for chunk in chunks:
            fine_tokens.extend(self.chunk_to_fine[chunk])
        return fine_tokens

    def print_mappings(self):
        """Print all mappings for debugging."""
        print("=== Root -> Chunks Mappings ===")
        for root, chunks in self.root_to_chunks.items():
            print(f"  Root {root} -> {chunks}")

        print("\n=== Chunk -> Fine Mappings ===")
        for chunk, fine in self.chunk_to_fine.items():
            print(f"  Chunk {chunk} -> {fine}")


# =============================================================================
# Bidirectional Self-Attention (No Causal Mask)
# =============================================================================

class BidirectionalSelfAttention(nn.Module):
    """
    Standard self-attention WITHOUT causal masking.
    All positions can attend to all other positions.
    """

    def __init__(self, config: FractalConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Bidirectional attention: is_causal=False
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False  # <-- KEY DIFFERENCE from nanoGPT
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: FractalConfig):
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
    def __init__(self, config: FractalConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# Time Embedding MLP
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimeEmbeddingMLP(nn.Module):
    """MLP to project timestep into embedding space."""

    def __init__(self, config: FractalConfig):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        t_emb = self.sinusoidal(t.float())
        return self.mlp(t_emb)


# =============================================================================
# Fractal Diffusion Model
# =============================================================================

class FractalDiffusionModel(nn.Module):
    """
    Bidirectional Transformer for Discrete Diffusion.

    Input: [condition_token, noisy_target_1, noisy_target_2, noisy_target_3, noisy_target_4]
    Plus: timestep t

    Output: Predicted clean tokens for positions 1-4
    """

    def __init__(self, config: FractalConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Time embedding
        self.time_mlp = TimeEmbeddingMLP(config)

        # Transformer blocks
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
        print(f"Fractal Diffusion Model: {n_params/1e3:.1f}K parameters")

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
        device = x.device

        # Token embeddings
        tok_emb = self.wte(x)  # (B, T, n_embd)

        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.wpe(pos)  # (T, n_embd)

        # Time embeddings - broadcast to all positions
        time_emb = self.time_mlp(t)  # (B, n_embd)
        time_emb = time_emb.unsqueeze(1)  # (B, 1, n_embd)

        # Combine embeddings
        x = tok_emb + pos_emb + time_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Only predict for target positions (indices 1-4)
        logits = self.lm_head(x[:, 1:, :])  # (B, 4, vocab_size)

        return logits


# =============================================================================
# Discrete Diffusion: Poisson Bit-Flip Noise
# =============================================================================

class DiscreteDiffusion:
    """
    Discrete diffusion via bit-flip (Poisson) noise on tokens.

    At timestep t, each token independently has probability p(t) of being
    replaced with a random token from its valid range.
    """

    def __init__(self, config: FractalConfig, dataset: FractalDataset):
        self.num_timesteps = config.num_timesteps
        self.dataset = dataset

        # Noise schedule: linear from 0 to ~0.9
        # At t=0: no noise, at t=T: almost completely random
        self.noise_schedule = torch.linspace(0, 0.9, config.num_timesteps)

    def add_noise(
        self,
        x: torch.Tensor,           # (B, 4) clean target tokens
        t: torch.Tensor,           # (B,) timesteps
        device: torch.device
    ) -> torch.Tensor:
        """
        Add noise by randomly replacing tokens.

        For each position, with probability noise_schedule[t]:
        - Replace with a random token from the same level's vocabulary
        """
        B, L = x.size()

        # Get noise probabilities for each sample
        probs = self.noise_schedule[t.cpu()].to(device)  # (B,)
        probs = probs.unsqueeze(1).expand(B, L)    # (B, L)

        # Determine which positions to corrupt
        mask = torch.rand(B, L, device=device) < probs

        # Determine what level these tokens are from (for valid replacement range)
        # Check if any token is in chunk range (10-19) or fine range (20-29)
        is_chunk_level = (x >= self.dataset.CHUNK_START) & (x < self.dataset.CHUNK_END)
        is_fine_level = (x >= self.dataset.FINE_START) & (x < self.dataset.FINE_END)

        # Generate random replacements
        # For chunks: random from 10-19
        # For fine: random from 20-29
        random_chunks = torch.randint(
            self.dataset.CHUNK_START,
            self.dataset.CHUNK_END,
            (B, L),
            device=device
        )
        random_fine = torch.randint(
            self.dataset.FINE_START,
            self.dataset.FINE_END,
            (B, L),
            device=device
        )

        # Select appropriate random tokens based on level
        random_tokens = torch.where(is_chunk_level, random_chunks, random_fine)

        # Apply noise
        noisy_x = torch.where(mask, random_tokens, x)

        return noisy_x

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


# =============================================================================
# Training Loop
# =============================================================================

def train(config: FractalConfig, device: torch.device):
    """Train the fractal diffusion model."""

    print("=" * 60)
    print("FRACTAL DIFFUSION EXPERIMENT")
    print("Testing the Universal Refinement Hypothesis")
    print("=" * 60)

    # Setup
    dataset = FractalDataset()
    dataset.print_mappings()

    model = FractalDiffusionModel(config).to(device)
    diffusion = DiscreteDiffusion(config, dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\nTraining for {config.max_iters} iterations...")
    print(f"Device: {device}")
    print("-" * 60)

    model.train()
    start_time = time.time()

    pbar = tqdm(range(config.max_iters), desc="Training", unit="iter")
    for iter_num in pbar:
        # Get batch
        conditions, targets = dataset.get_batch(config.batch_size, device)

        # Sample timesteps and add noise
        t = diffusion.sample_timesteps(config.batch_size, device)
        noisy_targets = diffusion.add_noise(targets, t, device)

        # Build input sequence: [condition, noisy_target_1, ..., noisy_target_4]
        x = torch.cat([conditions.unsqueeze(1), noisy_targets], dim=1)  # (B, 5)

        # Forward pass
        logits = model(x, t)  # (B, 4, vocab_size)

        # Compute cross-entropy loss against clean targets
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Logging
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            # Evaluate accuracy
            model.eval()
            with torch.no_grad():
                acc_l1, acc_l2 = evaluate_accuracy(model, diffusion, dataset, device, n_samples=50)
            model.train()

            elapsed = time.time() - start_time
            iters_per_sec = (iter_num + 1) / elapsed if elapsed > 0 else 0
            remaining = (config.max_iters - iter_num - 1) / iters_per_sec if iters_per_sec > 0 else 0

            tqdm.write(f"Iter {iter_num:5d} | Loss: {loss.item():.4f} | "
                       f"L0->L1: {acc_l1:.1%} | L1->L2: {acc_l2:.1%} | "
                       f"ETA: {remaining:.0f}s")

    print("-" * 60)
    print("Training complete!")
    return model, dataset, diffusion


# =============================================================================
# Evaluation and Generation
# =============================================================================

@torch.no_grad()
def evaluate_accuracy(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    dataset: FractalDataset,
    device: torch.device,
    n_samples: int = 100
) -> Tuple[float, float]:
    """
    Evaluate accuracy on both levels separately.
    Uses single-step denoising from t=T-1 for simplicity.
    """
    model.eval()

    correct_l1 = 0
    total_l1 = 0
    correct_l2 = 0
    total_l2 = 0

    for _ in range(n_samples):
        # Level 0 -> 1 test
        root = random.randint(0, 9)
        true_chunks = dataset.root_to_chunks[root]
        pred_chunks = generate_single(model, diffusion, root, device, dataset)

        correct_l1 += sum(p == t for p, t in zip(pred_chunks, true_chunks))
        total_l1 += 4

        # Level 1 -> 2 test
        chunk = random.randint(dataset.CHUNK_START, dataset.CHUNK_END - 1)
        true_fine = dataset.chunk_to_fine[chunk]
        pred_fine = generate_single(model, diffusion, chunk, device, dataset)

        correct_l2 += sum(p == t for p, t in zip(pred_fine, true_fine))
        total_l2 += 4

    return correct_l1 / total_l1, correct_l2 / total_l2


@torch.no_grad()
def generate_single(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    condition: int,
    device: torch.device,
    dataset: FractalDataset,
    num_steps: int = 20
) -> List[int]:
    """
    Generate 4 tokens given a condition using reverse diffusion.

    Starts from random noise and iteratively denoises.
    """
    model.eval()

    # Determine the output token range based on condition
    if condition < dataset.CHUNK_START:
        # Condition is a root -> output chunks
        token_range = (dataset.CHUNK_START, dataset.CHUNK_END)
    else:
        # Condition is a chunk -> output fine tokens
        token_range = (dataset.FINE_START, dataset.FINE_END)

    # Start with random tokens
    x = torch.randint(token_range[0], token_range[1], (1, 4), device=device)
    condition_tensor = torch.tensor([[condition]], device=device)

    # Reverse diffusion: denoise from t=T-1 down to t=0
    timesteps = list(range(diffusion.num_timesteps - 1, -1, -diffusion.num_timesteps // num_steps))
    if 0 not in timesteps:
        timesteps.append(0)

    for t_val in timesteps:
        t = torch.tensor([t_val], device=device)

        # Build input
        inp = torch.cat([condition_tensor, x], dim=1)  # (1, 5)

        # Get predictions
        logits = model(inp, t)  # (1, 4, vocab_size)

        # Sample from predictions (or take argmax)
        probs = F.softmax(logits, dim=-1)

        # For cleaner generation, use argmax at low t, sample at high t
        if t_val < diffusion.num_timesteps // 4:
            x = logits.argmax(dim=-1)
        else:
            x = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, 4)

    return x[0].tolist()


@torch.no_grad()
def run_recursive_test(
    model: FractalDiffusionModel,
    diffusion: DiscreteDiffusion,
    dataset: FractalDataset,
    device: torch.device
):
    """
    The Ultimate Test: Recursive expansion from Root to Fine tokens.

    1. Root(k) -> Generate 4 Chunks
    2. For each generated Chunk -> Generate 4 Fine tokens
    3. Compare to ground truth
    """
    print("\n" + "=" * 60)
    print("RECURSIVE GENERATION TEST (The Falsification Check)")
    print("=" * 60)

    total_correct = 0
    total_tokens = 0

    for root in range(10):
        print(f"\nRoot {root}:")

        # Ground truth
        true_chunks = dataset.root_to_chunks[root]
        true_fine = dataset.get_full_expansion(root)

        # Generate Level 1
        pred_chunks = generate_single(model, diffusion, root, device, dataset)
        chunks_correct = sum(p == t for p, t in zip(pred_chunks, true_chunks))

        print(f"  L1 True:  {true_chunks}")
        print(f"  L1 Pred:  {pred_chunks}  ({chunks_correct}/4 correct)")

        # Generate Level 2 from PREDICTED chunks (recursive!)
        pred_fine = []
        for chunk in pred_chunks:
            fine = generate_single(model, diffusion, chunk, device, dataset)
            pred_fine.extend(fine)

        # Compare to what we WOULD get from true chunks
        expected_fine = []
        for chunk in true_chunks:
            expected_fine.extend(dataset.chunk_to_fine[chunk])

        fine_correct = sum(p == t for p, t in zip(pred_fine, expected_fine))

        print(f"  L2 Expected: {expected_fine}")
        print(f"  L2 Pred:     {pred_fine}  ({fine_correct}/16 correct)")

        total_correct += chunks_correct + fine_correct
        total_tokens += 20

    overall_acc = total_correct / total_tokens
    print("\n" + "=" * 60)
    print(f"OVERALL RECURSIVE ACCURACY: {overall_acc:.1%} ({total_correct}/{total_tokens})")
    print("=" * 60)

    if overall_acc > 0.99:
        print("\n*** HYPOTHESIS VIABLE: Universal Refinement works! ***")
        print("Proceed to build the Fractal Language Model.")
    elif overall_acc > 0.8:
        print("\n*** PARTIAL SUCCESS: Shows promise but needs refinement ***")
    else:
        print("\n*** HYPOTHESIS FALSIFIED: Gradient conflict detected ***")
        print("Consider Mixture of Experts for different scales.")

    return overall_acc


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

    # Create config
    config = FractalConfig()

    # Train the model
    model, dataset, diffusion = train(config, device)

    # Run the recursive test
    run_recursive_test(model, diffusion, dataset, device)
