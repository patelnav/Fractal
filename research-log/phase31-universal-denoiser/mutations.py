"""
Mutation Engine for Universal Denoiser.

Applies various corruption types at a given noise level sigma.
Corruption types:
- replace: Random token → random other token
- delete: Remove token, shift left
- insert: Insert random token, shift right
- swap: Adjacent token swap
- mask: Replace with <MASK> token
"""

import random
import torch
from typing import List, Set, Tuple


def corrupt_sequence(
    tokens: torch.Tensor,
    sigma: float,
    vocab_size: int,
    special_tokens: Set[int],
    mask_token_id: int,
    pad_token_id: int,
    max_len: int = None,
) -> torch.Tensor:
    """
    Apply corruption to a token sequence at noise level sigma.

    Args:
        tokens: (T,) tensor of token ids
        sigma: noise level in [0, 1], controls corruption intensity
        vocab_size: size of vocabulary
        special_tokens: set of token ids to protect (BOS, EOS, PAD)
        mask_token_id: id of <MASK> token
        pad_token_id: id of <PAD> token
        max_len: max sequence length (for padding after ops)

    Returns:
        corrupted: (T,) or (max_len,) corrupted token sequence
    """
    tokens = tokens.clone().tolist()
    max_len = max_len or len(tokens)

    # Find non-special, non-pad positions
    editable_positions = [
        i for i, t in enumerate(tokens)
        if t not in special_tokens and t != pad_token_id
    ]

    if len(editable_positions) == 0:
        return torch.tensor(tokens, dtype=torch.long)

    # Number of corruptions based on sigma
    n_corrupt = max(1, int(sigma * len(editable_positions)))
    n_corrupt = min(n_corrupt, len(editable_positions))

    # Sample positions to corrupt
    positions_to_corrupt = random.sample(editable_positions, n_corrupt)

    # Apply corruptions
    # Weight operations: mask is most common (for MLM-style training)
    op_weights = {
        'mask': 0.5,      # 50% masking
        'replace': 0.2,   # 20% random replacement
        'swap': 0.15,     # 15% swaps
        'delete': 0.075,  # 7.5% deletions
        'insert': 0.075,  # 7.5% insertions
    }

    for pos in sorted(positions_to_corrupt, reverse=True):
        op = _sample_operation(op_weights)

        if op == 'mask':
            tokens[pos] = mask_token_id

        elif op == 'replace':
            # Replace with random non-special token
            new_token = _sample_regular_token(vocab_size, special_tokens)
            tokens[pos] = new_token

        elif op == 'swap':
            # Swap with next position if possible
            if pos + 1 < len(tokens) and tokens[pos + 1] not in special_tokens:
                tokens[pos], tokens[pos + 1] = tokens[pos + 1], tokens[pos]

        elif op == 'delete':
            # Delete (only if sequence won't become too short)
            if len([t for t in tokens if t != pad_token_id]) > 5:
                del tokens[pos]
                tokens.append(pad_token_id)  # Maintain length

        elif op == 'insert':
            # Insert random token (only if we have room)
            if len(tokens) < max_len:
                new_token = _sample_regular_token(vocab_size, special_tokens)
                tokens.insert(pos, new_token)
                # Trim if too long
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]

    # Ensure correct length
    if len(tokens) < max_len:
        tokens = tokens + [pad_token_id] * (max_len - len(tokens))
    elif len(tokens) > max_len:
        tokens = tokens[:max_len]

    return torch.tensor(tokens, dtype=torch.long)


def _sample_operation(weights: dict) -> str:
    """Sample an operation based on weights."""
    ops = list(weights.keys())
    probs = list(weights.values())
    return random.choices(ops, weights=probs, k=1)[0]


def _sample_regular_token(vocab_size: int, special_tokens: Set[int]) -> int:
    """Sample a non-special token."""
    regular_tokens = [i for i in range(vocab_size) if i not in special_tokens]
    return random.choice(regular_tokens)


def corrupt_batch(
    batch: torch.Tensor,
    sigma: torch.Tensor,
    vocab_size: int,
    special_tokens: Set[int],
    mask_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Apply corruption to a batch of sequences.

    Args:
        batch: (B, T) tensor of token ids
        sigma: (B,) tensor of noise levels
        vocab_size: size of vocabulary
        special_tokens: set of token ids to protect
        mask_token_id: id of <MASK> token
        pad_token_id: id of <PAD> token

    Returns:
        corrupted: (B, T) corrupted batch
    """
    B, T = batch.shape
    corrupted = []

    for i in range(B):
        corrupted_seq = corrupt_sequence(
            batch[i],
            sigma[i].item(),
            vocab_size,
            special_tokens,
            mask_token_id,
            pad_token_id,
            max_len=T,
        )
        corrupted.append(corrupted_seq)

    return torch.stack(corrupted)


# Simpler masking-only corruption for baseline comparison
def mask_sequence(
    tokens: torch.Tensor,
    sigma: float,
    mask_token_id: int,
    special_tokens: Set[int],
    pad_token_id: int,
) -> torch.Tensor:
    """
    Simple masking corruption (like Phase 30).
    Only replaces tokens with <MASK>, no structural changes.
    """
    tokens = tokens.clone()

    # Find editable positions
    editable_mask = torch.ones_like(tokens, dtype=torch.bool)
    for st in special_tokens:
        editable_mask &= (tokens != st)
    editable_mask &= (tokens != pad_token_id)

    # Random mask based on sigma
    rand = torch.rand_like(tokens, dtype=torch.float)
    mask = (rand < sigma) & editable_mask

    tokens[mask] = mask_token_id
    return tokens


def mask_batch(
    batch: torch.Tensor,
    sigma: torch.Tensor,
    mask_token_id: int,
    special_tokens: Set[int],
    pad_token_id: int,
) -> torch.Tensor:
    """
    Apply simple masking to a batch.

    Args:
        batch: (B, T) tensor
        sigma: (B,) or scalar noise levels

    Returns:
        masked: (B, T) masked batch
    """
    B, T = batch.shape
    device = batch.device

    # Expand sigma if scalar
    if sigma.dim() == 0:
        sigma = sigma.expand(B)

    # Build editable mask
    editable_mask = torch.ones_like(batch, dtype=torch.bool)
    for st in special_tokens:
        editable_mask &= (batch != st)
    editable_mask &= (batch != pad_token_id)

    # Random mask per position
    rand = torch.rand(B, T, device=device)
    sigma_expanded = sigma.unsqueeze(1).expand(B, T)
    mask = (rand < sigma_expanded) & editable_mask

    result = batch.clone()
    result[mask] = mask_token_id
    return result


if __name__ == "__main__":
    # Test corruption
    vocab_size = 19
    special_tokens = {0, 1, 2, 3}  # PAD, MASK, BOS, EOS
    mask_id = 1
    pad_id = 0

    # Example sequence: <BOS> ( + 5 3 ) = 8 <EOS> <PAD> ...
    tokens = torch.tensor([2, 14, 16, 5, 4, 15, 18, 9, 3, 0, 0, 0])

    print("Original:", tokens.tolist())

    for sigma in [0.1, 0.3, 0.5, 0.7, 0.9]:
        corrupted = corrupt_sequence(
            tokens, sigma, vocab_size, special_tokens, mask_id, pad_id, max_len=12
        )
        print(f"Sigma={sigma}: {corrupted.tolist()}")

    # Test batch
    batch = tokens.unsqueeze(0).expand(4, -1).clone()
    sigmas = torch.tensor([0.1, 0.3, 0.5, 0.7])
    corrupted_batch = corrupt_batch(
        batch, sigmas, vocab_size, special_tokens, mask_id, pad_id
    )
    print("\nBatch corruption:")
    for i, (s, c) in enumerate(zip(sigmas, corrupted_batch)):
        print(f"  Sigma={s.item():.1f}: {c.tolist()}")
