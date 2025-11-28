"""
Inference modes for Universal Denoiser.

Three unified modes:
1. generate(): Generation from full mask
2. repair(): Fix corrupted input
3. edit(): Local editing with anchored positions
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from model import UniversalDenoiser


def generate(
    model: UniversalDenoiser,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    K_steps: int = 5,
    temperature: float = 1.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Generate a sequence from scratch (full mask → complete).

    Args:
        model: Trained UniversalDenoiser
        length: Target sequence length (including BOS/EOS)
        mask_token_id: ID of <MASK> token
        pad_token_id: ID of <PAD> token
        bos_token_id: ID of <BOS> token
        eos_token_id: ID of <EOS> token
        K_steps: Number of refinement steps
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on

    Returns:
        Generated sequence (length,)
    """
    model.eval()

    with torch.no_grad():
        # Initialize with BOS + MASKs + EOS
        x = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
        x[0, 0] = bos_token_id
        x[0, -1] = eos_token_id

        # Iterative refinement: start with high sigma, decrease
        for k in range(K_steps):
            # Sigma decreases from 1.0 to ~0.1
            sigma = torch.tensor([1.0 - k * 0.8 / K_steps], device=device)

            logits, _ = model(x, sigma, K_iter=2)

            # Sample or argmax
            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)

            # Keep BOS/EOS anchored
            pred[0, 0] = bos_token_id
            pred[0, -1] = eos_token_id

            x = pred

    return x.squeeze(0)


def generate_batch(
    model: UniversalDenoiser,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    K_steps: int = 5,
    temperature: float = 1.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """Generate a batch of sequences."""
    model.eval()

    with torch.no_grad():
        x = torch.full((batch_size, length), mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = bos_token_id
        x[:, -1] = eos_token_id

        for k in range(K_steps):
            sigma = torch.full((batch_size,), 1.0 - k * 0.8 / K_steps, device=device)

            logits, _ = model(x, sigma, K_iter=2)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)

            pred[:, 0] = bos_token_id
            pred[:, -1] = eos_token_id
            x = pred

    return x


def repair(
    model: UniversalDenoiser,
    corrupted: torch.Tensor,
    pad_token_id: int,
    K_steps: int = 3,
    sigma_estimate: float = 0.3,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Repair a corrupted sequence.

    Args:
        model: Trained UniversalDenoiser
        corrupted: (T,) or (B, T) corrupted token sequence
        pad_token_id: ID of <PAD> token
        K_steps: Number of refinement steps
        sigma_estimate: Estimated noise level of input
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on

    Returns:
        Repaired sequence (same shape as input)
    """
    model.eval()

    # Handle single sequence
    if corrupted.dim() == 1:
        corrupted = corrupted.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    corrupted = corrupted.to(device)
    B, T = corrupted.shape

    with torch.no_grad():
        x = corrupted.clone()

        for k in range(K_steps):
            # Sigma stays roughly constant for repair (we assume known noise level)
            sigma = torch.full((B,), sigma_estimate, device=device)

            logits, _ = model(x, sigma, K_iter=2)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)

            # Keep PAD positions as PAD
            pad_mask = (corrupted == pad_token_id)
            pred[pad_mask] = pad_token_id

            x = pred

    if squeeze:
        return x.squeeze(0)
    return x


def edit(
    model: UniversalDenoiser,
    sequence: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    K_steps: int = 2,
    sigma: float = 0.5,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Edit specific positions while keeping others anchored.

    Args:
        model: Trained UniversalDenoiser
        sequence: (T,) or (B, T) token sequence
        mask_positions: (T,) or (B, T) boolean mask of positions to edit
        mask_token_id: ID of <MASK> token
        pad_token_id: ID of <PAD> token
        K_steps: Number of refinement steps
        sigma: Noise level for editing
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on

    Returns:
        Edited sequence (same shape as input)
    """
    model.eval()

    # Handle single sequence
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)
        mask_positions = mask_positions.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    sequence = sequence.to(device)
    mask_positions = mask_positions.to(device)
    B, T = sequence.shape

    with torch.no_grad():
        # Mask the positions to edit
        x = sequence.clone()
        x[mask_positions] = mask_token_id

        # Store anchors (non-masked, non-pad positions)
        anchor_mask = ~mask_positions & (sequence != pad_token_id)

        for k in range(K_steps):
            sigma_t = torch.full((B,), sigma, device=device)

            logits, _ = model(x, sigma_t, K_iter=2)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)

            # CRITICAL: Re-anchor non-masked positions
            pred[anchor_mask] = sequence[anchor_mask]

            # Keep PAD as PAD
            pad_mask = (sequence == pad_token_id)
            pred[pad_mask] = pad_token_id

            x = pred

    if squeeze:
        return x.squeeze(0)
    return x


def iterative_refinement(
    model: UniversalDenoiser,
    x: torch.Tensor,
    sigma_schedule: List[float],
    anchor_mask: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    General iterative refinement with custom sigma schedule.

    Args:
        model: Trained model
        x: Initial sequence (possibly masked/corrupted)
        sigma_schedule: List of sigma values for each step
        anchor_mask: Optional (B, T) boolean mask of positions to keep fixed
        pad_token_id: ID of PAD token
        temperature: Sampling temperature
        device: Device

    Returns:
        final: Final refined sequence
        trajectory: List of intermediate sequences
    """
    model.eval()

    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    x = x.to(device)
    B, T = x.shape

    trajectory = [x.clone()]

    with torch.no_grad():
        for sigma_val in sigma_schedule:
            sigma = torch.full((B,), sigma_val, device=device)

            logits, _ = model(x, sigma, K_iter=2)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, -1)

            # Apply anchors if provided
            if anchor_mask is not None:
                anchor = anchor_mask.to(device)
                pred[anchor] = x[anchor]

            # Keep PAD
            pred[x == pad_token_id] = pad_token_id

            x = pred
            trajectory.append(x.clone())

    if squeeze:
        return x.squeeze(0), [t.squeeze(0) for t in trajectory]
    return x, trajectory


if __name__ == "__main__":
    # Quick test
    from model import Config

    config = Config()
    config.vocab_size = 19
    model = UniversalDenoiser(config)

    # Token IDs
    PAD, MASK, BOS, EOS = 0, 1, 2, 3

    # Test generate
    print("Testing generate()...")
    gen = generate(model, length=20, mask_token_id=MASK, pad_token_id=PAD,
                   bos_token_id=BOS, eos_token_id=EOS, K_steps=3)
    print(f"Generated: {gen.tolist()}")

    # Test repair
    print("\nTesting repair()...")
    corrupted = torch.tensor([BOS, 14, 16, 5, 1, 15, 18, 9, EOS, PAD, PAD])  # Some masked
    repaired = repair(model, corrupted, pad_token_id=PAD, K_steps=3)
    print(f"Corrupted: {corrupted.tolist()}")
    print(f"Repaired:  {repaired.tolist()}")

    # Test edit
    print("\nTesting edit()...")
    sequence = torch.tensor([BOS, 14, 16, 5, 4, 15, 18, 9, EOS, PAD, PAD])
    mask_pos = torch.tensor([False, False, False, True, True, False, False, False, False, False, False])
    edited = edit(model, sequence, mask_pos, mask_token_id=MASK, pad_token_id=PAD, K_steps=2)
    print(f"Original: {sequence.tolist()}")
    print(f"Mask:     {mask_pos.tolist()}")
    print(f"Edited:   {edited.tolist()}")
