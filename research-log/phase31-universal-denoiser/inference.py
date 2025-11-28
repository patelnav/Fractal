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


def generate_maskgit(
    model: UniversalDenoiser,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',  # 'linear', 'cosine'
    device: str = 'cpu',
) -> torch.Tensor:
    """
    MaskGIT-style progressive unmasking generation.

    Instead of refining all positions at once, we:
    1. Start from full MASK
    2. At each step, unmask only the highest-confidence positions
    3. Keep unmasked positions as anchors for subsequent steps

    This creates structural anchors progressively, avoiding the
    "no bootstrap" problem of naive parallel generation.

    Args:
        model: Trained UniversalDenoiser
        length: Target sequence length (including BOS/EOS)
        mask_token_id: ID of <MASK> token
        pad_token_id: ID of <PAD> token
        bos_token_id: ID of <BOS> token
        eos_token_id: ID of <EOS> token
        num_steps: Number of unmasking steps
        temperature: Sampling temperature (0 = argmax)
        schedule: Unmasking schedule ('linear' or 'cosine')
        device: Device to run on

    Returns:
        Generated sequence (length,)
    """
    import math
    model.eval()

    with torch.no_grad():
        # Initialize: BOS + MASKs + EOS
        x = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
        x[0, 0] = bos_token_id
        x[0, -1] = eos_token_id

        # Track which positions are still masked (exclude BOS/EOS)
        # Positions 1 to length-2 are maskable
        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        num_maskable = maskable_positions.sum().item()

        # Current mask state: True = still masked
        is_masked = maskable_positions.clone()

        for step in range(num_steps):
            # Compute how many to unmask this step
            if schedule == 'linear':
                # Linear: unmask equal fraction each step
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:  # cosine
                # Cosine: more unmasking early, less later
                # ratio = 1 - cos(pi * (step+1) / (2 * num_steps))
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            current_unmasked = (~is_masked & maskable_positions).sum().item()
            to_unmask = max(1, target_unmasked - current_unmasked)

            # Only proceed if there are masked positions left
            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break

            to_unmask = min(to_unmask, num_currently_masked)

            # Get model predictions
            # Sigma decreases as we unmask (less noise = more confident context)
            sigma = torch.tensor([1.0 - step / num_steps], device=device)
            logits, _ = model(x, sigma, K_iter=2)

            # Compute confidence for each position (max probability)
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)  # (length,)

            # Only consider currently masked positions
            confidence[~is_masked] = -float('inf')

            # Select top-k highest confidence masked positions
            _, top_indices = confidence.topk(to_unmask)

            # Get predictions for those positions
            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                sampled = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
                predictions = sampled

            # Unmask selected positions
            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final pass: fill any remaining masked positions
        if is_masked.any():
            sigma = torch.tensor([0.1], device=device)
            logits, _ = model(x, sigma, K_iter=2)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1).squeeze(0)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def generate_maskgit_batch(
    model: UniversalDenoiser,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """Batched MaskGIT-style generation."""
    import math
    model.eval()

    with torch.no_grad():
        x = torch.full((batch_size, length), mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = bos_token_id
        x[:, -1] = eos_token_id

        # Track masks per sample
        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        num_maskable = maskable_positions.sum().item()

        is_masked = maskable_positions.unsqueeze(0).expand(batch_size, -1).clone()

        for step in range(num_steps):
            if schedule == 'linear':
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:  # cosine
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            sigma = torch.full((batch_size,), 1.0 - step / num_steps, device=device)
            logits, _ = model(x, sigma, K_iter=2)
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values  # (B, T)

            # Process each sample
            for b in range(batch_size):
                current_unmasked = (~is_masked[b] & maskable_positions).sum().item()
                to_unmask = max(1, target_unmasked - current_unmasked)
                num_currently_masked = is_masked[b].sum().item()
                if num_currently_masked == 0:
                    continue
                to_unmask = min(to_unmask, num_currently_masked)

                conf_b = confidence[b].clone()
                conf_b[~is_masked[b]] = -float('inf')
                _, top_indices = conf_b.topk(to_unmask)

                if temperature <= 0:
                    pred_b = logits[b].argmax(dim=-1)
                else:
                    pred_b = torch.multinomial(probs[b], 1).squeeze(-1)

                x[b, top_indices] = pred_b[top_indices]
                is_masked[b, top_indices] = False

        # Final pass
        remaining = is_masked.any(dim=1)
        if remaining.any():
            sigma = torch.full((batch_size,), 0.1, device=device)
            logits, _ = model(x, sigma, K_iter=2)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.stack([torch.multinomial(probs[b], 1).squeeze(-1) for b in range(batch_size)])
            for b in range(batch_size):
                if is_masked[b].any():
                    x[b, is_masked[b]] = final_pred[b, is_masked[b]]

    return x


def generate_maskgit_anchored(
    model: UniversalDenoiser,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    eq_token_id: int,
    lparen_token_id: int,
    rparen_token_id: int,
    plus_token_id: int,
    mult_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',
    anchor_equals: bool = True,
    priority_structural: bool = True,
    structural_boost: float = 0.5,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    MaskGIT-style generation with anchor seeding and priority unmasking.

    Improvements over vanilla MaskGIT:
    1. Anchor seeding: Fix BOS, EOS, and optionally '=' before generation starts
    2. Priority unmasking: Boost confidence of structural tokens (parens, operators)
       so skeleton forms before digits

    Args:
        model: Trained UniversalDenoiser
        length: Target sequence length
        mask_token_id, pad_token_id, bos_token_id, eos_token_id: Token IDs
        eq_token_id: ID of '=' token (for anchor seeding)
        lparen_token_id, rparen_token_id: Parenthesis token IDs
        plus_token_id, mult_token_id: Operator token IDs
        num_steps: Number of unmasking steps
        temperature: Sampling temperature
        schedule: 'linear' or 'cosine'
        anchor_equals: If True, place '=' at fixed position before generation
        priority_structural: If True, boost structural token confidence
        structural_boost: Amount to boost structural token confidence
        device: Device to run on

    Returns:
        Generated sequence (length,)
    """
    import math
    model.eval()

    structural_tokens = {lparen_token_id, rparen_token_id, plus_token_id, mult_token_id, eq_token_id}

    with torch.no_grad():
        # Initialize: BOS + MASKs + EOS
        x = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
        x[0, 0] = bos_token_id
        x[0, -1] = eos_token_id

        # Anchor seeding: place '=' at approximately 2/3 position
        if anchor_equals:
            eq_pos = (length * 2) // 3
            x[0, eq_pos] = eq_token_id

        # Track which positions are maskable (exclude BOS, EOS, and anchored '=')
        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        if anchor_equals:
            maskable_positions[eq_pos] = False

        num_maskable = maskable_positions.sum().item()
        is_masked = maskable_positions.clone()

        for step in range(num_steps):
            # Compute how many to unmask this step
            if schedule == 'linear':
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:  # cosine
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            current_unmasked = (~is_masked & maskable_positions).sum().item()
            to_unmask = max(1, target_unmasked - current_unmasked)

            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break

            to_unmask = min(to_unmask, num_currently_masked)

            # Get model predictions
            sigma = torch.tensor([1.0 - step / num_steps], device=device)
            logits, _ = model(x, sigma, K_iter=2)

            # Compute confidence for each position
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)  # (length,)

            # Priority unmasking: boost structural tokens
            if priority_structural:
                pred_tokens = logits.argmax(dim=-1).squeeze(0)  # (length,)
                is_structural = torch.zeros(length, dtype=torch.bool, device=device)
                for st in structural_tokens:
                    is_structural |= (pred_tokens == st)
                confidence = confidence + structural_boost * is_structural.float()

            # Only consider currently masked positions
            confidence[~is_masked] = -float('inf')

            # Select top-k highest confidence masked positions
            _, top_indices = confidence.topk(to_unmask)

            # Get predictions for those positions
            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                sampled = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
                predictions = sampled

            # Unmask selected positions
            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final pass: fill any remaining masked positions
        if is_masked.any():
            sigma = torch.tensor([0.1], device=device)
            logits, _ = model(x, sigma, K_iter=2)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1).squeeze(0)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def generate_maskgit_anchored_batch(
    model: UniversalDenoiser,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    eq_token_id: int,
    lparen_token_id: int,
    rparen_token_id: int,
    plus_token_id: int,
    mult_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',
    anchor_equals: bool = True,
    priority_structural: bool = True,
    structural_boost: float = 0.5,
    device: str = 'cpu',
) -> torch.Tensor:
    """Batched version of generate_maskgit_anchored."""
    import math
    model.eval()

    structural_tokens = {lparen_token_id, rparen_token_id, plus_token_id, mult_token_id, eq_token_id}

    with torch.no_grad():
        x = torch.full((batch_size, length), mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = bos_token_id
        x[:, -1] = eos_token_id

        eq_pos = (length * 2) // 3
        if anchor_equals:
            x[:, eq_pos] = eq_token_id

        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        if anchor_equals:
            maskable_positions[eq_pos] = False

        num_maskable = maskable_positions.sum().item()
        is_masked = maskable_positions.unsqueeze(0).expand(batch_size, -1).clone()

        for step in range(num_steps):
            if schedule == 'linear':
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            sigma = torch.full((batch_size,), 1.0 - step / num_steps, device=device)
            logits, _ = model(x, sigma, K_iter=2)
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values  # (B, T)

            # Priority unmasking
            if priority_structural:
                pred_tokens = logits.argmax(dim=-1)  # (B, T)
                is_structural = torch.zeros_like(pred_tokens, dtype=torch.bool)
                for st in structural_tokens:
                    is_structural |= (pred_tokens == st)
                confidence = confidence + structural_boost * is_structural.float()

            for b in range(batch_size):
                current_unmasked = (~is_masked[b] & maskable_positions).sum().item()
                to_unmask = max(1, target_unmasked - current_unmasked)
                num_currently_masked = is_masked[b].sum().item()
                if num_currently_masked == 0:
                    continue
                to_unmask = min(to_unmask, num_currently_masked)

                conf_b = confidence[b].clone()
                conf_b[~is_masked[b]] = -float('inf')
                _, top_indices = conf_b.topk(to_unmask)

                if temperature <= 0:
                    pred_b = logits[b].argmax(dim=-1)
                else:
                    pred_b = torch.multinomial(probs[b], 1).squeeze(-1)

                x[b, top_indices] = pred_b[top_indices]
                is_masked[b, top_indices] = False

        # Final pass
        remaining = is_masked.any(dim=1)
        if remaining.any():
            sigma = torch.full((batch_size,), 0.1, device=device)
            logits, _ = model(x, sigma, K_iter=2)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.stack([torch.multinomial(probs[b], 1).squeeze(-1) for b in range(batch_size)])
            for b in range(batch_size):
                if is_masked[b].any():
                    x[b, is_masked[b]] = final_pred[b, is_masked[b]]

    return x


def generate_maskgit_selfcond(
    model: UniversalDenoiser,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """
    MaskGIT-style generation with self-conditioning.

    Uses previous iteration's predictions as auxiliary input to the model,
    providing "memory" of what the model predicted before.

    Args:
        model: Trained UniversalDenoiser (must have use_self_cond=True)
        length: Target sequence length (including BOS/EOS)
        mask_token_id: ID of <MASK> token
        pad_token_id: ID of <PAD> token
        bos_token_id: ID of <BOS> token
        eos_token_id: ID of <EOS> token
        num_steps: Number of unmasking steps
        temperature: Sampling temperature (0 = argmax)
        schedule: Unmasking schedule ('linear' or 'cosine')
        device: Device to run on

    Returns:
        Generated sequence (length,)
    """
    import math
    model.eval()

    with torch.no_grad():
        # Initialize: BOS + MASKs + EOS
        x = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
        x[0, 0] = bos_token_id
        x[0, -1] = eos_token_id

        # Track which positions are still masked (exclude BOS/EOS)
        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        num_maskable = maskable_positions.sum().item()

        # Current mask state: True = still masked
        is_masked = maskable_positions.clone()

        # Self-conditioning: track previous logits
        prev_logits = None

        for step in range(num_steps):
            # Compute how many to unmask this step
            if schedule == 'linear':
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:  # cosine
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            current_unmasked = (~is_masked & maskable_positions).sum().item()
            to_unmask = max(1, target_unmasked - current_unmasked)

            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break

            to_unmask = min(to_unmask, num_currently_masked)

            # Get model predictions with self-conditioning
            sigma = torch.tensor([1.0 - step / num_steps], device=device)
            logits, _ = model(x, sigma, K_iter=2, prev_logits=prev_logits)

            # Save logits for next iteration
            prev_logits = logits.detach()

            # Compute confidence for each position (max probability)
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)  # (length,)

            # Only consider currently masked positions
            confidence[~is_masked] = -float('inf')

            # Select top-k highest confidence masked positions
            _, top_indices = confidence.topk(to_unmask)

            # Get predictions for those positions
            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                sampled = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
                predictions = sampled

            # Unmask selected positions
            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final pass: fill any remaining masked positions
        if is_masked.any():
            sigma = torch.tensor([0.1], device=device)
            logits, _ = model(x, sigma, K_iter=2, prev_logits=prev_logits)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1).squeeze(0)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def generate_maskgit_selfcond_batch(
    model: UniversalDenoiser,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """Batched MaskGIT-style generation with self-conditioning."""
    import math
    model.eval()

    with torch.no_grad():
        x = torch.full((batch_size, length), mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = bos_token_id
        x[:, -1] = eos_token_id

        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[1:length-1] = True
        num_maskable = maskable_positions.sum().item()

        is_masked = maskable_positions.unsqueeze(0).expand(batch_size, -1).clone()

        # Self-conditioning: track previous logits
        prev_logits = None

        for step in range(num_steps):
            if schedule == 'linear':
                target_unmasked = int((step + 1) / num_steps * num_maskable)
            else:  # cosine
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
                target_unmasked = int(ratio * num_maskable)

            sigma = torch.full((batch_size,), 1.0 - step / num_steps, device=device)
            logits, _ = model(x, sigma, K_iter=2, prev_logits=prev_logits)

            # Save for next iteration
            prev_logits = logits.detach()

            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values  # (B, T)

            # Process each sample
            for b in range(batch_size):
                current_unmasked = (~is_masked[b] & maskable_positions).sum().item()
                to_unmask = max(1, target_unmasked - current_unmasked)
                num_currently_masked = is_masked[b].sum().item()
                if num_currently_masked == 0:
                    continue
                to_unmask = min(to_unmask, num_currently_masked)

                conf_b = confidence[b].clone()
                conf_b[~is_masked[b]] = -float('inf')
                _, top_indices = conf_b.topk(to_unmask)

                if temperature <= 0:
                    pred_b = logits[b].argmax(dim=-1)
                else:
                    pred_b = torch.multinomial(probs[b], 1).squeeze(-1)

                x[b, top_indices] = pred_b[top_indices]
                is_masked[b, top_indices] = False

        # Final pass
        remaining = is_masked.any(dim=1)
        if remaining.any():
            sigma = torch.full((batch_size,), 0.1, device=device)
            logits, _ = model(x, sigma, K_iter=2, prev_logits=prev_logits)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.stack([torch.multinomial(probs[b], 1).squeeze(-1) for b in range(batch_size)])
            for b in range(batch_size):
                if is_masked[b].any():
                    x[b, is_masked[b]] = final_pred[b, is_masked[b]]

    return x


def generate_ar_tokens(
    model: UniversalDenoiser,
    num_tokens: int,
    bos_token_id: int,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> List[int]:
    """
    Generate first few tokens autoregressively using causal attention.

    Args:
        model: Trained UniversalDenoiser
        num_tokens: Number of tokens to generate after BOS
        bos_token_id: ID of <BOS> token
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on

    Returns:
        List of token IDs [BOS, t1, t2, ...]
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        sigma = torch.tensor([0.0], device=device)  # No noise for AR

        for _ in range(num_tokens):
            logits, _ = model(x, sigma, K_iter=1, causal=True)

            # Get prediction for last position
            if temperature <= 0:
                next_token = logits[0, -1].argmax()
            else:
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze()

            # Append to sequence
            x = torch.cat([x, next_token.view(1, 1)], dim=1)

    return x[0].tolist()


def generate_hybrid(
    model: UniversalDenoiser,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_ar_tokens: int = 3,
    maskgit_steps: int = 12,
    temperature: float = 0.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Hybrid generation: AR warmstart + MaskGIT refinement.

    1. Generate first few tokens autoregressively (structural anchors)
    2. Fill rest with MASKs
    3. Use MaskGIT to progressively unmask remaining positions

    Args:
        model: Trained UniversalDenoiser
        length: Target sequence length (including BOS/EOS)
        mask_token_id: ID of <MASK> token
        pad_token_id: ID of <PAD> token
        bos_token_id: ID of <BOS> token
        eos_token_id: ID of <EOS> token
        num_ar_tokens: Number of tokens to generate autoregressively after BOS
        maskgit_steps: Number of MaskGIT unmasking steps
        temperature: Sampling temperature
        schedule: MaskGIT schedule ('linear' or 'cosine')
        device: Device to run on

    Returns:
        Generated sequence (length,)
    """
    import math
    model.eval()

    with torch.no_grad():
        # Ensure num_ar_tokens doesn't exceed available space (length - 2 for BOS/EOS)
        effective_ar_tokens = min(num_ar_tokens, length - 2)

        # Phase 1: AR warmstart - generate first few tokens
        prefix = generate_ar_tokens(
            model,
            num_tokens=effective_ar_tokens,
            bos_token_id=bos_token_id,
            temperature=temperature,
            device=device,
        )

        # Phase 2: Build sequence with prefix + MASKs + EOS
        x = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
        prefix_len = min(len(prefix), length - 1)  # Leave room for EOS
        x[0, :prefix_len] = torch.tensor(prefix[:prefix_len], device=device)
        x[0, -1] = eos_token_id

        # Track which positions are maskable (exclude prefix and EOS)
        maskable_positions = torch.zeros(length, dtype=torch.bool, device=device)
        maskable_positions[prefix_len:length-1] = True
        num_maskable = maskable_positions.sum().item()

        if num_maskable == 0:
            return x.squeeze(0)

        # Current mask state
        is_masked = maskable_positions.clone()

        # Phase 3: MaskGIT refinement
        for step in range(maskgit_steps):
            if schedule == 'linear':
                target_unmasked = int((step + 1) / maskgit_steps * num_maskable)
            else:  # cosine
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * maskgit_steps))
                target_unmasked = int(ratio * num_maskable)

            current_unmasked = (~is_masked & maskable_positions).sum().item()
            to_unmask = max(1, target_unmasked - current_unmasked)

            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break

            to_unmask = min(to_unmask, num_currently_masked)

            # Get model predictions (bidirectional now)
            sigma = torch.tensor([1.0 - step / maskgit_steps], device=device)
            logits, _ = model(x, sigma, K_iter=2, causal=False)

            # Compute confidence
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)
            confidence[~is_masked] = -float('inf')

            # Select top-k
            _, top_indices = confidence.topk(to_unmask)

            # Get predictions
            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                sampled = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
                predictions = sampled

            # Unmask
            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final pass
        if is_masked.any():
            sigma = torch.tensor([0.1], device=device)
            logits, _ = model(x, sigma, K_iter=2)
            if temperature <= 0:
                final_pred = logits.argmax(dim=-1).squeeze(0)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                final_pred = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def generate_hybrid_batch(
    model: UniversalDenoiser,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    num_ar_tokens: int = 3,
    maskgit_steps: int = 12,
    temperature: float = 0.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """Batched hybrid generation (generates samples sequentially for simplicity)."""
    results = []
    for _ in range(batch_size):
        gen = generate_hybrid(
            model, length, mask_token_id, pad_token_id,
            bos_token_id, eos_token_id, num_ar_tokens,
            maskgit_steps, temperature, schedule, device
        )
        results.append(gen)
    return torch.stack(results)


def generate_with_rejection(
    model: UniversalDenoiser,
    energy_model,  # UniversalDenoiserWithEnergy or just the energy head
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    n_candidates: int = 5,
    num_ar_tokens: int = 3,
    maskgit_steps: int = 12,
    temperature: float = 0.8,  # Need some temperature for diversity
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Rejection sampling: generate N candidates, select lowest energy.

    Args:
        model: Trained UniversalDenoiser (for generation)
        energy_model: Model with compute_energy() method
        length: Target sequence length
        mask_token_id, pad_token_id, bos_token_id, eos_token_id: Token IDs
        n_candidates: Number of candidates to generate
        num_ar_tokens: AR tokens for hybrid generation
        maskgit_steps: MaskGIT unmasking steps
        temperature: Sampling temperature (>0 for diversity)
        schedule: MaskGIT schedule
        device: Device

    Returns:
        Best candidate (lowest energy)
    """
    model.eval()

    with torch.no_grad():
        # Generate N candidates
        candidates = []
        for _ in range(n_candidates):
            gen = generate_hybrid(
                model,
                length=length,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                num_ar_tokens=num_ar_tokens,
                maskgit_steps=maskgit_steps,
                temperature=temperature,
                schedule=schedule,
                device=device,
            )
            candidates.append(gen)

        # Stack into batch
        candidates_batch = torch.stack(candidates)  # (N, length)

        # Compute energy scores
        sigma = torch.zeros(n_candidates, device=device)  # Clean sequences
        energies = energy_model.compute_energy(candidates_batch, sigma, K_iter=2)

        # Select lowest energy
        best_idx = energies.argmin()
        return candidates[best_idx]


def generate_with_rejection_batch(
    model: UniversalDenoiser,
    energy_model,
    batch_size: int,
    length: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    n_candidates: int = 5,
    num_ar_tokens: int = 3,
    maskgit_steps: int = 12,
    temperature: float = 0.8,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """Batched version - generates batch_size samples, each with rejection sampling."""
    results = []
    for _ in range(batch_size):
        gen = generate_with_rejection(
            model, energy_model, length,
            mask_token_id, pad_token_id, bos_token_id, eos_token_id,
            n_candidates, num_ar_tokens, maskgit_steps,
            temperature, schedule, device
        )
        results.append(gen)
    return torch.stack(results)


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
