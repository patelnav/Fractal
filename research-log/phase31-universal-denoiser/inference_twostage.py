"""
Two-Stage Inference: Skeleton → Digit Filler

Stage 1: Generate skeleton from full mask
Stage 2: Fill digits into skeleton
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import math

from model import UniversalDenoiser, Config


def load_twostage_models(
    skeleton_path: str,
    filler_path: str,
    device: str = 'cpu',
) -> Tuple[UniversalDenoiser, UniversalDenoiser, dict, dict]:
    """Load both skeleton and filler models."""
    # Load skeleton model
    skel_ckpt = torch.load(skeleton_path, map_location=device)
    skel_config = Config()
    for k, v in skel_ckpt['config'].items():
        setattr(skel_config, k, v)
    skeleton_model = UniversalDenoiser(skel_config).to(device)
    skeleton_model.load_state_dict(skel_ckpt['model'])
    skeleton_model.eval()

    # Load filler model
    filler_ckpt = torch.load(filler_path, map_location=device)
    filler_config = Config()
    for k, v in filler_ckpt['config'].items():
        setattr(filler_config, k, v)
    filler_model = UniversalDenoiser(filler_config).to(device)
    filler_model.load_state_dict(filler_ckpt['model'])
    filler_model.eval()

    return skeleton_model, filler_model, skel_ckpt, filler_ckpt


def generate_skeleton_maskgit(
    model: UniversalDenoiser,
    length: int,
    vocab: dict,
    num_steps: int = 12,
    temperature: float = 0.0,
    schedule: str = 'cosine',
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Generate skeleton using MaskGIT-style progressive unmasking.

    Skeleton vocab: <PAD>, <MASK>, <BOS>, <EOS>, <DIGIT>, (, ), +, *, =
    """
    stoi = {s: i for i, s in enumerate(vocab)}
    pad_id = stoi["<PAD>"]
    mask_id = stoi["<MASK>"]
    bos_id = stoi["<BOS>"]
    eos_id = stoi["<EOS>"]

    model.eval()
    with torch.no_grad():
        # Initialize: BOS + MASKs + EOS
        x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
        x[0, 0] = bos_id
        x[0, -1] = eos_id

        maskable = torch.zeros(length, dtype=torch.bool, device=device)
        maskable[1:length-1] = True
        num_maskable = maskable.sum().item()
        is_masked = maskable.clone()

        for step in range(num_steps):
            if schedule == 'cosine':
                ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
            else:
                ratio = (step + 1) / num_steps
            target_unmasked = int(ratio * num_maskable)

            current_unmasked = (~is_masked & maskable).sum().item()
            to_unmask = max(1, target_unmasked - current_unmasked)

            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break
            to_unmask = min(to_unmask, num_currently_masked)

            sigma = torch.tensor([1.0 - step / num_steps], device=device)
            logits, _ = model(x, sigma, K_iter=2)

            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)
            confidence[~is_masked] = -float('inf')

            _, top_indices = confidence.topk(to_unmask)

            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                predictions = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)

            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final cleanup
        if is_masked.any():
            sigma = torch.tensor([0.1], device=device)
            logits, _ = model(x, sigma, K_iter=2)
            final_pred = logits.argmax(dim=-1).squeeze(0)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def fill_digits(
    model: UniversalDenoiser,
    skeleton: torch.Tensor,
    skel_vocab: list,
    filler_vocab: list,
    num_steps: int = 8,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Fill digits into skeleton.

    Input skeleton uses <DIGIT> placeholders.
    Output uses actual digit tokens.
    """
    skel_stoi = {s: i for i, s in enumerate(skel_vocab)}
    filler_stoi = {s: i for i, s in enumerate(filler_vocab)}

    digit_placeholder_skel = skel_stoi["<DIGIT>"]
    mask_id = filler_stoi["<MASK>"]
    pad_id = filler_stoi["<PAD>"]

    # Map skeleton tokens to filler vocabulary
    # Skeleton: <PAD>=0, <MASK>=1, <BOS>=2, <EOS>=3, <DIGIT>=4, (=5, )=6, +=7, *=8, ==9
    # Filler:   <PAD>=0, <MASK>=1, <BOS>=2, <EOS>=3, <DIGIT>=4, 0-9=5-14, (=15, )=16, +=17, *=18, ==19

    skel_to_filler = {}
    for tok in skel_vocab:
        if tok in filler_stoi:
            skel_to_filler[skel_stoi[tok]] = filler_stoi[tok]

    # Convert skeleton to filler vocab, with <DIGIT> becoming <MASK>
    length = len(skeleton)
    x = torch.zeros(1, length, dtype=torch.long, device=device)

    digit_positions = []
    for i, tok in enumerate(skeleton.tolist()):
        if tok == digit_placeholder_skel:
            x[0, i] = mask_id
            digit_positions.append(i)
        elif tok in skel_to_filler:
            x[0, i] = skel_to_filler[tok]
        else:
            x[0, i] = pad_id

    if len(digit_positions) == 0:
        return x.squeeze(0)

    # Fill digits using MaskGIT
    model.eval()
    with torch.no_grad():
        is_masked = torch.zeros(length, dtype=torch.bool, device=device)
        is_masked[digit_positions] = True
        num_to_fill = len(digit_positions)

        for step in range(num_steps):
            ratio = 1 - math.cos(math.pi * (step + 1) / (2 * num_steps))
            target_filled = int(ratio * num_to_fill)

            current_filled = (~is_masked).sum().item() - (length - num_to_fill)
            to_fill = max(1, target_filled - current_filled)

            num_currently_masked = is_masked.sum().item()
            if num_currently_masked == 0:
                break
            to_fill = min(to_fill, num_currently_masked)

            sigma = torch.tensor([0.0], device=device)  # Skeleton is conditioning
            logits, _ = model(x, sigma, K_iter=2)

            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max(dim=-1).values.squeeze(0)
            confidence[~is_masked] = -float('inf')

            _, top_indices = confidence.topk(to_fill)

            if temperature <= 0:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                predictions = torch.multinomial(probs.squeeze(0), 1).squeeze(-1)

            x[0, top_indices] = predictions[top_indices]
            is_masked[top_indices] = False

        # Final cleanup
        if is_masked.any():
            logits, _ = model(x, torch.tensor([0.0], device=device), K_iter=2)
            final_pred = logits.argmax(dim=-1).squeeze(0)
            x[0, is_masked] = final_pred[is_masked]

    return x.squeeze(0)


def generate_twostage(
    skeleton_model: UniversalDenoiser,
    filler_model: UniversalDenoiser,
    length: int,
    skel_vocab: list,
    filler_vocab: list,
    skel_steps: int = 12,
    filler_steps: int = 8,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full two-stage generation.

    Returns: (skeleton, final_output)
    """
    # Stage 1: Generate skeleton
    skeleton = generate_skeleton_maskgit(
        skeleton_model,
        length=length,
        vocab=skel_vocab,
        num_steps=skel_steps,
        temperature=temperature,
        device=device,
    )

    # Stage 2: Fill digits
    output = fill_digits(
        filler_model,
        skeleton=skeleton,
        skel_vocab=skel_vocab,
        filler_vocab=filler_vocab,
        num_steps=filler_steps,
        temperature=temperature,
        device=device,
    )

    return skeleton, output


def decode_skeleton(tokens: torch.Tensor, vocab: list) -> str:
    itos = {i: s for i, s in enumerate(vocab)}
    chars = []
    for t in tokens.tolist():
        if itos.get(t) == "<PAD>":
            break
        chars.append(itos.get(t, '?'))
    return ''.join(chars)


def decode_filler(tokens: torch.Tensor, vocab: list) -> str:
    itos = {i: s for i, s in enumerate(vocab)}
    chars = []
    for t in tokens.tolist():
        if itos.get(t) == "<PAD>":
            break
        chars.append(itos.get(t, '?'))
    return ''.join(chars)


if __name__ == "__main__":
    print("Two-stage inference module loaded.")
    print("Use load_twostage_models() and generate_twostage() for inference.")
