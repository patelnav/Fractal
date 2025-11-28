"""
JSON Repair Inference - the core repair loop.

Pipeline:
1. Parse JSON -> get error location on failure
2. Define error window (±N tokens)
3. Mask tokens in window
4. Run denoiser to fill masked region
5. Validate with parser
6. If still broken, iterate with new error location
7. Optionally use energy head to rank candidates
"""

import json
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from tokenizer_json import JSONTokenizer
from model_denoiser import JSONDenoiser, JSONDenoiserWithEnergy, JSONDenoiserConfig


@dataclass
class RepairResult:
    """Result of a JSON repair attempt."""
    success: bool           # Did the repair produce valid JSON?
    original: str           # Original broken JSON
    repaired: str           # Repaired JSON (or best attempt)
    iterations: int         # Number of repair iterations used
    tokens_changed: int     # Number of tokens changed
    error_message: Optional[str] = None  # Parse error if still broken


def get_parse_error_position(json_str: str) -> Optional[int]:
    """
    Try to parse JSON and return error position if it fails.

    Returns:
        Character position of error, or None if valid JSON
    """
    try:
        json.loads(json_str)
        return None
    except json.JSONDecodeError as e:
        return e.pos


def repair_json(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    max_iterations: int = 3,
    window_size: int = 5,
    sigma: float = 0.3,
    temperature: float = 0.0,
    device: str = 'cpu',
    max_len: int = 128,
) -> RepairResult:
    """
    Repair broken JSON using the denoiser model.

    Args:
        model: Trained JSONDenoiser
        tokenizer: JSONTokenizer instance
        broken_json: The broken JSON string
        max_iterations: Maximum repair iterations
        window_size: Number of tokens on each side of error to mask
        sigma: Noise level for denoiser (lower = more confident)
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on
        max_len: Maximum sequence length for model

    Returns:
        RepairResult with success status and repaired JSON
    """
    model.eval()

    # Check if already valid
    if get_parse_error_position(broken_json) is None:
        return RepairResult(
            success=True,
            original=broken_json,
            repaired=broken_json,
            iterations=0,
            tokens_changed=0,
        )

    # Tokenize
    original_ids = tokenizer.tokenize(broken_json)

    # Truncate if too long
    if len(original_ids) > max_len:
        original_ids = original_ids[:max_len - 1] + [tokenizer.eos_id]

    current_ids = list(original_ids)

    for iteration in range(max_iterations):
        # Decode current attempt
        current_json = tokenizer.detokenize(current_ids)

        # Check if valid
        error_pos = get_parse_error_position(current_json)
        if error_pos is None:
            # Success!
            tokens_changed = sum(1 for a, b in zip(original_ids, current_ids) if a != b)
            return RepairResult(
                success=True,
                original=broken_json,
                repaired=current_json,
                iterations=iteration + 1,
                tokens_changed=tokens_changed,
            )

        # Find error token position
        # Map character position to token position (approximate)
        char_to_token = _map_char_to_token(current_json, current_ids, tokenizer)
        error_token_idx = char_to_token.get(error_pos, len(current_ids) // 2)

        # Define window to mask
        start_idx = max(1, error_token_idx - window_size)  # Skip BOS
        end_idx = min(len(current_ids) - 1, error_token_idx + window_size + 1)  # Skip EOS

        # Create masked input
        masked_ids = list(current_ids)
        mask_positions = []
        for i in range(start_idx, end_idx):
            if current_ids[i] not in tokenizer.special_ids:
                masked_ids[i] = tokenizer.mask_id
                mask_positions.append(i)

        if not mask_positions:
            # Nothing to mask - try expanding window
            continue

        # Run denoiser
        with torch.no_grad():
            input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)
            sigma_tensor = torch.tensor([sigma], device=device)

            logits, _ = model(input_tensor, sigma_tensor)

            # Get predictions
            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)

            # Only update masked positions
            pred_ids = pred[0].tolist()
            for pos in mask_positions:
                current_ids[pos] = pred_ids[pos]

    # Reached max iterations without success
    final_json = tokenizer.detokenize(current_ids)
    error_pos = get_parse_error_position(final_json)
    tokens_changed = sum(1 for a, b in zip(original_ids, current_ids) if a != b)

    return RepairResult(
        success=False,
        original=broken_json,
        repaired=final_json,
        iterations=max_iterations,
        tokens_changed=tokens_changed,
        error_message=f"Parse error at position {error_pos}" if error_pos else None,
    )


def repair_json_beam(
    model: JSONDenoiserWithEnergy,
    tokenizer: JSONTokenizer,
    broken_json: str,
    beam_size: int = 3,
    max_iterations: int = 3,
    window_size: int = 5,
    sigma: float = 0.3,
    temperature: float = 0.8,
    device: str = 'cpu',
) -> RepairResult:
    """
    Repair broken JSON using beam search with energy-based ranking.

    Generates multiple candidate repairs and selects the best one
    based on: 1) parse success, 2) lowest energy score.

    Args:
        model: JSONDenoiserWithEnergy (has energy head)
        tokenizer: JSONTokenizer
        broken_json: The broken JSON string
        beam_size: Number of candidates to generate
        max_iterations: Maximum repair iterations
        window_size: Tokens on each side of error to mask
        sigma: Noise level for denoiser
        temperature: Sampling temperature (>0 for diversity)
        device: Device to run on

    Returns:
        RepairResult with best repair
    """
    model.eval()

    # Check if already valid
    if get_parse_error_position(broken_json) is None:
        return RepairResult(
            success=True,
            original=broken_json,
            repaired=broken_json,
            iterations=0,
            tokens_changed=0,
        )

    original_ids = tokenizer.tokenize(broken_json)

    for iteration in range(max_iterations):
        # Generate beam_size candidates
        candidates = []

        # Decode and find error
        current_json = tokenizer.detokenize(original_ids)
        error_pos = get_parse_error_position(current_json)
        if error_pos is None:
            return RepairResult(
                success=True,
                original=broken_json,
                repaired=current_json,
                iterations=iteration,
                tokens_changed=0,
            )

        # Map to token position
        char_to_token = _map_char_to_token(current_json, original_ids, tokenizer)
        error_token_idx = char_to_token.get(error_pos, len(original_ids) // 2)

        # Define window
        start_idx = max(1, error_token_idx - window_size)
        end_idx = min(len(original_ids) - 1, error_token_idx + window_size + 1)

        # Create masked input
        masked_ids = list(original_ids)
        mask_positions = []
        for i in range(start_idx, end_idx):
            if original_ids[i] not in tokenizer.special_ids:
                masked_ids[i] = tokenizer.mask_id
                mask_positions.append(i)

        if not mask_positions:
            continue

        # Generate candidates
        with torch.no_grad():
            input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)
            sigma_tensor = torch.tensor([sigma], device=device)

            for _ in range(beam_size):
                logits, _ = model.denoiser(input_tensor, sigma_tensor)

                # Sample with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)

                # Create candidate
                candidate_ids = list(original_ids)
                for pos in mask_positions:
                    candidate_ids[pos] = pred[0, pos].item()
                candidates.append(candidate_ids)

        # Evaluate candidates
        valid_candidates = []
        for cand_ids in candidates:
            cand_json = tokenizer.detokenize(cand_ids)
            if get_parse_error_position(cand_json) is None:
                valid_candidates.append(cand_ids)

        if valid_candidates:
            # Score by energy and pick lowest
            if len(valid_candidates) == 1:
                best_ids = valid_candidates[0]
            else:
                with torch.no_grad():
                    cand_tensor = torch.tensor(valid_candidates, dtype=torch.long, device=device)
                    sigma_batch = torch.full((len(valid_candidates),), 0.0, device=device)
                    energies = model.compute_energy(cand_tensor, sigma_batch)
                    best_idx = energies.argmin().item()
                    best_ids = valid_candidates[best_idx]

            repaired_json = tokenizer.detokenize(best_ids)
            tokens_changed = sum(1 for a, b in zip(original_ids, best_ids) if a != b)

            return RepairResult(
                success=True,
                original=broken_json,
                repaired=repaired_json,
                iterations=iteration + 1,
                tokens_changed=tokens_changed,
            )

        # No valid candidates - try with best by energy
        with torch.no_grad():
            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            sigma_batch = torch.full((len(candidates),), sigma, device=device)
            energies = model.compute_energy(cand_tensor, sigma_batch)
            best_idx = energies.argmin().item()
            original_ids = candidates[best_idx]  # Use best for next iteration

    # Failed
    final_json = tokenizer.detokenize(original_ids)
    tokens_changed = sum(1 for a, b in zip(tokenizer.tokenize(broken_json), original_ids) if a != b)

    return RepairResult(
        success=False,
        original=broken_json,
        repaired=final_json,
        iterations=max_iterations,
        tokens_changed=tokens_changed,
        error_message=f"Parse error after {max_iterations} iterations",
    )


def _map_char_to_token(
    json_str: str,
    token_ids: List[int],
    tokenizer: JSONTokenizer,
) -> Dict[int, int]:
    """
    Create a mapping from character positions to token indices.

    This is approximate since the tokenizer doesn't preserve exact positions.
    """
    # Reconstruct and track positions
    mapping = {}
    char_pos = 0

    for token_idx, tok_id in enumerate(token_ids):
        if tok_id in tokenizer.special_ids:
            continue

        tok = tokenizer.itos.get(tok_id, '')
        if not tok:
            continue

        # Map this range of characters to this token
        tok_len = len(tok) if tok not in ['<STR>', '<STR_EMPTY>', '<NUM>'] else 1
        for i in range(tok_len):
            if char_pos + i < len(json_str):
                mapping[char_pos + i] = token_idx
        char_pos += tok_len

    return mapping


def iterative_repair(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    token_ids: List[int],
    mask_positions: List[int],
    num_steps: int = 3,
    sigma_schedule: Optional[List[float]] = None,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> List[int]:
    """
    Iteratively refine masked positions.

    Similar to MaskGIT-style progressive unmasking but for repair.

    Args:
        model: JSONDenoiser
        tokenizer: JSONTokenizer
        token_ids: Input token sequence with some positions masked
        mask_positions: Indices of masked positions
        num_steps: Number of refinement steps
        sigma_schedule: Optional noise schedule (decreasing)
        temperature: Sampling temperature

    Returns:
        Refined token IDs
    """
    model.eval()

    if sigma_schedule is None:
        sigma_schedule = [0.5 - 0.4 * i / num_steps for i in range(num_steps)]

    current_ids = list(token_ids)

    with torch.no_grad():
        for step, sigma in enumerate(sigma_schedule):
            input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)
            sigma_tensor = torch.tensor([sigma], device=device)

            logits, _ = model(input_tensor, sigma_tensor)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)

            # Update masked positions
            for pos in mask_positions:
                current_ids[pos] = pred[0, pos].item()

    return current_ids


def test_repair():
    """Test the repair functionality with a simple model."""
    from model_denoiser import JSONDenoiserConfig

    tokenizer = JSONTokenizer()
    config = JSONDenoiserConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=128,
    )
    model = JSONDenoiser(config)

    print("=== JSON Repair Test ===")
    print(f"Note: Using untrained model, results will be random")
    print()

    test_cases = [
        '{"name": "Alice" "age": 30}',  # Missing comma
        '{"items": [1, 2, 3}',          # Missing bracket
        '{"key" "value"}',               # Missing colon
        '{"valid": true}',               # Already valid
    ]

    for broken in test_cases:
        print(f"Input:  {broken}")

        result = repair_json(
            model=model,
            tokenizer=tokenizer,
            broken_json=broken,
            max_iterations=2,
            window_size=3,
            sigma=0.3,
            device='cpu',
        )

        print(f"Output: {result.repaired}")
        print(f"Success: {result.success}, Iterations: {result.iterations}, Changed: {result.tokens_changed}")
        print()


if __name__ == "__main__":
    test_repair()
