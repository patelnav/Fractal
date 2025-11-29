"""
JSON Repair Inference - the core repair loop (Stage 3 Polished).

Pipeline:
1. Parse JSON -> get error location on failure
2. Define error window (±N tokens)
3. Mask tokens in window (HARD ANCHOR tokens outside)
4. Run denoiser to fill masked region
5. Validate with parser
6. If still broken, progressively expand window and retry
7. Use confidence-based beam ranking (energy head optional)

Key improvements in this version:
- Hard locality enforcement: tokens outside window NEVER change
- Progressive window expansion on failure
- Confidence-based beam ranking (no energy head required)
- Edit region tracking and reporting
"""

import json
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field

from tokenizer_json import JSONTokenizer
from model_denoiser import JSONDenoiser, JSONDenoiserWithEnergy, JSONDenoiserConfig


@dataclass
class EditRegion:
    """Describes a region that was edited during repair."""
    start_token: int        # Start token index
    end_token: int          # End token index (exclusive)
    original_tokens: List[int]  # Original token IDs in this region
    repaired_tokens: List[int]  # Repaired token IDs


@dataclass
class RepairResult:
    """Result of a JSON repair attempt."""
    success: bool           # Did the repair produce valid JSON?
    original: str           # Original broken JSON
    repaired: str           # Repaired JSON (or best attempt)
    iterations: int         # Number of repair iterations used
    tokens_changed: int     # Number of tokens changed
    error_message: Optional[str] = None  # Parse error if still broken
    edit_regions: List[EditRegion] = field(default_factory=list)  # What was edited
    confidence: float = 0.0  # Average confidence of predictions


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


def repair_json_full_denoise(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    sigma: float = 0.2,
    device: str = 'cpu',
    max_len: int = 128,
) -> RepairResult:
    """
    Repair broken JSON by full-sequence denoising.

    This is the simplest repair mode: pass the corrupted sequence to the model
    and let it denoise all positions at once. Works best when the model was
    trained on full-sequence denoising (not window-based).

    Args:
        model: Trained JSONDenoiser
        tokenizer: JSONTokenizer
        broken_json: The broken JSON string
        sigma: Noise level hint (lower = more confident predictions)
        device: Device to run on
        max_len: Maximum sequence length

    Returns:
        RepairResult
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
            confidence=1.0,
        )

    # Tokenize
    original_ids = tokenizer.tokenize(broken_json)

    # Truncate if too long
    if len(original_ids) > max_len:
        original_ids = original_ids[:max_len - 1] + [tokenizer.eos_id]

    # Pad to max_len for model
    padded_ids = original_ids + [tokenizer.pad_id] * (max_len - len(original_ids))

    # Full denoise
    with torch.no_grad():
        input_tensor = torch.tensor([padded_ids], dtype=torch.long, device=device)
        sigma_tensor = torch.tensor([sigma], device=device)

        logits, _ = model(input_tensor, sigma_tensor)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        # Get confidence
        pred_probs = probs.gather(2, pred.unsqueeze(-1)).squeeze(-1)
        # Only count non-pad positions
        mask = torch.tensor(padded_ids, device=device) != tokenizer.pad_id
        avg_conf = (pred_probs * mask).sum() / mask.sum()

    # Decode
    pred_ids = pred[0].tolist()

    # Trim to original length (or find EOS)
    try:
        eos_pos = pred_ids.index(tokenizer.eos_id)
        pred_ids = pred_ids[:eos_pos + 1]
    except ValueError:
        pred_ids = pred_ids[:len(original_ids)]

    repaired_json = tokenizer.detokenize(pred_ids)

    # Count changes
    tokens_changed = sum(1 for a, b in zip(original_ids, pred_ids) if a != b)

    # Check success
    success = get_parse_error_position(repaired_json) is None

    return RepairResult(
        success=success,
        original=broken_json,
        repaired=repaired_json,
        iterations=1,
        tokens_changed=tokens_changed,
        confidence=avg_conf.item(),
    )


def repair_json(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    max_iterations: int = 5,
    initial_window_size: int = 5,
    max_window_size: int = 15,
    sigma: float = 0.3,
    temperature: float = 0.0,
    device: str = 'cpu',
    max_len: int = 128,
) -> RepairResult:
    """
    Repair broken JSON using the denoiser model.

    POLISHED VERSION with:
    - Hard locality enforcement (tokens outside window NEVER change)
    - Progressive window expansion on failure
    - Edit region tracking

    Args:
        model: Trained JSONDenoiser
        tokenizer: JSONTokenizer instance
        broken_json: The broken JSON string
        max_iterations: Maximum repair iterations
        initial_window_size: Starting window size (±N tokens from error)
        max_window_size: Maximum window expansion
        sigma: Noise level for denoiser (lower = more confident)
        temperature: Sampling temperature (0 = argmax)
        device: Device to run on
        max_len: Maximum sequence length for model

    Returns:
        RepairResult with success status, repaired JSON, and edit regions
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
            confidence=1.0,
        )

    # Tokenize
    original_ids = tokenizer.tokenize(broken_json)

    # Truncate if too long
    if len(original_ids) > max_len:
        original_ids = original_ids[:max_len - 1] + [tokenizer.eos_id]

    # HARD LOCALITY: anchor_ids stores the immutable original tokens
    anchor_ids = list(original_ids)
    current_ids = list(original_ids)

    # Track all edit regions
    all_edit_regions: List[EditRegion] = []
    # Track which positions have been modified (for locality tracking)
    modified_positions: Set[int] = set()

    # Current window size (will expand on failure)
    window_size = initial_window_size

    # Track confidence scores
    total_confidence = 0.0
    num_predictions = 0

    for iteration in range(max_iterations):
        # Decode current attempt
        current_json = tokenizer.detokenize(current_ids)

        # Check if valid
        error_pos = get_parse_error_position(current_json)
        if error_pos is None:
            # Success!
            tokens_changed = len(modified_positions)
            return RepairResult(
                success=True,
                original=broken_json,
                repaired=current_json,
                iterations=iteration + 1,
                tokens_changed=tokens_changed,
                edit_regions=all_edit_regions,
                confidence=total_confidence / max(1, num_predictions),
            )

        # Find error token position
        char_to_token = _map_char_to_token(current_json, current_ids, tokenizer)
        error_token_idx = char_to_token.get(error_pos, len(current_ids) // 2)

        # Clamp to valid range
        error_token_idx = max(1, min(error_token_idx, len(current_ids) - 2))

        # Define window to mask (progressive expansion)
        start_idx = max(1, error_token_idx - window_size)  # Skip BOS
        end_idx = min(len(current_ids) - 1, error_token_idx + window_size + 1)  # Skip EOS

        # Create masked input - ONLY mask within window
        masked_ids = list(current_ids)
        mask_positions = []
        for i in range(start_idx, end_idx):
            if current_ids[i] not in tokenizer.special_ids:
                masked_ids[i] = tokenizer.mask_id
                mask_positions.append(i)

        if not mask_positions:
            # Nothing to mask - expand window and retry
            window_size = min(window_size + 3, max_window_size)
            continue

        # Run denoiser
        with torch.no_grad():
            input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)
            sigma_tensor = torch.tensor([sigma], device=device)

            logits, _ = model(input_tensor, sigma_tensor)

            # Get predictions and confidence
            probs = F.softmax(logits, dim=-1)

            if temperature <= 0:
                pred = logits.argmax(dim=-1)
                # Get confidence for predicted tokens
                pred_probs = probs.gather(2, pred.unsqueeze(-1)).squeeze(-1)
            else:
                scaled_probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(scaled_probs.view(-1, scaled_probs.size(-1)), 1).view(1, -1)
                pred_probs = probs.gather(2, pred.unsqueeze(-1)).squeeze(-1)

            # HARD LOCALITY: Only update masked positions, never touch anchored tokens
            pred_ids = pred[0].tolist()
            region_original = []
            region_repaired = []

            for pos in mask_positions:
                old_id = current_ids[pos]
                new_id = pred_ids[pos]

                region_original.append(old_id)
                region_repaired.append(new_id)

                if old_id != new_id:
                    current_ids[pos] = new_id
                    modified_positions.add(pos)

                # Track confidence
                total_confidence += pred_probs[0, pos].item()
                num_predictions += 1

            # Record this edit region
            if region_original != region_repaired:
                all_edit_regions.append(EditRegion(
                    start_token=start_idx,
                    end_token=end_idx,
                    original_tokens=region_original,
                    repaired_tokens=region_repaired,
                ))

        # Check if we made progress (error moved or was fixed)
        new_json = tokenizer.detokenize(current_ids)
        new_error_pos = get_parse_error_position(new_json)

        if new_error_pos is not None and new_error_pos == error_pos:
            # Same error position - expand window
            window_size = min(window_size + 2, max_window_size)
        else:
            # Error moved or was fixed - reset window
            window_size = initial_window_size

    # Reached max iterations without success
    final_json = tokenizer.detokenize(current_ids)
    error_pos = get_parse_error_position(final_json)
    tokens_changed = len(modified_positions)

    return RepairResult(
        success=False,
        original=broken_json,
        repaired=final_json,
        iterations=max_iterations,
        tokens_changed=tokens_changed,
        error_message=f"Parse error at position {error_pos}" if error_pos else None,
        edit_regions=all_edit_regions,
        confidence=total_confidence / max(1, num_predictions),
    )


def repair_json_beam(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    beam_size: int = 5,
    max_iterations: int = 5,
    initial_window_size: int = 5,
    max_window_size: int = 15,
    sigma: float = 0.3,
    temperature: float = 0.8,
    device: str = 'cpu',
    max_len: int = 128,
) -> RepairResult:
    """
    Repair broken JSON using beam search with confidence-based ranking.

    POLISHED VERSION:
    - Uses confidence scores instead of energy head for ranking
    - Hard locality enforcement
    - Progressive window expansion
    - Ranks candidates by: 1) parse success, 2) highest confidence, 3) minimal edits

    Args:
        model: JSONDenoiser (energy head optional, not required)
        tokenizer: JSONTokenizer
        broken_json: The broken JSON string
        beam_size: Number of candidates to generate per iteration
        max_iterations: Maximum repair iterations
        initial_window_size: Starting window size
        max_window_size: Maximum window expansion
        sigma: Noise level for denoiser
        temperature: Sampling temperature (>0 for diversity)
        device: Device to run on
        max_len: Maximum sequence length

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
            confidence=1.0,
        )

    original_ids = tokenizer.tokenize(broken_json)

    # Truncate if too long
    if len(original_ids) > max_len:
        original_ids = original_ids[:max_len - 1] + [tokenizer.eos_id]

    current_ids = list(original_ids)
    window_size = initial_window_size
    all_edit_regions: List[EditRegion] = []
    modified_positions: Set[int] = set()

    for iteration in range(max_iterations):
        # Decode and find error
        current_json = tokenizer.detokenize(current_ids)
        error_pos = get_parse_error_position(current_json)

        if error_pos is None:
            return RepairResult(
                success=True,
                original=broken_json,
                repaired=current_json,
                iterations=iteration,
                tokens_changed=len(modified_positions),
                edit_regions=all_edit_regions,
                confidence=1.0,
            )

        # Map to token position
        char_to_token = _map_char_to_token(current_json, current_ids, tokenizer)
        error_token_idx = char_to_token.get(error_pos, len(current_ids) // 2)
        error_token_idx = max(1, min(error_token_idx, len(current_ids) - 2))

        # Define window
        start_idx = max(1, error_token_idx - window_size)
        end_idx = min(len(current_ids) - 1, error_token_idx + window_size + 1)

        # Create masked input
        masked_ids = list(current_ids)
        mask_positions = []
        for i in range(start_idx, end_idx):
            if current_ids[i] not in tokenizer.special_ids:
                masked_ids[i] = tokenizer.mask_id
                mask_positions.append(i)

        if not mask_positions:
            window_size = min(window_size + 3, max_window_size)
            continue

        # Generate beam_size candidates with confidence scores
        candidates: List[Tuple[List[int], float]] = []  # (ids, confidence)

        with torch.no_grad():
            input_tensor = torch.tensor([masked_ids], dtype=torch.long, device=device)
            sigma_tensor = torch.tensor([sigma], device=device)

            logits, _ = model(input_tensor, sigma_tensor)
            probs = F.softmax(logits, dim=-1)

            for _ in range(beam_size):
                # Sample with temperature
                scaled_probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(scaled_probs.view(-1, scaled_probs.size(-1)), 1).view(1, -1)

                # Create candidate with HARD LOCALITY
                candidate_ids = list(current_ids)
                total_conf = 0.0

                for pos in mask_positions:
                    new_id = pred[0, pos].item()
                    candidate_ids[pos] = new_id
                    # Use unscaled probability as confidence
                    total_conf += probs[0, pos, new_id].item()

                avg_conf = total_conf / len(mask_positions)
                candidates.append((candidate_ids, avg_conf))

        # Also add argmax candidate (highest confidence single prediction)
        with torch.no_grad():
            argmax_pred = logits.argmax(dim=-1)
            argmax_ids = list(current_ids)
            argmax_conf = 0.0

            for pos in mask_positions:
                new_id = argmax_pred[0, pos].item()
                argmax_ids[pos] = new_id
                argmax_conf += probs[0, pos, new_id].item()

            argmax_conf /= len(mask_positions)
            candidates.append((argmax_ids, argmax_conf))

        # Evaluate candidates - separate valid and invalid
        valid_candidates: List[Tuple[List[int], float, int]] = []  # (ids, conf, edits)
        invalid_candidates: List[Tuple[List[int], float]] = []

        for cand_ids, conf in candidates:
            cand_json = tokenizer.detokenize(cand_ids)
            edits = sum(1 for i, (a, b) in enumerate(zip(original_ids, cand_ids)) if a != b)

            if get_parse_error_position(cand_json) is None:
                valid_candidates.append((cand_ids, conf, edits))
            else:
                invalid_candidates.append((cand_ids, conf))

        if valid_candidates:
            # Rank by: highest confidence, then minimal edits
            valid_candidates.sort(key=lambda x: (-x[1], x[2]))
            best_ids, best_conf, best_edits = valid_candidates[0]

            repaired_json = tokenizer.detokenize(best_ids)

            # Track what changed
            for pos in mask_positions:
                if current_ids[pos] != best_ids[pos]:
                    modified_positions.add(pos)

            all_edit_regions.append(EditRegion(
                start_token=start_idx,
                end_token=end_idx,
                original_tokens=[current_ids[p] for p in mask_positions],
                repaired_tokens=[best_ids[p] for p in mask_positions],
            ))

            return RepairResult(
                success=True,
                original=broken_json,
                repaired=repaired_json,
                iterations=iteration + 1,
                tokens_changed=len(modified_positions),
                edit_regions=all_edit_regions,
                confidence=best_conf,
            )

        # No valid candidates - use highest confidence invalid for next iteration
        invalid_candidates.sort(key=lambda x: -x[1])
        best_invalid_ids, _ = invalid_candidates[0]

        # Track modifications
        for pos in mask_positions:
            if current_ids[pos] != best_invalid_ids[pos]:
                modified_positions.add(pos)

        current_ids = best_invalid_ids

        # Check if error moved
        new_json = tokenizer.detokenize(current_ids)
        new_error_pos = get_parse_error_position(new_json)

        if new_error_pos is not None and new_error_pos == error_pos:
            window_size = min(window_size + 2, max_window_size)
        else:
            window_size = initial_window_size

    # Failed
    final_json = tokenizer.detokenize(current_ids)
    error_pos = get_parse_error_position(final_json)

    return RepairResult(
        success=False,
        original=broken_json,
        repaired=final_json,
        iterations=max_iterations,
        tokens_changed=len(modified_positions),
        error_message=f"Parse error at position {error_pos}" if error_pos else None,
        edit_regions=all_edit_regions,
        confidence=0.0,
    )


def _map_char_to_token(
    json_str: str,
    token_ids: List[int],
    tokenizer: JSONTokenizer,
) -> Dict[int, int]:
    """
    Create a mapping from character positions to token indices.

    Improved version: uses the lexer to get exact token positions.
    """
    mapping = {}

    # Get lexer tokens with positions
    try:
        lex_tokens = tokenizer._lex_json(json_str)
    except Exception:
        # Fallback to approximate mapping
        return _map_char_to_token_approx(json_str, token_ids, tokenizer)

    # Map lex token index to our token_ids index
    # token_ids starts with BOS, then structural tokens
    tok_idx = 1  # Skip BOS

    for lex_tok in lex_tokens:
        # Map all characters in this lex token's range to tok_idx
        for char_pos in range(lex_tok.start, lex_tok.end):
            if char_pos < len(json_str):
                mapping[char_pos] = tok_idx

        # Advance tok_idx based on token type
        if lex_tok.type in ('LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'COLON', 'COMMA'):
            tok_idx += 1
        elif lex_tok.type in ('TRUE', 'FALSE', 'NULL'):
            tok_idx += 1
        elif lex_tok.type == 'STRING':
            if len(lex_tok.value) == 0:
                tok_idx += 1  # <STR_EMPTY>
            else:
                # <STR> + chars + <STR>
                tok_idx += 2 + min(len(lex_tok.value), tokenizer.max_string_len)
        elif lex_tok.type == 'NUMBER':
            # <NUM> + digits + <NUM>
            tok_idx += 2 + min(len(lex_tok.value), tokenizer.max_number_len)
        elif lex_tok.type == 'ERROR':
            pass  # Skip errors

    return mapping


def _map_char_to_token_approx(
    json_str: str,
    token_ids: List[int],
    tokenizer: JSONTokenizer,
) -> Dict[int, int]:
    """
    Approximate fallback mapping from character positions to token indices.
    """
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


def repair_json_auto(
    model: JSONDenoiser,
    tokenizer: JSONTokenizer,
    broken_json: str,
    use_beam: bool = True,
    device: str = 'cpu',
    **kwargs,
) -> RepairResult:
    """
    Convenience function that picks the best repair strategy.

    Args:
        model: Trained JSONDenoiser
        tokenizer: JSONTokenizer
        broken_json: The broken JSON string
        use_beam: If True, use beam search for better results (slower)
        device: Device to run on
        **kwargs: Additional arguments passed to repair function

    Returns:
        RepairResult
    """
    if use_beam:
        return repair_json_beam(model, tokenizer, broken_json, device=device, **kwargs)
    else:
        return repair_json(model, tokenizer, broken_json, device=device, **kwargs)


def format_diff(result: RepairResult, tokenizer: JSONTokenizer) -> str:
    """Format the edit regions as a human-readable diff."""
    if not result.edit_regions:
        return "No changes made"

    lines = []
    for region in result.edit_regions:
        orig_tokens = [tokenizer.itos.get(t, '?') for t in region.original_tokens]
        new_tokens = [tokenizer.itos.get(t, '?') for t in region.repaired_tokens]

        lines.append(f"  Tokens {region.start_token}-{region.end_token}:")
        lines.append(f"    - {' '.join(orig_tokens)}")
        lines.append(f"    + {' '.join(new_tokens)}")

    return '\n'.join(lines)


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

    print("=== JSON Repair Test (Stage 3 Polished) ===")
    print(f"Note: Using untrained model, results will be random")
    print()

    test_cases = [
        ('{"name": "Alice" "age": 30}', "Missing comma"),
        ('{"items": [1, 2, 3}', "Missing bracket"),
        ('{"key" "value"}', "Missing colon"),
        ('{"valid": true}', "Already valid"),
        ('{"a": 1, "b": 2, "c": 3', "Truncated - missing braces"),
    ]

    print("--- Testing repair_json (greedy) ---")
    for broken, desc in test_cases:
        print(f"[{desc}]")
        print(f"  Input:  {broken}")

        result = repair_json(
            model=model,
            tokenizer=tokenizer,
            broken_json=broken,
            max_iterations=3,
            initial_window_size=5,
            sigma=0.3,
            device='cpu',
        )

        print(f"  Output: {result.repaired}")
        print(f"  Success: {result.success}, Iters: {result.iterations}, "
              f"Changed: {result.tokens_changed}, Conf: {result.confidence:.2f}")
        if result.edit_regions:
            print(f"  Edit regions: {len(result.edit_regions)}")
        print()

    print("--- Testing repair_json_beam (beam search) ---")
    for broken, desc in test_cases[:3]:  # Just first 3 for speed
        print(f"[{desc}]")
        print(f"  Input:  {broken}")

        result = repair_json_beam(
            model=model,
            tokenizer=tokenizer,
            broken_json=broken,
            beam_size=5,
            max_iterations=3,
            initial_window_size=5,
            sigma=0.3,
            temperature=0.8,
            device='cpu',
        )

        print(f"  Output: {result.repaired}")
        print(f"  Success: {result.success}, Iters: {result.iterations}, "
              f"Changed: {result.tokens_changed}, Conf: {result.confidence:.2f}")
        print()


def test_with_trained_model(model_path: str, device: str = 'cpu'):
    """Test with a trained model checkpoint."""
    import torch

    tokenizer = JSONTokenizer()

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = JSONDenoiser(config).to(device)

    # Handle torch.compile() prefix in state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    test_cases = [
        '{"name": "Alice" "age": 30}',  # Missing comma
        '{"items": [1, 2, 3}',          # Missing bracket
        '{"key" "value"}',               # Missing colon
        '{"users": [{"id": 1} {"id": 2}]}',  # Missing comma in array
        '{"config": {"debug" true, "level": 5}}',  # Missing colon
    ]

    print("=== Trained Model Test ===")
    for broken in test_cases:
        print(f"Input:  {broken}")

        # Try greedy first
        result = repair_json(
            model=model,
            tokenizer=tokenizer,
            broken_json=broken,
            device=device,
        )

        if not result.success:
            # Fall back to beam search
            result = repair_json_beam(
                model=model,
                tokenizer=tokenizer,
                broken_json=broken,
                device=device,
            )

        print(f"Output: {result.repaired}")
        print(f"Success: {result.success}, Iters: {result.iterations}, "
              f"Changed: {result.tokens_changed}, Conf: {result.confidence:.2f}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with trained model
        model_path = sys.argv[1]
        device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
        test_with_trained_model(model_path, device)
    else:
        # Test with untrained model
        test_repair()
