"""
JSON Repair API - Clean public interface for neural JSON repair.

This module provides a simple API for repairing broken JSON using
the REINFORCE-trained denoiser model.

Usage:
    from json_repair_api import repair, load_model

    # Load model (auto-finds best checkpoint)
    model, tokenizer = load_model()

    # Repair broken JSON
    result = repair('{"name": "Alice" "age": 30}', model, tokenizer)
    print(result)  # '{"name": "Alice", "age": 30}'
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Tuple, Union

from tokenizer_json import JSONTokenizer
from model_denoiser import JSONDenoiser, JSONDenoiserConfig
from inference_repair import repair_json_full_denoise, RepairResult


# Default checkpoint paths (relative to this file)
_THIS_DIR = Path(__file__).parent
_CHECKPOINT_PATHS = [
    _THIS_DIR / "checkpoints_reinforce" / "best_denoiser.pt",
    _THIS_DIR / "checkpoints" / "best_denoiser.pt",
]


def load_model(
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> Tuple[JSONDenoiser, JSONTokenizer]:
    """
    Load the JSON repair model and tokenizer.

    Args:
        checkpoint_path: Path to model checkpoint. If None, auto-finds best.
        device: Device to load model on ('cpu', 'cuda', 'mps')

    Returns:
        (model, tokenizer) tuple ready for repair

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    # Find checkpoint
    if checkpoint_path is None:
        for path in _CHECKPOINT_PATHS:
            if path.exists():
                checkpoint_path = str(path)
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found. Looked in: {_CHECKPOINT_PATHS}"
            )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = JSONDenoiser(config).to(device)

    # Handle torch.compile() prefix in state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Create tokenizer
    tokenizer = JSONTokenizer()

    return model, tokenizer


def repair(
    broken_json: str,
    model: Optional[JSONDenoiser] = None,
    tokenizer: Optional[JSONTokenizer] = None,
    device: str = 'cpu',
    return_result: bool = False,
) -> Union[str, RepairResult]:
    """
    Repair broken JSON string.

    Args:
        broken_json: The broken JSON string to repair
        model: Pre-loaded model (if None, loads default)
        tokenizer: Pre-loaded tokenizer (if None, loads default)
        device: Device to run on
        return_result: If True, return RepairResult with metadata

    Returns:
        Repaired JSON string, or RepairResult if return_result=True

    Example:
        >>> repair('{"key" "value"}')
        '{"key": "value"}'
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(device=device)

    # Get max_len from model config
    max_len = model.config.block_size

    # Repair
    result = repair_json_full_denoise(
        model=model,
        tokenizer=tokenizer,
        broken_json=broken_json,
        sigma=0.2,
        device=device,
        max_len=max_len,
    )

    if return_result:
        return result
    return result.repaired


def loads(broken_json: str, **kwargs) -> dict:
    """
    Parse broken JSON, repairing if necessary.

    Drop-in replacement for json.loads that auto-repairs.

    Args:
        broken_json: JSON string (possibly broken)
        **kwargs: Passed to repair()

    Returns:
        Parsed Python object

    Raises:
        json.JSONDecodeError: If repair fails

    Example:
        >>> loads('{"key" "value"}')
        {'key': 'value'}
    """
    # First try standard parse
    try:
        return json.loads(broken_json)
    except json.JSONDecodeError:
        pass

    # Repair and parse
    repaired = repair(broken_json, **kwargs)
    return json.loads(repaired)


def repair_file(
    input_path: str,
    output_path: Optional[str] = None,
    inline: bool = False,
    **kwargs,
) -> str:
    """
    Repair a JSON file.

    Args:
        input_path: Path to broken JSON file
        output_path: Path to write repaired JSON (if None, returns string)
        inline: If True, overwrite input file
        **kwargs: Passed to repair()

    Returns:
        Repaired JSON string
    """
    with open(input_path, 'r') as f:
        broken = f.read()

    repaired = repair(broken, **kwargs)

    if inline:
        output_path = input_path

    if output_path:
        with open(output_path, 'w') as f:
            f.write(repaired)

    return repaired


# Convenience: pre-load model on import if env var set
_PRELOADED_MODEL = None
_PRELOADED_TOKENIZER = None

if os.environ.get('JSON_REPAIR_PRELOAD'):
    try:
        _PRELOADED_MODEL, _PRELOADED_TOKENIZER = load_model()
    except FileNotFoundError:
        pass


def repair_quick(broken_json: str) -> str:
    """
    Quick repair using pre-loaded model (if available).

    Faster than repair() for repeated calls.
    """
    global _PRELOADED_MODEL, _PRELOADED_TOKENIZER

    if _PRELOADED_MODEL is None:
        _PRELOADED_MODEL, _PRELOADED_TOKENIZER = load_model()

    return repair(broken_json, _PRELOADED_MODEL, _PRELOADED_TOKENIZER)
