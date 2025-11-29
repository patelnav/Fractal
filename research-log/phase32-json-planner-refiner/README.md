# Phase 32: JSON Repair Engine

Neural JSON repair using a REINFORCE-trained bidirectional denoiser. Achieves **98.5% parse success** with minimal edits.

## Results

| Method | Parse@1 | Locality | Avg Edits |
|--------|---------|----------|-----------|
| Do Nothing | 37.2% | 64.1% | 0.0 |
| Heuristic | 88.3% | 76.3% | 11.6 |
| **Denoiser** | **98.5%** | **98.9%** | **0.4** |

## Quick Start

```python
import torch
from tokenizer_json import JSONTokenizer
from model_denoiser import JSONDenoiser
from inference_repair import repair_json_full_denoise

# Load model
checkpoint = torch.load('checkpoints_reinforce/best_denoiser.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']
model = JSONDenoiser(config)

state_dict = checkpoint['model_state_dict']
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

tokenizer = JSONTokenizer()

# Repair broken JSON
broken = '{"name": "Alice" "age": 30}'  # Missing comma
result = repair_json_full_denoise(model, tokenizer, broken, max_len=256)

print(result.repaired)  # {"name": "Alice", "age": 30}
print(result.success)   # True
print(result.confidence)  # ~0.95
```

## Files

| File | Description |
|------|-------------|
| `tokenizer_json.py` | JSON-aware tokenizer (105 vocab) |
| `model_denoiser.py` | Bidirectional transformer denoiser |
| `data_json.py` | Corruption engine and datasets |
| `train_denoiser.py` | Supervised training script |
| `train_denoiser_a100.py` | REINFORCE training (best results) |
| `inference_repair.py` | Repair functions |
| `benchmark.py` | Evaluation harness |

## Model Checkpoints

- `checkpoints/best_denoiser.pt` - 4-layer supervised (11% parse success)
- `checkpoints_reinforce/best_denoiser.pt` - 8-layer REINFORCE (**98.5% parse success**)

The REINFORCE model (26M params) was trained with:
- REINFORCE gradient on parse success reward
- 8 layers, 512 dim, 8 heads
- block_size=256

## API Reference

### `repair_json_full_denoise(model, tokenizer, broken_json, sigma=0.2, max_len=256)`

One-shot full-sequence repair. **Recommended** - achieves best results.

Returns `RepairResult` with:
- `success: bool` - Did repair produce valid JSON?
- `repaired: str` - Repaired JSON string
- `tokens_changed: int` - Number of tokens modified
- `confidence: float` - Model's average prediction confidence

### `repair_json(model, tokenizer, broken_json, ...)`

Iterative window-based repair with:
- Hard locality enforcement
- Progressive window expansion
- Edit region tracking

Lower accuracy (40%) - model wasn't trained for window-based infilling.

### `repair_json_beam(model, tokenizer, broken_json, beam_size=5, ...)`

Beam search with confidence-based candidate ranking.

## Key Insight

The model achieves best results with **full-sequence denoising** rather than window-based masking. Pass the entire corrupted sequence and let it denoise all positions simultaneously.
