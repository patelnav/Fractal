# Fractal Coder v2 - Execution Instructions

This directory contains the implementation of the "Sketch-then-Fill" Fractal Coder architecture using `Qwen-2.5-Coder`.

## Prerequisites

- **Hardware:** NVIDIA GPU (A100/H100 recommended) with sufficient VRAM for `Qwen-2.5-Coder-7B`.
- **Software:** Python 3.8+, CUDA drivers installed.

## Setup

1.  Install dependencies (use `uv`, not `pip`):
    ```bash
    source ~/.local/bin/env && uv pip install -r research-log/phase21-fractal-coder-v2/requirements.txt
    ```

2.  Ensure `HumanEval.jsonl` is available. 
    - The script tries to load it from `HumanEval.jsonl` in the current directory.
    - If not found, it attempts to download from HuggingFace `datasets` ("openai_humaneval").

## Running the Loop

### 1. Fractal Mode (Sketch-then-Fill)
This runs the core v2 logic: generating a plan (sketch) and then implementing it step-by-step.

```bash
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode fractal
```

### 2. Baseline Mode
This runs the standard generation (Prompt -> Code) for comparison.

```bash
python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode baseline
```

### Options
- `--limit N`: Run only the first N problems (useful for debugging).
  ```bash
  python research-log/phase21-fractal-coder-v2/fractal_loop_v2.py --mode fractal --limit 5
  ```

## Results
Results are saved incrementally to `research-log/phase21-fractal-coder-v2/results_v2.jsonl`.
Each entry contains:
- `task_id`: HumanEval ID.
- `passed`: Boolean result of execution.
- `code`: The generated code.
- `sketch`: The high-level plan (if fractal mode).
- `error`: Execution error (if any).

## Troubleshooting
- **vLLM Import Error:** Ensure you are in an environment with GPU support and `vllm` installed. The interface will fail safely (mock mode) if vLLM is missing, but generation will not work.
- **Timeout:** Execution has a 3s timeout per problem.
