# Phase 8: Self-Correcting Solver

**Goal:** Use the Ouroboros Energy Head (Phase 7) to rank candidate solutions from a Generator, improving reasoning accuracy.

## Files

- `solve_math.py`: Main evaluation script. Generates N candidates, scores them with Ouroboros, and compares "Greedy" vs "Min-Energy" accuracy.
- `generator.py`: Wrapper for LLM generation (HuggingFace, vLLM, OpenAI).
- `model.py`: The Ouroboros model architecture (copied from Phase 7).
- `tokenizer.py`: The tokenizer (copied from Phase 7).
- `utils.py`: Answer extraction and prompt formatting.
- `analyze_failures.py`: Diagnostic tool to analyze why the verifier agreed/disagreed with the baseline.
- `config/`: Configuration files for evaluation.

## How to Run

```bash
# 1. Quick Test (100 problems)
python solve_math.py --max_problems 100 --config config/eval_gsm8k.py

# 2. Full Evaluation
python solve_math.py --config config/eval_gsm8k.py

> **Important:** If you modify the prompt template in `utils.py`, you **MUST** clear the cache to see the effects.
> ```bash
> rm -rf cache/generations/
> ```

## Key Findings (from RESULTS.md)

- **+19.4% Accuracy Boost:** Improved Gemma-1B from 36% to 43% on GSM8K.
- **Mechanism:** 50% of gains came from filtering malformed answers (syntax), 50% from detecting logical flaws (semantics).
- **Speed:** "Turbo Mode" enables verifying 1000+ problems in minutes on a single A100.
