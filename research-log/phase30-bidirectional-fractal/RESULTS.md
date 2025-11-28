# Phase 30: Bidirectional Flash Flood Validation

**Date:** November 28, 2025
**Status:** Success
**Objective:** Verify if a Bidirectional Transformer (Discrete Diffusion) can preserve "Islands of Correctness" during parallel generation, solving the "Wavefront of Error" problem encountered in Phase 29.

## Experiment
- **Task:** Recursive Arithmetic (`(+ 5 (* 2 3)) = 11`).
- **Model:** 4-Layer Bidirectional Transformer (NanoGPT-based, no causal mask) vs Causal Baseline.
- **Training:** 
  - Bidirectional: Masked Language Modeling (MLM).
  - Causal: Next Token Prediction.
- **Inference:** 
  - Bidirectional: "Flash Flood" (Parallel Gibbs Sampling).
  - Causal: "Jacobi Decoding" (Parallel Naive).

## Results

### 1. Structural Stability (The "Islands" Test)
We masked the operands in a nested expression to see if the model could fill them without breaking the surrounding syntax (parentheses, operators).

**Target:** `(+ (* 2 2) (* 3 3)) = 13`
**Input:** `(+ (* <MASK> 2) (* <MASK> 3)) = <MASK><MASK>`

**Bidirectional (Flash Flood) - Step 1:**
`(+ (* 5 2) (* 2 3)) = 12`
- **Result:** The structure `(+ (* . .) (* . .)) = ..` is **perfectly preserved**. The model filled plausible numbers (`5, 2` and `2, 3`) and a plausible result (`12`). (Note: `10+6=16`, so math is off, but syntax is valid).

**Causal (Jacobi) - Step 1:**
`(* (* ( ( ) (* ( ( ) ) = 1 1 1`
- **Result:** **Catastrophic collapse.** Because the prefix contained MASKs, the Causal model hallucinated a completely broken sequence of parenthesis soup. The "Islands" were destroyed.

### 2. Numeric Convergence
On simpler examples, the Bidirectional model converges to the correct result in 1 step.

**Input:** `(* 3 4) = <MASK><MASK>`
**Output (Step 1):** `(* 3 4) = 12` (Correct)

### 3. Quantitative Comparison
We measured "Structural Accuracy" (Token Match Rate on non-padding tokens) over 5 parallel decoding steps.

```
Accuracy
  ^
  |
  |           o-----------o (Bidirectional: ~81% - Stable)
0.8|
  |
  |
  |       / 
0.6|
  |
  |
  |       | 
0.4|
  |
  |
  |       \ 
0.2|
  |       o-----------o (Causal: ~20% - Collapsed)
  +-------------------------> Steps
      0   1   2   3   4
```

| Step | Causal (Jacobi) | Bidirectional (Flood) |
|------|----------------|-----------------------|
| 0    | 74.2% (Init)   | 74.2% (Init)          |
| 1    | **19.4%**      | **80.6%**             |
| 2    | 14.6%          | 80.6%                 |
| 5    | 10.4%          | 80.6%                 |

## Key Implication
The "Wavefront of Error" (Phase 29) is a specific artifact of **Causal Masking**. 
- In Causal models, error at $t$ propagates to $t+1, t+2...$ instantly in parallel decoding.
- In Bidirectional models, tokens are anchored by *future* context. The error does not propagate; it is constrained.

**Conclusion:** The "Fractal Computer" MUST be Bidirectional.