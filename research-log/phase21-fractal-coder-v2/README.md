# Research Log: Phase 21 - Fractal Coder v2.x

## Objective
Scale the Fractal Coder architecture from synthetic tasks to real-world Python programming (HumanEval) using `Qwen-2.5-Coder`.

## Status
**COMPLETED (Success)**
Validated **Fractal Coder v3 (Sketch-Guided)** as the stable architecture for future repair loops.

## Experiments

### Experiment 1: Linear Assembly (v2)
- **Approach:** Generate linear steps (1, 2, 3) -> Generate snippets -> Append.
- **Result:** **FAILED (20% Pass)**.
- **Cause:** Linear assembly destroys Python's hierarchical structure (indentation/nesting).

### Experiment 2: Hierarchical Assembly (v2.5)
- **Approach:** Generate tree steps (1, 1.1, 1.2) -> Recursive generation -> Indent children.
- **Result:** **FAILED (0% Pass)**.
- **Cause:** LLMs generate full code blocks (including parents/siblings) even when asked for a specific child node. "Atomic Snippet Generation" is not a capability of current base models.

### Experiment 3: Sketch-Guided Generation (v3)
- **Approach:** Generate Sketch -> Feed as Prompt -> Generate Full Body (Single Shot).
- **Result:** **SUCCESS (66.7% Pass)**.
- **Insight:** Matches Baseline performance. Successfully decouples **Planning** (Sketch) from **Syntax** (Code), enabling a Repair Loop that operates on the Plan.

## Key Artifacts
- `fractal_guided_v3.py`: The validated implementation.
- `RESULTS_PHASE21.md`: Detailed report.
- `legacy/`: Failed experiments (v2, v2.5).

## Next Steps (Phase 22)
Implement the **Fractal Repair Loop**:
1.  Generate Code (v3).
2.  Execute Tests -> Get Error.
3.  Prompt: "Here is the Plan, Code, and Error. Fix the Plan."
4.  Regenerate Code from New Plan.
