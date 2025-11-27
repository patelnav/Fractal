# Results: Vector 7 (Hierarchical Editing)

## Objective
Demonstrate **"Surgical Editing"** in tree-structured generative models: The ability to modify a specific high-level node (Root) of a generated sequence without causing collateral changes to surrounding text ("Butterfly Effect").

## Methodology
1.  **Fractal Editor (`fractal_editor.py`):** Implements `patch_root(index, new_id)`. It acts on the generation trace (tree), replacing a single Root node and re-rendering *only* that subtree using the Fractal Engine.
2.  **Stability Test (`test_stability.py`):** 
    *   Generates a story ($N=10$ roots).
    *   Randomly selects a root index $i \in [1, 8]$.
    *   Replaces Root $i$ with a random new ID.
    *   Verifies byte-for-byte identity of Prefix ($0 \dots i-1$) and Suffix ($i+1 \dots 9$).
    *   Repeats for **10 trials**.
3.  **Baseline Comparison (`baseline_drift.py`):** 
    *   Runs a standard Autoregressive GPT (Manager).
    *   Generates sequence $A$ (20 tokens).
    *   Intervenes at step 5 (forces a different token).
    *   Regenerates the remaining 14 steps (Sequence $B$).
    *   Measures Suffix Match Rate between $A$ and $B$.
    *   Repeats for **20 trials**.

## Results

### 1. Fractal Stability (N=10)
*   **Metric:** Byte-for-byte identity of unedited segments.
*   **Result:** **100.0% Stability** (10/10 trials).
*   **Conclusion:** The Fractal Engine guarantees local edits. Because the "rendering" of Root $i$ depends only on Root $i$ (conditioned on its latent embedding), changes do not propagate to neighbors. Complexity is $O(\text{SubtreeSize})$, independent of document length.

### 2. Baseline Drift (Standard AR, N=20)
*   **Metric:** Exact match rate of the suffix (future tokens) after a single token change.
*   **Result:** **0.00% Match Rate** ($\sigma=0.00$).
*   **Conclusion:** In standard AR models, the "Butterfly Effect" is absolute. Changing one token alters the history for all subsequent attention steps, scrambling the probability distribution and rewriting the entire future timeline.

## Demo: The Fractal Stable Editor
We built an interactive CLI (`interactive_editor.py`) to demonstrate this live.
*   **Workflow:** User generates a Shakespeare-like story $\to$ Selects a chunk index $\to$ Editor patches the root $\to$ System displays a diff showing zero collateral damage.
*   **Significance:** Validates the architecture for applications requiring "Long-Form consistency with local editability" (e.g., Code generation, Screenwriting), which remains an open challenge for standard LLMs.

## Artifacts
*   **Stability Proof:** `python3 research-log/phase17-hierarchical-edit/test_stability.py`
*   **Drift Baseline:** `python3 research-log/phase17-hierarchical-edit/baseline_drift.py`
*   **Interactive Demo:** `python3 research-log/phase17-hierarchical-edit/interactive_editor.py`