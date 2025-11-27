# Plan: Vector 7 - Hierarchical Editing (Fractal Patching)

## Goal
Demonstrate **Surgical Editing**: The ability to modify a high-level structure (Root) and update only the corresponding low-level details (Text) without re-generating the entire sequence. This contrasts with Autoregressive models, where changing a token at index $t$ potentially alters all tokens $t+1 … N$ due to the butterfly effect of attention causal masking.

## The "Fractal Stability" Hypothesis
In a Fractal Model, high-level nodes (Roots) are independent conditional drivers for low-level nodes (Chunks/Chars).
If we change Root $R_i$ to $R'_i$:
1.  We must re-render $R'_i → 	ext{Text}'_i$.
2.  We do **not** need to change $R_j$ or $	ext{Text}_j$ for $j 
eq i$.
3.  Global context is preserved.

## Implementation
We will build a `FractalEditor` class that wraps the Hybrid System.

### 1. Data Structure: `FractalTrace`
Store the generation artifacts:
- `roots`: List of Root IDs.
- `chunks`: List of Lists of Chunk IDs (per root).
- `text_segments`: List of text strings (per root).

### 2. Operation: `patch_root(index, new_root_id)`
1.  **Lock:** Keep `roots[j]` for $j 
eq index$.
2.  **Replace:** Set `roots[index] = new_root_id`.
3.  **Re-Render (Local):**
    - Call `render_root(new_root_id)` $→$ `new_text`.
4.  **Update:** Update `text_segments[index] = new_text`.
5.  **Join:** Return `".join(text_segments)`.

### 3. Test Case: The "Butterfly Effect" Comparison
- **Control (AR):** (Simulated) Changing a token early in a sequence usually changes the entire ending.
- **Experiment (Fractal):** Change Root 2 of 10.
- **Metric:**
    - `Similarity(Old_Suffix, New_Suffix)`
    - For Fractal, this should be 100% (Identity).
    - For AR (Standard LLM), this is typically < 50%.

## Files
- `research-log/phase17-hierarchical-edit/fractal_editor.py`
- `research-log/phase17-hierarchical-edit/test_stability.py`
