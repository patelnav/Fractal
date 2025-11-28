# Plan: Fractal Coder v2.5 (Hierarchical Assembly)

## The Insight
`ANALYSIS.md` confirmed that linear sketches fail to guide Python code assembly because they lack structural (nesting) information.
**Hypothesis:** If the sketch itself is hierarchical (Tree-structured), we can deterministically map sketch depth to code indentation.

## Hierarchical Sketch Format
We will prompt Qwen to generate sketches using nested numbering:
```text
1. Initialize `result` set
2. Iterate through `l1` (element `i`):
    2.1. Check if `i` is in `l2`:
        2.1.1. If yes, add `i` to `result`
3. Return sorted list of `result`
```

## Assembly Logic (The "Tree Renderer")
1.  **Parse Sketch:** Extract steps and their *depth* (indent level or X.Y.Z count).
2.  **Indent State:**
    *   `current_indent = (step_depth - 1) * 4 spaces`.
3.  **Generation:**
    *   Prompt model for Step X.Y.Z.
    *   Prepend `current_indent` to the generated code.
4.  **Verification:**
    *   Does this handle "closing" blocks?
    *   Step 3 (depth 1) follows Step 2.1.1 (depth 3).
    *   We simply drop indent back to Level 1 (0 spaces) for Step 3.
    *   Python handles the dedent automatically by virtue of the next line being less indented.

## Advantages
1.  **Explicit Scope:** The sketch explicitly defines the scope of every action.
2.  **Deterministic Indentation:** No guessing based on `:` or `return`. The sketch *is* the source of truth for indentation.
3.  **Fractal Nature:** This is closer to the true Fractal ideal (recursive expansion).

## Implementation Plan
1.  **Prompt Engineering:** Update `generate_sketch` to enforce nested numbering/indentation.
2.  **Parser:** Update `parse_sketch` to detect hierarchy (indentation or numbering).
3.  **Loop:** Update `solve_problem_fractal` to apply indentation based on parsed depth.
