# Fractal Coder v2 - Problem Analysis

## The Core Problem: Step-by-Step Code Assembly vs Python's Significant Whitespace

The "Sketch-then-Fill" architecture generates code in two phases:
1. **Sketch**: Generate a high-level plan as numbered steps/comments
2. **Fill**: For each step, generate the corresponding code snippet and assemble them

This works beautifully for prose or languages with explicit block delimiters (e.g., `{}` in C/Java). **It fundamentally conflicts with Python's significant whitespace.**

## Why It Fails

### The Nested Control Flow Problem

Consider HumanEval problem `common` (find common elements between two lists):

**Generated Sketch:**
```
1. Initialize an empty set for results
2. Iterate through the first list
3. For each element, check if it's in the second list
4. If found, add to results
5. Return sorted results
```

**What Happens During Fill:**

| Step | Model Output | Required Context |
|------|--------------|------------------|
| 1 | `result = set()` | Base indent (4 spaces) |
| 2 | `for i in l1:` | Base indent (4 spaces) |
| 3 | `if i in l2:` | **Inside the for loop** (8 spaces) |
| 4 | `result.add(i)` | **Inside the if block** (12 spaces) |
| 5 | `return sorted(list(result))` | Back to base indent (4 spaces) |

**The Problem:** When generating Step 3, the model doesn't know it needs to be indented inside the `for` loop from Step 2. Each step is generated independently with only the accumulated `code_body` as context - but the model generates "flat" code without awareness of the required nesting depth.

### Concrete Failure Mode

```python
# What we get:
def common(l1, l2):
    result = set()
    for i in l1:
    if i in l2:        # IndentationError: expected indented block
    result.add(i)
    return sorted(list(result))

# What we need:
def common(l1, l2):
    result = set()
    for i in l1:
        if i in l2:    # Properly nested
            result.add(i)
    return sorted(list(result))
```

## Approaches Tried (All Failed or Degraded)

### 1. Plan-Guided Generation
**Idea:** Include the full sketch in each render prompt so the model knows what's coming.

**Result:** 0% pass rate

**Why it failed:** The model still generates each step's code snippet without explicit indentation instructions. Knowing the full plan doesn't translate to correct indentation.

### 2. Step Iteration with Duplicate Filtering
**Idea:** Filter out duplicate comments, apply base 4-space indent to all code.

**Result:** 20% pass rate

**Why it failed:** Flat indentation breaks nested structures. All code ends up at the same level.

### 3. Stateful Indent Tracking
**Idea:** Track indent level, increment on `:`, decrement on `return`/`break`.

**Result:** 0% pass rate

**Why it failed:** Heuristics don't capture Python's actual scoping rules. When does a `for` loop's scope end? After which step? The model doesn't emit explicit scope-end markers.

### 4. Dynamic Indentation Calculation
**Idea:** Look at the last line of `code_body`. If it ends with `:`, indent the next snippet by +4.

**Result:** 20% pass rate

**Why it failed:** Same underlying issue. If Step 2 generates `for i in l1:`, we correctly indent Step 3. But Step 3 generates `if i in l2:`, which should contain Step 4. And Step 4's code should be inside BOTH the for AND the if. The single-level lookahead doesn't handle multi-level nesting.

## The Fundamental Mismatch

| Sketch-then-Fill Assumes | Python Requires |
|--------------------------|-----------------|
| Steps are sequential, flat | Blocks are hierarchical, nested |
| Each step is independent | Each line's meaning depends on indent |
| Assemble by concatenation | Assemble by AST structure |

**The sketch decomposes the problem linearly, but Python code is a tree.**

## Baseline Comparison

### Small Sample (5 problems)
| Mode | Pass Rate | Notes |
|------|-----------|-------|
| Baseline (one-shot) | **60%** | Model generates complete, syntactically valid code |
| Fractal (best attempt) | **60%** | Matched baseline on small sample |

### Full HumanEval (164 problems)
| Mode | Pass Rate | Notes |
|------|-----------|-------|
| Baseline (one-shot) | **64.63%** | Consistent with small sample |
| Fractal (Sketch-then-Fill) | **12.20%** | Catastrophic degradation at scale |

The small sample was misleading - likely "easy" problems without deeply nested control structures. Full evaluation reveals the step-by-step assembly fundamentally breaks at scale.

## Potential Solutions (Not Yet Tried)

### A. Sketch-Guided Full Body Generation
Instead of assembling snippets, use the sketch as a prompt prefix for one-shot generation:
```
Here is my plan:
1. Initialize result set
2. Loop through first list
...

Now implement the function:
```
This preserves the "planning" benefit without the assembly problem.

### B. AST-Aware Assembly
Parse each snippet, detect incomplete blocks (unclosed `for`/`if`), and track scope explicitly. Complex to implement correctly.

### C. Hierarchical Sketch
Generate sketches that mirror code structure:
```
1. Initialize result
2. FOR each element in l1:
   2.1. IF element in l2:
        2.1.1. Add to result
3. Return sorted result
```
Then the nesting is explicit in the sketch.

### D. Post-hoc Repair
Generate flat code, then use a second pass to fix indentation. But this requires understanding the intended structure, which is what we're missing.

## Conclusion

The Sketch-then-Fill architecture is fundamentally misaligned with Python's syntax. The sketch represents a **linear decomposition** of the algorithm, but Python code requires a **hierarchical structure** with significant whitespace.

**Recommendation:** Pivot to Sketch-Guided Full Body Generation (Option A), which keeps the planning benefit while delegating the structural assembly to the model's native code generation capability.
