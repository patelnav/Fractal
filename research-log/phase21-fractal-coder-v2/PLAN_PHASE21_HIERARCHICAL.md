# Research Plan: Phase 21 - Fractal Coder v2.5 (Hierarchical Assembly)

## Objective
Fix the structural assembly bugs in Fractal Coder by replacing linear sketch execution with a **Tree-Based Hierarchical Assembly**.

## Hypotheses
1.  **Linearity Bug:** Current failures are due to linear generation of nested Python code (Steps 1, 2, 3 are treated flat).
2.  **Hierarchical Fix:** If the sketch is a tree (1, 2, 2.1, 2.2), indentation becomes deterministic (Depth -> Indentation).
3.  **Efficiency:** Hierarchical generation allows parallelizing "leaf" nodes, though v2.5 will process them sequentially for simplicity.

## Step 1: Hierarchical Sketching
**Prompt Update:**
Modify `generate_sketch` to explicitly request nested numbering:
- "Use nested numbering (1., 1.1., 1.2.) to show control flow."
- "Indentation in the sketch should reflect code nesting."

## Step 2: Tree Parsing
**Logic:**
- Parse lines like `2.1. Check condition`.
- Depth = Number of dots - 1 (or based on leading spaces).
- Construct a `StepNode` tree.

## Step 3: Hierarchical Rendering (The "Tree Renderer")
**Algorithm:**
```python
def render_tree(node, current_indent=0):
    code_body = ""
    
    # 1. Render current step
    # Prepend current_indent spaces
    snippet = model.render(..., context=parent_context) 
    code_body += snippet
    
    # 2. Render children
    for child in node.children:
        # Child indent = current_indent + 4
        # (Unless current step didn't open a block? 
        #  No, if it has children in the sketch, it implies a block)
        code_body += render_tree(child, current_indent + 4)
        
    return code_body
```

## Step 4: Evaluation
Run the `representative` test suite (15 problems).
**Success Metric:** Fractal Pass Rate >= 50% (approaching Baseline 66%).

## Tasks
- [ ] Modify `qwen_interface.py` (Prompt for hierarchy)
- [ ] Create `fractal_tree_v2.5.py` (New loop logic)
- [ ] Run Representative Test
