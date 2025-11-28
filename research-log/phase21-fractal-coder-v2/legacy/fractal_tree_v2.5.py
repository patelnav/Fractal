import os
import json
import argparse
import re
import time
from typing import List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from qwen_interface import QwenInterface
from humaneval_harness import load_humaneval_dataset, check_correctness

OUTPUT_FILE = "research-log/phase21-fractal-coder-v2/results_v2_hierarchical.jsonl"

@dataclass
class StepNode:
    id: str
    text: str
    depth: int
    children: List['StepNode'] = field(default_factory=list)
    parent: Optional['StepNode'] = None

def parse_hierarchical_sketch(sketch_text: str) -> List[StepNode]:
    """
    Parses hierarchical sketch into StepNode tree.
    """
    lines = sketch_text.strip().split('\n')
    roots = []
    # Stack stores the current 'active' node at each depth
    # We assume strict hierarchy: depth N is child of depth N-1
    # But numbering is more reliable than strict +1 depth
    
    # Map specific numbered IDs to nodes for parent lookup?
    # Or simpler: The last node seen at depth D-1 is the parent of current node at depth D.
    last_node_at_depth = {} # depth -> node
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        if not (clean_line[0].isdigit() or clean_line.startswith('-') or clean_line.startswith('*')):
             # Skip non-list lines
             continue

        # Parse ID and Content
        # Regex to find "1.1." or "1."
        match = re.match(r'^([\d\.]+)\s+(.*)', clean_line)
        if match:
            step_id = match.group(1)
            content = match.group(2)
            # Depth = count of dots? "1." (0 dots? no 1 dot) -> depth 0. "1.1." (2 dots) -> depth 1.
            # "1" -> depth 0.
            dots = step_id.count('.')
            if step_id.endswith('.'):
                depth = max(0, dots - 1)
            else:
                # "1.1" -> 1 dot -> depth 1
                depth = dots
        else:
            # Bullet point or just text
            step_id = "?"
            content = clean_line
            # Infer depth from indentation
            leading_spaces = len(line) - len(line.lstrip())
            depth = leading_spaces // 4
            
        node = StepNode(id=step_id, text=content, depth=depth)
        
        if depth == 0:
            roots.append(node)
            last_node_at_depth[0] = node
            # Clear deeper history
            keys_to_remove = [k for k in last_node_at_depth if k > 0]
            for k in keys_to_remove: del last_node_at_depth[k]
        else:
            # Find parent
            # Look for closest parent at depth < current
            parent_depth = depth - 1
            while parent_depth >= 0:
                if parent_depth in last_node_at_depth:
                    parent = last_node_at_depth[parent_depth]
                    parent.children.append(node)
                    node.parent = parent
                    break
                parent_depth -= 1
            
            if node.parent is None:
                # Fallback: treat as root
                roots.append(node)
                
            last_node_at_depth[depth] = node
            
    return roots

def render_tree(model: QwenInterface, roots: List[StepNode], signature: str, docstring: str, sketch: str) -> tuple[str, dict]:
    code_body = ""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    for root in roots:
        chunk, usage = _render_node(model, root, signature, docstring, sketch, base_indent=0, context=code_body)
        code_body += chunk
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        
    return code_body, total_usage

def _render_node(model: QwenInterface, node: StepNode, signature: str, docstring: str, sketch: str, base_indent: int, context: str) -> tuple[str, dict]:
    indent_str = " " * base_indent
    
    # 1. Render the current node
    # We instruct the model to implement THIS step.
    # We strip the model's indent and apply ours.
    
    # Prompt helper
    step_text = f"{node.id} {node.text}"
    snippet, usage = model.render_step(signature, docstring, sketch, step_text, context_so_far=context)
    
    snippet = snippet.replace("```python", "").replace("```", "")
    
    lines = snippet.split('\n')
    clean_lines = []
    step_norm = node.text.lower().replace("step", "").strip()
    
    for line in lines:
        if not line.strip(): continue
        # Filter duplication
        if step_norm in line.lower(): continue 
        clean_lines.append(line)
        
    # Apply Indent (Fixing Bug 2: Pre-indented code with DEDENT)
    final_block = ""

    if not clean_lines:
        # Just comment
        final_block += f"{indent_str}# {step_text}\n"
    else:
        # Find minimum indent across all non-empty lines (DEDENT)
        min_indent = float('inf')
        for line in clean_lines:
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces < min_indent:
                min_indent = leading_spaces
        if min_indent == float('inf'):
            min_indent = 0

        final_block += f"{indent_str}# {step_text}\n"
        for line in clean_lines:
            # DEDENT (strip common indent) then add our indent
            dedented = line[min_indent:] if len(line) >= min_indent else line.lstrip()
            final_block += f"{indent_str}{dedented}\n"
            
    # 2. Render Children
    # Determine child indent (Fixing Bug 1: Block Detection)
    # Does the PARENT code open a block?
    # Look at the last non-empty line of THIS snippet
    
    child_base_indent = base_indent # Default: sibling/flat
    
    if clean_lines:
        last_line = clean_lines[-1].strip()
        if last_line.endswith(':'):
            child_base_indent = base_indent + 4
        else:
            # Logic: If parent didn't open a block, but has children in the sketch...
            # What does it mean?
            # e.g., "1. Initialize vars" -> "1.1. x=1"
            # Usually implies grouping. Children stay at same level?
            # Or maybe parent was just a comment.
            # Let's stick to: Only indent if colon.
            child_base_indent = base_indent
    else:
        # Parent was just a comment.
        # If it's a grouping node ("2. Loop over items"), and children are "2.1 For item in items",
        # then children should be at same level?
        # Actually, usually "2. Loop" implies the code "for ..." IS step 2.
        # And "2.1" is inside.
        # If Step 2 generated NO code (just comment), and Step 2.1 generates code...
        # It implies Step 2 was just a label. Children inherit base indent.
        child_base_indent = base_indent

    combined_usage = usage.copy()
    
    # Recursively render children
    # Note: We pass the UPDATED context (including parent code)
    current_context = context + final_block
    
    for child in node.children:
        child_chunk, child_usage = _render_node(model, child, signature, docstring, sketch, child_base_indent, current_context)
        final_block += child_chunk
        current_context += child_chunk # Update context for next sibling
        
        combined_usage["prompt_tokens"] += child_usage["prompt_tokens"]
        combined_usage["completion_tokens"] += child_usage["completion_tokens"]
        
    return final_block, combined_usage

def solve_problem_hierarchical(model: QwenInterface, problem: dict) -> dict:
    start_time = time.time()
    full_prompt = problem['prompt']
    
    # Extract Sig/Doc (reuse regex)
    match = re.search(r'("""|""")', full_prompt)
    if match:
        split_idx = match.start()
        signature = full_prompt[:split_idx].strip()
        docstring = full_prompt[split_idx:].strip()
    else:
        signature, docstring = full_prompt.strip(), ""

    # 1. Generate Hierarchical Sketch
    sketch, usage = model.generate_hierarchical_sketch(signature, docstring)
    
    # 2. Parse
    roots = parse_hierarchical_sketch(sketch)
    
    if not roots:
        # Fallback
        roots = [StepNode("1", "Implement logic", 0)]
        
    # 3. Render Tree
    code_body, render_usage = render_tree(model, roots, signature, docstring, sketch)
    
    total_usage = {
        "prompt_tokens": usage["prompt_tokens"] + render_usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"] + render_usage["completion_tokens"]
    }
    
    end_time = time.time()
    
    return {
        "sketch": sketch,
        "generated_code": code_body,
        "method": "fractal_hierarchical",
        "latency": end_time - start_time,
        "usage": total_usage
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    model = QwenInterface()
    problems = load_humaneval_dataset()
    
    if args.representative:
        target_ids = [
            'HumanEval/0', 'HumanEval/1', 'HumanEval/10', 'HumanEval/11', 'HumanEval/12', 
            'HumanEval/26', 'HumanEval/29', 'HumanEval/32', 'HumanEval/33', 'HumanEval/37', 
            'HumanEval/39', 'HumanEval/40', 'HumanEval/43', 'HumanEval/46', 'HumanEval/129'
        ]
        problems = [p for p in problems if p['task_id'] in target_ids]
        
    if args.limit:
        problems = problems[:args.limit]
        
    for problem in tqdm(problems):
        result = solve_problem_hierarchical(model, problem)
        
        # Execute
        # Fix: clean up code_body? 
        # Ensure no double imports etc.
        
        exec_result = check_correctness(problem, result['generated_code'])
        result.update(exec_result)
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
