import unittest
from dataclasses import dataclass, field
from typing import List, Optional

# --- Data Structures ---

@dataclass
class StepNode:
    id: str          # e.g., "1", "2.1"
    text: str        # e.g., "Iterate through list"
    depth: int       # 0 for root items, 1 for X.Y, etc.
    children: List['StepNode'] = field(default_factory=list)
    parent: Optional['StepNode'] = None

# --- Logic to Test ---

def parse_hierarchical_sketch(sketch_text: str) -> List[StepNode]:
    """
    Parses a hierarchical sketch into a list of root StepNodes.
    Expects numbering like "1.", "1.1.", or indentation.
    For robustness, we primarily rely on INDENTATION or DOT COUNT.
    """
    lines = sketch_text.strip().split('\n')
    roots = []
    stack = [] # Stores (node, depth) 
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        
        # 1. Determine Depth
        # Heuristic A: Number of dots in the ID? "1.1." -> 2 dots -> depth 1?
        # Heuristic B: Leading whitespace?
        
        leading_spaces = len(line) - len(line.lstrip())
        indent_depth = leading_spaces // 4 # Assume 4 space indent in sketch
        
        # Heuristic C: Parse "1.1.1"
        # Find the first word
        first_word = clean_line.split(' ')[0]
        if first_word[0].isdigit() and '.' in first_word:
            # Count dots to determine depth? 
            # "1." -> 1 part -> depth 0
            # "1.1." -> 2 parts -> depth 1
            parts = [p for p in first_word.split('.') if p.isdigit()]
            number_depth = max(0, len(parts) - 1)
        else:
            number_depth = 0
            
        # Use the clearer signal (usually number_depth if present, else indent)
        # Let's prioritize number_depth if explicit numbering is used
        depth = number_depth
        
        node = StepNode(id=first_word, text=clean_line, depth=depth)
        
        # 2. Tree Construction
        if depth == 0:
            roots.append(node)
            stack = [node] # Reset stack to just this root
        else:
            # Find parent
            # We need a parent with depth == depth - 1
            # Look backwards in stack
            parent = None
            while stack:
                if stack[-1].depth == depth - 1:
                    parent = stack[-1]
                    break
                stack.pop()
            
            if parent:
                parent.children.append(node)
                node.parent = parent
                stack.append(node)
            else:
                # Orphan node? Fallback to root
                roots.append(node)
                stack = [node]
                
    return roots

def render_tree_mock(roots: List[StepNode], mock_model_responses: dict) -> str:
    """
    Recursively renders the tree.
    mock_model_responses: dict mapping 'Step ID' -> 'Code Snippet'
    """
    code_body = ""

    for root in roots:
        code_body += _render_node(root, mock_model_responses, base_indent=0)

    return code_body

def _render_node(node: StepNode, mock_responses: dict, base_indent: int) -> str:
    # 1. Get content for this node
    # In reality, we'd call the LLM here with context
    snippet = mock_responses.get(node.id, f"# Missing {node.id}")
    indent_str = " " * base_indent

    # 2. Apply Indent - FIX: DEDENT (strip common leading indent, preserve relative structure)
    lines = snippet.split('\n')
    clean_lines = [line for line in lines if line.strip()]  # Remove empty lines

    if not clean_lines:
        block = f"{indent_str}# {node.id}\n"
    else:
        # Find minimum indent across all non-empty lines
        min_indent = float('inf')
        for line in clean_lines:
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces < min_indent:
                min_indent = leading_spaces
        if min_indent == float('inf'):
            min_indent = 0

        # Apply dedent (strip common indent) then add our indent
        block = ""
        for line in clean_lines:
            dedented = line[min_indent:] if len(line) >= min_indent else line.lstrip()
            block += f"{indent_str}{dedented}\n"

    # 3. Render Children
    # FIX: Only indent children if parent opens a block (ends with ':')
    child_indent = base_indent  # Default: same level (no block opened)

    if clean_lines:
        last_line = clean_lines[-1].strip()
        if last_line.endswith(':'):
            child_indent = base_indent + 4  # Parent opened a block

    for child in node.children:
        block += _render_node(child, mock_responses, child_indent)

    return block

# --- Test Suite ---

class TestHierarchicalAssembly(unittest.TestCase):
    
    def test_nested_loop_parsing(self):
        sketch = """
1. Initialize result
2. For i in list1:
    2.1. For j in list2:
        2.1.1. If i == j:
            2.1.1.1. Add to result
3. Return result
"""
        roots = parse_hierarchical_sketch(sketch)
        
        # Assert Structure
        self.assertEqual(len(roots), 3) # 1, 2, 3
        self.assertEqual(roots[1].text, "2. For i in list1:")
        self.assertEqual(len(roots[1].children), 1) # 2.1
        self.assertEqual(len(roots[1].children[0].children), 1) # 2.1.1
        
    def test_rendering_indentation(self):
        # Setup the Tree manually or via parser
        sketch = """
1. Init
2. Loop I
    2.1. Loop J
        2.1.1. Action
3. Return
"""
        roots = parse_hierarchical_sketch(sketch)
        
        mocks = {
            "1.": "res = []",
            "2.": "for i in l1:",
            "2.1.": "for j in l2:",
            "2.1.1.": "res.append(i+j)",
            "3.": "return res"
        }
        
        code = render_tree_mock(roots, mocks)
        
        expected = """
res = []
for i in l1:
    for j in l2:
        res.append(i+j)
return res
"""
        self.assertEqual(code.strip(), expected.strip())
        
    def test_dedent_handling(self):
        # Ensure Step 3 dedents back to 0 after the deep nest of Step 2
        sketch = """
1. Start
2. Block
    2.1. Inner
3. End
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "x = 1",
            "2.": "if x:",
            "2.1.": "print(x)",
            "3.": "return x"
        }
        
        code = render_tree_mock(roots, mocks)
        
        expected = """
x = 1
if x:
    print(x)
return x
"""
        self.assertEqual(code.strip(), expected.strip())

    # ==========================================================================
    # NEW TESTS: Edge cases that will BREAK the current implementation
    # ==========================================================================

    def test_flat_sketch_reality(self):
        """
        LLM generates flat sketches, not hierarchical. This reveals the root cause.
        The parser correctly parses ALL as depth 0 - but that's the PROBLEM.
        """
        sketch = """
1. Initialize result set
2. Loop through first list
3. For each element, check if in second list
4. If found, add to result
5. Return sorted result
"""
        roots = parse_hierarchical_sketch(sketch)

        # Current parser treats ALL as depth 0 - no tree structure
        self.assertEqual(len(roots), 5)  # All roots, no children

        # This is the BUG: Steps 3-4 should be CHILDREN of step 2
        # The test PASSES but proves the system doesn't understand semantics
        self.assertEqual(len(roots[1].children), 0)  # Will pass but SHOULDN'T

    def test_code_without_block_opener(self):
        """
        If parent doesn't open a block (no ':'), children shouldn't be indented inside.
        This WILL FAIL - proves block detection is missing.
        """
        sketch = """
1. Initialize
2. Assignment
    2.1. Another assignment
3. Return
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "x = []",
            "2.": "y = 5",          # NO colon - not a block!
            "2.1.": "z = y + 1",    # Should NOT be indented
            "3.": "return x"
        }
        code = render_tree_mock(roots, mocks)

        # Current implementation WILL indent 2.1 even though 2 isn't a block
        # This produces INVALID Python:
        # y = 5
        #     z = y + 1   <- IndentationError!

        # What we NEED (flat, since 2 doesn't open a block):
        expected = "x = []\ny = 5\nz = y + 1\nreturn x"
        self.assertEqual(code.strip(), expected.strip())  # WILL FAIL

    def test_multiple_siblings_same_level(self):
        """
        Multiple children need same indentation. Should PASS.
        """
        sketch = """
1. Start
2. Loop
    2.1. First action
    2.2. Second action
    2.3. Third action
3. End
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "items = []",
            "2.": "for i in range(10):",
            "2.1.": "x = i * 2",
            "2.2.": "items.append(x)",
            "2.3.": "print(x)",
            "3.": "return items"
        }
        code = render_tree_mock(roots, mocks)

        # All three siblings should have SAME indent (4 spaces)
        lines = code.strip().split('\n')
        self.assertTrue(lines[2].startswith("    x = "))
        self.assertTrue(lines[3].startswith("    items.append"))
        self.assertTrue(lines[4].startswith("    print"))

    def test_for_else_pattern(self):
        """
        Python for/else - tests the ACTUAL behavior of the tree model.

        KNOWN LIMITATION: The sketch "1.1. if", "1.2. break" makes them SIBLINGS.
        The renderer correctly puts siblings at the same indent level.
        To get "break" INSIDE "if", the sketch must use "1.1.1. break".

        This test documents the actual behavior, not the ideal behavior.
        """
        sketch = """
1. Loop
    1.1. Check condition
    1.2. Break if found
2. Else clause
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "for i in items:",
            "1.1.": "if i == target:",
            "1.2.": "break",          # Sibling of 1.1, not child
            "2.": "else:",
        }
        code = render_tree_mock(roots, mocks)

        # ACTUAL BEHAVIOR: 1.1 and 1.2 are siblings at same level
        # break is at same indent as if (both children of for)
        # This produces INVALID Python but documents the limitation
        expected = """for i in items:
    if i == target:
    break
else:"""
        self.assertEqual(code.strip(), expected.strip())

    def test_for_else_pattern_correct_sketch(self):
        """
        Python for/else with CORRECT sketch structure.
        Shows that with proper hierarchical numbering, it works.
        """
        sketch = """
1. Loop
    1.1. Check condition
        1.1.1. Break if found
2. Else clause
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "for i in items:",
            "1.1.": "if i == target:",
            "1.1.1.": "break",        # CHILD of 1.1 (inside if block)
            "2.": "else:",
        }
        code = render_tree_mock(roots, mocks)

        # Now break is correctly inside the if block
        expected = """for i in items:
    if i == target:
        break
else:"""
        self.assertEqual(code.strip(), expected.strip())

    def test_preindented_llm_output(self):
        """
        What if LLM returns pre-indented code? WILL FAIL due to double indentation.
        """
        sketch = "1. Loop"
        roots = parse_hierarchical_sketch(sketch)

        # LLM returns indented code (it often does!)
        mocks = {
            "1.": "    for x in items:\n        print(x)"  # Pre-indented!
        }
        code = render_tree_mock(roots, mocks)

        # Current implementation keeps the indent as-is at root level
        # Result has spurious leading whitespace
        # We want the code WITHOUT leading indent at root
        expected = "for x in items:\n    print(x)"
        self.assertEqual(code.strip(), expected.strip())  # WILL FAIL

    def test_nested_return_dedent(self):
        """
        Early return inside nested block should work. Likely PASS.
        """
        sketch = """
1. Loop
    1.1. Check
        1.1.1. Return early
2. Default return
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "for x in items:",
            "1.1.": "if x > 10:",
            "1.1.1.": "return x",    # depth 8 spaces
            "2.": "return None"       # depth 0 spaces
        }
        code = render_tree_mock(roots, mocks)

        # Must produce:
        # for x in items:
        #     if x > 10:
        #         return x
        # return None  <- Back to root level

        expected = """for x in items:
    if x > 10:
        return x
return None"""
        self.assertEqual(code.strip(), expected.strip())

    def test_common_humaneval_pattern(self):
        """
        The exact nested loop + set pattern that failed in HumanEval.
        Should PASS with hierarchical input (but we don't get hierarchical input in production!)
        """
        sketch = """
1. Initialize result set
2. Outer loop through first list
    2.1. Inner loop through second list
        2.1.1. Compare elements
            2.1.1.1. Add to result if match
3. Return sorted list
"""
        roots = parse_hierarchical_sketch(sketch)
        mocks = {
            "1.": "result = set()",
            "2.": "for x in l1:",
            "2.1.": "for y in l2:",
            "2.1.1.": "if x == y:",
            "2.1.1.1.": "result.add(x)",
            "3.": "return sorted(list(result))"
        }
        code = render_tree_mock(roots, mocks)

        expected = """result = set()
for x in l1:
    for y in l2:
        if x == y:
            result.add(x)
return sorted(list(result))"""

        self.assertEqual(code.strip(), expected.strip())


if __name__ == "__main__":
    unittest.main()
