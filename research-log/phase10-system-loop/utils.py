#!/usr/bin/env python3
"""
Utility functions for GSM8K evaluation.

Handles answer extraction, comparison, and prompt formatting.
"""

import re
from typing import Optional, Tuple


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from a solution text.

    Handles various formats:
    - GSM8K format: "#### 42"
    - "The answer is 42"
    - "= 42" at end
    - Just a number at the end

    Returns:
        Normalized numeric string, or None if no answer found.
    """
    if not text:
        return None

    text = text.strip()

    # Pattern 1: GSM8K format "#### answer" (Robust to "Answer:" prefix)
    # Search for last occurrence
    matches = list(re.finditer(r'####\s*(?:answer:)?\s*(\$?-?[\d,]+\.?\d*)', text, re.IGNORECASE))
    if matches:
        return normalize_number(matches[-1].group(1))

    # Pattern 2: "The answer is X" or "the final answer is X"
    match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is\s+(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))

    # Pattern 3: "= X" at end of text (possibly with $ or other suffix)
    match = re.search(r'=\s*\$?\s*(-?[\d,]+\.?\d*)\s*(?:dollars?|cents?|%)?\s*\.?\s*$', text)
    if match:
        return normalize_number(match.group(1))

    # Pattern 4: "X." or just "X" at end (last number in text)
    match = re.search(r'(-?[\d,]+\.?\d*)\s*\.?\s*$', text)
    if match:
        return normalize_number(match.group(1))

    # Pattern 5: Boxed answer (LaTeX style) \boxed{X}
    match = re.search(r'\\boxed\{(-?[\d,]+\.?\d*)\}', text)
    if match:
        return normalize_number(match.group(1))

    # Pattern 6: Last number in the text (fallback)
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return normalize_number(numbers[-1])

    return None


def normalize_number(s: str) -> str:
    """
    Normalize a number string for comparison.

    - Removes commas (1,000 -> 1000)
    - Removes leading zeros (007 -> 7)
    - Handles decimals (.5 -> 0.5)
    - Preserves negatives
    """
    if not s:
        return ""

    # Remove commas
    s = s.replace(',', '')

    # Remove $ and other currency symbols
    s = re.sub(r'[$£€]', '', s)

    # Try to convert to number and back for normalization
    try:
        num = float(s)
        # If it's a whole number, return as int string
        if num == int(num):
            return str(int(num))
        else:
            # Return with reasonable precision
            return f"{num:.6f}".rstrip('0').rstrip('.')
    except ValueError:
        return s.strip()


def is_correct(candidate: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Check if a candidate answer matches the ground truth.

    Args:
        candidate: The candidate solution text
        ground_truth: The ground truth answer (already extracted)
        tolerance: Relative tolerance for floating point comparison

    Returns:
        True if the candidate's extracted answer matches ground truth.
    """
    candidate_answer = extract_final_answer(candidate)

    if candidate_answer is None:
        return False

    # Normalize ground truth too
    gt_normalized = normalize_number(ground_truth)

    # Direct string comparison first
    if candidate_answer == gt_normalized:
        return True

    # Numeric comparison with tolerance
    try:
        cand_num = float(candidate_answer)
        gt_num = float(gt_normalized)

        if gt_num == 0:
            return abs(cand_num) < tolerance

        return abs(cand_num - gt_num) / abs(gt_num) < tolerance
    except ValueError:
        return False


def format_prompt(question: str, include_cot: bool = True) -> str:
    """
    Format a question into a prompt for the generator.

    Args:
        question: The math question
        include_cot: Whether to prompt for chain-of-thought reasoning

    Returns:
        Formatted prompt string.
    """
    if include_cot:
        return f"""Question: {question}

Let's solve this step by step. At the end, please write the final answer after '####'.
Example: ...reasoning... #### 42
"""
    else:
        return f"""Question: {question}
Answer:"""


def parse_gsm8k_answer(answer_field: str) -> Tuple[str, str]:
    """
    Parse a GSM8K answer field into reasoning steps and final answer.

    Args:
        answer_field: The "answer" field from GSM8K dataset

    Returns:
        (reasoning_steps, final_answer) tuple
    """
    lines = answer_field.strip().split('\n')
    steps = []
    final_answer = ""

    for line in lines:
        line = line.strip()
        if line.startswith("####"):
            final_answer = line.replace("####", "").strip()
        elif line:
            steps.append(line)

    reasoning = '\n'.join(steps)
    return reasoning, final_answer


if __name__ == "__main__":
    # Test the extraction functions
    print("=" * 60)
    print("UTILS TEST")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("#### 42", "42"),
        ("The total is 48 + 24 = 72.\n#### 72", "72"),
        ("So the answer is 1,234", "1234"),
        ("The final answer is -15.", "-15"),
        ("= $42.50 dollars.", "42.5"),
        ("\\boxed{100}", "100"),
        ("Just some text with 99 at the end", "99"),
        ("Multiple numbers 1 2 3 but 456 is last.", "456"),
    ]

    print("\nExtraction tests:")
    for text, expected in test_cases:
        result = extract_final_answer(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{text[:40]}...' -> '{result}' (expected: '{expected}')")

    # Test is_correct
    print("\nCorrectness tests:")
    correct_tests = [
        ("#### 42", "42", True),
        ("The answer is 42.", "42", True),
        ("The answer is 43.", "42", False),
        ("#### 1,000", "1000", True),
        ("#### 3.14159", "3.14159", True),
        ("No number here", "42", False),
    ]

    for text, gt, expected in correct_tests:
        result = is_correct(text, gt)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] is_correct('{text[:30]}...', '{gt}') = {result}")

    print("\nFormat prompt test:")
    q = "Janet sells 16 eggs per day. How many in a week?"
    print(format_prompt(q))
