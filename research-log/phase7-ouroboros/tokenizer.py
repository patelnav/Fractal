#!/usr/bin/env python3
"""
Unified tokenizer for Ouroboros (math + code).

Uses tiktoken (GPT-2 BPE) for simplicity and portability.
Adds special tokens for domain markers and reasoning structure.
"""

import tiktoken
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for the Ouroboros tokenizer."""
    base_encoding: str = "gpt2"  # Tiktoken encoding
    max_seq_len: int = 512  # Maximum sequence length

    # Special tokens (added to vocab)
    # We'll encode these as unique strings that GPT-2 BPE handles
    MATH_START: str = "<|MATH|>"
    MATH_END: str = "<|/MATH|>"
    CODE_START: str = "<|CODE|>"
    CODE_END: str = "<|/CODE|>"
    STEP_SEP: str = "<|STEP|>"  # Separates reasoning steps
    ANSWER: str = "<|ANS|>"  # Marks final answer


class OuroborosTokenizer:
    """
    Tokenizer for Ouroboros reasoner.

    Wraps tiktoken with domain-aware encoding.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.enc = tiktoken.get_encoding(self.config.base_encoding)
        self.vocab_size = self.enc.n_vocab  # 50257 for GPT-2

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        return self.enc.decode(ids)

    def encode_math_problem(self, question: str, solution: str) -> List[int]:
        """
        Encode a math problem with its solution.

        Format: <|MATH|>Question<|STEP|>Step1<|STEP|>Step2...<|ANS|>Answer<|/MATH|>
        """
        # Parse GSM8K format: steps separated by newlines, answer after ####
        lines = solution.strip().split('\n')
        steps = []
        answer = ""

        for line in lines:
            line = line.strip()
            if line.startswith("####"):
                answer = line.replace("####", "").strip()
            elif line:
                steps.append(line)

        # Build formatted text
        text = f"{self.config.MATH_START}{question}"
        for step in steps:
            text += f"{self.config.STEP_SEP}{step}"
        text += f"{self.config.ANSWER}{answer}{self.config.MATH_END}"

        return self.encode(text)

    def encode_code_problem(self, prompt: str, solution: str) -> List[int]:
        """
        Encode a code problem with its solution.

        Format: <|CODE|>Prompt<|STEP|>Code<|/CODE|>
        """
        text = f"{self.config.CODE_START}{prompt}{self.config.STEP_SEP}{solution}{self.config.CODE_END}"
        return self.encode(text)

    def get_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from formatted text."""
        steps = text.split(self.config.STEP_SEP)
        return [s.strip() for s in steps if s.strip()]


def parse_gsm8k_solution(answer_text: str) -> Tuple[List[str], str]:
    """
    Parse GSM8K answer format into steps and final answer.

    GSM8K format:
    Step 1 explanation <<calculation>>result
    Step 2 explanation <<calculation>>result
    ...
    #### final_answer
    """
    lines = answer_text.strip().split('\n')
    steps = []
    final_answer = ""

    for line in lines:
        line = line.strip()
        if line.startswith("####"):
            final_answer = line.replace("####", "").strip()
        elif line:
            steps.append(line)

    return steps, final_answer


def create_math_contrastive_pairs(
    question: str,
    correct_solution: str,
    num_wrong: int = 3
) -> List[Tuple[str, str, bool]]:
    """
    Create contrastive pairs for math problem.

    Returns: List of (question, solution, is_correct) tuples

    Uses MORE OBVIOUS perturbations for clearer training signal.
    """
    import random
    import re

    steps, answer = parse_gsm8k_solution(correct_solution)

    pairs = []

    # Correct pair
    pairs.append((question, correct_solution, True))

    # Wrong pairs - use OBVIOUS perturbations
    perturbations = ['wrong_answer_big', 'remove_steps', 'contradictory', 'scramble_numbers']

    for i in range(num_wrong):
        perturbation = perturbations[i % len(perturbations)]
        wrong_steps = steps.copy()
        wrong_answer = answer

        if perturbation == 'wrong_answer_big':
            # Change answer significantly (multiply or divide by 10, or add 100)
            try:
                ans_val = int(float(answer))
                wrong_answer = str(random.choice([
                    ans_val * 10,
                    ans_val + 100,
                    ans_val - ans_val // 2 - 50,
                    ans_val * 2 + 37,
                    max(1, ans_val // 10),
                ]))
            except (ValueError, TypeError):
                wrong_answer = "999"

        elif perturbation == 'remove_steps' and len(wrong_steps) > 1:
            # Remove half the steps - creates incomplete reasoning
            keep_count = max(1, len(wrong_steps) // 2)
            wrong_steps = wrong_steps[:keep_count]
            # Also change answer to make it clearly wrong
            try:
                ans_val = int(float(answer))
                wrong_answer = str(ans_val + random.randint(50, 200))
            except:
                wrong_answer = "42"

        elif perturbation == 'contradictory':
            # Add contradictory statement at the end
            contradictions = [
                "Wait, that's wrong. The answer is actually 0.",
                "Actually I made an error, let me say the answer is 1000.",
                "Hmm, this doesn't make sense. Let me just guess 99.",
                "I'm confused, so the answer must be 42.",
            ]
            wrong_steps.append(random.choice(contradictions))
            wrong_answer = random.choice(["0", "42", "99", "1000", "999"])

        elif perturbation == 'scramble_numbers':
            # Replace ALL numbers in steps with random wrong numbers
            for idx in range(len(wrong_steps)):
                step = wrong_steps[idx]
                nums = re.findall(r'\d+', step)
                for num in nums:
                    try:
                        wrong_num = str(int(num) + random.randint(10, 100))
                        step = step.replace(num, wrong_num, 1)
                    except:
                        pass
                wrong_steps[idx] = step
            # Change answer too
            try:
                ans_val = int(float(answer))
                wrong_answer = str(ans_val + random.randint(50, 500))
            except:
                wrong_answer = "777"

        # Reconstruct solution
        wrong_solution = '\n'.join(wrong_steps) + f"\n#### {wrong_answer}"
        pairs.append((question, wrong_solution, False))

    return pairs


def create_code_contrastive_pairs(
    prompt: str,
    correct_code: str,
    test_code: str,
    num_wrong: int = 3
) -> List[Tuple[str, str, bool]]:
    """
    Create contrastive pairs for code problem.

    Returns: List of (prompt, code, is_correct) tuples

    Uses MORE OBVIOUS perturbations for clearer training signal.
    """
    import random

    pairs = []

    # Correct pair
    pairs.append((prompt, correct_code, True))

    # Wrong pairs - use OBVIOUS perturbations
    perturbations = ['return_constant', 'empty_body', 'wrong_logic', 'syntax_break']

    for i in range(num_wrong):
        perturbation = perturbations[i % len(perturbations)]
        wrong_code = correct_code

        if perturbation == 'return_constant':
            # Replace return value with a constant
            constants = ['None', '0', '[]', '""', 'True', 'False', '42', '-1']
            lines = wrong_code.split('\n')
            return_lines = [i for i, l in enumerate(lines) if 'return ' in l]
            if return_lines:
                idx = random.choice(return_lines)
                # Replace entire return statement
                indent = len(lines[idx]) - len(lines[idx].lstrip())
                lines[idx] = ' ' * indent + 'return ' + random.choice(constants)
                wrong_code = '\n'.join(lines)
            else:
                wrong_code = wrong_code + '\n    return None  # Wrong!'

        elif perturbation == 'empty_body':
            # Replace function body with pass
            lines = wrong_code.split('\n')
            if len(lines) > 1:
                # Keep first line, replace rest with pass
                wrong_code = lines[0] + '\n    pass  # TODO: implement this'
            else:
                wrong_code = '    pass  # Not implemented'

        elif perturbation == 'wrong_logic':
            # Add obviously wrong logic
            wrong_additions = [
                '\n    # BUG: This is wrong\n    result = result * -1',
                '\n    # ERROR: Inverting result\n    return not result',
                '\n    # MISTAKE: Returning opposite\n    if result: return False\n    return True',
                '\n    # WRONG: Always fail\n    raise ValueError("Always fails")',
            ]
            wrong_code = wrong_code.rstrip() + random.choice(wrong_additions)

        elif perturbation == 'syntax_break':
            # Add syntax errors that are visible
            syntax_errors = [
                '\n    !!!SYNTAX ERROR!!!',
                '\n    if True\n        pass',  # Missing colon
                '\n    def broken(:\n        pass',  # Invalid syntax
                '\n    for in range(10): pass',  # Missing variable
            ]
            wrong_code = wrong_code.rstrip() + random.choice(syntax_errors)

        pairs.append((prompt, wrong_code, False))

    return pairs


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = OuroborosTokenizer()

    print("=" * 60)
    print("OUROBOROS TOKENIZER TEST")
    print("=" * 60)

    print(f"\nVocab size: {tokenizer.vocab_size}")

    # Test math encoding
    question = "Natalia sold 48 clips in April, and half as many in May. How many total?"
    solution = """Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
#### 72"""

    tokens = tokenizer.encode_math_problem(question, solution)
    print(f"\nMath problem encoded to {len(tokens)} tokens")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded[:200]}...")

    # Test contrastive pairs
    pairs = create_math_contrastive_pairs(question, solution)
    print(f"\nGenerated {len(pairs)} contrastive pairs:")
    for q, s, is_correct in pairs:
        label = "CORRECT" if is_correct else "WRONG"
        print(f"  [{label}] {s[:50]}...")

    # Test code encoding
    prompt = 'def add(a, b):\n    """Add two numbers."""\n'
    code = "    return a + b"
    tokens = tokenizer.encode_code_problem(prompt, code)
    print(f"\nCode problem encoded to {len(tokens)} tokens")

    code_pairs = create_code_contrastive_pairs(prompt, code, "")
    print(f"\nGenerated {len(code_pairs)} code contrastive pairs:")
    for p, c, is_correct in code_pairs:
        label = "CORRECT" if is_correct else "WRONG"
        print(f"  [{label}] {c[:50]}...")
