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
    """
    import random

    steps, answer = parse_gsm8k_solution(correct_solution)

    pairs = []

    # Correct pair
    pairs.append((question, correct_solution, True))

    # Wrong pairs - perturb the steps
    for _ in range(num_wrong):
        wrong_steps = steps.copy()
        perturbation = random.choice(['swap_numbers', 'wrong_op', 'wrong_answer'])

        if perturbation == 'swap_numbers' and len(wrong_steps) > 0:
            # Swap digits in a random step
            idx = random.randint(0, len(wrong_steps) - 1)
            step = wrong_steps[idx]
            # Find numbers and swap some digits
            import re
            nums = re.findall(r'\d+', step)
            if nums:
                num = random.choice(nums)
                if len(num) > 1:
                    # Swap two digits
                    num_list = list(num)
                    i, j = random.sample(range(len(num_list)), 2)
                    num_list[i], num_list[j] = num_list[j], num_list[i]
                    new_num = ''.join(num_list)
                    step = step.replace(num, new_num, 1)
                else:
                    # Change single digit
                    new_digit = str((int(num) + random.randint(1, 5)) % 10)
                    step = step.replace(num, new_digit, 1)
                wrong_steps[idx] = step

        elif perturbation == 'wrong_op' and len(wrong_steps) > 0:
            # Change an operator
            idx = random.randint(0, len(wrong_steps) - 1)
            step = wrong_steps[idx]
            ops = ['+', '-', '*', '/']
            for op in ops:
                if op in step:
                    new_op = random.choice([o for o in ops if o != op])
                    step = step.replace(op, new_op, 1)
                    break
            wrong_steps[idx] = step

        elif perturbation == 'wrong_answer':
            # Change the final answer
            try:
                wrong_answer = str(int(float(answer)) + random.randint(-10, 10))
            except (ValueError, TypeError):
                wrong_answer = answer + "0"
            answer = wrong_answer

        # Reconstruct solution
        wrong_solution = '\n'.join(wrong_steps) + f"\n#### {answer}"
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
    """
    import random

    pairs = []

    # Correct pair
    pairs.append((prompt, correct_code, True))

    # Wrong pairs - perturb the code
    for _ in range(num_wrong):
        wrong_code = correct_code
        perturbation = random.choice(['syntax_error', 'off_by_one', 'wrong_return', 'wrong_operator'])

        if perturbation == 'syntax_error':
            # Remove a colon or add extra parenthesis
            if ':' in wrong_code:
                # Remove a colon from a random line
                lines = wrong_code.split('\n')
                colon_lines = [i for i, l in enumerate(lines) if ':' in l]
                if colon_lines:
                    idx = random.choice(colon_lines)
                    lines[idx] = lines[idx].replace(':', '', 1)
                    wrong_code = '\n'.join(lines)
            else:
                wrong_code = wrong_code + ')'

        elif perturbation == 'off_by_one':
            # Change range bounds or array indices
            import re
            # Find range(n) and change to range(n-1) or range(n+1)
            matches = list(re.finditer(r'range\((\d+)\)', wrong_code))
            if matches:
                m = random.choice(matches)
                n = int(m.group(1))
                new_n = n + random.choice([-1, 1])
                wrong_code = wrong_code[:m.start(1)] + str(new_n) + wrong_code[m.end(1):]

        elif perturbation == 'wrong_return':
            # Change what's returned
            if 'return ' in wrong_code:
                lines = wrong_code.split('\n')
                return_lines = [i for i, l in enumerate(lines) if 'return ' in l]
                if return_lines:
                    idx = random.choice(return_lines)
                    # Append some garbage
                    lines[idx] = lines[idx].rstrip() + ' + 1'
                    wrong_code = '\n'.join(lines)

        elif perturbation == 'wrong_operator':
            # Change == to != or vice versa
            if '==' in wrong_code:
                wrong_code = wrong_code.replace('==', '!=', 1)
            elif '!=' in wrong_code:
                wrong_code = wrong_code.replace('!=', '==', 1)
            elif ' and ' in wrong_code:
                wrong_code = wrong_code.replace(' and ', ' or ', 1)
            elif ' or ' in wrong_code:
                wrong_code = wrong_code.replace(' or ', ' and ', 1)

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
