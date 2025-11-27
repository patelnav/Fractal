#!/usr/bin/env python3
"""
vLLM-based Generator for Phase 10 Evaluation

Drop-in replacement for HuggingFaceGenerator that provides ~30x speedup.

Usage:
    from generator_vllm import VLLMGenerator
    generator = VLLMGenerator()
    candidates = generator.generate_batch(questions, n=16)
"""

import os
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams


class VLLMGenerator:
    """Fast batch generator using vLLM."""

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        temperature: float = 0.7,
        max_tokens: int = 256,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 2048,
    ):
        """
        Initialize vLLM generator.

        Args:
            model_name: HuggingFace model ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: Fraction of GPU memory to use (0.5 = 50%)
            max_model_len: Maximum sequence length
        """
        print(f"Loading {model_name} with vLLM...")

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )

        self.model_name = model_name
        print(f"vLLM generator ready!")

    def generate_batch(
        self,
        questions: List[str],
        n: int = 16,
        prompt_template: str = "Solve this math problem step by step:\n\n{question}\n\nSolution:"
    ) -> List[List[str]]:
        """
        Generate n candidate responses for each question.

        Args:
            questions: List of questions/prompts
            n: Number of candidates per question
            prompt_template: Template with {question} placeholder

        Returns:
            List of lists: [batch_size, n_candidates] where each element is a string
        """
        # Create all prompts
        prompts = []
        for q in questions:
            formatted = prompt_template.format(question=q)
            prompts.extend([formatted] * n)

        # Generate all at once (vLLM handles batching efficiently)
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Reshape to [batch_size, n_candidates]
        results = []
        idx = 0
        for _ in questions:
            candidates = []
            for i in range(n):
                text = outputs[idx + i].outputs[0].text
                candidates.append(text)
            results.append(candidates)
            idx += n

        return results


if __name__ == "__main__":
    # Quick test
    print("Testing VLLMGenerator...")

    generator = VLLMGenerator()

    questions = [
        "What is 25 * 4?",
        "If I have 10 apples and eat 3, how many are left?",
    ]

    results = generator.generate_batch(questions, n=4)

    for i, (q, candidates) in enumerate(zip(questions, results)):
        print(f"\nQ{i+1}: {q}")
        for j, c in enumerate(candidates):
            print(f"  [{j}] {c[:80]}...")
