#!/usr/bin/env python3
"""
Solution generators for GSM8K evaluation.

Provides different backends for generating candidate solutions:
- HuggingFaceGenerator: Uses local models via transformers
- OpenAIGenerator: Uses OpenAI API (for quick prototyping)
- CachedGenerator: Wraps any generator with disk caching
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any

from utils import format_prompt


class Generator(ABC):
    """Abstract base class for solution generators."""

    @abstractmethod
    def generate(self, question: str, n: int = 5) -> List[str]:
        """
        Generate n candidate solutions for a question.

        Args:
            question: The math question
            n: Number of candidates to generate

        Returns:
            List of n solution strings
        """
        pass

    def generate_batch(self, questions: List[str], n: int = 5) -> List[List[str]]:
        """
        Generate solutions for multiple questions.

        Default implementation calls generate() for each question.
        Subclasses can override for more efficient batching.
        """
        return [self.generate(q, n) for q in questions]


class HuggingFaceGenerator(Generator):
    """
    Generate solutions using HuggingFace transformers.

    Uses sampling with temperature to get diverse candidates.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            hf_token: HuggingFace API token for gated models
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # Get token from param or environment
        token = hf_token or os.environ.get("HF_TOKEN")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"Loading {model_name} on {device}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
            token=token,
        )

        if device == "mps":
            self.model = self.model.to(device)

        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {self.model.num_parameters() / 1e6:.1f}M parameters")

    def generate(self, question: str, n: int = 5) -> List[str]:
        """Generate n candidate solutions in parallel using batched generation."""
        import torch

        prompt = format_prompt(question)

        # Use chat template for instruction-tuned models
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        with torch.no_grad():
            # Generate all n candidates in a single batched call
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=n,  # Generate all candidates at once
            )

            # Decode all generated sequences
            candidates = []
            prompt_len = inputs['input_ids'].shape[1]
            for i in range(n):
                generated = outputs[i][prompt_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                candidates.append(text.strip())

        return candidates

    def generate_batch(self, questions: List[str], n: int = 5) -> List[List[str]]:
        """
        Generate solutions for multiple questions in TRUE parallel.

        Expands each question n times to create a flat batch, then generates
        all sequences in a single forward pass for maximum GPU utilization.
        """
        import torch

        if not questions:
            return []

        # Format all prompts with chat template and expand each n times
        formatted_prompts = []
        for q in questions:
            prompt = format_prompt(q)
            messages = [{"role": "user", "content": prompt}]
            # Apply chat template (returns token IDs as list)
            formatted = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # Return string, not tokens
            )
            formatted_prompts.append(formatted)

        expanded_prompts = []
        for p in formatted_prompts:
            expanded_prompts.extend([p] * n)  # Each question repeated n times

        # Tokenize with padding (left padding for generation)
        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        batch_size = len(questions)

        with torch.no_grad():
            # Generate all sequences in ONE call (true parallelism)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                # No num_return_sequences - we already expanded the batch
            )

            # outputs shape: (batch_size * n, seq_len)
            # Reshape and decode
            all_results = []
            prompt_lens = inputs['attention_mask'].sum(dim=1)  # Actual length per prompt

            for i in range(batch_size):
                candidates = []
                for j in range(n):
                    idx = i * n + j
                    prompt_len = prompt_lens[idx].item()
                    generated = outputs[idx][prompt_len:]
                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    candidates.append(text.strip())
                all_results.append(candidates)

        return all_results


class VLLMGenerator(Generator):
    """
    Generate solutions using vLLM for high-throughput inference.

    Uses PagedAttention and continuous batching for maximum GPU utilization.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hf_token: Optional[str] = None,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import os

        token = hf_token or os.environ.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        print(f"Loading {model_name} with vLLM...")

        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=2048,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        self.max_new_tokens = max_new_tokens
        print("vLLM model loaded!")

    def _apply_chat_template(self, question: str) -> str:
        """Apply chat template to a question."""
        prompt = format_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate(self, question: str, n: int = 5) -> List[str]:
        """Generate n candidate solutions."""
        formatted_prompt = self._apply_chat_template(question)
        # Generate n samples in one call
        outputs = self.llm.generate([formatted_prompt] * n, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_batch(self, questions: List[str], n: int = 5) -> List[List[str]]:
        """Generate solutions for multiple questions in parallel."""
        # Expand prompts with chat template: each question repeated n times
        prompts = []
        for q in questions:
            formatted_prompt = self._apply_chat_template(q)
            prompts.extend([formatted_prompt] * n)

        # Generate all at once - vLLM handles batching efficiently
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Reshape results
        results = []
        for i in range(len(questions)):
            candidates = []
            for j in range(n):
                idx = i * n + j
                candidates.append(outputs[idx].outputs[0].text.strip())
            results.append(candidates)

        return results


class OpenAIGenerator(Generator):
    """
    Generate solutions using OpenAI API.

    Good for quick prototyping before switching to local models.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 256,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI generator.

        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, question: str, n: int = 5) -> List[str]:
        """Generate n candidate solutions using OpenAI."""
        prompt = format_prompt(question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Solve problems step by step, showing your work. End with '#### [answer]' where [answer] is just the final number."
                },
                {"role": "user", "content": prompt}
            ],
            n=n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return [choice.message.content.strip() for choice in response.choices]


class CachedGenerator(Generator):
    """
    Wraps a generator with disk caching.

    Caches generated solutions to avoid regenerating during development.
    """

    def __init__(
        self,
        generator: Generator,
        cache_dir: str = "cache/generations",
        enabled: bool = True,
    ):
        """
        Initialize the cached generator.

        Args:
            generator: Underlying generator to wrap
            cache_dir: Directory for cache files
            enabled: Whether caching is enabled
        """
        self.generator = generator
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, question: str, n: int) -> str:
        """Generate a cache key for a question."""
        content = f"{question}|n={n}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self.cache_dir / f"{key}.json"

    def generate(self, question: str, n: int = 5) -> List[str]:
        """Generate with caching."""
        if not self.enabled:
            return self.generator.generate(question, n)

        key = self._cache_key(question, n)
        cache_file = self._cache_path(key)

        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return data['candidates']

        # Generate and cache
        candidates = self.generator.generate(question, n)

        with open(cache_file, 'w') as f:
            json.dump({
                'question': question,
                'n': n,
                'candidates': candidates,
            }, f, indent=2)

        return candidates


class DummyGenerator(Generator):
    """
    Dummy generator for testing.

    Generates fake solutions with controllable correctness rate.
    """

    def __init__(self, correct_rate: float = 0.3, seed: int = 42):
        """
        Initialize dummy generator.

        Args:
            correct_rate: Probability that a candidate is correct
            seed: Random seed
        """
        import random
        self.correct_rate = correct_rate
        self.rng = random.Random(seed)

    def generate(self, question: str, n: int = 5) -> List[str]:
        """Generate dummy solutions."""
        from utils import parse_gsm8k_answer

        candidates = []
        for i in range(n):
            if self.rng.random() < self.correct_rate:
                # Generate "correct" solution (would need ground truth)
                answer = self.rng.randint(1, 100)
            else:
                # Generate wrong solution
                answer = self.rng.randint(1, 100)

            # Create a fake solution
            solution = f"Step 1: We need to solve this problem.\n"
            solution += f"Step 2: After calculation, the answer is {answer}.\n"
            solution += f"#### {answer}"

            candidates.append(solution)

        return candidates


def load_generator(config: Dict[str, Any]) -> Generator:
    """
    Load a generator based on configuration.

    Args:
        config: Dictionary with generator configuration

    Returns:
        Configured generator instance
    """
    generator_type = config.get("type", "huggingface")

    if generator_type == "huggingface":
        gen = HuggingFaceGenerator(
            model_name=config.get("model_name", "google/gemma-3-1b-it"),
            device=config.get("device", "auto"),
            max_new_tokens=config.get("max_new_tokens", 256),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            do_sample=config.get("do_sample", True),
            hf_token=config.get("hf_token"),
        )
    elif generator_type == "vllm":
        gen = VLLMGenerator(
            model_name=config.get("model_name", "google/gemma-3-1b-it"),
            max_new_tokens=config.get("max_new_tokens", 256),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            hf_token=config.get("hf_token"),
        )
    elif generator_type == "openai":
        gen = OpenAIGenerator(
            model=config.get("model", "gpt-3.5-turbo"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 256),
        )
    elif generator_type == "dummy":
        gen = DummyGenerator(
            correct_rate=config.get("correct_rate", 0.3),
            seed=config.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

    # Wrap with caching if enabled
    if config.get("cache_enabled", True):
        gen = CachedGenerator(
            gen,
            cache_dir=config.get("cache_dir", "cache/generations"),
            enabled=True,
        )

    return gen


if __name__ == "__main__":
    # Test with dummy generator
    print("=" * 60)
    print("GENERATOR TEST")
    print("=" * 60)

    # Test dummy generator
    gen = DummyGenerator(correct_rate=0.4, seed=42)
    question = "Janet has 16 apples. She gives away 4. How many does she have?"

    print(f"\nQuestion: {question}")
    print("\nCandidates:")
    candidates = gen.generate(question, n=5)
    for i, c in enumerate(candidates):
        print(f"\n--- Candidate {i+1} ---")
        print(c)

    # Test caching
    print("\n" + "=" * 60)
    print("CACHE TEST")
    print("=" * 60)

    cached_gen = CachedGenerator(gen, cache_dir="/tmp/test_cache")
    candidates1 = cached_gen.generate(question, n=3)
    candidates2 = cached_gen.generate(question, n=3)  # Should hit cache

    print(f"\nFirst call: {len(candidates1)} candidates")
    print(f"Second call: {len(candidates2)} candidates (from cache)")
    print(f"Same results: {candidates1 == candidates2}")
