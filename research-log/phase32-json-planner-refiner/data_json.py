"""
JSON Dataset for the JSON Repair Engine.

Generates (clean_json, corrupted_json, sigma) triples for training.

Corruption types:
1. Missing comma/colon
2. Extra comma/colon
3. Missing brace/bracket
4. Extra brace/bracket
5. Missing quote
6. Truncated string
7. Swapped tokens
8. Multi-error combinations
"""

import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from tokenizer_json import JSONTokenizer


# Sample JSON templates for generating training data
JSON_TEMPLATES = [
    # Simple key-value
    '{"key": "value"}',
    '{"name": "Alice", "age": 30}',
    '{"active": true, "count": 0}',

    # Nested objects
    '{"user": {"id": 1, "name": "Bob"}}',
    '{"config": {"debug": false, "level": 5}}',

    # Arrays
    '[1, 2, 3]',
    '["a", "b", "c"]',
    '[true, false, null]',

    # Mixed
    '{"items": [1, 2, 3], "total": 6}',
    '{"users": [{"id": 1}, {"id": 2}]}',

    # Larger structures
    '{"a": 1, "b": 2, "c": 3, "d": 4}',
    '{"nested": {"deep": {"value": 42}}}',
]


def generate_random_json(
    max_depth: int = 3,
    max_keys: int = 5,
    max_array_len: int = 5,
    max_string_len: int = 10,
) -> str:
    """
    Generate a random valid JSON string.

    Args:
        max_depth: Maximum nesting depth
        max_keys: Maximum keys per object
        max_array_len: Maximum array length
        max_string_len: Maximum string length

    Returns:
        Valid JSON string
    """
    def random_string():
        length = random.randint(1, max_string_len)
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
        return ''.join(random.choice(chars) for _ in range(length))

    def random_number():
        if random.random() < 0.5:
            return random.randint(-1000, 1000)
        else:
            return round(random.uniform(-100, 100), 2)

    def random_value(depth: int):
        if depth >= max_depth:
            # Only primitives at max depth
            choice = random.randint(0, 4)
        else:
            choice = random.randint(0, 6)

        if choice == 0:
            return random_string()
        elif choice == 1:
            return random_number()
        elif choice == 2:
            return True
        elif choice == 3:
            return False
        elif choice == 4:
            return None
        elif choice == 5:
            # Object
            num_keys = random.randint(1, max_keys)
            return {random_string(): random_value(depth + 1) for _ in range(num_keys)}
        else:
            # Array
            length = random.randint(0, max_array_len)
            return [random_value(depth + 1) for _ in range(length)]

    # Start with either object or array
    if random.random() < 0.7:
        num_keys = random.randint(1, max_keys)
        data = {random_string(): random_value(1) for _ in range(num_keys)}
    else:
        length = random.randint(1, max_array_len)
        data = [random_value(1) for _ in range(length)]

    return json.dumps(data, separators=(',', ':'))


class JSONCorruptionEngine:
    """
    Engine for corrupting JSON in realistic ways.

    Corruption operations:
    - delete_comma: Remove a comma
    - insert_comma: Add an extra comma
    - delete_colon: Remove a colon
    - insert_colon: Add an extra colon
    - delete_brace: Remove { or }
    - insert_brace: Add extra { or }
    - delete_bracket: Remove [ or ]
    - insert_bracket: Add extra [ or ]
    - delete_quote: Remove a quote from a string
    - truncate_string: Cut off string content
    - swap_adjacent: Swap two adjacent tokens
    - mask_token: Replace with <MASK>
    """

    def __init__(self, tokenizer: JSONTokenizer):
        self.tokenizer = tokenizer

    def corrupt(
        self,
        token_ids: List[int],
        sigma: float,
        corruption_type: Optional[str] = None,
    ) -> Tuple[List[int], str]:
        """
        Corrupt a token sequence.

        Args:
            token_ids: Clean token sequence
            sigma: Corruption intensity (0-1)
            corruption_type: Specific corruption to apply, or None for random

        Returns:
            (corrupted_ids, corruption_name)
        """
        # Number of corruptions based on sigma
        # sigma=0.1 -> ~1 corruption, sigma=0.5 -> ~3 corruptions
        num_corruptions = max(1, int(sigma * 5))

        corrupted = list(token_ids)

        corruption_types = [
            'delete_comma',
            'insert_comma',
            'delete_colon',
            'insert_colon',
            'delete_brace',
            'delete_bracket',
            'swap_adjacent',
            'mask_token',
        ]

        applied = []

        for _ in range(num_corruptions):
            if corruption_type:
                ctype = corruption_type
            else:
                ctype = random.choice(corruption_types)

            if ctype == 'delete_comma' and self._has_token(corrupted, self.tokenizer.comma_id):
                corrupted = self._delete_token(corrupted, self.tokenizer.comma_id)
                applied.append('delete_comma')

            elif ctype == 'insert_comma':
                corrupted = self._insert_token(corrupted, self.tokenizer.comma_id)
                applied.append('insert_comma')

            elif ctype == 'delete_colon' and self._has_token(corrupted, self.tokenizer.colon_id):
                corrupted = self._delete_token(corrupted, self.tokenizer.colon_id)
                applied.append('delete_colon')

            elif ctype == 'insert_colon':
                corrupted = self._insert_token(corrupted, self.tokenizer.colon_id)
                applied.append('insert_colon')

            elif ctype == 'delete_brace':
                if self._has_token(corrupted, self.tokenizer.lbrace_id):
                    if random.random() < 0.5:
                        corrupted = self._delete_token(corrupted, self.tokenizer.lbrace_id)
                    else:
                        corrupted = self._delete_token(corrupted, self.tokenizer.rbrace_id)
                    applied.append('delete_brace')

            elif ctype == 'delete_bracket':
                if self._has_token(corrupted, self.tokenizer.lbracket_id):
                    if random.random() < 0.5:
                        corrupted = self._delete_token(corrupted, self.tokenizer.lbracket_id)
                    else:
                        corrupted = self._delete_token(corrupted, self.tokenizer.rbracket_id)
                    applied.append('delete_bracket')

            elif ctype == 'swap_adjacent':
                corrupted = self._swap_adjacent(corrupted)
                applied.append('swap_adjacent')

            elif ctype == 'mask_token':
                corrupted = self._mask_random(corrupted)
                applied.append('mask_token')

        return corrupted, '+'.join(applied) if applied else 'none'

    def _has_token(self, ids: List[int], token_id: int) -> bool:
        """Check if token exists in sequence (excluding special tokens)."""
        # Skip BOS and EOS positions
        return token_id in ids[1:-1]

    def _delete_token(self, ids: List[int], token_id: int) -> List[int]:
        """Delete one occurrence of a token."""
        # Find all positions (excluding BOS/EOS)
        positions = [i for i, t in enumerate(ids) if t == token_id and 0 < i < len(ids) - 1]
        if not positions:
            return ids
        pos = random.choice(positions)
        return ids[:pos] + ids[pos + 1:]

    def _insert_token(self, ids: List[int], token_id: int) -> List[int]:
        """Insert a token at a random position."""
        # Insert between BOS and EOS
        if len(ids) <= 2:
            return ids
        pos = random.randint(1, len(ids) - 1)
        return ids[:pos] + [token_id] + ids[pos:]

    def _swap_adjacent(self, ids: List[int]) -> List[int]:
        """Swap two adjacent tokens."""
        if len(ids) <= 3:
            return ids
        # Pick a position (excluding BOS and last position before EOS)
        pos = random.randint(1, len(ids) - 3)
        result = list(ids)
        result[pos], result[pos + 1] = result[pos + 1], result[pos]
        return result

    def _mask_random(self, ids: List[int]) -> List[int]:
        """Replace a random token with MASK."""
        if len(ids) <= 2:
            return ids
        pos = random.randint(1, len(ids) - 2)
        result = list(ids)
        result[pos] = self.tokenizer.mask_id
        return result


class JSONRepairDataset(Dataset):
    """
    Dataset for training the JSON denoiser.

    Generates (clean, corrupted, sigma) triples.
    """

    def __init__(
        self,
        num_samples: int,
        max_len: int = 128,
        sigma_min: float = 0.1,
        sigma_max: float = 0.5,
        use_templates: bool = True,
        max_depth: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            max_len: Maximum sequence length (will pad/truncate)
            sigma_min: Minimum corruption intensity
            sigma_max: Maximum corruption intensity
            use_templates: If True, mix templates with random JSON
            max_depth: Max nesting depth for random JSON
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.num_samples = num_samples
        self.max_len = max_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.tokenizer = JSONTokenizer()
        self.corruption_engine = JSONCorruptionEngine(self.tokenizer)

        # Generate clean JSON samples
        self.clean_jsons = []
        for i in range(num_samples):
            if use_templates and random.random() < 0.3:
                # Use a template
                json_str = random.choice(JSON_TEMPLATES)
            else:
                # Generate random JSON
                json_str = generate_random_json(max_depth=max_depth)
            self.clean_jsons.append(json_str)

        if seed is not None:
            random.seed()  # Reset seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            clean: (max_len,) clean token sequence
            corrupted: (max_len,) corrupted token sequence
            sigma: scalar corruption intensity
        """
        json_str = self.clean_jsons[idx]

        # Tokenize
        clean_ids = self.tokenizer.tokenize(json_str)

        # Sample corruption intensity
        sigma = random.uniform(self.sigma_min, self.sigma_max)

        # Corrupt
        corrupted_ids, _ = self.corruption_engine.corrupt(clean_ids, sigma)

        # Pad/truncate
        clean_ids = self._pad_or_truncate(clean_ids)
        corrupted_ids = self._pad_or_truncate(corrupted_ids)

        return (
            torch.tensor(clean_ids, dtype=torch.long),
            torch.tensor(corrupted_ids, dtype=torch.long),
            torch.tensor(sigma, dtype=torch.float32),
        )

    def _pad_or_truncate(self, ids: List[int]) -> List[int]:
        """Pad or truncate to max_len."""
        if len(ids) < self.max_len:
            return ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        else:
            # Truncate but keep EOS
            return ids[:self.max_len - 1] + [self.tokenizer.eos_id]

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token IDs to JSON string."""
        return self.tokenizer.detokenize(ids.tolist())


class JSONEvalDataset(Dataset):
    """
    Evaluation dataset with controlled corruption types.

    For systematic evaluation of repair capabilities.
    """

    def __init__(
        self,
        num_samples: int = 500,
        max_len: int = 128,
        corruption_types: Optional[List[str]] = None,
        sigma: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of samples per corruption type
            max_len: Maximum sequence length
            corruption_types: List of corruption types to test
            sigma: Fixed corruption intensity
            seed: Random seed
        """
        random.seed(seed)

        self.max_len = max_len
        self.sigma = sigma

        self.tokenizer = JSONTokenizer()
        self.corruption_engine = JSONCorruptionEngine(self.tokenizer)

        if corruption_types is None:
            corruption_types = [
                'delete_comma',
                'insert_comma',
                'delete_colon',
                'delete_brace',
                'delete_bracket',
                'swap_adjacent',
                'mask_token',
            ]

        self.corruption_types = corruption_types

        # Generate samples: num_samples per corruption type
        self.samples = []
        samples_per_type = num_samples // len(corruption_types)

        for ctype in corruption_types:
            for _ in range(samples_per_type):
                json_str = generate_random_json(max_depth=2)
                clean_ids = self.tokenizer.tokenize(json_str)
                corrupted_ids, applied = self.corruption_engine.corrupt(
                    clean_ids, sigma, corruption_type=ctype
                )
                self.samples.append({
                    'clean_json': json_str,
                    'clean_ids': clean_ids,
                    'corrupted_ids': corrupted_ids,
                    'corruption_type': ctype,
                    'applied': applied,
                })

        random.seed()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            clean: (max_len,) clean token sequence
            corrupted: (max_len,) corrupted token sequence
            sigma: scalar
            corruption_type: str
        """
        sample = self.samples[idx]

        clean_ids = self._pad_or_truncate(sample['clean_ids'])
        corrupted_ids = self._pad_or_truncate(sample['corrupted_ids'])

        return (
            torch.tensor(clean_ids, dtype=torch.long),
            torch.tensor(corrupted_ids, dtype=torch.long),
            torch.tensor(self.sigma, dtype=torch.float32),
            sample['corruption_type'],
        )

    def _pad_or_truncate(self, ids: List[int]) -> List[int]:
        if len(ids) < self.max_len:
            return ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        else:
            return ids[:self.max_len - 1] + [self.tokenizer.eos_id]

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.detokenize(ids.tolist())


def collate_fn(batch):
    """Collate function for DataLoader."""
    clean = torch.stack([b[0] for b in batch])
    corrupted = torch.stack([b[1] for b in batch])
    sigma = torch.stack([b[2] for b in batch])
    return clean, corrupted, sigma


def test_dataset():
    """Test the JSON dataset."""
    print("=== JSONRepairDataset Test ===")
    ds = JSONRepairDataset(num_samples=10, max_len=64, seed=42)

    print(f"Tokenizer vocab size: {ds.tokenizer.vocab_size}")
    print()

    for i in range(5):
        clean, corrupted, sigma = ds[i]
        print(f"Sample {i}:")
        print(f"  Original JSON: {ds.clean_jsons[i][:60]}...")
        print(f"  Clean tokens: {clean[:20].tolist()}...")
        print(f"  Corrupted:    {corrupted[:20].tolist()}...")
        print(f"  Sigma: {sigma.item():.2f}")
        print(f"  Decoded clean:     {ds.decode(clean)[:60]}...")
        print(f"  Decoded corrupted: {ds.decode(corrupted)[:60]}...")
        print()

    print("=== JSONEvalDataset Test ===")
    eval_ds = JSONEvalDataset(num_samples=21, max_len=64)

    print(f"Total samples: {len(eval_ds)}")
    print()

    # Show one sample per corruption type
    seen = set()
    for i in range(len(eval_ds)):
        clean, corrupted, sigma, ctype = eval_ds[i]
        if ctype not in seen:
            seen.add(ctype)
            print(f"Corruption: {ctype}")
            print(f"  Clean:     {eval_ds.decode(clean)[:50]}...")
            print(f"  Corrupted: {eval_ds.decode(corrupted)[:50]}...")
            print()


def test_parse_success():
    """Test how often corrupted JSON fails to parse vs clean."""
    import json

    ds = JSONRepairDataset(num_samples=100, max_len=128, seed=42)

    clean_parse = 0
    corrupted_parse = 0

    for i in range(len(ds)):
        clean, corrupted, sigma = ds[i]
        clean_json = ds.decode(clean)
        corrupted_json = ds.decode(corrupted)

        try:
            json.loads(clean_json)
            clean_parse += 1
        except json.JSONDecodeError:
            pass

        try:
            json.loads(corrupted_json)
            corrupted_parse += 1
        except json.JSONDecodeError:
            pass

    print(f"\n=== Parse Success Test ===")
    print(f"Clean JSON parse rate: {clean_parse}/{len(ds)} ({100*clean_parse/len(ds):.1f}%)")
    print(f"Corrupted JSON parse rate: {corrupted_parse}/{len(ds)} ({100*corrupted_parse/len(ds):.1f}%)")


if __name__ == "__main__":
    test_dataset()
    test_parse_success()
