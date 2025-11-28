"""
Two-Stage Dataset for Coarse→Fine Generation.

Stage 1: Skeleton Generator
- Input: Full mask
- Output: Structure only (parens, ops, equals) with <DIGIT> placeholders

Stage 2: Digit Filler
- Input: Skeleton with <DIGIT> placeholders
- Output: Complete expression with digits filled in
"""

import random
import torch
from torch.utils.data import Dataset


OPS = ['+', '*']


class SkeletonDataset(Dataset):
    """
    Dataset for skeleton generation.

    Vocabulary: <PAD>, <MASK>, <BOS>, <EOS>, <DIGIT>, (, ), +, *, =

    Example:
        Expression: (+(12)3)=15
        Skeleton:   <BOS> ( + ( <DIGIT> <DIGIT> ) <DIGIT> ) = <DIGIT> <DIGIT> <EOS>
    """

    def __init__(
        self,
        num_samples: int,
        max_depth: int = 6,
        min_depth: int = 1,
        max_int: int = 10,
        max_len: int = 64,
        sigma_min: float = 0.1,
        sigma_max: float = 0.9,
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_int = max_int
        self.max_len = max_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Skeleton vocabulary (no individual digits)
        self.vocab = [
            "<PAD>", "<MASK>", "<BOS>", "<EOS>", "<DIGIT>",
            "(", ")", "+", "*", "="
        ]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.mask_id = self.stoi["<MASK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.digit_id = self.stoi["<DIGIT>"]

        self.special_tokens = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        # Pre-generate expressions
        self.data = []
        for _ in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            expr, val = self._generate_expression(depth)
            text = f"{expr}={val}"
            self.data.append(text)

    def _generate_expression(self, depth: int):
        if depth == 0 or (depth > 0 and random.random() < 0.2):
            val = random.randint(0, self.max_int)
            return str(val), val

        op = random.choice(OPS)
        left_depth = depth - 1
        right_depth = random.randint(0, depth - 1)

        if random.random() < 0.5:
            left_depth, right_depth = right_depth, left_depth

        left_str, left_val = self._generate_expression(left_depth)
        right_str, right_val = self._generate_expression(right_depth)

        expr = f"({op}{left_str}{right_str})"

        if op == '+':
            val = left_val + right_val
        else:
            val = left_val * right_val

        return expr, val

    def _to_skeleton(self, text: str) -> list:
        """Convert expression to skeleton tokens."""
        tokens = [self.bos_id]
        for char in text:
            if char in '0123456789':
                tokens.append(self.digit_id)
            elif char in self.stoi:
                tokens.append(self.stoi[char])
            # Skip spaces
        tokens.append(self.eos_id)
        return tokens

    def _tokenize(self, tokens: list) -> torch.Tensor:
        """Pad/truncate token list."""
        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            clean: Skeleton tokens
            corrupted: Masked skeleton
            sigma: Noise level
        """
        text = self.data[idx]
        skeleton_tokens = self._to_skeleton(text)
        clean = self._tokenize(skeleton_tokens)

        # Sample sigma and corrupt
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        corrupted = self._mask_sequence(clean, sigma)

        return clean, corrupted, torch.tensor(sigma, dtype=torch.float32)

    def _mask_sequence(self, seq: torch.Tensor, sigma: float) -> torch.Tensor:
        """Mask non-special tokens with probability sigma."""
        corrupted = seq.clone()
        for i in range(len(seq)):
            if seq[i].item() not in self.special_tokens and random.random() < sigma:
                corrupted[i] = self.mask_id
        return corrupted

    def decode(self, tokens: torch.Tensor) -> str:
        chars = []
        for t in tokens.tolist():
            if t == self.pad_id:
                break
            chars.append(self.itos.get(t, '?'))
        return ''.join(chars)


class DigitFillerDataset(Dataset):
    """
    Dataset for digit filling.

    Input: Skeleton with <DIGIT> placeholders
    Output: Complete expression with actual digits

    Full vocabulary includes digits.
    """

    def __init__(
        self,
        num_samples: int,
        max_depth: int = 6,
        min_depth: int = 1,
        max_int: int = 10,
        max_len: int = 64,
        sigma_min: float = 0.1,
        sigma_max: float = 0.9,
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_int = max_int
        self.max_len = max_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Full vocabulary with digits
        self.vocab = [
            "<PAD>", "<MASK>", "<BOS>", "<EOS>", "<DIGIT>",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "(", ")", "+", "*", "="
        ]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.mask_id = self.stoi["<MASK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.digit_placeholder_id = self.stoi["<DIGIT>"]

        self.special_tokens = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        # Pre-generate expressions
        self.data = []
        for _ in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            expr, val = self._generate_expression(depth)
            text = f"{expr}={val}"
            self.data.append(text)

    def _generate_expression(self, depth: int):
        if depth == 0 or (depth > 0 and random.random() < 0.2):
            val = random.randint(0, self.max_int)
            return str(val), val

        op = random.choice(OPS)
        left_depth = depth - 1
        right_depth = random.randint(0, depth - 1)

        if random.random() < 0.5:
            left_depth, right_depth = right_depth, left_depth

        left_str, left_val = self._generate_expression(left_depth)
        right_str, right_val = self._generate_expression(right_depth)

        expr = f"({op}{left_str}{right_val})"

        if op == '+':
            val = left_val + right_val
        else:
            val = left_val * right_val

        return expr, val

    def _tokenize_full(self, text: str) -> torch.Tensor:
        """Tokenize with actual digits."""
        tokens = [self.bos_id]
        for char in text:
            if char in self.stoi:
                tokens.append(self.stoi[char])
        tokens.append(self.eos_id)

        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens, dtype=torch.long)

    def _to_skeleton_input(self, full_tokens: torch.Tensor) -> torch.Tensor:
        """Replace digit tokens with <DIGIT> placeholder."""
        skeleton = full_tokens.clone()
        digit_ids = {self.stoi[d] for d in '0123456789'}
        for i in range(len(skeleton)):
            if skeleton[i].item() in digit_ids:
                skeleton[i] = self.digit_placeholder_id
        return skeleton

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            clean: Full expression with digits
            skeleton: Skeleton with <DIGIT> placeholders (input)
            sigma: Always 0 (skeleton is the conditioning)
        """
        text = self.data[idx]
        clean = self._tokenize_full(text)
        skeleton = self._to_skeleton_input(clean)

        # Sigma is 0 because skeleton provides full structure
        sigma = torch.tensor(0.0, dtype=torch.float32)

        return clean, skeleton, sigma

    def decode(self, tokens: torch.Tensor) -> str:
        chars = []
        for t in tokens.tolist():
            if t == self.pad_id:
                break
            chars.append(self.itos.get(t, '?'))
        return ''.join(chars)


def collate_fn(batch):
    """Collate function for DataLoader."""
    clean = torch.stack([b[0] for b in batch])
    corrupted = torch.stack([b[1] for b in batch])
    sigma = torch.stack([b[2] for b in batch])
    return clean, corrupted, sigma


if __name__ == "__main__":
    print("=" * 60)
    print("Skeleton Dataset Test")
    print("=" * 60)

    skel_ds = SkeletonDataset(num_samples=5, max_depth=3)
    print(f"Vocab size: {len(skel_ds.vocab)}")
    print(f"Vocab: {skel_ds.vocab}")
    print()

    for i in range(3):
        clean, corrupted, sigma = skel_ds[i]
        print(f"Example {i}:")
        print(f"  Expression: {skel_ds.data[i]}")
        print(f"  Skeleton:   {skel_ds.decode(clean)}")
        print(f"  Corrupted:  {skel_ds.decode(corrupted)}")
        print(f"  Sigma:      {sigma.item():.2f}")
        print()

    print("=" * 60)
    print("Digit Filler Dataset Test")
    print("=" * 60)

    filler_ds = DigitFillerDataset(num_samples=5, max_depth=3)
    print(f"Vocab size: {len(filler_ds.vocab)}")
    print(f"Vocab: {filler_ds.vocab}")
    print()

    for i in range(3):
        clean, skeleton, sigma = filler_ds[i]
        print(f"Example {i}:")
        print(f"  Expression: {filler_ds.data[i]}")
        print(f"  Full:       {filler_ds.decode(clean)}")
        print(f"  Skeleton:   {filler_ds.decode(skeleton)}")
        print()
