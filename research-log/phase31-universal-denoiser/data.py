"""
Dataset for Universal Denoiser training.

Returns (clean, corrupted, sigma) triples for denoising training.
Extends Phase 30's FractalMathDataset with corruption.
"""

import random
import torch
from torch.utils.data import Dataset
from mutations import corrupt_sequence, mask_sequence


OPS = ['+', '*']


class UniversalDenoiserDataset(Dataset):
    """
    Dataset that generates recursive arithmetic expressions
    and returns (clean, corrupted, sigma) triples.
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
        corruption_mode: str = 'mixed',  # 'mixed', 'mask_only'
        high_noise_bias: float = 0.0,  # Fraction of samples to bias toward σ ∈ [0.8, 1.0]
        ar_prefix_ratio: float = 0.0,  # Fraction of samples with AR-prefix corruption
        ar_prefix_len: int = 3,  # Number of clean tokens to keep as AR prefix
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_int = max_int
        self.max_len = max_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.corruption_mode = corruption_mode
        self.high_noise_bias = high_noise_bias
        self.ar_prefix_ratio = ar_prefix_ratio
        self.ar_prefix_len = ar_prefix_len

        # Vocabulary
        self.vocab = [
            "<PAD>", "<MASK>", "<BOS>", "<EOS>",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "(", ")", "+", "*", "="
        ]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.mask_id = self.stoi["<MASK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]

        self.special_tokens = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        # Pre-generate expressions
        self.data = []
        for _ in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            expr, val = self._generate_expression(depth)
            text = f"{expr}={val}"
            self.data.append(text)

    def _generate_expression(self, depth: int):
        """Generate a recursive arithmetic expression."""
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

        expr = f"({op} {left_str} {right_str})"

        if op == '+':
            val = left_val + right_val
        elif op == '*':
            val = left_val * right_val
        else:
            val = 0

        return expr, val

    def _tokenize(self, text: str) -> torch.Tensor:
        """Convert text to token ids."""
        tokens = [self.bos_id]
        for char in text:
            if char == ' ':
                continue
            if char in self.stoi:
                tokens.append(self.stoi[char])
        tokens.append(self.eos_id)

        # Pad/truncate
        if len(tokens) < self.max_len:
            tokens += [self.pad_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            clean: (T,) clean token sequence
            corrupted: (T,) corrupted token sequence
            sigma: scalar noise level
        """
        text = self.data[idx]
        clean = self._tokenize(text)

        # AR-prefix mode: keep first N tokens clean, mask the rest
        # This trains the model to work with hybrid AR+MaskGIT generation
        if self.ar_prefix_ratio > 0 and random.random() < self.ar_prefix_ratio:
            corrupted = self._ar_prefix_corrupt(clean)
            # For AR-prefix, sigma represents the fraction masked (high)
            # Find actual sequence length (non-PAD)
            seq_len = (clean != self.pad_id).sum().item()
            prefix_len = min(self.ar_prefix_len + 1, seq_len - 1)  # +1 for BOS
            sigma = 1.0 - (prefix_len / seq_len)  # ~0.8-0.95 depending on length
        else:
            # Normal corruption mode
            # Sample noise level (with optional high-noise bias for generation training)
            if self.high_noise_bias > 0 and random.random() < self.high_noise_bias:
                # High-noise regime: σ ∈ [0.8, sigma_max] to train on near-full masking
                sigma = random.uniform(0.8, self.sigma_max)
            else:
                # Normal regime
                sigma = random.uniform(self.sigma_min, self.sigma_max)

            # Apply corruption
            if self.corruption_mode == 'mask_only':
                corrupted = mask_sequence(
                    clean, sigma, self.mask_id, self.special_tokens, self.pad_id
                )
            else:  # 'mixed'
                corrupted = corrupt_sequence(
                    clean, sigma,
                    vocab_size=len(self.vocab),
                    special_tokens=self.special_tokens,
                    mask_token_id=self.mask_id,
                    pad_token_id=self.pad_id,
                    max_len=self.max_len,
                )

        return clean, corrupted, torch.tensor(sigma, dtype=torch.float32)

    def _ar_prefix_corrupt(self, clean: torch.Tensor) -> torch.Tensor:
        """
        AR-prefix corruption: keep first N tokens clean, mask the rest.

        This simulates the input distribution during hybrid generation:
        - BOS + first ar_prefix_len tokens are kept (AR-generated prefix)
        - Remaining tokens (except EOS) are masked
        - EOS is kept

        Example with ar_prefix_len=3:
        Clean:     <BOS> ( + 1 0 ) = 1 0 <EOS> <PAD> ...
        Corrupted: <BOS> ( + 1 <MASK> <MASK> <MASK> <MASK> <MASK> <EOS> <PAD> ...
        """
        corrupted = clean.clone()
        seq_len = (clean != self.pad_id).sum().item()

        # Prefix: BOS + ar_prefix_len tokens
        # Keep positions [0, ar_prefix_len] clean (BOS at 0)
        prefix_end = min(self.ar_prefix_len + 1, seq_len - 1)  # +1 for BOS, -1 to leave room for EOS

        # Find EOS position
        eos_pos = seq_len - 1

        # Mask everything between prefix and EOS
        for i in range(prefix_end, eos_pos):
            corrupted[i] = self.mask_id

        return corrupted

    def decode(self, tokens: torch.Tensor) -> str:
        """Convert token ids back to string."""
        chars = []
        for t in tokens.tolist():
            if t == self.pad_id:
                break
            if t in [self.bos_id, self.eos_id, self.mask_id]:
                chars.append(self.itos[t])
            else:
                chars.append(self.itos[t])
        return ''.join(chars)


class EvalDataset(Dataset):
    """
    Evaluation dataset with fixed expressions for reproducible benchmarking.
    Generates clean sequences only (corruption applied at eval time).
    """

    def __init__(
        self,
        num_samples: int = 500,
        max_depth: int = 6,
        min_depth: int = 1,
        max_int: int = 10,
        max_len: int = 64,
        seed: int = 42,
    ):
        random.seed(seed)

        self.max_len = max_len
        self.vocab = [
            "<PAD>", "<MASK>", "<BOS>", "<EOS>",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "(", ")", "+", "*", "="
        ]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.mask_id = self.stoi["<MASK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]

        self.special_tokens = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        # Generate fixed test set
        self.data = []
        self.texts = []
        for _ in range(num_samples):
            depth = random.randint(min_depth, max_depth)
            expr, val = self._generate_expression(depth)
            text = f"{expr}={val}"
            self.texts.append(text)
            self.data.append(self._tokenize(text))

        random.seed()  # Reset seed

    def _generate_expression(self, depth: int):
        if depth == 0 or (depth > 0 and random.random() < 0.2):
            val = random.randint(0, 10)
            return str(val), val

        op = random.choice(OPS)
        left_depth = depth - 1
        right_depth = random.randint(0, depth - 1)

        if random.random() < 0.5:
            left_depth, right_depth = right_depth, left_depth

        left_str, left_val = self._generate_expression(left_depth)
        right_str, right_val = self._generate_expression(right_depth)

        expr = f"({op} {left_str} {right_str})"

        if op == '+':
            val = left_val + right_val
        elif op == '*':
            val = left_val * right_val
        else:
            val = 0

        return expr, val

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = [self.bos_id]
        for char in text:
            if char == ' ':
                continue
            if char in self.stoi:
                tokens.append(self.stoi[char])
        tokens.append(self.eos_id)

        if len(tokens) < self.max_len:
            tokens += [self.pad_id] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    # Test dataset
    ds = UniversalDenoiserDataset(num_samples=10, max_depth=4, corruption_mode='mixed')

    print(f"Vocab size: {len(ds.vocab)}")
    print(f"Vocab: {ds.vocab}")
    print()

    for i in range(5):
        clean, corrupted, sigma = ds[i]
        print(f"Example {i}:")
        print(f"  Text: {ds.data[i]}")
        print(f"  Clean:     {ds.decode(clean)}")
        print(f"  Corrupted: {ds.decode(corrupted)}")
        print(f"  Sigma: {sigma.item():.2f}")
        print()

    # Test eval dataset
    print("\nEval Dataset:")
    eval_ds = EvalDataset(num_samples=5, seed=42)
    for i in range(3):
        tokens = eval_ds[i]
        print(f"  {i}: {eval_ds.decode(tokens)}")
