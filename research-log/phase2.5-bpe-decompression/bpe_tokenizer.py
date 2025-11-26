"""
BPE Tokenizer for Phase 2.5: Token Decompression Test

This creates a clean hierarchy with 0% UNK tokens:
- Train BPE on Shakespeare text
- Each BPE token maps to its character sequence
- Perfect coverage guaranteed
"""

import pickle
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch


@dataclass
class BPEConfig:
    vocab_size: int = 512  # Number of BPE tokens (merges)
    max_token_len: int = 16  # Max characters per BPE token (for padding)
    min_frequency: int = 2  # Minimum frequency for merge


class MinimalBPE:
    """
    Minimal BPE implementation inspired by minbpe.
    Trains directly on text, produces token->chars mapping.
    """

    def __init__(self, config: BPEConfig):
        self.config = config
        self.merges: Dict[Tuple[int, int], int] = {}  # (pair) -> new_token_id
        self.vocab: Dict[int, bytes] = {}  # token_id -> bytes
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def _get_stats(self, ids: List[int]) -> Counter:
        """Count consecutive pairs."""
        counts = Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace all occurrences of pair with new_id."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str) -> None:
        """Train BPE on text."""
        # Build character vocabulary (base tokens)
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        num_base_tokens = len(chars)

        # Initialize vocab with single characters (as bytes for consistency)
        for i, c in enumerate(chars):
            self.vocab[i] = c.encode('utf-8')

        # Convert text to initial token ids
        ids = [self.char_to_id[c] for c in text]

        # Perform merges until we reach desired vocab size
        num_merges = self.config.vocab_size - num_base_tokens
        print(f"Training BPE: {num_base_tokens} base tokens, {num_merges} merges to perform")

        for merge_idx in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                print(f"No more pairs to merge at iteration {merge_idx}")
                break

            # Find most frequent pair
            top_pair = max(stats.keys(), key=lambda p: stats[p])

            if stats[top_pair] < self.config.min_frequency:
                print(f"Stopping: top pair frequency {stats[top_pair]} below threshold")
                break

            # Create new token
            new_id = num_base_tokens + merge_idx
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            # Apply merge
            ids = self._merge(ids, top_pair, new_id)

            if merge_idx % 100 == 0:
                print(f"  Merge {merge_idx}: {self.vocab[top_pair[0]]} + {self.vocab[top_pair[1]]} -> token {new_id} (freq={stats[top_pair]})")

        print(f"Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        # Start with character ids
        ids = [self.char_to_id[c] for c in text]

        # Apply merges in order learned
        for pair, new_id in self.merges.items():
            ids = self._merge(ids, pair, new_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        return b''.join(self.vocab[i] for i in ids).decode('utf-8')

    def get_token_chars(self, token_id: int) -> str:
        """Get the character string for a token."""
        return self.vocab[token_id].decode('utf-8')

    def get_token_char_ids(self, token_id: int) -> List[int]:
        """Get the character ids for a token."""
        chars = self.get_token_chars(token_id)
        return [self.char_to_id[c] for c in chars]


@dataclass
class DecompressionDataset:
    """
    Dataset for BPE token -> character decompression.

    Each sample is:
    - condition: BPE token id (single int)
    - target: character ids (padded to max_len)
    - target_mask: which positions are real vs padding
    """
    bpe_token_ids: torch.Tensor  # (N,) - BPE token ids as conditions
    char_sequences: torch.Tensor  # (N, max_len) - character id sequences
    sequence_lengths: torch.Tensor  # (N,) - actual lengths before padding

    # Vocabulary info
    num_chars: int  # Character vocabulary size
    num_bpe_tokens: int  # BPE vocabulary size
    max_len: int  # Maximum sequence length
    pad_id: int  # Padding token id

    def __len__(self):
        return len(self.bpe_token_ids)


def build_decompression_dataset(
    text_path: str,
    bpe_config: BPEConfig,
    save_path: str = None
) -> Tuple[MinimalBPE, DecompressionDataset]:
    """
    Build the decompression dataset from text.

    Returns:
        tokenizer: Trained BPE tokenizer
        dataset: DecompressionDataset with all samples
    """
    print(f"Loading text from {text_path}...")
    text = Path(text_path).read_text()
    print(f"  {len(text):,} characters")

    # Train BPE
    print("\nTraining BPE tokenizer...")
    tokenizer = MinimalBPE(bpe_config)
    tokenizer.train(text)

    # Build decompression samples
    # For each BPE token, create a sample: token_id -> char_ids
    print("\nBuilding decompression samples...")

    num_chars = len(tokenizer.char_to_id)
    num_bpe = len(tokenizer.vocab)
    max_len = bpe_config.max_token_len
    pad_id = num_chars  # Use num_chars as pad token (outside char vocab)

    # Collect unique BPE tokens and their decompositions
    bpe_ids = []
    char_seqs = []
    seq_lens = []

    for token_id in range(num_bpe):
        chars = tokenizer.get_token_chars(token_id)
        char_ids = tokenizer.get_token_char_ids(token_id)

        if len(char_ids) > max_len:
            # Skip tokens that are too long (rare edge case)
            continue

        bpe_ids.append(token_id)

        # Pad to max_len
        padded = char_ids + [pad_id] * (max_len - len(char_ids))
        char_seqs.append(padded)
        seq_lens.append(len(char_ids))

    print(f"  {len(bpe_ids)} unique BPE tokens")
    print(f"  Character vocab size: {num_chars}")
    print(f"  Max token length: {max(seq_lens)} chars")
    print(f"  Avg token length: {sum(seq_lens)/len(seq_lens):.1f} chars")

    # Also encode the full text and create samples from actual usage
    # This gives us frequency-weighted training data
    print("\nEncoding full text for training samples...")
    encoded = tokenizer.encode(text)
    print(f"  {len(encoded):,} BPE tokens in text")

    # Create training samples from text occurrences
    train_bpe_ids = []
    train_char_seqs = []
    train_seq_lens = []

    for token_id in encoded:
        char_ids = tokenizer.get_token_char_ids(token_id)
        if len(char_ids) > max_len:
            continue

        train_bpe_ids.append(token_id)
        padded = char_ids + [pad_id] * (max_len - len(char_ids))
        train_char_seqs.append(padded)
        train_seq_lens.append(len(char_ids))

    print(f"  {len(train_bpe_ids):,} training samples")

    dataset = DecompressionDataset(
        bpe_token_ids=torch.tensor(train_bpe_ids, dtype=torch.long),
        char_sequences=torch.tensor(train_char_seqs, dtype=torch.long),
        sequence_lengths=torch.tensor(train_seq_lens, dtype=torch.long),
        num_chars=num_chars,
        num_bpe_tokens=num_bpe,
        max_len=max_len,
        pad_id=pad_id
    )

    if save_path:
        print(f"\nSaving to {save_path}...")
        save_data = {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'config': bpe_config
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print("  Saved!")

    return tokenizer, dataset


def load_decompression_dataset(path: str) -> Tuple[MinimalBPE, DecompressionDataset, BPEConfig]:
    """Load saved tokenizer and dataset."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['tokenizer'], data['dataset'], data['config']


# --- Statistics and Visualization ---

def print_dataset_stats(tokenizer: MinimalBPE, dataset: DecompressionDataset):
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"\nVocabulary:")
    print(f"  Characters: {dataset.num_chars}")
    print(f"  BPE tokens: {dataset.num_bpe_tokens}")
    print(f"  Pad token ID: {dataset.pad_id}")

    print(f"\nSequences:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Max length: {dataset.max_len}")

    lens = dataset.sequence_lengths.tolist()
    print(f"  Length distribution:")
    for l in range(1, max(lens) + 1):
        count = lens.count(l)
        if count > 0:
            pct = count / len(lens) * 100
            bar = '#' * int(pct / 2)
            print(f"    {l:2d} chars: {count:6,} ({pct:5.1f}%) {bar}")

    print(f"\nSample tokens:")
    for i in range(min(10, dataset.num_bpe_tokens)):
        token_id = i
        chars = tokenizer.get_token_chars(token_id)
        char_ids = tokenizer.get_token_char_ids(token_id)
        print(f"  Token {token_id}: '{repr(chars)}' -> {char_ids}")


if __name__ == "__main__":
    import sys

    # Default paths
    text_path = "data/tinyshakespeare.txt"
    save_path = "data/bpe_decompression.pkl"

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print(f"Created {data_dir}/")

    # Download tinyshakespeare if needed
    if not Path(text_path).exists():
        print(f"Downloading tinyshakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, text_path)
        print(f"  Saved to {text_path}")

    # Build dataset
    config = BPEConfig(
        vocab_size=512,
        max_token_len=16,
        min_frequency=2
    )

    tokenizer, dataset = build_decompression_dataset(
        text_path=text_path,
        bpe_config=config,
        save_path=save_path
    )

    print_dataset_stats(tokenizer, dataset)

    # Test encoding/decoding roundtrip
    print("\n" + "=" * 50)
    print("ROUNDTRIP TEST")
    print("=" * 50)
    test_text = "First Citizen:\nBefore we proceed any further, hear me speak."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {repr(test_text)}")
    print(f"Encoded:  {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"Decoded:  {repr(decoded)}")
    print(f"Match:    {test_text == decoded}")
