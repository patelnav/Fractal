"""
Hierarchical BPE Tokenizer for Phase 4: Fractal Engine

Creates a 2-level recursive BPE hierarchy:
- Level 2 (Fine): Characters (~65 tokens)
- Level 1 (Chunks): BPE on characters → 1024 tokens (e.g., "The", "ing", " bear")
- Level 0 (Roots): BPE on Level 1 tokens → 1024 tokens (e.g., "The king", "Exeunt")

This ensures semantic density at every level, not arbitrary chunking.
The key insight: Roots are BPE tokens OF BPE tokens - dense semantic concepts.
"""

import pickle
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch


@dataclass
class HierarchicalBPEConfig:
    """Configuration for 2-level hierarchical BPE."""
    # Level 1: Characters -> Chunks (like standard BPE)
    chunk_vocab_size: int = 1024  # Number of chunk tokens
    max_chunk_len: int = 16  # Max characters per chunk
    min_chunk_freq: int = 2  # Minimum frequency for chunk merges

    # Level 0: Chunks -> Roots (BPE on BPE tokens)
    root_vocab_size: int = 1024  # Number of root tokens
    root_expansion_size: int = 4  # Each root expands to N chunks
    min_root_freq: int = 2  # Minimum frequency for root merges

    # Padding
    pad_char: str = '\x00'  # Padding character


class HierarchicalBPE:
    """
    Two-level hierarchical BPE tokenizer.

    Level 2 (chars): Single characters
    Level 1 (chunks): BPE tokens over characters
    Level 0 (roots): BPE tokens over chunks

    This creates semantic units at every level.
    """

    def __init__(self, config: HierarchicalBPEConfig):
        self.config = config

        # Character level (Level 2)
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Chunk level (Level 1) - BPE over characters
        self.chunk_merges: Dict[Tuple[int, int], int] = {}
        self.chunk_vocab: Dict[int, bytes] = {}  # chunk_id -> bytes
        self.num_base_chars: int = 0

        # Root level (Level 0) - BPE over chunks
        self.root_merges: Dict[Tuple[int, int], int] = {}
        self.root_vocab: Dict[int, List[int]] = {}  # root_id -> list of chunk_ids
        self.num_base_chunks: int = 0

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
        """
        Train hierarchical BPE on text.

        1. Build character vocabulary
        2. Train Level 1 BPE (chars -> chunks)
        3. Encode full text with Level 1
        4. Train Level 0 BPE (chunks -> roots)
        """
        print("=" * 60)
        print("TRAINING HIERARCHICAL BPE")
        print("=" * 60)

        # ==== LEVEL 2: Character Vocabulary ====
        print("\n--- Level 2: Building character vocabulary ---")
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.num_base_chars = len(chars)

        # Initialize chunk vocab with single characters
        for i, c in enumerate(chars):
            self.chunk_vocab[i] = c.encode('utf-8')

        print(f"  Character vocabulary: {self.num_base_chars} tokens")

        # ==== LEVEL 1: BPE over Characters (Chunks) ====
        print(f"\n--- Level 1: Training chunk BPE ({self.config.chunk_vocab_size} vocab) ---")

        # Convert text to character ids
        ids = [self.char_to_id[c] for c in text]

        num_chunk_merges = self.config.chunk_vocab_size - self.num_base_chars
        print(f"  Performing {num_chunk_merges} merges...")

        for merge_idx in range(num_chunk_merges):
            stats = self._get_stats(ids)
            if not stats:
                print(f"  No more pairs to merge at iteration {merge_idx}")
                break

            top_pair = max(stats.keys(), key=lambda p: stats[p])
            if stats[top_pair] < self.config.min_chunk_freq:
                print(f"  Stopping: frequency {stats[top_pair]} below threshold")
                break

            new_id = self.num_base_chars + merge_idx
            self.chunk_merges[top_pair] = new_id
            self.chunk_vocab[new_id] = self.chunk_vocab[top_pair[0]] + self.chunk_vocab[top_pair[1]]

            ids = self._merge(ids, top_pair, new_id)

            if merge_idx % 200 == 0:
                token_str = self.chunk_vocab[new_id].decode('utf-8', errors='replace')
                print(f"    Merge {merge_idx}: '{repr(token_str)}' (freq={stats[top_pair]})")

        self.num_base_chunks = len(self.chunk_vocab)
        print(f"  Final chunk vocabulary: {self.num_base_chunks} tokens")

        # ==== LEVEL 0: BPE over Chunks (Roots) ====
        print(f"\n--- Level 0: Training root BPE ({self.config.root_vocab_size} vocab) ---")

        # Encode full text with Level 1 BPE
        chunk_ids = self.encode_to_chunks(text)
        print(f"  Text encoded to {len(chunk_ids):,} chunk tokens")

        # Initialize root vocab (each root starts as a single chunk)
        for i in range(self.num_base_chunks):
            self.root_vocab[i] = [i]  # Root i = just chunk i

        num_root_merges = self.config.root_vocab_size
        print(f"  Performing {num_root_merges} merges...")

        root_ids = list(chunk_ids)  # Copy

        for merge_idx in range(num_root_merges):
            stats = self._get_stats(root_ids)
            if not stats:
                print(f"  No more pairs to merge at iteration {merge_idx}")
                break

            top_pair = max(stats.keys(), key=lambda p: stats[p])
            if stats[top_pair] < self.config.min_root_freq:
                print(f"  Stopping: frequency {stats[top_pair]} below threshold")
                break

            new_id = self.num_base_chunks + merge_idx
            self.root_merges[top_pair] = new_id

            # Root vocab: concatenate the chunk lists
            self.root_vocab[new_id] = self.root_vocab[top_pair[0]] + self.root_vocab[top_pair[1]]

            root_ids = self._merge(root_ids, top_pair, new_id)

            if merge_idx % 200 == 0:
                # Decode root to text for display
                root_text = self.decode_root(new_id)
                print(f"    Merge {merge_idx}: '{repr(root_text[:30])}' (freq={stats[top_pair]})")

        num_roots = len([k for k in self.root_vocab.keys() if k >= self.num_base_chunks])
        print(f"  Final root vocabulary: {num_roots} merged + {self.num_base_chunks} base = {len(self.root_vocab)} total")

        print("\n" + "=" * 60)
        print("HIERARCHICAL BPE TRAINING COMPLETE")
        print("=" * 60)

    def encode_to_chunks(self, text: str) -> List[int]:
        """Encode text to Level 1 chunk ids."""
        ids = [self.char_to_id[c] for c in text]
        for pair, new_id in self.chunk_merges.items():
            ids = self._merge(ids, pair, new_id)
        return ids

    def encode_to_roots(self, text: str) -> List[int]:
        """Encode text to Level 0 root ids."""
        chunk_ids = self.encode_to_chunks(text)
        root_ids = list(chunk_ids)
        for pair, new_id in self.root_merges.items():
            root_ids = self._merge(root_ids, pair, new_id)
        return root_ids

    def decode_chunks(self, chunk_ids: List[int]) -> str:
        """Decode chunk ids to text."""
        return b''.join(self.chunk_vocab[i] for i in chunk_ids).decode('utf-8')

    def decode_root(self, root_id: int) -> str:
        """Decode a single root id to text."""
        chunk_ids = self.root_vocab[root_id]
        return self.decode_chunks(chunk_ids)

    def get_root_chunks(self, root_id: int) -> List[int]:
        """Get the chunk ids that a root expands to."""
        return self.root_vocab[root_id]

    def get_chunk_chars(self, chunk_id: int) -> str:
        """Get the character string for a chunk."""
        return self.chunk_vocab[chunk_id].decode('utf-8')

    def get_chunk_char_ids(self, chunk_id: int) -> List[int]:
        """Get the character ids for a chunk."""
        chars = self.get_chunk_chars(chunk_id)
        return [self.char_to_id[c] for c in chars]


@dataclass
class FractalDataset:
    """
    Dataset for hierarchical fractal training.

    Contains samples at two levels:
    - Level 0: root_id -> [chunk_id_1, chunk_id_2, chunk_id_3, chunk_id_4]
    - Level 1: chunk_id -> [char_id_1, char_id_2, ..., char_id_n]

    Both levels have energy training data (correct and wrong pairs).
    """
    # Level 0 (Root -> Chunks)
    root_ids: torch.Tensor  # (N0,) root token ids
    root_expansions: torch.Tensor  # (N0, 4) chunk id expansions (padded)
    root_expansion_lens: torch.Tensor  # (N0,) actual expansion lengths

    # Level 1 (Chunk -> Chars)
    chunk_ids: torch.Tensor  # (N1,) chunk token ids
    chunk_chars: torch.Tensor  # (N1, max_char_len) character id sequences
    chunk_char_lens: torch.Tensor  # (N1,) actual character lengths

    # Vocabulary info
    num_chars: int
    num_chunks: int
    num_roots: int
    max_chunk_len: int  # Max chars per chunk
    expansion_size: int  # Root expansion size (4)
    pad_char_id: int
    pad_chunk_id: int

    def __len__(self):
        return len(self.root_ids)


def build_fractal_dataset(
    text_path: str,
    config: HierarchicalBPEConfig,
    save_path: str = None
) -> Tuple[HierarchicalBPE, FractalDataset]:
    """
    Build the fractal dataset from text.

    Creates training data for both levels of the hierarchy.
    """
    print(f"Loading text from {text_path}...")
    text = Path(text_path).read_text()
    print(f"  {len(text):,} characters")

    # Train hierarchical BPE
    tokenizer = HierarchicalBPE(config)
    tokenizer.train(text)

    # === Build Level 1 Dataset (Chunk -> Chars) ===
    print("\n--- Building Level 1 Dataset (Chunk -> Chars) ---")

    chunk_ids_list = []
    chunk_chars_list = []
    chunk_lens_list = []

    pad_char_id = tokenizer.num_base_chars  # Use next id as pad

    for chunk_id in range(tokenizer.num_base_chunks):
        chars = tokenizer.get_chunk_chars(chunk_id)
        char_ids = tokenizer.get_chunk_char_ids(chunk_id)

        if len(char_ids) > config.max_chunk_len:
            continue  # Skip too-long chunks

        chunk_ids_list.append(chunk_id)

        # Pad to max length
        padded = char_ids + [pad_char_id] * (config.max_chunk_len - len(char_ids))
        chunk_chars_list.append(padded)
        chunk_lens_list.append(len(char_ids))

    print(f"  {len(chunk_ids_list)} chunk samples")

    # Also add samples from actual text usage (frequency-weighted)
    chunk_seq = tokenizer.encode_to_chunks(text)
    train_chunk_ids = []
    train_chunk_chars = []
    train_chunk_lens = []

    for chunk_id in chunk_seq:
        char_ids = tokenizer.get_chunk_char_ids(chunk_id)
        if len(char_ids) > config.max_chunk_len:
            continue

        train_chunk_ids.append(chunk_id)
        padded = char_ids + [pad_char_id] * (config.max_chunk_len - len(char_ids))
        train_chunk_chars.append(padded)
        train_chunk_lens.append(len(char_ids))

    print(f"  + {len(train_chunk_ids):,} frequency-weighted samples")

    # === Build Level 0 Dataset (Root -> Chunks) ===
    print("\n--- Building Level 0 Dataset (Root -> Chunks) ---")

    root_ids_list = []
    root_expansions_list = []
    root_lens_list = []

    pad_chunk_id = tokenizer.num_base_chunks  # Use next id as pad
    expansion_size = config.root_expansion_size

    # Collect roots that expand to exactly `expansion_size` chunks or less
    for root_id, chunk_list in tokenizer.root_vocab.items():
        if len(chunk_list) > expansion_size:
            continue  # Skip roots that expand to more than 4 chunks
        if len(chunk_list) == 0:
            continue

        root_ids_list.append(root_id)

        # Pad to expansion_size
        padded = chunk_list[:expansion_size] + [pad_chunk_id] * (expansion_size - len(chunk_list))
        root_expansions_list.append(padded)
        root_lens_list.append(len(chunk_list))

    print(f"  {len(root_ids_list)} root samples (expansion <= {expansion_size})")

    # Also add frequency-weighted samples from text
    root_seq = tokenizer.encode_to_roots(text)
    train_root_ids = []
    train_root_expansions = []
    train_root_lens = []

    for root_id in root_seq:
        chunk_list = tokenizer.root_vocab[root_id]
        if len(chunk_list) > expansion_size:
            continue
        if len(chunk_list) == 0:
            continue

        train_root_ids.append(root_id)
        padded = chunk_list[:expansion_size] + [pad_chunk_id] * (expansion_size - len(chunk_list))
        train_root_expansions.append(padded)
        train_root_lens.append(len(chunk_list))

    print(f"  + {len(train_root_ids):,} frequency-weighted samples")

    # Create dataset
    dataset = FractalDataset(
        # Level 0
        root_ids=torch.tensor(train_root_ids, dtype=torch.long),
        root_expansions=torch.tensor(train_root_expansions, dtype=torch.long),
        root_expansion_lens=torch.tensor(train_root_lens, dtype=torch.long),
        # Level 1
        chunk_ids=torch.tensor(train_chunk_ids, dtype=torch.long),
        chunk_chars=torch.tensor(train_chunk_chars, dtype=torch.long),
        chunk_char_lens=torch.tensor(train_chunk_lens, dtype=torch.long),
        # Vocab info
        num_chars=tokenizer.num_base_chars,
        num_chunks=tokenizer.num_base_chunks,
        num_roots=len(tokenizer.root_vocab),
        max_chunk_len=config.max_chunk_len,
        expansion_size=expansion_size,
        pad_char_id=pad_char_id,
        pad_chunk_id=pad_chunk_id
    )

    if save_path:
        print(f"\nSaving to {save_path}...")
        save_data = {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'config': config
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print("  Saved!")

    return tokenizer, dataset


def load_fractal_dataset(path: str) -> Tuple[HierarchicalBPE, FractalDataset, HierarchicalBPEConfig]:
    """Load saved tokenizer and dataset."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['tokenizer'], data['dataset'], data['config']


def print_fractal_stats(tokenizer: HierarchicalBPE, dataset: FractalDataset):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("FRACTAL DATASET STATISTICS")
    print("=" * 60)

    print(f"\nVocabulary:")
    print(f"  Level 2 (Characters): {dataset.num_chars}")
    print(f"  Level 1 (Chunks):     {dataset.num_chunks}")
    print(f"  Level 0 (Roots):      {dataset.num_roots}")

    print(f"\nLevel 1 (Chunk -> Chars):")
    print(f"  Total samples: {len(dataset.chunk_ids):,}")
    print(f"  Max char length: {dataset.max_chunk_len}")
    lens = dataset.chunk_char_lens.tolist()
    avg_len = sum(lens) / len(lens)
    print(f"  Avg char length: {avg_len:.1f}")

    print(f"\nLevel 0 (Root -> Chunks):")
    print(f"  Total samples: {len(dataset.root_ids):,}")
    print(f"  Expansion size: {dataset.expansion_size}")
    lens = dataset.root_expansion_lens.tolist()
    avg_len = sum(lens) / len(lens)
    print(f"  Avg expansion length: {avg_len:.1f}")

    print(f"\nSample Level 1 (Chunk -> Chars):")
    for i in range(min(5, len(dataset.chunk_ids))):
        chunk_id = dataset.chunk_ids[i].item()
        chars = tokenizer.get_chunk_chars(chunk_id)
        char_ids = dataset.chunk_chars[i, :dataset.chunk_char_lens[i]].tolist()
        print(f"  Chunk {chunk_id}: '{repr(chars)}' -> {char_ids}")

    print(f"\nSample Level 0 (Root -> Chunks):")
    for i in range(min(5, len(dataset.root_ids))):
        root_id = dataset.root_ids[i].item()
        exp_len = dataset.root_expansion_lens[i].item()
        chunk_ids = dataset.root_expansions[i, :exp_len].tolist()
        root_text = tokenizer.decode_root(root_id)
        chunk_texts = [tokenizer.get_chunk_chars(c) for c in chunk_ids]
        print(f"  Root {root_id}: '{repr(root_text[:40])}' -> {chunk_ids}")
        print(f"    = {[repr(t) for t in chunk_texts]}")


if __name__ == "__main__":
    # Default paths
    text_path = "data/tinyshakespeare.txt"
    save_path = "data/fractal_hierarchy.pkl"

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
    config = HierarchicalBPEConfig(
        chunk_vocab_size=1024,
        max_chunk_len=16,
        min_chunk_freq=2,
        root_vocab_size=1024,
        root_expansion_size=4,
        min_root_freq=2
    )

    tokenizer, dataset = build_fractal_dataset(
        text_path=text_path,
        config=config,
        save_path=save_path
    )

    print_fractal_stats(tokenizer, dataset)

    # Test roundtrip
    print("\n" + "=" * 60)
    print("ROUNDTRIP TEST")
    print("=" * 60)
    test_text = "First Citizen:\nBefore we proceed any further, hear me speak."

    # Encode to chunks
    chunk_ids = tokenizer.encode_to_chunks(test_text)
    decoded = tokenizer.decode_chunks(chunk_ids)
    print(f"\nOriginal:    {repr(test_text)}")
    print(f"Chunk IDs:   {chunk_ids[:15]}... ({len(chunk_ids)} tokens)")
    print(f"Decoded:     {repr(decoded)}")
    print(f"Match:       {test_text == decoded}")

    # Encode to roots
    root_ids = tokenizer.encode_to_roots(test_text)
    print(f"\nRoot IDs:    {root_ids[:10]}... ({len(root_ids)} tokens)")

    # Decode roots back
    all_chunks = []
    for rid in root_ids:
        all_chunks.extend(tokenizer.get_root_chunks(rid))
    decoded_from_roots = tokenizer.decode_chunks(all_chunks)
    print(f"Via roots:   {repr(decoded_from_roots)}")
    print(f"Match:       {test_text == decoded_from_roots}")
