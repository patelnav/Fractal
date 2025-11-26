"""
Fractal Shakespeare: Hierarchical Tokenization for Discrete Diffusion

Creates a 3-level hierarchy from tinyshakespeare:
- Level 2 (Fine): Characters (~65 unique)
- Level 1 (Chunks): 4-character blocks (top 2048 + <UNK>)
- Level 0 (Roots): 4-chunk blocks (top 2048 + <UNK>)

The model learns to expand:
  Root -> [Chunk, Chunk, Chunk, Chunk]
  Chunk -> [Char, Char, Char, Char]
"""

import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import torch


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical tokenization."""
    chunk_size: int = 4          # Characters per chunk
    chunks_per_root: int = 4     # Chunks per root
    max_chunk_vocab: int = 2048  # Top N chunks to keep
    max_root_vocab: int = 2048   # Top N roots to keep
    data_path: str = "data/shakespeare.txt"
    cache_path: str = "data/shakespeare_hierarchy.pkl"


class ShakespeareHierarchy:
    """
    Builds and manages the 3-level hierarchical tokenization of Shakespeare.

    Vocabulary Layout:
    - [0, char_vocab_size): Character tokens
    - [char_vocab_size, char_vocab_size + chunk_vocab_size): Chunk tokens
    - [char_vocab_size + chunk_vocab_size, ...): Root tokens

    Special tokens:
    - <UNK> exists at each level (last ID in that level's range)
    - <PAD> for padding sequences
    """

    def __init__(self, config: HierarchyConfig = None):
        self.config = config or HierarchyConfig()

        # Will be populated by build() or load()
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        self.chunk_to_id: Dict[str, int] = {}  # 4-char string -> chunk ID
        self.id_to_chunk: Dict[int, str] = {}

        self.root_to_id: Dict[Tuple[int, ...], int] = {}  # (chunk_id, chunk_id, chunk_id, chunk_id) -> root ID
        self.id_to_root: Dict[int, Tuple[int, ...]] = {}

        # Vocabulary sizes (set after build)
        self.char_vocab_size: int = 0
        self.chunk_vocab_size: int = 0
        self.root_vocab_size: int = 0

        # Offsets for unified vocabulary
        self.chunk_offset: int = 0
        self.root_offset: int = 0

        # Special token IDs (relative to their level)
        self.char_unk_id: int = 0
        self.chunk_unk_id: int = 0
        self.root_unk_id: int = 0

        # Full text converted to IDs at each level
        self.char_ids: List[int] = []
        self.chunk_ids: List[int] = []
        self.root_ids: List[int] = []

        # Training samples
        self.root_to_chunks_samples: List[Tuple[int, List[int]]] = []  # (root_id, [chunk_ids])
        self.chunk_to_chars_samples: List[Tuple[int, List[int]]] = []  # (chunk_id, [char_ids])

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size across all levels."""
        return self.char_vocab_size + self.chunk_vocab_size + self.root_vocab_size

    def build(self, text: str = None) -> 'ShakespeareHierarchy':
        """Build the hierarchy from text."""
        if text is None:
            with open(self.config.data_path, 'r', encoding='utf-8') as f:
                text = f.read()

        print(f"Building hierarchy from {len(text):,} characters...")

        # Step 1: Build character vocabulary
        self._build_char_vocab(text)

        # Step 2: Build chunk vocabulary (4-char blocks)
        self._build_chunk_vocab(text)

        # Step 3: Convert text to chunk IDs
        self._convert_to_chunk_ids(text)

        # Step 4: Build root vocabulary (4-chunk blocks)
        self._build_root_vocab()

        # Step 5: Convert chunk sequence to root IDs
        self._convert_to_root_ids()

        # Step 6: Build training samples
        self._build_training_samples(text)

        # Set offsets for unified vocabulary
        self.chunk_offset = self.char_vocab_size
        self.root_offset = self.char_vocab_size + self.chunk_vocab_size

        self._print_stats()
        return self

    def _build_char_vocab(self, text: str):
        """Build character-level vocabulary."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.char_vocab_size = len(chars)

        # No UNK at char level (we keep all characters)
        self.char_unk_id = -1  # Not used

        # Convert text to char IDs
        self.char_ids = [self.char_to_id[ch] for ch in text]

        print(f"  Character vocab: {self.char_vocab_size} tokens")

    def _build_chunk_vocab(self, text: str):
        """Build chunk vocabulary from 4-character blocks."""
        chunk_size = self.config.chunk_size

        # Count all 4-char blocks (sliding window, stride = chunk_size for non-overlapping)
        chunk_counts = Counter()
        for i in range(0, len(text) - chunk_size + 1, chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunk_counts[chunk] += 1

        # Keep top N most frequent
        top_chunks = chunk_counts.most_common(self.config.max_chunk_vocab)

        # Build vocabulary: frequent chunks get IDs 0 to N-1, UNK gets ID N
        self.chunk_to_id = {chunk: i for i, (chunk, _) in enumerate(top_chunks)}
        self.chunk_unk_id = len(top_chunks)  # UNK is the last ID
        self.chunk_vocab_size = len(top_chunks) + 1  # +1 for UNK

        self.id_to_chunk = {i: chunk for chunk, i in self.chunk_to_id.items()}
        self.id_to_chunk[self.chunk_unk_id] = "<UNK>"

        # Stats
        total_chunks = sum(chunk_counts.values())
        covered_chunks = sum(count for _, count in top_chunks)
        coverage = covered_chunks / total_chunks * 100

        print(f"  Chunk vocab: {self.chunk_vocab_size} tokens (top {self.config.max_chunk_vocab} + UNK)")
        print(f"  Chunk coverage: {coverage:.1f}% of text")

    def _convert_to_chunk_ids(self, text: str):
        """Convert text to sequence of chunk IDs."""
        chunk_size = self.config.chunk_size
        self.chunk_ids = []

        for i in range(0, len(text) - chunk_size + 1, chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunk_id = self.chunk_to_id.get(chunk, self.chunk_unk_id)
                self.chunk_ids.append(chunk_id)

        unk_count = sum(1 for cid in self.chunk_ids if cid == self.chunk_unk_id)
        print(f"  Chunk sequence length: {len(self.chunk_ids):,}")
        print(f"  UNK chunks: {unk_count:,} ({unk_count/len(self.chunk_ids)*100:.1f}%)")

    def _build_root_vocab(self):
        """Build root vocabulary from 4-chunk blocks."""
        chunks_per_root = self.config.chunks_per_root

        # Count all 4-chunk blocks
        root_counts = Counter()
        for i in range(0, len(self.chunk_ids) - chunks_per_root + 1, chunks_per_root):
            root = tuple(self.chunk_ids[i:i + chunks_per_root])
            if len(root) == chunks_per_root:
                root_counts[root] += 1

        # Keep top N most frequent
        top_roots = root_counts.most_common(self.config.max_root_vocab)

        # Build vocabulary
        self.root_to_id = {root: i for i, (root, _) in enumerate(top_roots)}
        self.root_unk_id = len(top_roots)
        self.root_vocab_size = len(top_roots) + 1

        self.id_to_root = {i: root for root, i in self.root_to_id.items()}
        self.id_to_root[self.root_unk_id] = tuple([self.chunk_unk_id] * chunks_per_root)

        # Stats
        total_roots = sum(root_counts.values())
        covered_roots = sum(count for _, count in top_roots)
        coverage = covered_roots / total_roots * 100 if total_roots > 0 else 0

        print(f"  Root vocab: {self.root_vocab_size} tokens (top {self.config.max_root_vocab} + UNK)")
        print(f"  Root coverage: {coverage:.1f}% of text")

    def _convert_to_root_ids(self):
        """Convert chunk sequence to root IDs."""
        chunks_per_root = self.config.chunks_per_root
        self.root_ids = []

        for i in range(0, len(self.chunk_ids) - chunks_per_root + 1, chunks_per_root):
            root = tuple(self.chunk_ids[i:i + chunks_per_root])
            if len(root) == chunks_per_root:
                root_id = self.root_to_id.get(root, self.root_unk_id)
                self.root_ids.append(root_id)

        unk_count = sum(1 for rid in self.root_ids if rid == self.root_unk_id)
        print(f"  Root sequence length: {len(self.root_ids):,}")
        print(f"  UNK roots: {unk_count:,} ({unk_count/len(self.root_ids)*100:.1f}%)")

    def _build_training_samples(self, text: str):
        """Build training samples for both levels."""
        chunk_size = self.config.chunk_size
        chunks_per_root = self.config.chunks_per_root

        # Root -> Chunks samples
        # Only include samples where condition (root) is NOT <UNK>
        for i, root_id in enumerate(self.root_ids):
            if root_id != self.root_unk_id:
                # Get the corresponding chunk IDs
                chunk_start = i * chunks_per_root
                chunk_ids = self.chunk_ids[chunk_start:chunk_start + chunks_per_root]
                if len(chunk_ids) == chunks_per_root:
                    self.root_to_chunks_samples.append((root_id, chunk_ids))

        # Chunk -> Chars samples
        # Only include samples where condition (chunk) is NOT <UNK>
        for i, chunk_id in enumerate(self.chunk_ids):
            if chunk_id != self.chunk_unk_id:
                # Get the corresponding characters
                char_start = i * chunk_size
                char_end = char_start + chunk_size
                if char_end <= len(text):
                    chunk_str = text[char_start:char_end]
                    char_ids = [self.char_to_id[ch] for ch in chunk_str]
                    self.chunk_to_chars_samples.append((chunk_id, char_ids))

        print(f"  Root->Chunks samples: {len(self.root_to_chunks_samples):,}")
        print(f"  Chunk->Chars samples: {len(self.chunk_to_chars_samples):,}")

    def _print_stats(self):
        """Print vocabulary statistics."""
        print(f"\n=== Vocabulary Summary ===")
        print(f"  Characters: {self.char_vocab_size} (IDs 0-{self.char_vocab_size-1})")
        print(f"  Chunks: {self.chunk_vocab_size} (IDs {self.chunk_offset}-{self.chunk_offset + self.chunk_vocab_size - 1})")
        print(f"  Roots: {self.root_vocab_size} (IDs {self.root_offset}-{self.root_offset + self.root_vocab_size - 1})")
        print(f"  Total vocab: {self.total_vocab_size}")

    def save(self, path: str = None):
        """Save the hierarchy to disk."""
        path = path or self.config.cache_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            'config': self.config,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'chunk_to_id': self.chunk_to_id,
            'id_to_chunk': self.id_to_chunk,
            'root_to_id': self.root_to_id,
            'id_to_root': self.id_to_root,
            'char_vocab_size': self.char_vocab_size,
            'chunk_vocab_size': self.chunk_vocab_size,
            'root_vocab_size': self.root_vocab_size,
            'chunk_offset': self.chunk_offset,
            'root_offset': self.root_offset,
            'char_unk_id': self.char_unk_id,
            'chunk_unk_id': self.chunk_unk_id,
            'root_unk_id': self.root_unk_id,
            'char_ids': self.char_ids,
            'chunk_ids': self.chunk_ids,
            'root_ids': self.root_ids,
            'root_to_chunks_samples': self.root_to_chunks_samples,
            'chunk_to_chars_samples': self.chunk_to_chars_samples,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved hierarchy to {path}")

    @classmethod
    def load(cls, path: str = "data/shakespeare_hierarchy.pkl") -> 'ShakespeareHierarchy':
        """Load a saved hierarchy."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        obj = cls(data['config'])
        for key, value in data.items():
            if key != 'config':
                setattr(obj, key, value)

        print(f"Loaded hierarchy from {path}")
        obj._print_stats()
        return obj

    def get_batch(
        self,
        batch_size: int,
        level: str,  # 'root_to_chunk' or 'chunk_to_char'
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training batch for the specified level.

        Returns:
            conditions: (B,) tensor of condition token IDs (global vocab)
            targets: (B, 4) tensor of target token IDs (global vocab)
        """
        if level == 'root_to_chunk':
            samples = self.root_to_chunks_samples
            cond_offset = self.root_offset
            target_offset = self.chunk_offset
        else:  # chunk_to_char
            samples = self.chunk_to_chars_samples
            cond_offset = self.chunk_offset
            target_offset = 0  # chars start at 0

        # Random sample
        batch_indices = random.choices(range(len(samples)), k=batch_size)

        conditions = []
        targets = []

        for idx in batch_indices:
            cond_id, target_ids = samples[idx]
            # Add offsets to convert to global vocab
            conditions.append(cond_id + cond_offset)
            targets.append([tid + target_offset for tid in target_ids])

        return (
            torch.tensor(conditions, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device)
        )

    def get_mixed_batch(
        self,
        batch_size: int,
        device: torch.device,
        root_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a mixed batch with both Root->Chunk and Chunk->Char samples.

        Args:
            batch_size: Total batch size
            device: Torch device
            root_ratio: Fraction of batch that is Root->Chunk samples

        Returns:
            conditions: (B,) tensor of condition token IDs
            targets: (B, 4) tensor of target token IDs
        """
        n_root = int(batch_size * root_ratio)
        n_chunk = batch_size - n_root

        # Get Root->Chunk samples
        root_conds, root_targets = self.get_batch(n_root, 'root_to_chunk', device)

        # Get Chunk->Char samples
        chunk_conds, chunk_targets = self.get_batch(n_chunk, 'chunk_to_char', device)

        # Concatenate
        conditions = torch.cat([root_conds, chunk_conds], dim=0)
        targets = torch.cat([root_targets, chunk_targets], dim=0)

        # Shuffle
        perm = torch.randperm(batch_size, device=device)
        conditions = conditions[perm]
        targets = targets[perm]

        return conditions, targets

    def decode_chars(self, char_ids: List[int]) -> str:
        """Decode character IDs to string."""
        return ''.join(self.id_to_char.get(cid, '?') for cid in char_ids)

    def decode_chunk(self, chunk_id: int) -> str:
        """Decode a chunk ID to its 4-character string."""
        # Remove offset if present
        local_id = chunk_id - self.chunk_offset if chunk_id >= self.chunk_offset else chunk_id
        return self.id_to_chunk.get(local_id, "<UNK>")

    def decode_root(self, root_id: int) -> str:
        """Decode a root ID to its 16-character string."""
        # Remove offset if present
        local_id = root_id - self.root_offset if root_id >= self.root_offset else root_id
        chunk_ids = self.id_to_root.get(local_id, tuple([self.chunk_unk_id] * 4))

        result = []
        for cid in chunk_ids:
            chunk_str = self.id_to_chunk.get(cid, "????")
            if chunk_str == "<UNK>":
                chunk_str = "????"
            result.append(chunk_str)
        return ''.join(result)

    def encode_text(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode arbitrary text to all three levels.

        Returns:
            char_ids, chunk_ids, root_ids (all with global vocab offsets)
        """
        # Pad text to multiple of 16
        pad_len = (16 - len(text) % 16) % 16
        text = text + ' ' * pad_len

        # Characters
        char_ids = [self.char_to_id.get(ch, 0) for ch in text]

        # Chunks
        chunk_ids = []
        for i in range(0, len(text), 4):
            chunk = text[i:i+4]
            cid = self.chunk_to_id.get(chunk, self.chunk_unk_id)
            chunk_ids.append(cid + self.chunk_offset)

        # Roots
        root_ids = []
        for i in range(0, len(chunk_ids), 4):
            root = tuple(cid - self.chunk_offset for cid in chunk_ids[i:i+4])
            rid = self.root_to_id.get(root, self.root_unk_id)
            root_ids.append(rid + self.root_offset)

        return char_ids, chunk_ids, root_ids


def build_and_save():
    """Build the hierarchy and save to disk."""
    config = HierarchyConfig()
    hierarchy = ShakespeareHierarchy(config)
    hierarchy.build()
    hierarchy.save()
    return hierarchy


if __name__ == "__main__":
    # Build the hierarchy
    hierarchy = build_and_save()

    # Demo: show some samples
    print("\n=== Sample Root->Chunk Mappings ===")
    for i in range(min(5, len(hierarchy.root_to_chunks_samples))):
        root_id, chunk_ids = hierarchy.root_to_chunks_samples[i]
        root_str = hierarchy.decode_root(root_id)
        chunks_str = [hierarchy.decode_chunk(cid) for cid in chunk_ids]
        print(f"  Root {root_id}: '{root_str}' -> {chunks_str}")

    print("\n=== Sample Chunk->Char Mappings ===")
    for i in range(min(5, len(hierarchy.chunk_to_chars_samples))):
        chunk_id, char_ids = hierarchy.chunk_to_chars_samples[i]
        chunk_str = hierarchy.decode_chunk(chunk_id)
        chars_str = hierarchy.decode_chars(char_ids)
        print(f"  Chunk {chunk_id}: '{chunk_str}' -> '{chars_str}'")

    # Test batch generation
    print("\n=== Test Batch Generation ===")
    device = torch.device('cpu')
    conds, targets = hierarchy.get_mixed_batch(8, device)
    print(f"  Conditions shape: {conds.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Sample conditions: {conds[:4].tolist()}")
