"""
Fractal Dataloader for Phase 4: Fractal Engine

Provides balanced batches for training across both hierarchy levels:
- Level 0: Root -> Chunks (33%)
- Level 1: Chunk -> Chars (33%)
- Contrastive pairs: Wrong pairings for energy training (33%)

This interleaved training forces the shared weights to find
the "General Refinement" manifold that works across abstraction levels.
"""

import torch
from typing import Tuple, Dict
from hierarchical_bpe import FractalDataset


class FractalBatch:
    """
    Batch for fractal training with samples from both levels.

    Contains:
    - Level 0 (Root -> Chunks): root_ids, chunk_targets, correct pairs
    - Level 1 (Chunk -> Chars): chunk_ids, char_targets, correct pairs
    - Contrastive: wrong pairings for both levels
    - Level indicators: which level each sample belongs to
    """

    def __init__(
        self,
        # Level 0: Root -> Chunks
        level0_conditions: torch.Tensor,  # (B0,) root ids
        level0_targets: torch.Tensor,  # (B0, expansion_size) chunk ids
        level0_target_lens: torch.Tensor,  # (B0,) actual lengths
        level0_wrong_targets: torch.Tensor,  # (B0, expansion_size) wrong chunks

        # Level 1: Chunk -> Chars
        level1_conditions: torch.Tensor,  # (B1,) chunk ids
        level1_targets: torch.Tensor,  # (B1, max_char_len) char ids
        level1_target_lens: torch.Tensor,  # (B1,) actual lengths
        level1_wrong_targets: torch.Tensor,  # (B1, max_char_len) wrong chars

        # Sizes
        level0_batch_size: int,
        level1_batch_size: int
    ):
        self.level0_conditions = level0_conditions
        self.level0_targets = level0_targets
        self.level0_target_lens = level0_target_lens
        self.level0_wrong_targets = level0_wrong_targets

        self.level1_conditions = level1_conditions
        self.level1_targets = level1_targets
        self.level1_target_lens = level1_target_lens
        self.level1_wrong_targets = level1_wrong_targets

        self.level0_batch_size = level0_batch_size
        self.level1_batch_size = level1_batch_size


def get_level0_batch(
    dataset: FractalDataset,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a batch for Level 0 (Root -> Chunks) with contrastive pairs.

    Returns:
        conditions: (B,) root ids
        correct_targets: (B, expansion_size) correct chunk expansions
        target_lens: (B,) actual expansion lengths
        wrong_targets: (B, expansion_size) wrong chunk expansions
    """
    n_samples = len(dataset.root_ids)

    # Sample indices for correct pairs
    idx_correct = torch.randint(n_samples, (batch_size,))

    # Sample indices for wrong pairs (ensure different)
    idx_wrong = torch.randint(n_samples, (batch_size,))
    same_mask = idx_correct == idx_wrong
    while same_mask.any():
        idx_wrong[same_mask] = torch.randint(n_samples, (same_mask.sum(),))
        same_mask = idx_correct == idx_wrong

    conditions = dataset.root_ids[idx_correct].to(device)
    correct_targets = dataset.root_expansions[idx_correct].to(device)
    target_lens = dataset.root_expansion_lens[idx_correct].to(device)
    wrong_targets = dataset.root_expansions[idx_wrong].to(device)

    return conditions, correct_targets, target_lens, wrong_targets


def get_level1_batch(
    dataset: FractalDataset,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a batch for Level 1 (Chunk -> Chars) with contrastive pairs.

    Returns:
        conditions: (B,) chunk ids
        correct_targets: (B, max_char_len) correct char sequences
        target_lens: (B,) actual char lengths
        wrong_targets: (B, max_char_len) wrong char sequences
    """
    n_samples = len(dataset.chunk_ids)

    # Sample indices for correct pairs
    idx_correct = torch.randint(n_samples, (batch_size,))

    # Sample indices for wrong pairs (ensure different)
    idx_wrong = torch.randint(n_samples, (batch_size,))
    same_mask = idx_correct == idx_wrong
    while same_mask.any():
        idx_wrong[same_mask] = torch.randint(n_samples, (same_mask.sum(),))
        same_mask = idx_correct == idx_wrong

    conditions = dataset.chunk_ids[idx_correct].to(device)
    correct_targets = dataset.chunk_chars[idx_correct].to(device)
    target_lens = dataset.chunk_char_lens[idx_correct].to(device)
    wrong_targets = dataset.chunk_chars[idx_wrong].to(device)

    return conditions, correct_targets, target_lens, wrong_targets


def get_fractal_batch(
    dataset: FractalDataset,
    batch_size: int,
    device: torch.device,
    level0_ratio: float = 0.5  # Ratio of Level 0 samples
) -> FractalBatch:
    """
    Get a balanced batch from both hierarchy levels.

    Args:
        dataset: FractalDataset
        batch_size: Total batch size
        device: Device to use
        level0_ratio: Fraction of batch for Level 0 (default 0.5)

    Returns:
        FractalBatch with samples from both levels
    """
    level0_size = int(batch_size * level0_ratio)
    level1_size = batch_size - level0_size

    # Get Level 0 batch (Root -> Chunks)
    l0_cond, l0_targets, l0_lens, l0_wrong = get_level0_batch(
        dataset, level0_size, device
    )

    # Get Level 1 batch (Chunk -> Chars)
    l1_cond, l1_targets, l1_lens, l1_wrong = get_level1_batch(
        dataset, level1_size, device
    )

    return FractalBatch(
        level0_conditions=l0_cond,
        level0_targets=l0_targets,
        level0_target_lens=l0_lens,
        level0_wrong_targets=l0_wrong,
        level1_conditions=l1_cond,
        level1_targets=l1_targets,
        level1_target_lens=l1_lens,
        level1_wrong_targets=l1_wrong,
        level0_batch_size=level0_size,
        level1_batch_size=level1_size
    )


class FractalDataLoader:
    """
    DataLoader for fractal training.

    Provides batches balanced across both hierarchy levels,
    with contrastive pairs for energy training.
    """

    def __init__(
        self,
        dataset: FractalDataset,
        batch_size: int = 64,
        level0_ratio: float = 0.5,
        device: torch.device = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.level0_ratio = level0_ratio
        self.device = device or torch.device('cpu')

    def get_batch(self) -> FractalBatch:
        """Get a single batch."""
        return get_fractal_batch(
            self.dataset,
            self.batch_size,
            self.device,
            self.level0_ratio
        )

    def __iter__(self):
        """Infinite iterator over batches."""
        while True:
            yield self.get_batch()


def create_attention_mask(
    lengths: torch.Tensor,
    max_len: int,
    device: torch.device,
    add_condition: bool = True
) -> torch.Tensor:
    """
    Create attention mask for variable-length sequences.

    Args:
        lengths: (B,) actual sequence lengths
        max_len: Maximum sequence length
        device: Device
        add_condition: If True, adds 1 position for condition token

    Returns:
        mask: (B, 1 + max_len) or (B, max_len) attention mask
    """
    batch_size = lengths.shape[0]
    positions = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    seq_mask = positions < lengths.unsqueeze(1)  # (B, max_len)

    if add_condition:
        cond_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        mask = torch.cat([cond_mask, seq_mask], dim=1)
    else:
        mask = seq_mask

    return mask


# Test
if __name__ == "__main__":
    from hierarchical_bpe import load_fractal_dataset, build_fractal_dataset, HierarchicalBPEConfig
    from pathlib import Path

    # Build or load dataset
    data_path = Path("data/fractal_hierarchy.pkl")
    if not data_path.exists():
        print("Building dataset first...")
        config = HierarchicalBPEConfig()
        tokenizer, dataset = build_fractal_dataset(
            "data/tinyshakespeare.txt",
            config,
            str(data_path)
        )
    else:
        tokenizer, dataset, config = load_fractal_dataset(str(data_path))

    print(f"Dataset loaded:")
    print(f"  Level 0 samples: {len(dataset.root_ids):,}")
    print(f"  Level 1 samples: {len(dataset.chunk_ids):,}")

    # Test dataloader
    device = torch.device('cpu')
    loader = FractalDataLoader(dataset, batch_size=32, device=device)

    batch = loader.get_batch()

    print(f"\nBatch info:")
    print(f"  Level 0 batch size: {batch.level0_batch_size}")
    print(f"  Level 1 batch size: {batch.level1_batch_size}")

    print(f"\nLevel 0 (Root -> Chunks):")
    print(f"  Conditions shape: {batch.level0_conditions.shape}")
    print(f"  Targets shape: {batch.level0_targets.shape}")
    print(f"  Sample condition: {batch.level0_conditions[0].item()}")
    print(f"  Sample target: {batch.level0_targets[0].tolist()}")
    print(f"  Sample wrong: {batch.level0_wrong_targets[0].tolist()}")

    print(f"\nLevel 1 (Chunk -> Chars):")
    print(f"  Conditions shape: {batch.level1_conditions.shape}")
    print(f"  Targets shape: {batch.level1_targets.shape}")
    print(f"  Sample condition: {batch.level1_conditions[0].item()}")
    print(f"  Sample target: {batch.level1_targets[0, :batch.level1_target_lens[0]].tolist()}")

    # Test attention mask
    mask = create_attention_mask(
        batch.level1_target_lens,
        dataset.max_chunk_len,
        device,
        add_condition=True
    )
    print(f"\nAttention mask shape: {mask.shape}")
    print(f"  Sample mask: {mask[0].tolist()}")
