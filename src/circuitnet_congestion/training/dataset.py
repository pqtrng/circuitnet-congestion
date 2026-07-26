"""Gold-layer patch dataset for routing congestion prediction.

The Gold layer stores one compressed .npz per 128x128 patch:

    features : float32 [3, 128, 128]  z-scored (cell_density, rudy) + binary macro_region
    gt       : float32 [128, 128]     raw congestion = max(horizontal, vertical)
    mask     : uint8   [128, 128]     1 = real die area, 0 = edge zero-padding

This module is read-only with respect to statistics. Normalisation constants are
fitted during the Gold build on the train split alone (see gold/norm_stats.json)
and are never recomputed here -- there must be exactly one place in the codebase
that is allowed to look at the distribution of the data.

The mask is returned as a separate tensor and is deliberately NOT applied to the
ground truth. Padded pixels and genuinely uncongested pixels are both zero; once
they are merged, neither the loss nor the evaluation can tell them apart. 42% of
patches contain padding, so this distinction is load-bearing.

No augmentation is applied. Congestion maps live on physical die coordinates and
the feature channels are anchored to that layout, so the rotation/flip family
used for wafer maps does not transfer. Augmentation, if introduced later, is an
explicit ablation with its own justification -- not a silent default.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")
FEATURE_CHANNELS = ("cell_density", "rudy", "macro_region")
NUM_FEATURE_CHANNELS = len(FEATURE_CHANNELS)
PATCH_SIZE = 128

KEY_FEATURES = "features"
KEY_GT = "gt"
KEY_MASK = "mask"


class CongestionPatchDataset(Dataset):
    """Lazily reads Gold patches from ``<root>/<split>/*.npz``.

    Patches are read on access rather than cached: the full Gold layer is
    roughly 16 GB once decompressed, so materialising it in memory is not an
    option. Keep ``__init__`` cheap -- it runs again in every dataloader worker.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        limit: int | None = None,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Gold split directory not found: {split_dir}")

        paths = sorted(split_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz patches found under {split_dir}")

        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive, got {limit}")
            # Deterministic head slice: smoke runs and CI only. Never report
            # metrics from a limited dataset -- the slice is ordered by filename
            # and is therefore biased towards a single design.
            paths = paths[:limit]

        self._paths: tuple[Path, ...] = tuple(paths)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    def path_of(self, index: int) -> Path:
        """Map a dataset index back to its source file, for per-patch analysis."""
        return self._paths[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        path = self._paths[index]

        with np.load(path) as payload:
            missing = [k for k in (KEY_FEATURES, KEY_GT, KEY_MASK) if k not in payload]
            if missing:
                raise KeyError(f"{path}: missing arrays {missing}; present {list(payload.files)}")
            features = np.ascontiguousarray(payload[KEY_FEATURES], dtype=np.float32)
            gt = np.ascontiguousarray(payload[KEY_GT], dtype=np.float32)
            # Cast the mask to float32 here rather than in the loss: it is used
            # multiplicatively and summed over 16384 pixels, which overflows an
            # unsigned 8-bit accumulator and promotes unpredictably under AMP.
            mask = np.ascontiguousarray(payload[KEY_MASK], dtype=np.float32)

        self._validate(path, features, gt, mask)

        # Ground truth and mask carry an explicit channel dimension so they
        # already match the model output shape [B, 1, H, W]. Relying on implicit
        # broadcasting in the loss is how silent shape bugs get in.
        return {
            "features": torch.from_numpy(features),
            "gt": torch.from_numpy(gt).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "index": index,
        }

    def _validate(
        self,
        path: Path,
        features: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        size = self.patch_size
        expected = (NUM_FEATURE_CHANNELS, size, size)
        if features.shape != expected:
            raise ValueError(f"{path}: features shape {features.shape}, expected {expected}")
        if gt.shape != (size, size):
            raise ValueError(f"{path}: gt shape {gt.shape}, expected {(size, size)}")
        if mask.shape != (size, size):
            raise ValueError(f"{path}: mask shape {mask.shape}, expected {(size, size)}")


def seed_worker(worker_id: int) -> None:
    """Give each dataloader worker a distinct but run-reproducible seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: CongestionPatchDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    seed: int = 42,
    drop_last: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Wrap a patch dataset in a dataloader with deterministic shuffling.

    Reading is I/O bound (one compressed file per sample), so worker processes
    exist to hide decompression latency behind accelerator compute.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
