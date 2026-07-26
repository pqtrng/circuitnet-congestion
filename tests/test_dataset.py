"""Tests for the Gold-layer patch dataset.

All fixtures are synthetic and written to a temporary directory: continuous
integration runs on CPU without access to the real Gold layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from circuitnet_congestion.training.dataset import (
    NUM_FEATURE_CHANNELS,
    CongestionPatchDataset,
    build_dataloader,
)

SIZE = 128


def _write_patch(path: Path, *, coverage_rows: int, gt_fill: float) -> dict[str, np.ndarray]:
    """Write one synthetic patch and return the arrays that were stored."""
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    features = rng.standard_normal((NUM_FEATURE_CHANNELS, SIZE, SIZE)).astype(np.float32)

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[:coverage_rows, :] = 1

    gt = np.zeros((SIZE, SIZE), dtype=np.float32)
    gt[:coverage_rows, :4] = gt_fill

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, gt=gt, mask=mask)
    return {"features": features, "gt": gt, "mask": mask}


@pytest.fixture
def gold_root(tmp_path: Path) -> Path:
    """A miniature Gold layer: 4 train patches, 2 val, 1 test."""
    counts = {"train": 4, "val": 2, "test": 1}
    for split, count in counts.items():
        for i in range(count):
            _write_patch(
                tmp_path / split / f"design_{i:03d}.npz",
                coverage_rows=SIZE if i % 2 == 0 else SIZE // 4,
                gt_fill=0.05,
            )
    return tmp_path


def test_split_lengths(gold_root: Path) -> None:
    assert len(CongestionPatchDataset(gold_root, "train")) == 4
    assert len(CongestionPatchDataset(gold_root, "val")) == 2
    assert len(CongestionPatchDataset(gold_root, "test")) == 1


def test_item_shapes_and_dtypes(gold_root: Path) -> None:
    item = CongestionPatchDataset(gold_root, "train")[0]

    assert item["features"].shape == (NUM_FEATURE_CHANNELS, SIZE, SIZE)
    assert item["gt"].shape == (1, SIZE, SIZE)
    assert item["mask"].shape == (1, SIZE, SIZE)
    for key in ("features", "gt", "mask"):
        assert item[key].dtype == torch.float32


def test_mask_is_float_and_binary(gold_root: Path) -> None:
    """The mask is cast at read time: it is summed over 16384 pixels downstream,
    which an unsigned 8-bit accumulator cannot hold."""
    mask = CongestionPatchDataset(gold_root, "train")[1]["mask"]

    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert 0.0 < float(mask.mean()) < 1.0


def test_ground_truth_is_not_premultiplied_by_mask(gold_root: Path) -> None:
    """Padded pixels and genuinely uncongested pixels are both zero. Applying the
    mask to the ground truth merges them irreversibly, so the dataset must return
    the stored ground truth untouched."""
    path = gold_root / "train" / "zzz_probe.npz"
    stored = _write_patch(path, coverage_rows=SIZE // 2, gt_fill=0.25)

    dataset = CongestionPatchDataset(gold_root, "train")
    index = dataset.paths.index(path)
    item = dataset[index]

    np.testing.assert_array_equal(item["gt"].squeeze(0).numpy(), stored["gt"])
    np.testing.assert_array_equal(item["mask"].squeeze(0).numpy(), stored["mask"])


def test_ordering_is_sorted_and_stable(gold_root: Path) -> None:
    names = [p.name for p in CongestionPatchDataset(gold_root, "train").paths]
    assert names == sorted(names)


def test_limit_takes_deterministic_head(gold_root: Path) -> None:
    full = CongestionPatchDataset(gold_root, "train")
    limited = CongestionPatchDataset(gold_root, "train", limit=2)

    assert len(limited) == 2
    assert limited.paths == full.paths[:2]


def test_rejects_unknown_split(gold_root: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        CongestionPatchDataset(gold_root, "holdout")


def test_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CongestionPatchDataset(tmp_path, "train")


def test_rejects_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="No .npz patches"):
        CongestionPatchDataset(tmp_path, "train")


def test_rejects_missing_array(tmp_path: Path) -> None:
    path = tmp_path / "train" / "broken.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, features=np.zeros((3, SIZE, SIZE), dtype=np.float32))

    with pytest.raises(KeyError, match="missing arrays"):
        CongestionPatchDataset(tmp_path, "train")[0]


def test_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "train" / "wrong.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        features=np.zeros((3, 64, 64), dtype=np.float32),
        gt=np.zeros((64, 64), dtype=np.float32),
        mask=np.ones((64, 64), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="features shape"):
        CongestionPatchDataset(tmp_path, "train")[0]


def test_dataloader_batches(gold_root: Path) -> None:
    dataset = CongestionPatchDataset(gold_root, "train")
    loader = build_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))

    assert batch["features"].shape == (2, NUM_FEATURE_CHANNELS, SIZE, SIZE)
    assert batch["gt"].shape == (2, 1, SIZE, SIZE)
    assert batch["mask"].shape == (2, 1, SIZE, SIZE)
    assert batch["index"].tolist() == [0, 1]
