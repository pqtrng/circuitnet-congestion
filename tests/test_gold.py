"""Unit tests for Gold: crop/pad/mask, dedup, split, norm. Synthetic, CPU-safe."""

from pathlib import Path

import numpy as np

from circuitnet_congestion.data.gold import (
    PATCH,
    assign_split,
    crop_chip,
    dedup_chips,
    fit_norm_stats,
)


def test_dedup_keeps_one_per_sha():
    man = [
        {"sample_id": "b", "sha1": "x", "design": "D"},
        {"sample_id": "a", "sha1": "x", "design": "D"},
        {"sample_id": "c", "sha1": "y", "design": "D"},
    ]
    out = dedup_chips(man)
    assert len(out) == 2
    # deterministic: alphabetically-first id for sha 'x'
    xs = [r for r in out if r["sha1"] == "x"]
    assert xs[0]["sample_id"] == "a"


def test_assign_split():
    cfg = {"train": ["A", "B"], "val": ["C"], "test": ["D"]}
    assert assign_split("A", cfg) == "train"
    assert assign_split("C", cfg) == "val"
    assert assign_split("Z", cfg) is None


def test_crop_exact_multiple():
    arr = {"gt": np.ones((256, 256), np.float32), "cell_density": np.ones((256, 256), np.float32)}
    patches = crop_chip(arr, patch=128)
    assert len(patches) == 4  # 2x2
    for _, _, piece, mask in patches:
        assert piece["gt"].shape == (128, 128)
        assert mask.sum() == 128 * 128  # fully valid, no pad


def test_crop_edge_pad_and_mask():
    # 130x130 -> 2x2 patches, edges padded, mask marks real vs pad
    arr = {"gt": np.ones((130, 130), np.float32), "cell_density": np.ones((130, 130), np.float32)}
    patches = crop_chip(arr, patch=128)
    assert len(patches) == 4
    # the bottom-right patch is mostly padding: real region 2x2
    br = [p for p in patches if p[0] == 1 and p[1] == 1][0]
    _, _, piece, mask = br
    assert piece["gt"].shape == (128, 128)
    assert mask.sum() == 2 * 2  # only 2x2 real pixels
    # padded region is zero
    assert piece["gt"][mask == 0].sum() == 0


def test_fit_norm_uses_valid_pixels_only(tmp_path: Path):
    d = tmp_path / "train"
    d.mkdir()
    # one patch: real region value=10, padded region value=0 (but mask=0 there)
    cd = np.zeros((PATCH, PATCH), np.float32)
    cd[:4, :4] = 10.0
    mask = np.zeros((PATCH, PATCH), np.uint8)
    mask[:4, :4] = 1
    np.savez_compressed(d / "p.npz", cell_density=cd, rudy=cd, macro_region=cd, mask=mask)
    stats = fit_norm_stats(d)
    # mean over VALID pixels only = 10.0 (not diluted by padded zeros)
    assert abs(stats["cell_density"]["mean"] - 10.0) < 1e-6
