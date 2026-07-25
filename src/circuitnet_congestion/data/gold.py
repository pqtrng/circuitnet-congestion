"""Gold layer: dedup, split, crop to fixed patches with valid masks, normalize.

Pipeline (order matters):
  1. Dedup unique chips by Silver sha1 (duplicates inflate the dataset).
  2. Split design-wise (a chip's patches all inherit its split — no leakage).
  3. Crop each chip into non-overlapping PxP patches; edge patches are zero-padded
     to PxP and a valid mask marks real (1) vs padded (0) pixels.
  4. Normalize features train-only: z-score cell_density & rudy over VALID train
     pixels; macro_region left as-is (binary); GT left raw (target, comparable to
     literature). Mask travels with each patch for masked loss (T6) and masked
     metrics (T8).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

PATCH = 128
FEATURE_KEYS = ["cell_density", "rudy", "macro_region"]
ZSCORE_KEYS = ["cell_density", "rudy"]  # macro_region stays binary


@dataclass
class PatchRecord:
    split: str
    design: str
    source_id: str
    row: int
    col: int
    valid_frac: float


def dedup_chips(manifest: list[dict]) -> list[dict]:
    """Keep one representative per sha1, deterministic by sample_id."""
    by_sha: dict[str, dict] = {}
    for r in sorted(manifest, key=lambda x: x["sample_id"]):
        by_sha.setdefault(r["sha1"], r)
    return list(by_sha.values())


def assign_split(design: str, split_cfg: dict[str, list[str]]) -> str | None:
    for split, designs in split_cfg.items():
        if design in designs:
            return split
    return None


def crop_chip(
    arrays: dict[str, np.ndarray], patch: int = PATCH
) -> list[tuple[int, int, dict[str, np.ndarray], np.ndarray]]:
    """Non-overlapping crop. Edge patches zero-padded to patch size.

    Returns list of (row_idx, col_idx, {key: patch_array}, valid_mask).
    """
    h, w = arrays["gt"].shape
    out = []
    for i in range(0, h, patch):
        for j in range(0, w, patch):
            hh = min(patch, h - i)
            ww = min(patch, w - j)
            mask = np.zeros((patch, patch), dtype=np.uint8)
            mask[:hh, :ww] = 1
            piece = {}
            for k, a in arrays.items():
                p = np.zeros((patch, patch), dtype=np.float32)
                p[:hh, :ww] = a[i : i + hh, j : j + ww]
                piece[k] = p
            out.append((i // patch, j // patch, piece, mask))
    return out


def fit_norm_stats(train_patch_dir: Path) -> dict[str, dict[str, float]]:
    """z-score stats over VALID pixels of train patches, per z-score channel."""
    sums = {k: 0.0 for k in ZSCORE_KEYS}
    sqs = {k: 0.0 for k in ZSCORE_KEYS}
    counts = {k: 0 for k in ZSCORE_KEYS}
    for f in sorted(train_patch_dir.glob("*.npz")):
        with np.load(f) as d:
            mask = d["mask"].astype(bool)
            for k in ZSCORE_KEYS:
                vals = d[k][mask]
                sums[k] += float(vals.sum())
                sqs[k] += float((vals**2).sum())
                counts[k] += vals.size
    stats = {}
    for k in ZSCORE_KEYS:
        n = max(counts[k], 1)
        mean = sums[k] / n
        var = max(sqs[k] / n - mean**2, 0.0)
        std = math.sqrt(var) or 1.0
        stats[k] = {"mean": mean, "std": std}
    return stats


def apply_norm(patch_dir: Path, stats: dict[str, dict[str, float]]) -> None:
    """Rewrite patches: z-score channels normalized, macro & gt untouched, mask kept."""
    for f in sorted(patch_dir.glob("*.npz")):
        with np.load(f) as d:
            data = {k: d[k] for k in d.files}
        for k in ZSCORE_KEYS:
            data[k] = ((data[k] - stats[k]["mean"]) / stats[k]["std"]).astype(np.float32)
        # stack features in fixed order -> [3, P, P]
        features = np.stack([data[k] for k in FEATURE_KEYS], axis=0).astype(np.float32)
        np.savez_compressed(f, features=features, gt=data["gt"], mask=data["mask"])


def build(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    silver_dir = Path(cfg["paths"].get("silver_dir", "data/silver"))
    gold_dir = Path(cfg["paths"].get("gold_dir", "data/gold"))
    split_cfg = cfg["dataset"]["designs"]

    manifest = json.loads((silver_dir / "manifest.json").read_text())
    chips = dedup_chips(manifest)
    print(f"unique chips after dedup: {len(chips)}")

    records: list[PatchRecord] = []
    for split in ("train", "val", "test"):
        (gold_dir / split).mkdir(parents=True, exist_ok=True)

    for chip in chips:
        design, sid = chip["design"], chip["sample_id"]
        split = assign_split(design, split_cfg)
        if split is None:
            continue
        with np.load(silver_dir / design / f"{sid}.npz") as d:
            arrays = {k: d[k] for k in [*FEATURE_KEYS, "gt"]}
        for row, col, piece, mask in crop_chip(arrays):
            out_name = f"{sid}__r{row}_c{col}.npz"
            np.savez_compressed(gold_dir / split / out_name, mask=mask, **piece)
            records.append(PatchRecord(split, design, sid, row, col, float(mask.mean())))

    # normalize: fit train-only, apply to all splits
    stats = fit_norm_stats(gold_dir / "train")
    for split in ("train", "val", "test"):
        apply_norm(gold_dir / split, stats)

    (gold_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    manifest_out = {
        "patch_size": PATCH,
        "feature_keys": FEATURE_KEYS,
        "zscore_keys": ZSCORE_KEYS,
        "n_patches": len(records),
        "counts": {s: sum(1 for r in records if r.split == s) for s in ("train", "val", "test")},
        "patches": [asdict(r) for r in records],
    }
    (gold_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print(f"patches: {manifest_out['counts']}")
    print(f"norm_stats: {stats}")
    return manifest_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold: dedup/split/crop/normalize.")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = ap.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
