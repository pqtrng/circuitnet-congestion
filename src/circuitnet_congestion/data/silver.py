"""Silver layer: validate Bronze samples, merge congestion GT, fingerprint lineage.

For each Bronze sample: validates the 5 raw maps (float, finite, non-negative, 2D,
mutually equal shape), computes the ground-truth congestion map as the per-pixel
max of horizontal and vertical overflow, and writes a Silver .npz holding the 3
input features plus the single gt map. A per-sample metadata table is validated
with a Pandera schema (Pandera is used for tabular metadata, not raw image arrays).
No normalization, resizing, or splitting — those need the split (Gold).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml

FEATURE_KEYS = ["cell_density", "rudy", "macro_region"]
CONG_KEYS = ["congestion_h", "congestion_v"]
SILVER_KEYS = [*FEATURE_KEYS, "gt"]

# Pandera schema for the per-sample metadata table (tabular, not the arrays).
SAMPLE_SCHEMA = pa.DataFrameSchema(
    {
        "design": pa.Column(str),
        "sample_id": pa.Column(str, unique=True),
        "height": pa.Column(int, pa.Check.gt(0)),
        "width": pa.Column(int, pa.Check.gt(0)),
        "gt_max": pa.Column(float, pa.Check.ge(0.0)),
        "gt_frac_nonzero": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
        "sha1": pa.Column(str, pa.Check.str_length(40, 40)),
    },
    strict=True,
    coerce=True,
)


@dataclass
class SilverRecord:
    design: str
    sample_id: str
    height: int
    width: int
    gt_max: float
    gt_frac_nonzero: float
    sha1: str


def _validate_map(name: str, arr: np.ndarray) -> None:
    """Raw-array invariants. Raises ValueError on violation."""
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D, got {arr.ndim}D")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{name}: expected float dtype, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN or Inf")
    if (arr < 0).any():
        raise ValueError(f"{name}: contains negative values")


def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_sample(bronze_path: Path, design: str, sample_id: str, out_dir: Path) -> SilverRecord:
    with np.load(bronze_path) as d:
        maps = {k: d[k] for k in [*FEATURE_KEYS, *CONG_KEYS]}

    # Clean: some maps load as int (numpy stores all-zero overflow maps as int64).
    # Cast to float64 so all Silver arrays share one dtype. Values are unchanged
    # (integer 0 -> float 0.0); this only normalizes the storage dtype.
    for name, arr in maps.items():
        if not np.issubdtype(arr.dtype, np.floating):
            maps[name] = arr.astype(np.float64)

    for name, arr in maps.items():
        _validate_map(name, arr)

    shapes = {a.shape for a in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"{sample_id}: shape mismatch {sorted(shapes)}")

    gt = np.maximum(maps["congestion_h"], maps["congestion_v"])
    silver = {k: maps[k] for k in FEATURE_KEYS}
    silver["gt"] = gt

    out_path = out_dir / f"{sample_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **silver)

    h, w = gt.shape
    return SilverRecord(
        design=design,
        sample_id=sample_id,
        height=int(h),
        width=int(w),
        gt_max=float(gt.max()),
        gt_frac_nonzero=float((gt > 0).mean()),
        sha1=_sha1_of_file(out_path),
    )


def build(config_path: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text())
    bronze_dir = Path(cfg["paths"].get("bronze_dir", "data/bronze"))
    silver_dir = Path(cfg["paths"].get("silver_dir", "data/silver"))

    manifest = json.loads((bronze_dir / "manifest.json").read_text())
    records: list[SilverRecord] = []

    for s in manifest["samples"]:
        design, sid = s["design"], s["sample_id"]
        bronze_path = bronze_dir / design / f"{sid}.npz"
        rec = process_sample(bronze_path, design, sid, silver_dir / design)
        records.append(rec)

    df = pd.DataFrame([r.__dict__ for r in records])
    SAMPLE_SCHEMA.validate(df, lazy=True)  # raises SchemaErrors with all failures

    silver_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(silver_dir / "manifest.json", orient="records", indent=2)
    print(f"Silver: {len(df)} samples validated -> {silver_dir / 'manifest.json'}")
    print(
        f"  gt_frac_nonzero: min={df.gt_frac_nonzero.min():.5f} "
        f"max={df.gt_frac_nonzero.max():.5f} mean={df.gt_frac_nonzero.mean():.5f}"
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Silver validation + GT merge.")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = ap.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
