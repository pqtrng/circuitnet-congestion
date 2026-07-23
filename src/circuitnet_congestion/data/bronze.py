"""Bronze layer: land raw CircuitNet-N14 congestion samples, unmodified.

For each sample, reads the 3 input feature maps and 2 congestion label maps
(horizontal / vertical, kept separate — merging is a Silver decision), verifies
they share one shape, and repacks them into a single .npz per sample. Values are
NOT normalized, resized, merged, or split. Samples with missing maps or mismatched
shapes are skipped and recorded with a reason. A manifest inventories everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

# Relative paths inside each design dir. One .npz per sample lives under each.
FEATURE_SUBDIRS = {
    "cell_density": "cell_density",
    "rudy": "RUDY/RUDY",
    "macro_region": "macro_region",
}
CONGESTION_BASE = "congestion/congestion_global_routing/overflow_based"
LABEL_SUBDIRS = {
    "congestion_h": f"{CONGESTION_BASE}/congestion_GR_horizontal_overflow",
    "congestion_v": f"{CONGESTION_BASE}/congestion_GR_vertical_overflow",
}
ALL_KEYS = list(FEATURE_SUBDIRS) + list(LABEL_SUBDIRS)
NPZ_ARRAY_KEY = "data"  # each source .npz stores its map under this key


@dataclass
class SampleRecord:
    design: str
    sample_id: str
    shape: list[int]
    keys: list[str]
    sha1: str


@dataclass
class SkipRecord:
    design: str
    sample_id: str
    reason: str


def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_map(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as d:
        return d[NPZ_ARRAY_KEY]


def _sample_ids(design_dir: Path) -> list[str]:
    """Sample ids come from the cell_density dir (one .npz per sample)."""
    cd = design_dir / FEATURE_SUBDIRS["cell_density"]
    return sorted(p.stem for p in cd.glob("*.npz"))


def ingest_design(
    design: str, extracted_root: Path, bronze_dir: Path
) -> tuple[list[SampleRecord], list[SkipRecord]]:
    design_dir = extracted_root / design
    out_dir = bronze_dir / design
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    skips: list[SkipRecord] = []

    for sid in _sample_ids(design_dir):
        maps: dict[str, np.ndarray] = {}
        missing = []
        for key, sub in {**FEATURE_SUBDIRS, **LABEL_SUBDIRS}.items():
            p = design_dir / sub / f"{sid}.npz"
            if not p.exists():
                missing.append(key)
            else:
                maps[key] = _load_map(p)

        if missing:
            skips.append(SkipRecord(design, sid, f"missing maps: {','.join(missing)}"))
            continue

        shapes = {m.shape for m in maps.values()}
        if len(shapes) != 1:
            skips.append(SkipRecord(design, sid, f"shape mismatch across maps: {sorted(shapes)}"))
            continue

        shape = maps["cell_density"].shape
        out_path = out_dir / f"{sid}.npz"
        # Repack: same raw values, grouped. No transformation.
        np.savez_compressed(out_path, **maps)
        records.append(
            SampleRecord(
                design=design,
                sample_id=sid,
                shape=list(shape),
                keys=ALL_KEYS,
                sha1=_sha1_of_file(out_path),
            )
        )

    return records, skips


def _tarball_for(raw_dir: Path, design: str) -> Path:
    return raw_dir / "CircuitNet-N14" / "routability_features" / f"{design}.tar.gz"


def ingest(config_path: Path, only_design: str | None = None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    raw_dir = Path(cfg["paths"]["raw_dir"])
    bronze_dir = Path(cfg["paths"].get("bronze_dir", "data/bronze"))
    designs_cfg = cfg["dataset"]["designs"]
    all_designs = [d for group in designs_cfg.values() for d in group]
    designs = [only_design] if only_design else all_designs

    bronze_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[SampleRecord] = []
    all_skips: list[SkipRecord] = []

    for design in designs:
        tarball = _tarball_for(raw_dir, design)
        if not tarball.exists():
            raise FileNotFoundError(f"Tarball not found: {tarball}")
        print(f"[{design}] extracting {tarball.name} ...")
        with tempfile.TemporaryDirectory(prefix="bronze_") as tmp:
            tmp_root = Path(tmp)
            with tarfile.open(tarball, "r:gz") as tf:
                tf.extractall(tmp_root, filter="data")
            recs, skips = ingest_design(design, tmp_root, bronze_dir)
        all_records.extend(recs)
        all_skips.extend(skips)
        print(f"[{design}] ingested={len(recs)} skipped={len(skips)}")

    manifest = {
        "layer": "bronze",
        "keys": ALL_KEYS,
        "n_samples": len(all_records),
        "n_skipped": len(all_skips),
        "samples": [asdict(r) for r in all_records],
        "skipped": [asdict(s) for s in all_skips],
    }
    manifest_path = bronze_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path} (samples={len(all_records)}, skipped={len(all_skips)})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Bronze ingest for CircuitNet-N14 congestion.")
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--design", type=str, default=None, help="Ingest one design only.")
    args = ap.parse_args()
    ingest(args.config, only_design=args.design)


if __name__ == "__main__":
    main()
