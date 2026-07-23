"""Unit tests for Bronze ingest logic. Synthetic .npz, no real data, CPU-safe."""

from pathlib import Path

import numpy as np

from circuitnet_congestion.data.bronze import (
    ALL_KEYS,
    FEATURE_SUBDIRS,
    LABEL_SUBDIRS,
    ingest_design,
)


def _write_map(root: Path, design: str, sub: str, sid: str, arr: np.ndarray) -> None:
    d = root / design / sub
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{sid}.npz", data=arr)


def _write_full_sample(root: Path, design: str, sid: str, shape=(8, 8)) -> None:
    for sub in {**FEATURE_SUBDIRS, **LABEL_SUBDIRS}.values():
        _write_map(root, design, sub, sid, np.random.rand(*shape))


def test_ingest_full_sample(tmp_path: Path):
    extracted = tmp_path / "ex"
    bronze = tmp_path / "bronze"
    _write_full_sample(extracted, "DesignA", "s1", shape=(8, 8))

    recs, skips = ingest_design("DesignA", extracted, bronze)
    assert len(recs) == 1
    assert len(skips) == 0
    r = recs[0]
    assert r.shape == [8, 8]
    assert set(r.keys) == set(ALL_KEYS)
    # repacked file has all 5 keys
    with np.load(bronze / "DesignA" / "s1.npz") as d:
        assert set(d.files) == set(ALL_KEYS)


def test_skip_missing_map(tmp_path: Path):
    extracted = tmp_path / "ex"
    bronze = tmp_path / "bronze"
    # write all but macro_region
    for key, sub in {**FEATURE_SUBDIRS, **LABEL_SUBDIRS}.items():
        if key == "macro_region":
            continue
        _write_map(extracted, "DesignA", sub, "s1", np.random.rand(8, 8))

    recs, skips = ingest_design("DesignA", extracted, bronze)
    assert len(recs) == 0
    assert len(skips) == 1
    assert "macro_region" in skips[0].reason


def test_skip_shape_mismatch(tmp_path: Path):
    extracted = tmp_path / "ex"
    bronze = tmp_path / "bronze"
    _write_full_sample(extracted, "DesignA", "s1", shape=(8, 8))
    # overwrite one map with a different shape
    _write_map(extracted, "DesignA", LABEL_SUBDIRS["congestion_v"], "s1", np.random.rand(8, 9))

    recs, skips = ingest_design("DesignA", extracted, bronze)
    assert len(recs) == 0
    assert len(skips) == 1
    assert "shape mismatch" in skips[0].reason
