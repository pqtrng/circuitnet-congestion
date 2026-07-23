"""Unit tests for Silver validation + GT merge. Synthetic arrays, CPU-safe."""

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from circuitnet_congestion.data.silver import (
    SAMPLE_SCHEMA,
    SILVER_KEYS,
    _validate_map,
    process_sample,
)


def _make_bronze(path: Path, shape=(8, 8), h_val=0.5, v_val=0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    maps = {
        "cell_density": rng.random(shape),
        "rudy": rng.random(shape),
        "macro_region": rng.random(shape),
        "congestion_h": np.full(shape, h_val),
        "congestion_v": np.full(shape, v_val),
    }
    np.savez_compressed(path, **maps)


def test_gt_is_pixelwise_max(tmp_path: Path):
    bronze = tmp_path / "b" / "s1.npz"
    _make_bronze(bronze, h_val=0.5, v_val=0.3)
    rec = process_sample(bronze, "D", "s1", tmp_path / "silver")
    with np.load(tmp_path / "silver" / "s1.npz") as d:
        assert set(d.files) == set(SILVER_KEYS)
        assert np.allclose(d["gt"], 0.5)  # max(0.5, 0.3)
    assert rec.gt_max == pytest.approx(0.5)


def test_validate_rejects_nan():
    a = np.array([[0.0, np.nan], [0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        _validate_map("x", a)


def test_validate_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        _validate_map("x", np.array([[-1.0, 0.0]]))


def test_validate_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        _validate_map("x", np.zeros((3, 3, 3)))


def test_schema_rejects_bad_frac():
    bad = pd.DataFrame(
        [
            {
                "design": "D",
                "sample_id": "s1",
                "height": 8,
                "width": 8,
                "gt_max": 0.5,
                "gt_frac_nonzero": 1.5,
                "sha1": "a" * 40,
            }
        ]
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        SAMPLE_SCHEMA.validate(bad, lazy=True)


def test_int_allzero_map_is_cast_to_float(tmp_path: Path):
    # Reproduce the real-data quirk: an all-zero congestion map stored as int64.
    bronze = tmp_path / "b" / "s1.npz"
    bronze.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    maps = {
        "cell_density": rng.random((8, 8)),
        "rudy": rng.random((8, 8)),
        "macro_region": rng.random((8, 8)),
        "congestion_h": np.zeros((8, 8), dtype=np.int64),  # int, all zero
        "congestion_v": np.full((8, 8), 0.3),
    }
    np.savez_compressed(bronze, **maps)
    rec = process_sample(bronze, "D", "s1", tmp_path / "silver")
    # gt = max(0, 0.3) = 0.3, no crash from int dtype
    assert rec.gt_max == pytest.approx(0.3)
