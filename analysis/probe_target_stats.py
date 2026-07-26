"""Probe: distribution of the congestion target, measured per split.

Writes a JSON artefact that the matching audit renders. The split is
deliberate: probes need the Gold layer, which is excluded from the repository,
while audits must run on a fresh clone. Committing the measurement rather than
the measurement code's inputs is what keeps `make report` reproducible for a
reader who has no data.

Values are accumulated as an exact histogram rather than a sample. Targets are
routing overflow divided by track capacity, so they are quantised to fractions
with small denominators and the number of distinct values is modest. Quantiles,
the most frequent levels and the share of squared error above a threshold are
therefore exact over the whole split, not estimated from a subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
ERROR_SHARE_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)
TOP_VALUES = 10
HOTSPOT_THRESHOLD = 0.05
THRESHOLD_WINDOW = 0.015
MAX_DENOMINATOR = 200
ROUNDING = 9

OUTPUT = Path("results/probes/target_stats.json")


def _histogram_statistics(histogram: Counter[float], valid_pixels: int) -> dict[str, Any]:
    """Derive every reported quantity from the exact value histogram."""
    values = np.array(sorted(histogram), dtype=np.float64)
    counts = np.array([histogram[v] for v in values], dtype=np.int64)
    nonzero_total = int(counts.sum())

    cumulative = np.cumsum(counts)
    quantiles = {
        f"p{q * 100:g}": float(values[int(np.searchsorted(cumulative, q * nonzero_total))])
        for q in QUANTILES
    }

    squared = values**2 * counts
    total_squared = float(squared.sum())
    shares = {}
    for threshold in ERROR_SHARE_THRESHOLDS:
        selected = values > threshold
        shares[f"{threshold:g}"] = {
            "pixel_fraction": float(counts[selected].sum() / valid_pixels),
            "squared_error_share": float(squared[selected].sum() / total_squared),
        }

    order = np.argsort(-counts)[:TOP_VALUES]
    levels = [
        {
            "value": float(values[i]),
            "count": int(counts[i]),
            "fraction": str(Fraction(float(values[i])).limit_denominator(MAX_DENOMINATOR)),
        }
        for i in order
    ]
    # Whether the decision boundary coincides with a quantisation level. The
    # comparison is strict, so pixels sitting exactly on the threshold count as
    # background and an epsilon shift would move all of them at once.
    near = (values >= HOTSPOT_THRESHOLD - THRESHOLD_WINDOW) & (
        values <= HOTSPOT_THRESHOLD + THRESHOLD_WINDOW
    )
    neighbourhood = [
        {
            "value": float(v),
            "count": int(c),
            "fraction": str(Fraction(float(v)).limit_denominator(MAX_DENOMINATOR)),
        }
        for v, c in zip(values[near], counts[near], strict=True)
    ]
    on_threshold = int(counts[np.isclose(values, HOTSPOT_THRESHOLD, rtol=0, atol=1e-6)].sum())

    return {
        "nonzero_pixels": nonzero_total,
        "nonzero_fraction_of_valid": nonzero_total / valid_pixels,
        "distinct_nonzero_values": int(values.size),
        "max": float(values[-1]),
        "pixels_at_or_above_one": int(counts[values >= 1.0].sum()),
        "quantiles_of_nonzero": quantiles,
        "most_frequent_levels": levels,
        "error_share_above": shares,
        "pixels_on_threshold": on_threshold,
        "levels_near_threshold": neighbourhood,
    }


def probe_split(gold_dir: Path, split: str, stride: int) -> dict[str, Any]:
    paths = sorted((gold_dir / split).glob("*.npz"))[::stride]
    if not paths:
        raise FileNotFoundError(f"no patches under {gold_dir / split}")

    histogram: Counter[float] = Counter()
    total_pixels = 0
    valid_pixels = 0
    squared_sum = 0.0

    for path in paths:
        with np.load(path) as payload:
            # Only the target and the mask are decompressed; the feature stack
            # is the bulk of each file and is not needed here.
            target = payload["gt"].astype(np.float64)
            mask = payload["mask"].astype(bool)

        valid = target[mask]
        total_pixels += target.size
        valid_pixels += valid.size
        squared_sum += float(np.square(valid).sum())
        histogram.update(np.round(valid[valid > 0], ROUNDING).tolist())

    statistics = _histogram_statistics(histogram, valid_pixels)
    statistics.update(
        {
            "patches": len(paths),
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "mask_coverage": valid_pixels / total_pixels,
            # Equal to the masked mean squared error of predicting zero
            # everywhere. Reported per split because it differs between them,
            # which makes absolute error incomparable across splits.
            "zero_predictor_mse": squared_sum / valid_pixels,
        }
    )
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the congestion target distribution.")
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="take every Nth patch; the default scans every patch",
    )
    args = parser.parse_args()

    record = {
        "gold_dir": str(args.gold_dir),
        "stride": args.stride,
        "splits": {split: probe_split(args.gold_dir, split, args.stride) for split in SPLITS},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")
    for split, statistics in record["splits"].items():
        print(
            f"  {split:5s} patches={statistics['patches']:>6} "
            f"valid={statistics['valid_pixels']:>12,} "
            f"nonzero={statistics['nonzero_fraction_of_valid'] * 100:.3f}% "
            f"zero_pred_mse={statistics['zero_predictor_mse']:.4e} "
            f"max={statistics['max']:.4f}"
        )


if __name__ == "__main__":
    main()
