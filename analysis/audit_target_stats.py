"""Audit: what the congestion target actually looks like.

Renders the measurement written by analysis.probe_target_stats. Four findings
drive decisions elsewhere in the codebase, so each is reported with the numbers
that produced it rather than as an assertion.

The target is quantised. It is routing overflow divided by track capacity, so
it lands on fractions with small denominators that vary by design and by metal
layer. This is why the hotspot threshold cannot be placed in a gap between
levels, and why the evaluation reports a range of thresholds.

The zero-predictor error differs by an order of magnitude between splits, so
absolute error is not comparable across them. Every error figure is reported
against the baseline of its own split.

Squared error is dominated by a vanishing fraction of pixels on the training
split, which is why the weighted loss uses a binary class weight rather than
one proportional to the target: squaring already amplifies large values.

The splits are design-wise, and their target distributions differ sharply. The
split used for checkpoint selection is the sparsest of the three, which is a
limitation of the evaluation rather than a property of the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/target_stats.json")
HOTSPOT_THRESHOLD = 0.05
LEVELS_SHOWN = 6


def _overview(splits: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "Per split, measured over valid pixels only:",
        "",
        f"  {'split':<6} {'patches':>8} {'valid px':>14} {'coverage':>9} "
        f"{'nonzero':>9} {'zero-pred MSE':>14} {'max':>8}",
    ]
    for name, split in splits.items():
        lines.append(
            f"  {name:<6} {split['patches']:>8,} {split['valid_pixels']:>14,} "
            f"{split['mask_coverage']:>9.4f} "
            f"{split['nonzero_fraction_of_valid'] * 100:>8.3f}% "
            f"{split['zero_predictor_mse']:>14.4e} {split['max']:>8.2f}"
        )

    baselines = [s["zero_predictor_mse"] for s in splits.values()]
    lines += [
        "",
        f"The zero-predictor error spans a factor of {max(baselines) / min(baselines):.1f} "
        "across splits, so an absolute error figure says nothing without the",
        "baseline of the split it was measured on. Every error below is a ratio.",
    ]
    return lines


def _quantisation(splits: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "Quantisation. Targets are routing overflow over track capacity, so they",
        "take fractional values whose denominators vary by design and metal layer:",
        "",
    ]
    for name, split in splits.items():
        levels = ", ".join(
            f"{level['fraction']} ({level['count']:,})"
            for level in split["most_frequent_levels"][:LEVELS_SHOWN]
        )
        lines.append(f"  {name:<6} {split['distinct_nonzero_values']:>4} distinct levels")
        lines.append(f"         {levels}")
    return lines


def _threshold(splits: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"The threshold at {HOTSPOT_THRESHOLD} is itself a level, not a gap between",
        "levels. Neighbouring values on the training split are dense enough that no",
        "gap exists to place it in:",
        "",
    ]

    train = splits["train"]
    near = train["levels_near_threshold"]
    index = next(
        i for i, level in enumerate(near) if abs(level["value"] - HOTSPOT_THRESHOLD) < 1e-6
    )
    for level in near[max(0, index - 2): index + 3]:
        marker = "  <-- threshold" if abs(level["value"] - HOTSPOT_THRESHOLD) < 1e-6 else ""
        lines.append(
            f"  {level['value']:.6f}  {level['fraction']:>7}  {level['count']:>10,}{marker}"
        )

    on_threshold = {name: split["pixels_on_threshold"] for name, split in splits.items()}
    lines += [
        "",
        "  pixels sitting exactly on it: "
        + ", ".join(f"{name} {count:,}" for name, count in on_threshold.items()),
        "",
        "The definition is therefore physical rather than geometric: a hotspot is a",
        "cell whose routing demand exceeds its track capacity by more than five",
        "percent. The level at exactly five percent is excluded by the strict",
        "comparison, deliberately. Both round to the same single-precision value, so",
        "the boundary does not depend on rounding -- but roughly two percent of",
        "positive pixels sit on it, which is why evaluation sweeps thresholds",
        "instead of reporting one.",
    ]
    return lines


def _error_concentration(splits: dict[str, dict[str, Any]]) -> list[str]:
    thresholds = sorted(splits["train"]["error_share_above"], key=float)
    lines = [
        "Concentration of squared error. Each cell is the fraction of valid pixels",
        "above the threshold, and the share of total squared error they carry:",
        "",
        f"  {'split':<6} " + " ".join(f"{'>' + t:>18}" for t in thresholds),
    ]
    for name, split in splits.items():
        cells = []
        for threshold in thresholds:
            entry = split["error_share_above"][threshold]
            cells.append(
                f"{entry['pixel_fraction'] * 100:>7.3f}% /"
                f"{entry['squared_error_share'] * 100:>6.1f}%"
            )
        lines.append(f"  {name:<6} " + " ".join(f"{cell:>18}" for cell in cells))

    train_tail = splits["train"]["error_share_above"]["0.2"]
    test_tail = splits["test"]["error_share_above"]["0.2"]
    lines += [
        "",
        f"On the training split {train_tail['pixel_fraction'] * 100:.3f}% of pixels "
        f"carry {train_tail['squared_error_share'] * 100:.1f}% of the squared error.",
        "Squaring already amplifies large targets, so a weight proportional to the",
        "target would compound an existing bias rather than correct one. The weighted",
        "loss uses a binary class weight for that reason.",
        "",
        f"That concentration is a property of the training split, not of the task. On",
        f"the test split the same tail carries only "
        f"{test_tail['squared_error_share'] * 100:.1f}% -- its error sits in the",
        "moderate hotspot band instead, which is the band that matters in practice.",
    ]
    return lines


def _shift(splits: dict[str, dict[str, Any]]) -> list[str]:
    sparsest = min(splits, key=lambda n: splits[n]["nonzero_fraction_of_valid"])
    densest = max(splits, key=lambda n: splits[n]["nonzero_fraction_of_valid"])
    ratio = (
            splits[densest]["nonzero_fraction_of_valid"]
            / splits[sparsest]["nonzero_fraction_of_valid"]
    )

    lines = [
        "Distribution shift. The splits are design-wise, so nothing forces their",
        f"target distributions to match, and they do not: '{densest}' is {ratio:.1f}x",
        f"denser in hotspots than '{sparsest}'.",
        "",
    ]
    for name, split in splits.items():
        lines.append(
            f"  {name:<6} nonzero {split['nonzero_fraction_of_valid'] * 100:>6.3f}%  "
            f"max {split['max']:>6.2f}  pixels at or above 1.0: "
            f"{split['pixels_at_or_above_one']:,}"
        )

    lines += [
        "",
        "Checkpoints are selected on the validation split, which is the sparsest of",
        "the three. Model selection therefore happens on a distribution unlike the",
        "one the model is finally scored against. This is a limitation of the",
        "protocol and is not corrected for.",
    ]
    return lines


def run() -> str:
    splits = json.loads(PROBE.read_text())["splits"]

    sections = [
        _overview(splits),
        _quantisation(splits),
        _threshold(splits),
        _error_concentration(splits),
        _shift(splits),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


if __name__ == "__main__":
    print(run())
