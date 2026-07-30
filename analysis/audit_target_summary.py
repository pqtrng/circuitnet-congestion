"""Headline audit: the per-split target table rendered for the README.

Reads results/probes/target_stats.json and emits a markdown table of patch
counts, non-zero fractions and zero-predictor errors per split, followed by
the spread of the zero-predictor error across splits. The README's argument
that absolute error is not comparable between splits rests on that spread,
so the figure is rendered here rather than written by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBE = Path("results/probes/target_stats.json")


def run() -> str:
    splits = json.loads(PROBE.read_text())["splits"]
    lines = [
        "| split | patches | non-zero pixels | zero-predictor error |",
        "|-------|---------|-----------------|----------------------|",
    ]
    errors = []
    for name in ("train", "val", "test"):
        s = splits[name]
        errors.append(s["zero_predictor_mse"])
        lines.append(
            f"| {name} | {s['patches']:,} | {s['nonzero_fraction_of_valid']:.3%} "
            f"| {s['zero_predictor_mse']:.2e} |"
        )
    spread = max(errors) / min(errors)
    lines += [
        "",
        f"The zero-predictor error spans a factor of {spread:.1f} across splits, so "
        "**absolute error is not comparable between them**.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
