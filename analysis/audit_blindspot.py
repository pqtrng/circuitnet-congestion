"""Why the pixel metric is uninformative here, rendered per split.

Reads results/probes/target_stats.json. The argument the README makes at this
table is that a metric can be almost perfect and useless at once, and the two
columns that show it are a property of the target alone, measured before any
model exists: how often predicting zero is exactly right, and how few pixels
are the hotspots the task is about.

Both are rendered rather than written. The correct fraction is one minus the
non-zero fraction of valid pixels; the hotspot fraction is the pixel share
above the hotspot threshold, the same strict comparison the metric uses. A
reader meeting the ratios elsewhere in the README meets here the reason a
ratio, not an absolute error, is the only comparable quantity.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBE = Path("results/probes/target_stats.json")
THRESHOLD_KEY = "0.05"
SPLITS = ("train", "val", "test")


def run() -> str:
    splits = json.loads(PROBE.read_text())["splits"]
    lines = [
        "| split | valid pixels | zero-predictor correct | pixels that are hotspots |",
        "|---|---|---|---|",
    ]
    for name in SPLITS:
        split = splits[name]
        correct = 1 - split["nonzero_fraction_of_valid"]
        hotspot = split["error_share_above"][THRESHOLD_KEY]["pixel_fraction"]
        lines.append(f"| {name} | {split['valid_pixels']:,} | {correct:.1%} | {hotspot:.3%} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
