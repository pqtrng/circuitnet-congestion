"""Headline audit: pixel error and hotspot detection in one table.

Reads results/probes/selection_gap.json for the trained runs and
results/probes/target_stats.json for the reference row. Values are never
written by hand in the README; this module is the only place they enter it.

The first row is the predictor that outputs zero everywhere. Without it the
baseline is present only as the denominator of a ratio, and a reader has to
reconstruct what the ratios are measured against. Its figures are not typed
here either. The ratio is that predictor's own error over the baseline the
training loop recorded -- two artefacts measuring the same quantity, so a
reference row departing from parity means they have diverged. The hotspot
figures come from HotspotCounts, the properties the training loop itself
reports, given the counts a zero prediction produces: no true positive, no
predicted positive, and every hotspot pixel in the split missed.

Precision on that row is undefined rather than bad, because nothing was
predicted positive. HotspotCounts returns zero there, a value inside the
legitimate range, so printing it in a precision column would read as a
measurement. It renders as a dash.
"""

from __future__ import annotations

import json
from pathlib import Path

from circuitnet_congestion.training.metrics import HotspotCounts

SELECTION = Path("results/probes/selection_gap.json")
TARGET_STATS = Path("results/probes/target_stats.json")

SPLIT = "val"
THRESHOLD_KEY = "0.05"
LOSS_LABELS = {"masked_mse": "plain squared error", "masked_weighted_mse": "weighted"}
RULES = (("val_mse", "lowest error"), ("val_f1", "highest F1"))
HEADER = (
    "| run | selected by | epoch | pixel error / zero-predictor | hotspot precision | recall | F1 |"
)
DIVIDER = "|---|---|---|---|---|---|---|"


def _cells(ratio: float, precision: str, recall: float, f1: float) -> str:
    marked = f"**{ratio:.3f}**" if ratio > 1.0 else f"{ratio:.3f}"
    return f"{marked} | {precision} | {recall:.4f} | {f1:.4f} |"


def run() -> str:
    selection = json.loads(SELECTION.read_text())["runs"]
    split = json.loads(TARGET_STATS.read_text())["splits"][SPLIT]

    baselines = {
        record["baseline_val_zero_predictor_mse"]
        for record in selection.values()
        if not record["superseded"]
    }
    if len(baselines) != 1:
        raise ValueError(f"canonical runs disagree on the baseline: {sorted(baselines)}")
    baseline = baselines.pop()
    hotspot_fraction = split["error_share_above"][THRESHOLD_KEY]["pixel_fraction"]
    missed = round(hotspot_fraction * split["valid_pixels"])
    reference = HotspotCounts(false_negative=missed)

    lines = [
        HEADER,
        DIVIDER,
        "| predict zero everywhere | — | — | "
        + _cells(split["zero_predictor_mse"] / baseline, "—", reference.recall, reference.f1),
    ]

    for name in sorted(selection):
        run_record = selection[name]
        if run_record["superseded"]:
            continue
        label = LOSS_LABELS[run_record["loss"]]
        for rule, rule_label in RULES:
            chosen = run_record["selections"][rule]
            lines.append(
                f"| {label} | {rule_label} | {chosen['epoch']} | "
                + _cells(
                    chosen["val_mse_over_baseline"],
                    f"{chosen['val_precision']:.4f}",
                    chosen["val_recall"],
                    chosen["val_f1"],
                )
            )

    lines += [
        "",
        "The reference row predicts zero at every pixel: on the validation split that leaves "
        f"{1 - split['nonzero_fraction_of_valid']:.1%} of valid pixels exactly right while "
        f"finding none of the {hotspot_fraction:.3%} that are hotspots. Bold marks a pixel "
        "error above it -- the metric rating a trained model worse than predicting nothing.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
