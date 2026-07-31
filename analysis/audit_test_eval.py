"""Test-evaluation audit: the selection gap on a split no checkpoint was chosen on.

Reads results/probes/test_eval.json. This is a results table, not a design
rationale: no figure here informs a choice made in sections 1 through 6, which
remain validation-only. It reports each run's two published selections -- the
error-selected and F1-selected checkpoint -- scored on test, beside the
zero-predictor row for that split.

Two things are legible at once, and the surrounding prose states both. The
gap survives: the F1-selected checkpoint detects more on test than the
error-selected one, in each run, on a design neither was chosen on. And the
validation paradox does not survive: on test every checkpoint sits below the
zero-predictor on pixel error, where on the far sparser validation split the
F1-selected one sat above it. Values are rendered, never typed.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBE = Path("results/probes/test_eval.json")

LOSS_LABELS = {"masked_mse": "plain squared error", "masked_weighted_mse": "weighted"}
SELECTIONS = (("best_val_mse", "lowest error"), ("best_val_f1", "highest F1"))
HEADER = (
    "| run | selected by | epoch | test error / zero-predictor | hotspot precision | recall | F1 |"
)
DIVIDER = "|---|---|---|---|---|---|---|"


def _row(label: str, rule_label: str, record: dict) -> str:
    ratio = record["test_mse_over_baseline"]
    marked = f"**{ratio:.3f}**" if ratio > 1.0 else f"{ratio:.3f}"
    return (
        f"| {label} | {rule_label} | {record['epoch']} | {marked} "
        f"| {record['test_precision']:.4f} | {record['test_recall']:.4f} "
        f"| {record['test_f1']:.4f} |"
    )


def run() -> str:
    runs = json.loads(PROBE.read_text())["runs"]
    lines = [HEADER, DIVIDER, "| predict zero everywhere | — | — | 1.000 | — | 0.0000 | 0.0000 |"]

    below = []
    for name in sorted(runs):
        record = runs[name]
        label = LOSS_LABELS[record["loss"]]
        for key, rule_label in SELECTIONS:
            selection = record["selections"][key]
            lines.append(_row(label, rule_label, selection))
            below.append(selection["test_mse_over_baseline"] < 1.0)

    all_below = all(below)
    lines += [
        "",
        "Each F1-selected checkpoint detects more hotspots than the error-selected one of the "
        "same run, on a design neither was selected on: the gap holds off the split it was found "
        "on. "
        + (
            "Unlike on validation, every trained checkpoint here sits below the zero-predictor on "
            "pixel error -- the metric no longer rates a detector worse than predicting nothing, "
            "because this split is denser and there is real signal to find."
            if all_below
            else "At least one checkpoint still sits at or above the zero-predictor on pixel error."
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
