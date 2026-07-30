"""Headline audit: the selection table rendered for the README.

Reads results/probes/selection_gap.json and emits a markdown table of the two
canonical runs under the two selection rules the README compares. Values are
never written by hand in the README; this module is the only place they enter
it. Bold marks the hotspot F1 of the F1-selected rows, and the error ratio of
an F1-selected row when it exceeds the zero-predictor while the same run's
error-selected row sits below it -- that crossing is the point of the table.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBE = Path("results/probes/selection_gap.json")

LOSS_LABELS = {"masked_mse": "plain squared error", "masked_weighted_mse": "weighted"}
RULES = (("val_mse", "lowest error"), ("val_f1", "highest F1"))


def run() -> str:
    record = json.loads(PROBE.read_text())
    lines = [
        "| run | selected by | epoch | val error / zero-predictor | hotspot F1 |",
        "|---------------------|--------------|-------|----------------------------|------------|",
    ]
    for name in sorted(record["runs"]):
        r = record["runs"][name]
        if r["superseded"]:
            continue
        label = LOSS_LABELS[r["loss"]]
        error_rule_ratio = r["selections"]["val_mse"]["val_mse_over_baseline"]
        for rule, rule_label in RULES:
            sel = r["selections"][rule]
            ratio = sel["val_mse_over_baseline"]
            f1 = sel["val_f1"]
            crossing = rule == "val_f1" and ratio > 1.0 and error_rule_ratio < 1.0
            ratio_cell = f"**{ratio:.3f}**" if crossing else f"{ratio:.3f}"
            f1_cell = f"**{f1:.4f}**" if rule == "val_f1" else f"{f1:.4f}"
            lines.append(f"| {label} | {rule_label} | {sel['epoch']} | {ratio_cell} | {f1_cell} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
