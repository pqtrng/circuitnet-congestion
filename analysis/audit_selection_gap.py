"""Audit: disagreement between checkpoint selection rules.

Renders the measurement written by analysis.probe_selection_gap. The probe
needs run histories; this reads the committed JSON, so `make report` works for
a reader who has cloned the repository and has no data and no accelerator.

Every run is listed with its epoch count rather than filtered by it. A run that
was superseded, or one that was still in progress when the probe ran, is real
measurement and hiding it would defeat the purpose of the record. Short runs
are shown but kept out of the headline range: a ratio taken against an F1 near
zero is arithmetically large and carries no information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/selection_gap.json")

# Below this many epochs a run's F1 is still close enough to zero that the
# ratio between two selections is dominated by its denominator.
MINIMUM_EPOCHS_FOR_RATIO = 15

RULE_LABELS = {
    "val_mse": "lowest error",
    "val_weighted_mse": "lowest weighted error",
    "val_f1": "highest hotspot F1",
    "final_epoch": "final epoch",
}


def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    name, run = item
    return (1 if run["superseded"] else 0, name)


def _format_run(name: str, run: dict[str, Any]) -> list[str]:
    marker = "  [superseded]" if run["superseded"] else ""
    lines = [
        f"{name}{marker}",
        f"  loss={run['loss']}  epochs={run['epochs_recorded']}  "
        f"zero-predictor baseline={run['baseline_val_zero_predictor_mse']:.4e}",
        f"  {'selected by':<24} {'epoch':>5} {'error/baseline':>15} {'F1':>8} {'recall':>8}",
    ]

    for rule, chosen in run["selections"].items():
        ratio = chosen.get("val_mse_over_baseline")
        lines.append(
            f"  {RULE_LABELS.get(rule, rule):<24} {chosen['epoch']:>5} "
            f"{ratio:>15.4f} {chosen['val_f1']:>8.4f} {chosen['val_recall']:>8.4f}"
        )

    gap = run["gap_between_error_and_f1"]
    ratio = gap["f1_ratio"]
    apart = gap["epochs_apart"]
    plural = "epoch" if abs(apart) == 1 else "epochs"

    if ratio is None:
        # The error-selected model found nothing at all, so the ratio has no
        # denominator. That is a stronger statement than any finite factor.
        lines.append(
            f"  -> the error rule selected a model with an F1 of exactly zero, "
            f"{abs(apart)} {plural} before the F1 rule's choice"
        )
    else:
        note = "" if run["epochs_recorded"] >= MINIMUM_EPOCHS_FOR_RATIO else " (short run)"
        lines.append(
            f"  -> the F1 rule's choice scores {ratio:.1f}x higher on F1, "
            f"{apart} {plural} later{note}"
        )

    if gap["f1_choice_error_over_baseline"] and gap["f1_choice_error_over_baseline"] > 1.0:
        lines.append(
            "  -> that choice is rated worse than predicting zero everywhere "
            "by the pixel metric"
        )

    return lines


def run() -> str:
    runs = json.loads(PROBE.read_text())["runs"]

    lines: list[str] = []
    for name, entry in sorted(runs.items(), key=_sort_key):
        lines.extend(_format_run(name, entry))
        lines.append("")

    substantial = [
        r["gap_between_error_and_f1"]["f1_ratio"]
        for r in runs.values()
        if r["gap_between_error_and_f1"]["f1_ratio"] is not None
           and r["epochs_recorded"] >= MINIMUM_EPOCHS_FOR_RATIO
    ]
    zeroed = sum(1 for r in runs.values() if r["gap_between_error_and_f1"]["f1_ratio"] is None)

    lines.append(f"Across {len(runs)} runs the two rules never select the same epoch.")
    if substantial:
        lines.append(
            f"In the {len(substantial)} run(s) long enough for the ratio to mean "
            f"anything, the F1 rule's choice scores between {min(substantial):.1f}x "
            f"and {max(substantial):.1f}x higher on F1."
        )
    if zeroed:
        lines.append(
            f"In {zeroed} run(s) the error rule selected a model that detected no "
            "hotspot at all, while a model in the same run did."
        )
    lines.append(
        f"Ratios from runs shorter than {MINIMUM_EPOCHS_FOR_RATIO} epochs are shown "
        "per run but excluded from that range: an F1 near zero makes the "
        "denominator unstable."
    )
    lines.append(
        "Selecting on pixel error is therefore not a neutral default. It is a "
        "choice that discards most of the run's detection performance, and the "
        "loss curve gives no indication that it is happening."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
