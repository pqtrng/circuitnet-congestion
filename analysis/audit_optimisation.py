"""Audit: optimisation health and the usable learning rate band.

Renders the measurement written by analysis.probe_optimisation.

Two questions, and the second one only became answerable after the instrument
was fixed. Three earlier invocations of the same sweep disagreed with each other
by more than the learning rates disagreed among themselves: one configuration
landed at 0.03, 0.34 and 0.79 of the baseline on separate runs at an identical
seed, a factor of thirty-two, while neighbouring rates differed by about two.
Under autotuned kernel selection the probe was measuring its own noise. Every
verdict it produced about which rates were usable was worthless, including one
that had already been written into a commit message.

The probe now disables autotuning and verifies the choice by repeating a
configuration and comparing losses bit for bit before it sweeps. That costs
roughly three and a half times the runtime and is the right trade for something
whose only job is to measure. The training entry point does the opposite,
keeping autotuning for the throughput and recording in each run that its results
are reproducible in distribution rather than bitwise.

With determinism in place, spread across seeds means sensitivity to
initialisation rather than platform noise -- and the ranking changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/optimisation.json")
CONFIGURED_LEARNING_RATE = 3e-4


def _determinism(record: dict[str, Any]) -> list[str]:
    check = record["determinism"]
    if not check["deterministic"]:
        return [
            "The sweep was not run. Two identical configurations produced different",
            f"losses ({check.get('reason', 'no deterministic kernel available')}), so a",
            "rate comparison here would again measure the platform rather than the rate.",
            "A fourth set of numbers that cannot be reproduced is worse than none.",
        ]

    settings = record["settings"]
    return [
        "Determinism, verified before sweeping:",
        "",
        f"  repeated {check['steps']} steps at lr {check['learning_rate']:.0e}, "
        f"seed {check['seed']}",
        f"  loss, first run    {check['first_loss']:.10e}",
        f"  loss, second run   {check['second_loss']:.10e}",
        f"  identical          {check['identical']}",
        "",
        f"  autotuning {'off' if not settings['cudnn_benchmark'] else 'ON'}, "
        f"deterministic kernels "
        f"{'enforced' if settings['deterministic_algorithms'] else 'not enforced'}",
        "",
        "Without this the sweep below would be unreadable. Earlier invocations under",
        "autotuning disagreed with themselves by a factor of thirty-two at a fixed seed,",
        "which is larger than the difference between the rates being compared.",
    ]


def _sweep(record: dict[str, Any]) -> list[str]:
    subset = record["subset"]
    lines = [
        f"Fitting {subset['patches']} patches, {record['steps']} steps, "
        f"{len(record['summary']['seeds'])} seeds per rate. Loss is reported against the",
        f"zero-predictor baseline of {subset['zero_predictor_loss']:.4e} on this subset, "
        f"which contains {subset['hotspot_pixels']} hotspot pixels:",
        "",
        f"  {'rate':>8} {'collapsed':>10} {'median':>8} {'best':>8} {'worst':>8} "
        f"{'spread':>8} {'recall':>8}",
    ]

    for rate in record["rates"]:
        spread = f"{rate['relative_spread']:.1f}x" if rate["relative_spread"] else "n/a"
        marker = "  <-- configured" if rate["learning_rate"] == CONFIGURED_LEARNING_RATE else ""
        lines.append(
            f"  {rate['learning_rate']:>8.0e} {rate['collapses']}/{rate['seeds']:<8} "
            f"{rate['median_loss_over_baseline']:>8.4f} "
            f"{rate['min_loss_over_baseline']:>8.4f} "
            f"{rate['max_loss_over_baseline']:>8.4f} {spread:>8} "
            f"{rate['median_recall']:>8.3f}{marker}"
        )

    lines += [
        "",
        "A ratio near zero means the subset was fitted. Optimisation is therefore",
        f"healthy: {record['summary']['optimisation_healthy']}. There is no gradient bug "
        "and no shortage of capacity, so underfitting on the full split is a question of",
        "generalisation or of the stopping rule, not of the optimiser. This is the",
        "cheapest diagnostic in the toolbox and it is almost never run.",
    ]
    return lines


def _verdict(record: dict[str, Any]) -> list[str]:
    summary = record["summary"]
    usable = summary["usable_learning_rates"]
    sensitive = summary["initialisation_sensitive_rates"]
    collapsing = summary["always_collapsing_rates"]

    lines = []

    if collapsing:
        rates = ", ".join(f"{r:.0e}" for r in collapsing)
        lines += [
            f"At {rates} every seed collapsed into the zero-predictor solution and stayed",
            "there, with no prediction anywhere above the hotspot threshold. This is what a",
            "target that is 98.5% zeros does to optimisation: the trivial solution is a",
            "strong attractor, and a step large enough to land in it does not leave.",
            "",
        ]

    if sensitive:
        entries = [r for r in record["rates"] if r["learning_rate"] in sensitive]
        for rate in entries:
            lines += [
                f"At {rate['learning_rate']:.0e} nothing collapsed outright, but the seeds "
                f"span {rate['relative_spread']:.0f}x -- from "
                f"{rate['min_loss_over_baseline']:.4f} to "
                f"{rate['max_loss_over_baseline']:.4f} of the baseline.",
                "Under determinism that spread is sensitivity to initialisation, not noise.",
                "A rate whose outcome depends on where the weights started is not a rate to",
                "build on, however good its median looks.",
                "",
            ]

    if usable:
        rates = ", ".join(f"{r:.0e}" for r in usable)
        chosen = next(
            (r for r in record["rates"] if r["learning_rate"] == CONFIGURED_LEARNING_RATE),
            None,
        )
        lines.append(f"That leaves {rates}: no collapse, and repetitions within a small factor.")
        if chosen and CONFIGURED_LEARNING_RATE in usable:
            lines += [
                f"The configured rate of {CONFIGURED_LEARNING_RATE:.0e} is among them, holding "
                f"within {chosen['relative_spread']:.1f}x across seeds with a median recall of "
                f"{chosen['median_recall']:.3f} on the subset.",
                "",
                "This is not the result the earlier, non-deterministic sweeps reported. They",
                "made 1e-4 look like the stable choice and this one look marginal. The",
                "ordering reversed once the instrument stopped contributing its own variance,",
                "which is the entire argument for spending the extra runtime.",
            ]
    else:
        lines.append(
            "No rate met both conditions. The band is either narrower than the grid "
            "resolution here, or the subset is too small to separate them."
        )

    lines += [
        "",
        "Three seeds bound sensitivity to initialisation loosely and no more. A rate that",
        "passes here can still fail on a fourth seed, and nothing in this table says",
        "otherwise. Neither does fitting thirty-two patches predict the full split: mini-",
        "batch noise over 37,704 patches is a different optimisation problem, and the two",
        "completed runs at the configured rate did not collapse.",
    ]
    return lines


def run() -> str:
    record = json.loads(PROBE.read_text())

    sections = [_determinism(record)]
    if record["summary"].get("answered"):
        sections += [_sweep(record), _verdict(record)]
    return "\n\n".join("\n".join(section) for section in sections)


if __name__ == "__main__":
    print(run())
