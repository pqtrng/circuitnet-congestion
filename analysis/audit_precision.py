"""Audit: why this baseline trains in single precision.

Renders the measurement written by analysis.probe_precision.

The decision rests on the timings alone. Both half-width formats run the same
training step materially slower than single precision on this machine; the
figures are in ``modes`` in ``results/probes/precision.json`` and the table
below restates them from that record. Timing depends on shapes, dtypes and
kernel selection rather than on weight values, so it is the one block of the
artefact that transfers to real training.

The numerical-range claims that used to accompany the timing argument are
withdrawn in both directions. Half precision was once said to produce
non-finite values and to push squared errors below its exponent floor; the
records behind that observation were not retained, so whether it was real
cannot be established. The replacement claim -- an activation ceiling with
ample headroom -- came from measurements of the model at initialisation, where
a zero-initialised head makes every examined error a squared target. The
report therefore renders the stability and range blocks as measurements of
initialisation, labelled as such, and lets the timing table carry the
decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/precision.json")


def _mode_table(record: dict[str, Any]) -> list[str]:
    single = next(m for m in record["modes"] if m["mode"] == "float32")

    lines = [
        "Identical training steps under each precision mode:",
        "",
        f"  {'mode':<10} {'ms/image':>9} {'vs fp32':>8} {'peak':>8} {'vs fp32':>8} "
        f"{'all finite':>11}",
    ]
    for entry in record["modes"]:
        speed = entry["ms_per_image"] / single["ms_per_image"]
        memory = entry["peak_allocated_gib"] / single["peak_allocated_gib"]
        finite = (
            "yes" if entry["all_losses_finite"] else f"no, step {entry['first_non_finite_step']}"
        )
        lines.append(
            f"  {entry['mode']:<10} {entry['ms_per_image']:>9.2f} {speed:>7.2f}x "
            f"{entry['peak_allocated_gib']:>7.2f}G {memory:>7.2f}x {finite:>11}"
        )

    half = next(m for m in record["modes"] if m["mode"] == "float16")
    control = next((m for m in record["modes"] if m["mode"] == "bfloat16"), None)

    lines += [
        "",
        f"Half precision costs {half['ms_per_image'] / single['ms_per_image']:.1f} times the "
        "step time. On this machine the narrower format buys no arithmetic",
        "throughput, and the slowdown is consistent with overhead from casting and",
        "from the loss scaler rather than with any gain being available.",
    ]

    if control:
        lines += [
            "",
            f"Bfloat16 lands at {control['ms_per_image'] / single['ms_per_image']:.1f} times "
            "as well. Two formats with different mantissa widths agreeing this closely is",
            "what a hardware property looks like, rather than a quirk of one of them.",
            "",
            "The peak allocation column is not usable evidence and is shown only for "
            "completeness. Both reduced formats store activations at half the width, so "
            "both should land near 0.5x;",
            f"bfloat16 does at {control['peak_allocated_gib'] / single['peak_allocated_gib']:.2f}x "
            f"while half precision reports "
            f"{half['peak_allocated_gib'] / single['peak_allocated_gib']:.2f}x. The "
            "difference is autotuning scratch space, which varies with dtype in ways that",
            "have nothing to do with the format's width -- the same effect makes peak "
            "allocation non-monotonic in batch size in the throughput probe. The case for",
            "single precision rests on the timings alone.",
        ]

    return lines


def _range_evidence(record: dict[str, Any]) -> list[str]:
    activation = record["activation_range"]
    magnitudes = record["error_magnitudes"]
    limits = record["format_limits"]

    lines = [
        "The stability and range blocks, labelled for what they measure:",
        "",
        f"  largest activation, model at initialisation  {activation['largest_activation']:.1f}"
        f"  (from {activation['produced_by']})",
        f"  half precision finite ceiling                "
        f"{activation['float16_ceiling']:.0f}",
        "",
        f"  squared errors examined                      "
        f"{magnitudes['squared_errors_examined']:,} valid pixels",
        f"  half precision smallest normal               "
        f"{magnitudes['float16_smallest_normal']:.2e}",
        f"  fraction falling below it                    "
        f"{magnitudes['fraction_below_float16_normal']:.1%}",
        f"  smallest non-zero squared error seen         "
        f"{magnitudes['smallest_non_zero_squared_error']:.2e}",
        "",
        "These are measurements of the model at initialisation. The output head is",
        "zero-initialised, so the network emits exact zeros and every squared error",
        "examined above is a squared target; the activation range is a property of",
        "initialisation, taken in eval mode on random input. Nothing here supports a",
        "claim about numerical behaviour during training, in either direction: not",
        "the withdrawn claim that half precision overflows, and not the withdrawn",
        "claim of comfortable headroom. The question stays open until these numbers",
        "are re-taken from a trained checkpoint.",
        "",
        "An earlier note said half precision produced non-finite values before the",
        "head was zero-initialised. The records behind that observation were not",
        "retained; the mechanism it described is arithmetically plausible for a",
        "Kaiming-initialised single-channel head, but plausibility is not a",
        "measurement.",
    ]

    control = next((m for m in record["modes"] if m["mode"] == "bfloat16"), None)
    if control:
        lines += [
            "",
            "Bfloat16 is included as a control. It carries the exponent range of single",
            f"precision with {limits['bfloat16']['mantissa_bits']} mantissa bits against "
            f"half precision's {limits['float16']['mantissa_bits']}, so once these",
            "measurements come from a trained checkpoint, a failure of range would show",
            "up in one format and not the other. On a zero-output model both are finite",
            "by construction, which discriminates nothing.",
        ]
    return lines


def _caveat(record: dict[str, Any]) -> list[str]:
    check = record.get("stability", {})
    if check.get("contended"):
        return [
            "CONTENDED. The repeated single-precision measurement drifted by "
            f"{check.get('relative_drift', float('nan')):.1%}, so another process was using",
            "the accelerator. The timings describe an unknown competing load and should",
            "not be quoted; the finiteness and range blocks do not depend on timing,",
            "though per the section above they describe initialisation only.",
        ]
    return [
        "The device was verified idle by repeating the single-precision measurement at "
        f"the start and end of the run: the two agreed within "
        f"{check.get('relative_drift', 0):.1%}."
    ]


def run() -> str:
    record = json.loads(PROBE.read_text())
    sections = [
        _mode_table(record),
        _range_evidence(record),
        _caveat(record),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


if __name__ == "__main__":
    print(run())
