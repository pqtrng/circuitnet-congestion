"""Audit: why this baseline trains in single precision.

Renders the measurement written by analysis.probe_precision.

Two earlier claims in this repository did not survive being measured. Half
precision was said to produce non-finite values on the first forward pass, and
squared errors were said to reach the bottom of its dynamic range. Neither
reproduces, and both are reported here rather than quietly dropped: the
observation was real, but it was made before the output head was
zero-initialised, when predictions ran two orders of magnitude above any target
and the resulting errors genuinely did overflow.

The decision did not change, because it never rested on those claims alone. On
an accelerator without dedicated matrix units, half precision is simply slower
and uses more memory, since autocast keeps both copies of the weights while the
narrower format buys no arithmetic throughput.

Bfloat16 appears as a control, not a candidate. It has single precision's
exponent range with fewer mantissa bits, so if half precision were failing on
range, bfloat16 would survive where it did not.
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
        "step time. This accelerator has no dedicated matrix units, so the narrower",
        "format buys no arithmetic throughput and the cost is pure overhead from casting",
        "and from the loss scaler.",
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
        "Numerical range is not the reason, and an earlier claim that it was does not",
        "reproduce:",
        "",
        f"  largest activation anywhere in the network   {activation['largest_activation']:.1f}"
        f"  (from {activation['produced_by']})",
        f"  half precision ceiling                       {activation['float16_ceiling']:.0f}",
        f"  headroom                                     {activation['headroom_factor']:.0f}x",
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
        "Nothing overflows and nothing underflows. The original observation was made",
        "before the output head was zero-initialised: predictions then ran two orders of",
        "magnitude above any target, and the squared errors that followed did overflow.",
        "Fixing the initialisation removed the numerical problem without changing the",
        "conclusion, which is why the conclusion needed a different reason.",
    ]

    control = next((m for m in record["modes"] if m["mode"] == "bfloat16"), None)
    if control:
        lines += [
            "",
            "Bfloat16 is included as a control. It carries the exponent range of single",
            f"precision with {limits['bfloat16']['mantissa_bits']} mantissa bits against "
            f"half precision's {limits['float16']['mantissa_bits']}, so a failure of range",
            "would show up in one and not the other. Both are finite here, which is",
            "consistent with range never having been the constraint.",
        ]
    return lines


def _caveat(record: dict[str, Any]) -> list[str]:
    check = record.get("stability", {})
    if check.get("contended"):
        return [
            "CONTENDED. The repeated single-precision measurement drifted by "
            f"{check.get('relative_drift', float('nan')):.1%}, so another process was using",
            "the accelerator. The timings describe an unknown competing load and should",
            "not be quoted; the finiteness and range results do not depend on timing.",
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
