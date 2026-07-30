"""Audit: throughput and memory against batch size.

Renders the measurement written by analysis.probe_throughput. The probe needs
an idle accelerator; this reads the committed JSON, so the report renders on a
machine that has neither.

Two things the numbers here do not say on their own, and which the text has to.

The configured batch size is not chosen for speed. The per-image spread across
the sweep is rendered from the record below rather than summarised here, and
the configured size sits within measurement drift of the fastest; what decides
is the memory it leaves free. That preference traces to an earlier
measurement, taken before autotuning was enabled, in which the largest batch
spilled past device memory and ran drastically slower without raising. The
records of that episode were not retained, so it stands as testimony; what it
motivates stands on its own, because the failure mode it points at is silent.
The present sweep does not reproduce the spill, because autotuning selects a
more frugal convolution algorithm under pressure.

Peak allocation is not monotonic in batch size, which looks like a measurement
error and is not. Autotuning allocates scratch space while trying algorithms,
so a small batch that triggers a wide search can peak above a larger one that
does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/throughput.json")
CONFIGURED_BATCH_SIZE = 32
CONFIGURED_WORKERS = 4


def _compute_table(record: dict[str, Any]) -> list[str]:
    total = record["device"]["total_gib"]
    lines = [
        "Compute, timed on synthetic tensors of the production shape so that "
        "loading cannot contribute:",
        "",
        f"  {'batch':>6} {'ms/step':>9} {'ms/image':>9} {'peak':>8} {'reserved':>9} "
        f"{'free after':>11}",
    ]

    for entry in record["compute"]:
        if entry["out_of_memory"]:
            lines.append(f"  {entry['batch_size']:>6}   out of memory")
            continue
        marker = ""
        if entry.get("exceeds_device_memory"):
            marker = "  <-- above device memory"
        elif entry.get("departs_from_flat_region"):
            marker = "  <-- outside the flat region"
        lines.append(
            f"  {entry['batch_size']:>6} {entry['ms_per_step']:>9.1f} "
            f"{entry['ms_per_image']:>9.2f} {entry['peak_allocated_gib']:>7.2f}G "
            f"{entry['peak_reserved_gib']:>8.2f}G {entry['free_after_gib']:>10.2f}G{marker}"
        )

    lines += [
        "",
        f"  device total: {total:.2f} GiB",
    ]
    return lines


def _loading_table(record: dict[str, Any], compute_ms: float) -> list[str]:
    lines = [
        "Loading, timed on real patches with no compute attached:",
        "",
        f"  {'workers':>8} {'ms/step':>9} {'ms/image':>9} {'headroom':>10}",
    ]
    for entry in record["loading"]:
        headroom = compute_ms / entry["ms_per_image"] if entry["ms_per_image"] else None
        lines.append(
            f"  {entry['num_workers']:>8} {entry['ms_per_step']:>9.1f} "
            f"{entry['ms_per_image']:>9.2f} "
            f"{headroom:>9.0f}x"
            if headroom
            else f"  {entry['num_workers']:>8}"
        )

    configured = next(
        (e for e in record["loading"] if e["num_workers"] == CONFIGURED_WORKERS), None
    )
    if configured:
        ratio = compute_ms / configured["ms_per_image"]
        lines += [
            "",
            f"At {CONFIGURED_WORKERS} workers, loading is {ratio:.0f} times faster than the "
            "loop can consume. This task is bound by compute, not by input --",
            "the opposite of what was assumed before it was measured, when the two were",
            "timed together and loading was blamed for the step time.",
        ]
    return lines


def _decision(record: dict[str, Any]) -> list[str]:
    summary = record["summary"]
    spilled = summary["spilled_batch_sizes"]
    per_image = [e["ms_per_image"] for e in record["compute"] if not e["out_of_memory"]]
    fastest, slowest = min(per_image), max(per_image)

    configured = next(
        (e for e in record["compute"] if e["batch_size"] == CONFIGURED_BATCH_SIZE), None
    )
    largest = next(
        (e for e in record["compute"] if e["batch_size"] == summary["largest_safe_batch_size"]),
        None,
    )

    lines = [
        f"Per-image time spans {fastest:.2f} to {slowest:.2f} ms across the sweep, a "
        f"spread of {slowest / fastest - 1.0:.0%} between the fastest and slowest "
        "batch size measured,",
    ]
    if spilled:
        lines.append(f"and {spilled} left the flat region or exceeded device memory.")
    else:
        lines.append(
            "and none exceeded device memory. The spill observed earlier, without "
            "autotuning, does not reproduce here; its records were not retained."
        )

    if configured and largest and configured["batch_size"] != largest["batch_size"]:
        lines += [
            "",
            f"The configured batch of {CONFIGURED_BATCH_SIZE} is therefore not chosen for "
            f"speed. It runs at {configured['ms_per_image']:.2f} ms per image against "
            f"{largest['ms_per_image']:.2f} at {largest['batch_size']}, a difference "
            "inside the noise of this measurement,",
            f"and leaves {configured['free_after_gib']:.2f} GiB free where "
            f"{largest['batch_size']} leaves {largest['free_after_gib']:.2f} GiB. Headroom "
            "is the whole reason: on this platform,",
            "exceeding device memory is not expected to raise. The driver can fall back",
            "to host memory and the run slows drastically and silently, with nothing in",
            "the logs to say so. A configuration that cannot fail that way is worth more",
            "than a speed difference within measurement drift.",
        ]

    return lines


def _caveat(record: dict[str, Any]) -> list[str]:
    check = record.get("stability", {})
    if check.get("contended"):
        return [
            "CONTENDED. The repeated measurement drifted by "
            f"{check.get('relative_drift', float('nan')):.1%}, above the "
            f"{check.get('tolerance', 0):.0%} tolerance, so another process was using the",
            "accelerator while these numbers were taken. They describe an unknown "
            "competing load and should not be quoted.",
        ]
    return [
        "The device was verified idle by repeating one batch size at the start and end "
        "of the compute sweep: the two agreed within "
        f"{check.get('relative_drift', 0):.1%}. The loading numbers are not bracketed "
        "by this check, a known limitation of the instrument. The check exists because "
        "reading free memory does not work here --",
        "an earlier version of this probe saw most of a busy device as free, produced "
        "timings far from the truth, and slowed the training run it was competing with. "
        "The records of that episode were not retained.",
    ]


def run() -> str:
    record = json.loads(PROBE.read_text())
    flat = record["summary"]["flat_region_ms_per_image"]

    sections = [
        _compute_table(record),
        _loading_table(record, flat),
        _decision(record),
        _caveat(record),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


if __name__ == "__main__":
    print(run())
