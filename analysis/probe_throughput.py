"""Probe: training throughput and memory as a function of batch size.

Exceeding accelerator memory does not raise on every platform. The driver may
fall back to host memory and keep running, which shows up only as throughput
collapsing, not as an error. A configuration chosen from a memory figure alone
can therefore run much slower than one chosen from a measurement, with nothing
in the logs to say so. This probe measures both.

Two quantities are measured separately and from different sources. Compute is
timed on synthetic tensors so that loading cannot contribute to it, and loading
is timed on the real Gold layer because that is the thing worth knowing. Timing
them together is what first led to the wrong conclusion that this task is bound
by input rather than by compute.

Detecting that the device is busy turned out to be the hard part. On a
virtualised accelerator a fresh process does not necessarily observe memory
held by another process: an earlier version of this probe read most of the
device as free while a training run was using it, produced numbers far from
the truth, and slowed that training run in the process. The records of that
episode were not retained, so it stands as testimony; the design consequence
stands on its own. Sampling free memory repeatedly does not fix the blindness
either, because a caching allocator holds a steady reservation.

The check that does work is repetition. One batch size is timed twice, at the
start and at the end of the compute sweep. Contention slows everything
consistently, so the two timings diverge; on an idle device they agree within
a few percent. If they disagree the record is written with `contended` set and
must not be reported. The check brackets the compute sweep only: the loading
sweep runs after the second timing and is not covered by it, which is a known
limitation of this instrument rather than a property of the numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from circuitnet_congestion.models.unet import UNet
from circuitnet_congestion.training.dataset import CongestionPatchDataset, build_dataloader
from circuitnet_congestion.training.losses import MaskedMSELoss

OUTPUT = Path("results/probes/throughput.json")

BATCH_SIZES = (8, 16, 24, 32, 40, 48, 64)
WORKER_COUNTS = (0, 2, 4, 8)

# Repeated at the start and the end of the sweep to detect a competing load.
STABILITY_BATCH_SIZE = 16
STABILITY_TOLERANCE = 0.05

WARMUP_STEPS = 5
MEASURED_STEPS = 20
LOADER_BATCHES = 30

# Retained as a cheap first filter. It catches an obviously occupied device but
# cannot be relied on alone; see the module docstring.
ADVISORY_FREE_FRACTION = 0.80

# A per-image time this many times the flat-region median is treated as spill
# rather than as scaling.
SPILL_FACTOR = 2.0

BYTES_PER_GIB = 2**30


def device_state() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": free / BYTES_PER_GIB,
        "total_gib": total / BYTES_PER_GIB,
        "free_fraction": free / total,
    }


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """One optimiser step.

    Defined at module level rather than as a closure so that the caller is free
    to release its tensors afterwards.
    """
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(features), target, mask).backward()
    optimizer.step()


def measure_compute(batch_size: int, device: torch.device) -> dict[str, Any]:
    """Time a training step on synthetic tensors of the production shape."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = MaskedMSELoss()

    features = torch.randn(batch_size, 3, 128, 128, device=device)
    target = torch.zeros(batch_size, 1, 128, 128, device=device)
    mask = torch.ones(batch_size, 1, 128, 128, device=device)

    try:
        for _ in range(WARMUP_STEPS):
            _training_step(model, optimizer, loss_fn, features, target, mask)
        torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(MEASURED_STEPS):
            _training_step(model, optimizer, loss_fn, features, target, mask)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError:
        # Raising here is the benign failure. The interesting case is the
        # platform that does not raise and silently spills instead.
        del model, optimizer, features, target, mask
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "out_of_memory": True}

    result = {
        "batch_size": batch_size,
        "out_of_memory": False,
        "ms_per_step": elapsed / MEASURED_STEPS * 1000,
        "ms_per_image": elapsed / MEASURED_STEPS / batch_size * 1000,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / BYTES_PER_GIB,
        "free_after_gib": device_state()["free_gib"],
    }

    del model, optimizer, features, target, mask
    torch.cuda.empty_cache()
    return result


def measure_loading(
    dataset: CongestionPatchDataset, batch_size: int, num_workers: int
) -> dict[str, Any]:
    """Time pure iteration over the real Gold layer, with no compute attached."""
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=0,
        pin_memory=False,
    )
    iterator = iter(loader)

    for _ in range(WARMUP_STEPS):
        next(iterator)

    started = time.perf_counter()
    for _ in range(LOADER_BATCHES):
        next(iterator)
    elapsed = time.perf_counter() - started

    del iterator, loader
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "ms_per_step": elapsed / LOADER_BATCHES * 1000,
        "ms_per_image": elapsed / LOADER_BATCHES / batch_size * 1000,
    }


def stability(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare the repeated measurement; divergence means a competing load."""
    if first["out_of_memory"] or second["out_of_memory"]:
        return {"contended": True, "reason": "the repeated measurement ran out of memory"}

    a, b = first["ms_per_image"], second["ms_per_image"]
    drift = abs(a - b) / min(a, b)
    return {
        "batch_size": first["batch_size"],
        "first_ms_per_image": a,
        "second_ms_per_image": b,
        "relative_drift": drift,
        "tolerance": STABILITY_TOLERANCE,
        "contended": drift > STABILITY_TOLERANCE,
    }


def analyse(compute: list[dict[str, Any]], total_gib: float) -> dict[str, Any]:
    """Identify the flat region and flag configurations that leave it."""
    usable = [r for r in compute if not r["out_of_memory"]]
    if not usable:
        return {"flat_region_ms_per_image": None, "spilled_batch_sizes": []}

    times = sorted(r["ms_per_image"] for r in usable)
    median = times[len(times) // 2]

    spilled = []
    for result in usable:
        over_memory = result["peak_allocated_gib"] > total_gib
        slow = result["ms_per_image"] > median * SPILL_FACTOR
        result["exceeds_device_memory"] = over_memory
        result["departs_from_flat_region"] = slow
        if over_memory or slow:
            spilled.append(result["batch_size"])

    flat = [r for r in usable if r["batch_size"] not in spilled]
    chosen = min(flat, key=lambda r: (r["ms_per_image"], -r["free_after_gib"])) if flat else None

    return {
        "flat_region_ms_per_image": median,
        "spilled_batch_sizes": spilled,
        "largest_safe_batch_size": max((r["batch_size"] for r in flat), default=None),
        "fastest_per_image": None if chosen is None else chosen["batch_size"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe training throughput and memory.")
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--loader-batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no accelerator available; this probe measures accelerator throughput")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    start_state = device_state()

    print(
        f"device: {start_state['total_gib']:.2f} GiB total, {start_state['free_gib']:.2f} GiB free"
    )
    if start_state["free_fraction"] < ADVISORY_FREE_FRACTION:
        print("  the device appears occupied; the stability check below is the decisive one")
    print()

    reference = measure_compute(STABILITY_BATCH_SIZE, device)

    print("compute, synthetic tensors of the production shape")
    print(f"  {'batch':>6} {'ms/step':>9} {'ms/img':>8} {'peak':>8} {'reserved':>9} {'free':>8}")
    compute = []
    for batch_size in BATCH_SIZES:
        result = measure_compute(batch_size, device)
        compute.append(result)
        if result["out_of_memory"]:
            print(f"  {batch_size:>6}   out of memory")
            continue
        print(
            f"  {batch_size:>6} {result['ms_per_step']:>9.1f} {result['ms_per_image']:>8.2f} "
            f"{result['peak_allocated_gib']:>7.2f}G {result['peak_reserved_gib']:>8.2f}G "
            f"{result['free_after_gib']:>7.2f}G"
        )

    repeat = measure_compute(STABILITY_BATCH_SIZE, device)
    check = stability(reference, repeat)

    print("\nloading, real patches, no compute attached")
    print(f"  {'workers':>8} {'ms/step':>9} {'ms/img':>8}")
    dataset = CongestionPatchDataset(args.gold_dir, "train")
    loading = []
    for num_workers in WORKER_COUNTS:
        result = measure_loading(dataset, args.loader_batch_size, num_workers)
        loading.append(result)
        print(f"  {num_workers:>8} {result['ms_per_step']:>9.1f} {result['ms_per_image']:>8.2f}")

    summary = analyse(compute, start_state["total_gib"])

    record = {
        "device": start_state,
        "steps": {"warmup": WARMUP_STEPS, "measured": MEASURED_STEPS},
        "stability": check,
        "compute": compute,
        "loading": loading,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    if check["contended"]:
        raise SystemExit(
            f"\nCONTENDED: the repeated measurement at batch {STABILITY_BATCH_SIZE} drifted "
            f"by {check.get('relative_drift', float('nan')):.1%}, above the "
            f"{STABILITY_TOLERANCE:.0%} tolerance. Another process was using the "
            f"accelerator, so these numbers describe an unknown competing load rather "
            f"than this configuration. Re-run on an idle device; the record was written "
            f"with contended set so it cannot be mistaken for a result."
        )

    print(f"flat region: {summary['flat_region_ms_per_image']:.2f} ms/image")
    print(f"spilled at batch sizes: {summary['spilled_batch_sizes'] or 'none observed'}")
    print(f"largest safe batch size: {summary['largest_safe_batch_size']}")


if __name__ == "__main__":
    main()
