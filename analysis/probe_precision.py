"""Probe: timings and numerical observations behind the single-precision choice.

What this records, stated precisely, because its output has been misread
before -- twice, in opposite directions.

Throughput. Each precision mode runs identical training steps on identical
shapes and is timed. Step time depends on shapes, dtypes and kernel selection,
not on the values of the weights, so these timings transfer to real training.
They are recorded under ``modes`` and they are the only part of this artefact
that supports the precision decision.

Activations and error magnitudes. Both are taken from the model as
constructed. The output head is zero-initialised, so the network emits exact
zeros, every squared error examined here is a squared target, and the recorded
activation range is a property of initialisation, taken in eval mode on random
input. Neither quantity describes a trained model, and neither supports a
claim about numerical behaviour during training -- not the original claim that
half precision overflows, and not the later claim of comfortable headroom.
Both readings have been withdrawn from the module docstrings that carried
them. Whether half precision fails on a trained checkpoint stays open until
these measurements are re-taken from one.

Bfloat16 is included as a control rather than as a candidate: it carries the
exponent range of single precision with fewer mantissa bits, so a range
failure and a precision failure would separate the two half-width formats. The
control becomes informative only once the quantities feeding it describe
training, which, per the paragraph above, they do not yet.
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

OUTPUT = Path("results/probes/precision.json")

BATCH_SIZE = 32
WARMUP_STEPS = 5
MEASURED_STEPS = 20
RANGE_BATCHES = 20

STABILITY_TOLERANCE = 0.05
ADVISORY_FREE_FRACTION = 0.80
BYTES_PER_GIB = 2**30

MODES: dict[str, torch.dtype | None] = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Explicit rather than derived: the bit layouts are fixed by the standard and
# reading them back out of finfo is more obscure than stating them.
MANTISSA_BITS = {"float32": 23, "float16": 10, "bfloat16": 7}

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def device_state() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": free / BYTES_PER_GIB,
        "total_gib": total / BYTES_PER_GIB,
        "free_fraction": free / total,
    }


def format_limits() -> dict[str, dict[str, float]]:
    """The representable range of each format, for the record."""
    limits = {}
    for name, dtype in DTYPES.items():
        info = torch.finfo(dtype)
        mantissa = MANTISSA_BITS[name]
        limits[name] = {
            "max": float(info.max),
            "smallest_normal": float(info.tiny),
            "smallest_subnormal": float(info.tiny) * 2.0**-mantissa,
            "mantissa_bits": mantissa,
        }
    return limits


def largest_activation(device: torch.device) -> dict[str, Any]:
    """The largest value the network produces at initialisation, on random input.

    Recorded against fp16's finite ceiling for the record. The model is freshly
    constructed and never trained, so this bounds nothing about training; see
    the module docstring.
    """
    torch.manual_seed(0)
    model = UNet().to(device).eval()
    peak: dict[str, Any] = {"value": 0.0, "module": None}

    def record(name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            largest = float(output.detach().abs().max())
            if largest > peak["value"]:
                peak["value"] = largest
                peak["module"] = name

        return hook

    handles = [
        module.register_forward_hook(record(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d | nn.BatchNorm2d | nn.ReLU)
    ]

    with torch.inference_mode():
        model(torch.randn(4, 3, 128, 128, device=device))

    for handle in handles:
        handle.remove()
    del model
    torch.cuda.empty_cache()

    ceiling = float(torch.finfo(torch.float16).max)
    value = float(peak["value"])
    return {
        "largest_activation": value,
        "produced_by": peak["module"],
        "float16_ceiling": ceiling,
        "headroom_factor": ceiling / value if value else None,
        "exceeds_float16": value > ceiling,
    }


def error_magnitudes(gold_dir: Path, device: torch.device) -> dict[str, Any]:
    """Where squared errors sit relative to half precision's exponent floor."""
    torch.manual_seed(0)
    model = UNet().to(device).eval()
    loader = build_dataloader(
        CongestionPatchDataset(gold_dir, "train"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        seed=0,
    )

    normal = float(torch.finfo(torch.float16).tiny)
    below_normal = 0
    non_zero_targets = 0
    counted = 0
    smallest = float("inf")

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= RANGE_BATCHES:
                break
            target = batch["gt"].to(device)
            mask = batch["mask"].to(device)
            squared = (model(batch["features"].to(device)) - target).square()

            values = squared[mask > 0]
            positive = values[values > 0]

            counted += int(values.numel())
            below_normal += int((positive < normal).sum())
            non_zero_targets += int((target[mask > 0] > 0).sum())
            if positive.numel():
                smallest = min(smallest, float(positive.min()))

    del model, loader
    torch.cuda.empty_cache()

    return {
        "squared_errors_examined": counted,
        "float16_smallest_normal": normal,
        "fraction_below_float16_normal": below_normal / counted if counted else None,
        "smallest_non_zero_squared_error": None if smallest == float("inf") else smallest,
        "non_zero_targets_seen": non_zero_targets,
    }


def _amp_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    features: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """One optimiser step under an autocast mode, returning the loss.

    Defined at module level rather than as a closure so that the caller can
    release its tensors afterwards.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        loss = loss_fn(model(features), target, mask)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


def train_steps(mode: str, dtype: torch.dtype | None, device: torch.device) -> dict[str, Any]:
    """Run identical steps under one precision mode and report what happened."""
    torch.manual_seed(0)
    model = UNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = MaskedMSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)

    features = torch.randn(BATCH_SIZE, 3, 128, 128, device=device)
    hit = (torch.rand(BATCH_SIZE, 1, 128, 128, device=device) < 0.016).float()
    target = hit * torch.rand(BATCH_SIZE, 1, 128, 128, device=device) * 0.1
    mask = torch.ones(BATCH_SIZE, 1, 128, 128, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    losses: list[float] = []
    first_non_finite: int | None = None

    for index in range(WARMUP_STEPS):
        loss = _amp_step(model, optimizer, loss_fn, scaler, features, target, mask, dtype)
        if first_non_finite is None and not torch.isfinite(loss):
            # Half precision fails on the forward pass, so the failure is
            # already visible during warmup and is recorded from there.
            first_non_finite = index - WARMUP_STEPS
    torch.cuda.synchronize()

    started = time.perf_counter()
    for index in range(MEASURED_STEPS):
        loss = _amp_step(model, optimizer, loss_fn, scaler, features, target, mask, dtype)
        losses.append(float(loss.detach()))
        if first_non_finite is None and not torch.isfinite(loss):
            first_non_finite = index
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated() / BYTES_PER_GIB
    del model, optimizer, features, target, mask
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "ms_per_image": elapsed / MEASURED_STEPS / BATCH_SIZE * 1000,
        "peak_allocated_gib": peak,
        "all_losses_finite": first_non_finite is None,
        "first_non_finite_step": first_non_finite,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe numerical precision behaviour.")
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no accelerator available; this probe measures accelerator behaviour")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    state = device_state()

    print(f"device: {state['total_gib']:.2f} GiB total, {state['free_gib']:.2f} GiB free")
    if state["free_fraction"] < ADVISORY_FREE_FRACTION:
        print("  the device appears occupied; the stability check below is the decisive one")
    print()

    reference = train_steps("float32", None, device)

    print("training steps under each precision mode")
    print(f"  {'mode':<10} {'ms/img':>8} {'peak':>8} {'finite':>8} {'first loss':>12}")
    modes = []
    for mode, dtype in MODES.items():
        result = train_steps(mode, dtype, device)
        modes.append(result)
        finite = "yes" if result["all_losses_finite"] else f"no@{result['first_non_finite_step']}"
        print(
            f"  {mode:<10} {result['ms_per_image']:>8.2f} "
            f"{result['peak_allocated_gib']:>7.2f}G {finite:>8} {result['first_loss']:>12.4e}"
        )

    repeat = train_steps("float32", None, device)
    drift = abs(reference["ms_per_image"] - repeat["ms_per_image"]) / min(
        reference["ms_per_image"], repeat["ms_per_image"]
    )
    check = {
        "first_ms_per_image": reference["ms_per_image"],
        "second_ms_per_image": repeat["ms_per_image"],
        "relative_drift": drift,
        "tolerance": STABILITY_TOLERANCE,
        "contended": drift > STABILITY_TOLERANCE,
    }

    activation = largest_activation(device)
    print(
        f"\nlargest activation {activation['largest_activation']:.1f} from "
        f"{activation['produced_by']}, against a half-precision ceiling of "
        f"{activation['float16_ceiling']:.0f}"
    )

    magnitudes = error_magnitudes(args.gold_dir, device)
    fraction = magnitudes["fraction_below_float16_normal"]
    print(
        f"squared errors below half precision's smallest normal "
        f"({magnitudes['float16_smallest_normal']:.1e}): {fraction:.1%} of "
        f"{magnitudes['squared_errors_examined']:,} valid pixels"
    )

    record = {
        "device": state,
        "stability": check,
        "format_limits": format_limits(),
        "modes": modes,
        "activation_range": activation,
        "error_magnitudes": magnitudes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    if check["contended"]:
        raise SystemExit(
            f"\nCONTENDED: the repeated single-precision measurement drifted by "
            f"{drift:.1%}, above the {STABILITY_TOLERANCE:.0%} tolerance. Another "
            f"process was using the accelerator, so the timings describe an unknown "
            f"competing load. Re-run on an idle device."
        )

    half = next(m for m in modes if m["mode"] == "float16")
    single = next(m for m in modes if m["mode"] == "float32")
    print(
        f"half precision runs at {half['ms_per_image'] / single['ms_per_image']:.2f}x "
        f"the single-precision time and uses "
        f"{half['peak_allocated_gib'] / single['peak_allocated_gib']:.2f}x the memory"
    )


if __name__ == "__main__":
    main()
