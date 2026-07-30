"""Probe: is optimisation healthy, and how wide is the usable learning rate?

Two questions, deliberately separate. Conflating them once produced the wrong
conclusion that the model was underfitting because the learning rate was too
low, when raising it made things worse.

Can the model fit a small subset? A network that reaches near-zero error on
thirty-two patches has no gradient bug and enough capacity; underfitting on the
full split is then a question of generalisation or of the stopping rule, not of
the optimiser. This is the cheapest diagnostic available and it is almost never
run.

How wide is the usable band? Rates an order of magnitude apart can either
converge to the same place or collapse into the zero-predictor solution, which
is what a target with 98.5% zeros does to optimisation: the trivial solution is
a strong attractor. A collapsed run and a merely slow one have similar final
losses, so they are separated by whether any prediction has left the floor.

This probe runs deterministically, and that requirement was earned rather than
assumed. Three earlier invocations of the same sweep disagreed with each other
by more than the rates disagreed among themselves: one configuration landed at
0.03, 0.34 and 0.79 of the baseline on separate runs at an identical seed, a
factor of thirty-two, while the gap between neighbouring rates was closer to
two. Under autotuned kernel selection the instrument was measuring its own
noise, and every conclusion drawn from it about which rates were usable was
worthless.

Determinism is therefore verified before the sweep runs, by repeating one
configuration and comparing the losses bit for bit. If they differ the sweep is
skipped and the record says the question could not be answered on this
platform. A fourth set of numbers that cannot be reproduced is worse than none.

Note that the training entry point does not do this: it keeps autotuning for the
throughput, and its run record states that results are reproducible in
distribution rather than bitwise. The difference is that training is trying to
finish, while this is trying to measure.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS reductions require this before any CUDA context exists,
# so it is set at import time rather than in main().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402

from circuitnet_congestion.models.unet import UNet  # noqa: E402
from circuitnet_congestion.training.dataset import CongestionPatchDataset  # noqa: E402
from circuitnet_congestion.training.losses import (  # noqa: E402
    DEFAULT_HOTSPOT_THRESHOLD,
    MaskedMSELoss,
    masked_mean,
)
from circuitnet_congestion.training.metrics import hotspot_counts, masked_max  # noqa: E402

OUTPUT = Path("results/probes/optimisation.json")

SUBSET_PATCHES = 32
LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_SEEDS = (0, 42, 2024)
STEPS = 800
TRACE_AT = (50, 100, 200, 400, 600, 800)

# Cheap gate before the full sweep: same configuration, twice, compared exactly.
DETERMINISM_STEPS = 100
DETERMINISM_LR = 3e-4
DETERMINISM_SEED = 42

# A subset loss this far below the zero-predictor baseline counts as fitted.
FITTED_RATIO = 0.10
# Above this fraction of the baseline, with nothing off the floor, the run has
# not left the trivial solution whatever its final loss looks like.
COLLAPSED_RATIO = 0.90
# Seed-to-seed spread above this factor means the rate is sensitive to
# initialisation, which under determinism is a real property rather than noise.
STABLE_SPREAD_FACTOR = 3.0

BYTES_PER_GIB = 2**30


def device_state() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": free / BYTES_PER_GIB,
        "total_gib": total / BYTES_PER_GIB,
        "free_fraction": free / total,
    }


def configure_determinism() -> dict[str, Any]:
    """Remove every source of run-to-run variation this probe can control."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def load_subset(gold_dir: Path, device: torch.device) -> dict[str, Any]:
    """A fixed subset held entirely on the accelerator; no loading in the loop."""
    dataset = CongestionPatchDataset(gold_dir, "train", limit=SUBSET_PATCHES)
    features = torch.stack([dataset[i]["features"] for i in range(len(dataset))]).to(device)
    target = torch.stack([dataset[i]["gt"] for i in range(len(dataset))]).to(device)
    mask = torch.stack([dataset[i]["mask"] for i in range(len(dataset))]).to(device)

    counts = hotspot_counts(target, target, mask, DEFAULT_HOTSPOT_THRESHOLD)
    return {
        "features": features,
        "target": target,
        "mask": mask,
        "baseline": float(masked_mean(target.square(), mask)),
        "patches": len(dataset),
        "hotspot_pixels": counts.target_positive,
        "target_max": float(masked_max(target, mask)),
    }


def fit_subset(
    subset: dict[str, Any],
    lr: float,
    seed: int,
    device: torch.device,
    steps: int = STEPS,
) -> dict[str, Any]:
    """Drive one learning rate at one seed against the subset."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = UNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MaskedMSELoss()

    features, target, mask = subset["features"], subset["target"], subset["mask"]
    baseline = subset["baseline"]
    trace: dict[str, float] = {}
    final_train_loss = 0.0

    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), target, mask)
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.detach())
        if step in TRACE_AT and step <= steps:
            trace[str(step)] = final_train_loss / baseline
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.inference_mode():
        prediction = model(features)
        final = float(loss_fn(prediction, target, mask))
        counts = hotspot_counts(prediction, target, mask, DEFAULT_HOTSPOT_THRESHOLD)
        largest = float(masked_max(prediction, mask))

    del model, optimizer
    torch.cuda.empty_cache()

    ratio = final / baseline
    # A collapsed run and a slow one have similar final losses. What separates
    # them is whether any prediction has left the floor at all.
    collapsed = ratio > COLLAPSED_RATIO and largest < DEFAULT_HOTSPOT_THRESHOLD
    descending = len(trace) >= 2 and list(trace.values())[-1] < list(trace.values())[-2] * 0.95

    return {
        "learning_rate": lr,
        "seed": seed,
        "steps": steps,
        "loss_over_baseline": ratio,
        "final_train_loss": final_train_loss,
        "prediction_max": largest,
        "recall": counts.recall,
        "precision": counts.precision,
        "fitted": ratio < FITTED_RATIO,
        "collapsed": collapsed,
        "still_descending": descending,
        "trace_loss_over_baseline": trace,
        "seconds": elapsed,
    }


def verify_determinism(subset: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run one configuration twice and compare the losses exactly.

    Anything short of bit equality means the sweep would again be measuring the
    platform rather than the learning rate.
    """
    try:
        first = fit_subset(subset, DETERMINISM_LR, DETERMINISM_SEED, device, DETERMINISM_STEPS)
        second = fit_subset(subset, DETERMINISM_LR, DETERMINISM_SEED, device, DETERMINISM_STEPS)
    except RuntimeError as error:
        # Raised when an operation in the model has no deterministic kernel.
        return {"deterministic": False, "reason": str(error)[:400]}

    a, b = first["final_train_loss"], second["final_train_loss"]
    return {
        "deterministic": a == b,
        "steps": DETERMINISM_STEPS,
        "learning_rate": DETERMINISM_LR,
        "seed": DETERMINISM_SEED,
        "first_loss": a,
        "second_loss": b,
        "identical": a == b,
        "seconds_each": (first["seconds"] + second["seconds"]) / 2,
    }


def classify(result: dict[str, Any]) -> str:
    if result["collapsed"]:
        return "collapsed"
    if result["fitted"]:
        return "fitted"
    return "descending" if result["still_descending"] else "stalled"


def aggregate(results: list[dict[str, Any]], lr: float) -> dict[str, Any]:
    """Per-rate summary across seeds.

    Under determinism the spread across seeds measures sensitivity to
    initialisation, which is a property of the rate on this loss surface.
    """
    runs = [r for r in results if r["learning_rate"] == lr]
    ratios = [r["loss_over_baseline"] for r in runs]
    collapses = sum(1 for r in runs if r["collapsed"])
    spread = max(ratios) / min(ratios) if min(ratios) > 0 else None

    return {
        "learning_rate": lr,
        "seeds": len(runs),
        "collapses": collapses,
        "collapse_rate": collapses / len(runs),
        "fitted_count": sum(1 for r in runs if r["fitted"]),
        "median_loss_over_baseline": statistics.median(ratios),
        "min_loss_over_baseline": min(ratios),
        "max_loss_over_baseline": max(ratios),
        "relative_spread": spread,
        "stable": spread is not None and spread <= STABLE_SPREAD_FACTOR,
        "median_recall": statistics.median(r["recall"] for r in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe optimisation health and rate band.")
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no accelerator available; this probe measures accelerator behaviour")

    device = torch.device("cuda")
    settings = configure_determinism()
    state = device_state()

    print(f"device: {state['total_gib']:.2f} GiB total, {state['free_gib']:.2f} GiB free")
    subset = load_subset(args.gold_dir, device)
    print(
        f"subset: {subset['patches']} patches, "
        f"zero-predictor loss {subset['baseline']:.4e}, "
        f"{subset['hotspot_pixels']} hotspot pixels, target max {subset['target_max']:.4f}"
    )

    print(f"\nverifying determinism at lr={DETERMINISM_LR:.0e} over {DETERMINISM_STEPS} steps")
    check = verify_determinism(subset, device)

    record: dict[str, Any] = {
        "device": state,
        "settings": settings,
        "determinism": check,
        "subset": {
            "patches": subset["patches"],
            "zero_predictor_loss": subset["baseline"],
            "hotspot_pixels": subset["hotspot_pixels"],
            "target_max": subset["target_max"],
        },
        "steps": STEPS,
    }

    if not check["deterministic"]:
        record["summary"] = {
            "answered": False,
            "note": (
                "The sweep was not run. Two identical configurations produced "
                "different losses, so a rate comparison on this platform would "
                "again measure the platform rather than the rate. Earlier "
                "invocations under autotuning disagreed with each other by a factor "
                "of thirty-two at an identical seed, more than the rates differ "
                "among themselves."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        detail = check.get("reason") or (
            f"{check['first_loss']:.10e} vs {check['second_loss']:.10e}"
        )
        raise SystemExit(f"NOT DETERMINISTIC, sweep skipped: {detail}\nwrote {args.out}")

    print(f"  identical: {check['first_loss']:.10e}  ({check['seconds_each']:.0f}s each)")

    total = len(LEARNING_RATES) * len(args.seeds)
    print(f"\n{total} runs, {len(args.seeds)} seeds per rate; a ratio near zero is healthy\n")
    print(f"  {'lr':>8} {'seed':>6} {'loss/base':>10} {'pred_max':>9} {'recall':>8} {'state':>11}")

    results = []
    for lr in LEARNING_RATES:
        for seed in args.seeds:
            result = fit_subset(subset, lr, seed, device)
            results.append(result)
            print(
                f"  {lr:>8.0e} {seed:>6} {result['loss_over_baseline']:>10.4f} "
                f"{result['prediction_max']:>9.4f} {result['recall']:>8.3f} "
                f"{classify(result):>11}"
            )
        print()

    rates = [aggregate(results, lr) for lr in LEARNING_RATES]
    fits = [r for r in rates if r["median_loss_over_baseline"] < FITTED_RATIO]
    usable = [r["learning_rate"] for r in fits if r["collapse_rate"] == 0.0 and r["stable"]]
    sensitive = [r["learning_rate"] for r in fits if not r["stable"]]

    print(f"  {'lr':>8} {'collapse':>9} {'median':>8} {'range':>18} {'spread':>7} {'recall':>8}")
    for rate in rates:
        spread = f"{rate['relative_spread']:.1f}x" if rate["relative_spread"] else "n/a"
        print(
            f"  {rate['learning_rate']:>8.0e} "
            f"{rate['collapses']}/{rate['seeds']:<7} "
            f"{rate['median_loss_over_baseline']:>8.4f} "
            f"{rate['min_loss_over_baseline']:>8.4f}-{rate['max_loss_over_baseline']:<8.4f} "
            f"{spread:>7} {rate['median_recall']:>8.3f}"
        )

    record["results"] = results
    record["rates"] = rates
    record["summary"] = {
        "answered": True,
        "optimisation_healthy": any(r["fitted"] for r in results),
        "usable_learning_rates": usable,
        "initialisation_sensitive_rates": sensitive,
        "always_collapsing_rates": [r["learning_rate"] for r in rates if r["collapse_rate"] == 1.0],
        "seeds": args.seeds,
        "note": (
            "Measured with autotuning disabled and deterministic kernels enforced, "
            f"verified by repeating one configuration over {DETERMINISM_STEPS} steps and "
            "comparing the losses exactly. Spread across seeds is therefore "
            "sensitivity to initialisation rather than platform noise. Three seeds "
            "bound that sensitivity loosely and no more."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print(f"usable: {usable}  initialisation-sensitive: {sensitive or 'none'}")


if __name__ == "__main__":
    main()
