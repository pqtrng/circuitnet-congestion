"""Probe: is optimisation healthy, and how wide is the usable learning rate?

Two questions, deliberately separate. Conflating them once produced the wrong
conclusion that the model was underfitting because the learning rate was too
low, when raising it made things worse.

Can the model fit a small subset? A network that reaches near-zero error on
thirty-two patches has no gradient bug and enough capacity; underfitting on the
full split is then a question of generalisation or of the stopping rule, not of
the optimiser. If it cannot, something is broken and no amount of training
budget will fix it. This is the cheapest diagnostic available and it is almost
never run.

How wide is the usable band? Not to find an optimal rate -- the answer here is
that the band is narrow. Rates an order of magnitude apart either converge to
the same place or collapse into the zero-predictor solution and stay there,
which is what a target with 98.5% zeros does to optimisation: the trivial
solution is a strong attractor. A run that has collapsed and a run that is
merely slow look alike at the final loss, so the probe separates them by
checking whether any prediction has moved off the floor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from circuitnet_congestion.models.unet import UNet
from circuitnet_congestion.training.dataset import CongestionPatchDataset
from circuitnet_congestion.training.losses import (
    DEFAULT_HOTSPOT_THRESHOLD,
    MaskedMSELoss,
    masked_mean,
)
from circuitnet_congestion.training.metrics import hotspot_counts, masked_max

OUTPUT = Path("results/probes/optimisation.json")

SUBSET_PATCHES = 32
LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
STEPS = 800
TRACE_AT = (50, 100, 200, 400, 600, 800)
SEED = 42

# A subset loss this far below the zero-predictor baseline counts as fitted.
FITTED_RATIO = 0.10
# Below this fraction of the baseline the run has not moved off the trivial
# solution, whatever its final loss looks like.
COLLAPSED_RATIO = 0.90

STABILITY_TOLERANCE = 0.05
ADVISORY_FREE_FRACTION = 0.80
BYTES_PER_GIB = 2**30


def device_state() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": free / BYTES_PER_GIB,
        "total_gib": total / BYTES_PER_GIB,
        "free_fraction": free / total,
    }


def load_subset(gold_dir: Path, device: torch.device) -> dict[str, Any]:
    """A fixed subset held entirely on the accelerator; no loading in the loop."""
    dataset = CongestionPatchDataset(gold_dir, "train", limit=SUBSET_PATCHES)
    features = torch.stack([dataset[i]["features"] for i in range(len(dataset))]).to(device)
    target = torch.stack([dataset[i]["gt"] for i in range(len(dataset))]).to(device)
    mask = torch.stack([dataset[i]["mask"] for i in range(len(dataset))]).to(device)

    baseline = float(masked_mean(target.square(), mask))
    counts = hotspot_counts(target, target, mask, DEFAULT_HOTSPOT_THRESHOLD)

    return {
        "features": features,
        "target": target,
        "mask": mask,
        "baseline": baseline,
        "patches": len(dataset),
        "hotspot_pixels": counts.target_positive,
        "target_max": float(masked_max(target, mask)),
    }


def fit_subset(subset: dict[str, Any], lr: float, device: torch.device) -> dict[str, Any]:
    """Drive one learning rate against the subset and record the trajectory."""
    torch.manual_seed(SEED)
    model = UNet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MaskedMSELoss()

    features, target, mask = subset["features"], subset["target"], subset["mask"]
    baseline = subset["baseline"]
    trace: dict[str, float] = {}

    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), target, mask)
        loss.backward()
        optimizer.step()
        if step in TRACE_AT:
            trace[str(step)] = float(loss) / baseline
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

    return {
        "learning_rate": lr,
        "final_loss": final,
        "loss_over_baseline": ratio,
        "prediction_max": largest,
        "recall": counts.recall,
        "precision": counts.precision,
        "fitted": ratio < FITTED_RATIO,
        "collapsed": collapsed,
        "still_descending": trace[str(STEPS)] < trace[str(TRACE_AT[-2])] * 0.95,
        "trace_loss_over_baseline": trace,
        "seconds": elapsed,
    }


def summarise(results: list[dict[str, Any]], subset: dict[str, Any]) -> dict[str, Any]:
    usable = [r for r in results if r["fitted"]]
    collapsed = [r["learning_rate"] for r in results if r["collapsed"]]

    best = min(results, key=lambda r: r["loss_over_baseline"])
    return {
        "optimisation_healthy": bool(usable),
        "usable_learning_rates": [r["learning_rate"] for r in usable],
        "collapsed_learning_rates": collapsed,
        "band_width_decades": (
            None
            if len(usable) < 2
            else round(
                torch.log10(
                    torch.tensor(max(r["learning_rate"] for r in usable))
                    / torch.tensor(min(r["learning_rate"] for r in usable))
                ).item(),
                2,
            )
        ),
        "best_learning_rate": best["learning_rate"],
        "best_loss_over_baseline": best["loss_over_baseline"],
        "best_recall": best["recall"],
        "subset_hotspot_pixels": subset["hotspot_pixels"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe optimisation health and rate band.")
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

    subset = load_subset(args.gold_dir, device)
    print(
        f"\nsubset: {subset['patches']} patches, "
        f"zero-predictor loss {subset['baseline']:.4e}, "
        f"{subset['hotspot_pixels']} hotspot pixels, target max {subset['target_max']:.4f}"
    )
    print("a ratio near zero means optimisation is healthy\n")

    reference = fit_subset(subset, LEARNING_RATES[0], device)

    print(f"  {'lr':>8} {'loss/base':>10} {'pred_max':>9} {'recall':>8} {'state':>12} {'s':>5}")
    results = []
    for lr in LEARNING_RATES:
        result = fit_subset(subset, lr, device)
        results.append(result)
        if result["collapsed"]:
            state_text = "collapsed"
        elif result["fitted"]:
            state_text = "fitted"
        elif result["still_descending"]:
            state_text = "descending"
        else:
            state_text = "stalled"
        print(
            f"  {lr:>8.0e} {result['loss_over_baseline']:>10.4f} "
            f"{result['prediction_max']:>9.4f} {result['recall']:>8.3f} "
            f"{state_text:>12} {result['seconds']:>5.0f}"
        )

    repeat = fit_subset(subset, LEARNING_RATES[0], device)
    drift = abs(reference["seconds"] - repeat["seconds"]) / min(
        reference["seconds"], repeat["seconds"]
    )
    check = {
        "first_seconds": reference["seconds"],
        "second_seconds": repeat["seconds"],
        "relative_drift": drift,
        "tolerance": STABILITY_TOLERANCE,
        "contended": drift > STABILITY_TOLERANCE,
    }

    summary = summarise(results, subset)
    record = {
        "device": state,
        "stability": check,
        "subset": {
            "patches": subset["patches"],
            "zero_predictor_loss": subset["baseline"],
            "hotspot_pixels": subset["hotspot_pixels"],
            "target_max": subset["target_max"],
        },
        "steps": STEPS,
        "seed": SEED,
        "results": results,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    if check["contended"]:
        raise SystemExit(
            f"\nCONTENDED: the repeated measurement drifted by {drift:.1%}, above the "
            f"{STABILITY_TOLERANCE:.0%} tolerance. Another process was using the "
            f"accelerator. The convergence results are still valid, being independent "
            f"of wall time, but re-run on an idle device before quoting any timing."
        )

    print(
        f"optimisation healthy: {summary['optimisation_healthy']} | "
        f"usable rates: {summary['usable_learning_rates']} | "
        f"collapsed at: {summary['collapsed_learning_rates'] or 'none'}"
    )


if __name__ == "__main__":
    main()
