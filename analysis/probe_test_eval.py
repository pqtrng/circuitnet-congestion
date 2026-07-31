"""Probe: evaluate the published checkpoints on the held-out test split.

Every model figure in this repository is a validation number, and the central
result -- that the checkpoint-selection rule changes the model more than the
loss does -- has been shown only on validation. This probe carries that
question to a second, independent design: it loads each published selection
(the error-selected and F1-selected checkpoint of each run) and scores it on
test, so the selection gap can be read on a split no checkpoint was chosen on.

Like the other probes it needs the Gold layer and an accelerator, is run by
hand via `make probe-test`, and writes JSON under results/probes/. It reuses
the training loop's own evaluate() and zero_predictor_baseline() so the numbers
are produced exactly as validation's were -- same masked reduction, same strict
hotspot threshold, same accumulation. The checkpoints it reads are verified
against the digest each run.json recorded, so a file swapped on disk is a
loud failure here rather than a silently wrong row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from circuitnet_congestion.device import get_device
from circuitnet_congestion.models.unet import UNet
from circuitnet_congestion.training.dataset import CongestionPatchDataset, build_dataloader
from circuitnet_congestion.training.train import (
    evaluate,
    load_config,
    zero_predictor_baseline,
)

RESULTS = Path("results")
OUTPUT = RESULTS / "probes" / "test_eval.json"

# The two selection rules the headline table compares, by checkpoint filename.
SELECTIONS = ("best_val_mse", "best_val_f1")
CONFIGS = {"unet_a": Path("configs/unet_a.yaml"), "unet_b": Path("configs/unet_b.yaml")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, run_record: dict[str, Any]) -> str:
    """Return the checkpoint digest, asserting it matches the run record.

    The recorded digest is the only link between a reported number and the
    weights behind it. Scoring a file whose digest has drifted would attach a
    test result to a model the repository never published.
    """
    recorded = run_record["checkpoints"].get(path.name, {}).get("sha256")
    if recorded is None:
        raise KeyError(f"{path.name} has no recorded digest in the run record")
    actual = _sha256(path)
    if actual != recorded:
        raise ValueError(f"{path.name} digest {actual[:12]} != recorded {recorded[:12]}")
    return actual


def probe_run(name: str, config_path: Path, device: torch.device) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = RESULTS / name
    run_record = json.loads((run_dir / "run.json").read_text())

    test_set = CongestionPatchDataset(config.data.gold_dir, "test")
    test_loader = build_dataloader(
        test_set,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )
    baseline = zero_predictor_baseline(test_loader, device)

    selections: dict[str, Any] = {}
    for selection in SELECTIONS:
        checkpoint = run_dir / "checkpoints" / f"{selection}.pt"
        digest = _verify(checkpoint, run_record)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = UNet(
            in_channels=config.model.in_channels,
            base_channels=config.model.base_channels,
            depth=config.model.depth,
        ).to(device)
        model.load_state_dict(state["model_state"])
        result = evaluate(
            model,
            test_loader,
            device,
            threshold=config.eval.hotspot_threshold,
            positive_weight=config.eval.positive_weight,
        )
        selections[selection] = {
            "epoch": int(state["epoch"]),
            "sha256": digest,
            "test_mse": result.mse,
            "test_weighted_mse": result.weighted_mse,
            "test_prediction_max": result.prediction_max,
            "test_recall": result.counts.recall,
            "test_precision": result.counts.precision,
            "test_f1": result.counts.f1,
            "test_mse_over_baseline": result.mse / baseline,
        }

    return {
        "loss": run_record.get("config", {}).get("loss", {}).get("name"),
        "test_patches": len(test_set),
        "baseline_test_zero_predictor_mse": baseline,
        "selections": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate published checkpoints on test.")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    device = get_device()
    runs = {name: probe_run(name, path, device) for name, path in CONFIGS.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"wrote {args.out}\n")

    for name, run in runs.items():
        print(f"{name}  loss={run['loss']}  baseline={run['baseline_test_zero_predictor_mse']:.6e}")
        for selection, record in run["selections"].items():
            print(
                f"    {selection:14s} ep{record['epoch']:>3}  "
                f"test_mse={record['test_mse']:.4e} "
                f"({record['test_mse_over_baseline']:.4f}x base)  "
                f"f1={record['test_f1']:.4f}  recall={record['test_recall']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
