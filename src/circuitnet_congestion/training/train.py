"""Training entry point for the congestion U-Net baseline.

Every constant in the default configuration was measured rather than guessed,
and the ones that were not are labelled as such in the run record.

Batch size 32 and four loader workers. Throughput is flat at roughly 4.9 ms per
image from batch 16 through 48 and then degrades by a factor of seven at batch
64, where peak allocation exceeds the accelerator's physical memory and the
driver silently spills to host memory instead of raising. Batch 32 sits in the
flat region with the largest remaining headroom. Loading costs about 8 ms per
step against 158 ms of compute, so four workers are already nineteen times
faster than they need to be.

Single precision. Half precision produced non-finite values on the first
forward pass, buys no memory that is needed here, and runs at the same rate as
single precision on accelerators without dedicated matrix units. Targets sit
around 1e-2 and squared errors reach 1e-6, which is the bottom of half
precision's dynamic range.

Both objectives are evaluated on both validation metrics regardless of which one
is being trained, otherwise the two runs cannot be placed in the same table.
Early stopping also tracks the shared metric for both, so that a difference in
epoch count between runs never reflects a difference in stopping rule.

Two checkpoints are kept when the objective differs from the selection metric.
The one selected on the shared metric supports the main comparison; the one
selected on the run's own objective supports a sensitivity line beside it.
Selecting the weighted run on a metric it does not optimise would catch it
before it specialises, which is precisely the axis the comparison is about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from circuitnet_congestion.device import get_device
from circuitnet_congestion.models.unet import UNet, count_parameters
from circuitnet_congestion.training.dataset import CongestionPatchDataset, build_dataloader
from circuitnet_congestion.training.losses import (
    DEFAULT_HOTSPOT_THRESHOLD,
    DEFAULT_POSITIVE_WEIGHT,
    LOSS_MSE,
    MaskedMeanAccumulator,
    MaskedWeightedMSELoss,
    build_loss,
    hotspot_weight,
)
from circuitnet_congestion.training.metrics import HotspotCounts, hotspot_counts, masked_max

# Measured on the reference configuration; see the module docstring. The guard
# exists because exceeding accelerator memory degrades throughput silently
# rather than raising, which otherwise costs a full night to notice. It is
# evaluated on the second epoch: the first pays for autotuned kernel selection,
# worker startup and context initialisation, and is several times slower on any
# healthy run.
REFERENCE_MS_PER_IMAGE = 4.9
THROUGHPUT_WARNING_MS_PER_IMAGE = 10.0

# Epochs to allow before warning that no prediction has crossed the threshold.
SHRINKAGE_WARNING_EPOCH = 5

CHECKPOINT_BEST_SHARED = "best_val_mse.pt"
CHECKPOINT_BEST_OBJECTIVE = "best_val_objective.pt"
CHECKPOINT_LAST = "last.pt"

RUN_RECORD = "run.json"
HISTORY = "history.jsonl"

BYTES_PER_GIB = 2**30


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataConfig:
    gold_dir: str = "data/gold"
    batch_size: int = 32
    num_workers: int = 4


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 32
    depth: int = 4


@dataclass(frozen=True)
class LossConfig:
    name: str = LOSS_MSE
    threshold: float | None = None
    positive_weight: float | None = None

    def build(self) -> nn.Module:
        kwargs: dict[str, float] = {}
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold
        if self.positive_weight is not None:
            kwargs["positive_weight"] = self.positive_weight
        return build_loss(self.name, **kwargs)


@dataclass(frozen=True)
class OptimConfig:
    """A patience of zero or less disables early stopping.

    At this scale no validation metric provides a usable stopping signal.
    Validation error fluctuates around a flat level from the first epoch while
    the training objective is still descending, so a patience counter measures
    noise and halts at an arbitrary point: in the superseded runs it selected a
    noise minimum for the plain objective and cut the weighted run short while
    its hotspot recall was still climbing. A fixed budget with the whole
    trajectory recorded is the honest alternative.
    """

    lr: float = 3e-4
    epochs: int = 60
    patience: int = 0


@dataclass(frozen=True)
class EvalConfig:
    """Shared across runs so that the comparison table is like for like."""

    hotspot_threshold: float = DEFAULT_HOTSPOT_THRESHOLD
    positive_weight: float = DEFAULT_POSITIVE_WEIGHT


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str = "results"
    tensorboard_dir: str = "runs"


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(cls: type, payload: dict[str, Any] | None, name: str) -> Any:
    """Build one configuration section, rejecting keys it does not define.

    Silently ignoring a misspelled key would produce a run that looks configured
    and is not.
    """
    payload = payload or {}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown keys in section '{name}': {unknown}; expected {sorted(known)}")
    return cls(**payload)


def load_config(path: Path) -> TrainConfig:
    document = yaml.safe_load(path.read_text()) or {}

    known = {"run_name", "seed", "data", "model", "loss", "optim", "eval", "output"}
    unknown = sorted(set(document) - known)
    if unknown:
        raise ValueError(f"unknown top-level keys in {path}: {unknown}")
    if "run_name" not in document:
        raise ValueError(f"{path} must define run_name")

    return TrainConfig(
        run_name=str(document["run_name"]),
        seed=int(document.get("seed", 42)),
        data=_section(DataConfig, document.get("data"), "data"),
        model=_section(ModelConfig, document.get("model"), "model"),
        loss=_section(LossConfig, document.get("loss"), "loss"),
        optim=_section(OptimConfig, document.get("optim"), "optim"),
        eval=_section(EvalConfig, document.get("eval"), "eval"),
        output=_section(OutputConfig, document.get("output"), "output"),
    )


def replace_top_level(config: TrainConfig, **overrides: Any) -> TrainConfig:
    """Replace top-level fields while preserving the nested dataclasses.

    dataclasses.replace would be equivalent, but asdict-based copying would
    flatten the sections into plain dictionaries.
    """
    current = {f.name: getattr(config, f.name) for f in fields(config)}
    return TrainConfig(**{**current, **overrides})


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def git_state() -> dict[str, Any]:
    """Revision and cleanliness of the worktree that produced a run.

    Numbers generated from a modified worktree have to say so.
    """

    def run(*command: str) -> str | None:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    revision = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "revision": revision,
        "dirty": None if status is None else bool(status),
    }


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Digest of a checkpoint.

    Model artefacts are excluded from the repository, so the digest recorded in
    the run record is the only link between a reported number and the weights
    that produced it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def accelerator_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_properties(device).total_memory / BYTES_PER_GIB


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalResult:
    mse: float
    weighted_mse: float
    prediction_max: float
    counts: HotspotCounts

    def as_record(self) -> dict[str, Any]:
        return {
            "val_mse": self.mse,
            "val_weighted_mse": self.weighted_mse,
            "val_prediction_max": self.prediction_max,
            "val_true_positive": self.counts.true_positive,
            "val_false_positive": self.counts.false_positive,
            "val_false_negative": self.counts.false_negative,
            "val_recall": self.counts.recall,
            "val_precision": self.counts.precision,
            "val_f1": self.counts.f1,
        }


def objective_weight(loss_fn: nn.Module, target: torch.Tensor) -> torch.Tensor | None:
    """The weight tensor a loss applies.

    Needed to accumulate the training objective over an epoch as a single
    global mean rather than as an average of per-batch losses.
    """
    if isinstance(loss_fn, MaskedWeightedMSELoss):
        return hotspot_weight(
            target,
            threshold=loss_fn.threshold,
            positive_weight=loss_fn.positive_weight,
        )
    return None


@torch.inference_mode()
def zero_predictor_baseline(loader: DataLoader, device: torch.device) -> float:
    """Validation error of predicting zero everywhere, over the whole split.

    Computed across every batch rather than a prefix: patches are ordered by
    filename, so any prefix is a single design and its statistics are not the
    split's.
    """
    accumulator = MaskedMeanAccumulator()
    for batch in loader:
        target = batch["gt"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        accumulator.update(target.square(), mask)
    return accumulator.compute()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    positive_weight: float,
) -> EvalResult:
    model.eval()
    mse = MaskedMeanAccumulator()
    weighted = MaskedMeanAccumulator()
    counts = HotspotCounts()
    largest = float("-inf")

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target = batch["gt"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        prediction = model(features)
        squared = (prediction - target).square()

        mse.update(squared, mask)
        weighted.update(
            squared,
            mask,
            hotspot_weight(target, threshold=threshold, positive_weight=positive_weight),
        )
        counts = counts + hotspot_counts(prediction, target, mask, threshold)
        largest = max(largest, float(masked_max(prediction, mask)))

    return EvalResult(
        mse=mse.compute(),
        weighted_mse=weighted.compute(),
        prediction_max=0.0 if largest == float("-inf") else largest,
        counts=counts,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    """Run one epoch; return the epoch objective and the number of images seen."""
    model.train()
    accumulator = MaskedMeanAccumulator()
    images = 0

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        target = batch["gt"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = loss_fn(prediction, target, mask)
        loss.backward()
        optimizer.step()

        accumulator.update((prediction - target).square(), mask, objective_weight(loss_fn, target))
        images += int(features.shape[0])

    return accumulator.compute(), images


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    result: EvalResult,
    best_shared: float,
    best_objective: float,
    include_optimizer: bool = True,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "config": asdict(config),
        "metrics": result.as_record(),
        "best_shared": best_shared,
        "best_objective": best_objective,
    }
    if include_optimizer:
        # Only the resume checkpoint carries optimiser moments. They are twice
        # the size of the weights and are never read when a checkpoint is
        # loaded for evaluation.
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)

    return {
        "epoch": epoch,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the congestion U-Net baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None, help="override run_name")
    parser.add_argument("--epochs", type=int, default=None, help="override optim.epochs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap patches per split; marks the run as a smoke test",
    )
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config = replace_top_level(config, run_name=args.run_name)
    if args.epochs is not None:
        config = replace_top_level(
            config,
            optim=OptimConfig(
                lr=config.optim.lr,
                epochs=args.epochs,
                patience=config.optim.patience,
            ),
        )

    smoke = args.limit is not None

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = get_device()
    if device.type == "cuda":
        # Input shapes are fixed, so autotuning pays for itself. Bitwise
        # reproducibility is consequently not claimed; the run record says so.
        torch.backends.cudnn.benchmark = True

    train_set = CongestionPatchDataset(config.data.gold_dir, "train", limit=args.limit)
    val_set = CongestionPatchDataset(config.data.gold_dir, "val", limit=args.limit)
    train_loader = build_dataloader(
        train_set,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )
    val_loader = build_dataloader(
        val_set,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )

    model = UNet(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        depth=config.model.depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)
    loss_fn = config.loss.build()

    # The plain objective and the shared selection metric are the same quantity,
    # so only the weighted run needs a second checkpoint.
    tracks_separate_objective = isinstance(loss_fn, MaskedWeightedMSELoss)

    results_dir = Path(config.output.results_dir) / config.run_name
    checkpoint_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / HISTORY

    start_epoch = 1
    best_shared = float("inf")
    best_objective = float("inf")
    last_path = checkpoint_dir / CHECKPOINT_LAST

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_shared = float(state["best_shared"])
        best_objective = float(state["best_objective"])
        print(f"resumed from epoch {state['epoch']}")
    elif history_path.exists():
        history_path.unlink()

    baseline = zero_predictor_baseline(val_loader, device)
    print(
        f"run={config.run_name} device={device.type} loss={loss_fn} "
        f"train={len(train_set)} val={len(val_set)} params={count_parameters(model):,}"
    )
    print(f"zero-predictor validation error over the whole split: {baseline:.6e}")

    writer = SummaryWriter(Path(config.output.tensorboard_dir) / config.run_name)
    record: dict[str, Any] = {
        "run_name": config.run_name,
        "smoke": smoke,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_type": device.type,
            "accelerator_memory_gib": accelerator_memory_gib(device),
        },
        "model": {"parameters": count_parameters(model)},
        "data": {"train_patches": len(train_set), "val_patches": len(val_set)},
        "baseline": {"val_zero_predictor_mse": baseline},
        "notes": {
            "epoch_budget": (
                "The epoch count is a time budget that fits two runs in one session, "
                "not a convergence criterion."
            ),
            "learning_rate": (
                "Fixed. Chosen from the scale of the targets and never tuned against "
                "this dataset, so it is not claimed to be optimal."
            ),
            "reproducibility": (
                "Seeded, but autotuned convolution selection means results are "
                "reproducible in distribution rather than bitwise."
            ),
            "checkpoints": (
                "Weights are excluded from the repository; the digests below are the "
                "link between these numbers and the weights that produced them."
            ),
        },
    }
    (results_dir / RUN_RECORD).write_text(json.dumps(record, indent=2) + "\n")

    checkpoints: dict[str, dict[str, Any]] = {}
    epochs_without_improvement = 0
    stopped_early = False
    started = time.perf_counter()
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, config.optim.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, images = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_seconds = time.perf_counter() - epoch_started
        ms_per_image = train_seconds * 1000 / max(images, 1)

        result = evaluate(
            model,
            val_loader,
            device,
            threshold=config.eval.hotspot_threshold,
            positive_weight=config.eval.positive_weight,
        )
        last_epoch = epoch

        entry = {
            "epoch": epoch,
            "train_objective": train_loss,
            **result.as_record(),
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "ms_per_image": ms_per_image,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

        writer.add_scalar("loss/train_objective", train_loss, epoch)
        writer.add_scalar("loss/val_mse", result.mse, epoch)
        writer.add_scalar("loss/val_weighted_mse", result.weighted_mse, epoch)
        writer.add_scalar("loss/val_zero_predictor_mse", baseline, epoch)
        writer.add_scalar("hotspot/prediction_max", result.prediction_max, epoch)
        writer.add_scalar("hotspot/recall", result.counts.recall, epoch)
        writer.add_scalar("hotspot/precision", result.counts.precision, epoch)
        writer.add_scalar("hotspot/f1", result.counts.f1, epoch)
        writer.add_scalar("throughput/ms_per_image", ms_per_image, epoch)

        objective = result.weighted_mse if tracks_separate_objective else result.mse

        if result.mse < best_shared:
            best_shared = result.mse
            epochs_without_improvement = 0
            checkpoints[CHECKPOINT_BEST_SHARED] = save_checkpoint(
                checkpoint_dir / CHECKPOINT_BEST_SHARED,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                result=result,
                best_shared=best_shared,
                best_objective=best_objective,
                include_optimizer=False,
            )
        else:
            epochs_without_improvement += 1

        if objective < best_objective:
            best_objective = objective
            if tracks_separate_objective:
                checkpoints[CHECKPOINT_BEST_OBJECTIVE] = save_checkpoint(
                    checkpoint_dir / CHECKPOINT_BEST_OBJECTIVE,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    result=result,
                    best_shared=best_shared,
                    best_objective=best_objective,
                    include_optimizer=False,
                )

        checkpoints[CHECKPOINT_LAST] = save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            result=result,
            best_shared=best_shared,
            best_objective=best_objective,
        )

        print(
            f"epoch {epoch:3d}/{config.optim.epochs}  train={train_loss:.6e}  "
            f"val_mse={result.mse:.6e}  val_wmse={result.weighted_mse:.6e}  "
            f"pred_max={result.prediction_max:.4f}  recall={result.counts.recall:.3f}  "
            f"{train_seconds:.1f}s ({ms_per_image:.2f} ms/img)"
        )

        if epoch == start_epoch + 1 and ms_per_image > THROUGHPUT_WARNING_MS_PER_IMAGE:
            print(
                f"  warning: {ms_per_image:.1f} ms/image against a reference of "
                f"{REFERENCE_MS_PER_IMAGE}. Throughput this far below reference usually "
                f"means peak allocation exceeds device memory and is spilling to host "
                f"memory, which degrades speed without raising. Try a smaller batch size."
            )

        if (
            result.prediction_max < config.eval.hotspot_threshold
            and epoch >= SHRINKAGE_WARNING_EPOCH
        ):
            print(
                f"  warning: no predicted pixel has crossed {config.eval.hotspot_threshold} "
                f"after {epoch} epochs. The model may have collapsed to the conditional "
                f"mean, which minimises squared error while detecting nothing."
            )

        if config.optim.patience > 0 and epochs_without_improvement >= config.optim.patience:
            stopped_early = True
            print(f"early stop: no improvement in {config.optim.patience} epochs")
            break

    writer.close()

    record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["training"] = {
        "epochs_run": last_epoch,
        "stopped_early": stopped_early,
        "wall_seconds": time.perf_counter() - started,
    }
    record["best"] = {"val_mse": best_shared, "val_objective": best_objective}
    record["checkpoints"] = checkpoints
    (results_dir / RUN_RECORD).write_text(json.dumps(record, indent=2) + "\n")

    print(f"best val_mse={best_shared:.6e}  baseline={baseline:.6e}")
    print(f"record written to {results_dir / RUN_RECORD}")


if __name__ == "__main__":
    main()
