"""Tests for the training entry point.

Two things matter most here. Configuration parsing must reject a key it does not
understand rather than silently producing a run that looks configured and is
not. And the loop must produce a complete run record, since the weights it
refers to are excluded from the repository and the record is the only link
between a reported number and the model that produced it.

Everything runs on CPU against a miniature synthetic Gold layer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from circuitnet_congestion.models.unet import UNet
from circuitnet_congestion.training import train as train_module
from circuitnet_congestion.training.dataset import CongestionPatchDataset, build_dataloader
from circuitnet_congestion.training.losses import (
    LOSS_MSE,
    LOSS_WEIGHTED_MSE,
    MaskedMSELoss,
    MaskedWeightedMSELoss,
)
from circuitnet_congestion.training.metrics import HotspotCounts
from circuitnet_congestion.training.train import (
    DataConfig,
    EvalResult,
    OptimConfig,
    TrainConfig,
    evaluate,
    load_config,
    objective_weight,
    replace_top_level,
    save_checkpoint,
    sha256_file,
    zero_predictor_baseline,
)

SIZE = 128
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _write_gold(root: Path, *, train_count: int = 4, val_count: int = 2) -> Path:
    """A miniature Gold layer with sparse, partially padded targets."""
    gold = root / "gold"
    rng = np.random.default_rng(0)

    for split, count in (("train", train_count), ("val", val_count)):
        for index in range(count):
            features = rng.standard_normal((3, SIZE, SIZE)).astype(np.float32)

            mask = np.ones((SIZE, SIZE), dtype=np.uint8)
            if index % 2:
                mask[SIZE // 2 :, :] = 0

            gt = np.zeros((SIZE, SIZE), dtype=np.float32)
            gt[:8, :8] = 1 / 44
            gt[0, :3] = 3 / 44

            path = gold / split / f"patch_{index:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, features=features, gt=gt, mask=mask)

    return gold


def _write_config(root: Path, gold: Path, *, epochs: int = 2, name: str = "tiny") -> Path:
    path = root / f"{name}.yaml"
    path.write_text(
        f"""
run_name: {name}
seed: 42
data:
  gold_dir: {gold}
  batch_size: 2
  num_workers: 0
model:
  in_channels: 3
  base_channels: 4
  depth: 2
loss:
  name: masked_mse
optim:
  lr: 3.0e-4
  epochs: {epochs}
  patience: 12
eval:
  hotspot_threshold: 0.05
  positive_weight: 10.0
output:
  results_dir: {root / "results"}
  tensorboard_dir: {root / "runs"}
"""
    )
    return path


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_load_config_reads_every_section(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, tmp_path / "gold"))

    assert config.run_name == "tiny"
    assert config.seed == 42
    assert config.data.batch_size == 2
    assert config.model.base_channels == 4
    assert config.loss.name == LOSS_MSE
    assert config.optim.lr == pytest.approx(3e-4)
    assert config.eval.hotspot_threshold == pytest.approx(0.05)


def test_load_config_applies_defaults_for_omitted_sections(tmp_path: Path) -> None:
    path = tmp_path / "minimal.yaml"
    path.write_text("run_name: minimal\n")

    config = load_config(path)

    assert config.data == DataConfig()
    assert config.optim == OptimConfig()
    assert config.loss.name == LOSS_MSE


def test_load_config_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text("run_name: x\nmodl:\n  depth: 3\n")

    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_config(path)


def test_load_config_rejects_an_unknown_section_key(tmp_path: Path) -> None:
    """A misspelled key must fail loudly; the alternative is a run that reports
    a configuration it did not use."""
    path = tmp_path / "typo.yaml"
    path.write_text("run_name: x\noptim:\n  learning_rate: 0.01\n")

    with pytest.raises(ValueError, match="unknown keys in section 'optim'"):
        load_config(path)


def test_load_config_requires_a_run_name(tmp_path: Path) -> None:
    path = tmp_path / "anonymous.yaml"
    path.write_text("seed: 1\n")

    with pytest.raises(ValueError, match="must define run_name"):
        load_config(path)


@pytest.mark.parametrize("name", ["unet_a.yaml", "unet_b.yaml"])
def test_repository_configurations_are_valid(name: str) -> None:
    """Guards the committed configurations against schema drift."""
    config = load_config(REPOSITORY_ROOT / "configs" / name)

    assert config.eval.hotspot_threshold == pytest.approx(0.05)
    assert config.optim.epochs == 60
    assert isinstance(config.loss.build(), (MaskedMSELoss, MaskedWeightedMSELoss))


def test_weighted_configuration_builds_the_weighted_loss() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "unet_b.yaml")
    loss_fn = config.loss.build()

    assert isinstance(loss_fn, MaskedWeightedMSELoss)
    assert loss_fn.positive_weight == pytest.approx(10.0)
    assert loss_fn.threshold == pytest.approx(0.05)


def test_loss_config_omits_unset_parameters() -> None:
    from circuitnet_congestion.training.train import LossConfig

    assert isinstance(LossConfig(name=LOSS_MSE).build(), MaskedMSELoss)
    assert isinstance(LossConfig(name=LOSS_WEIGHTED_MSE).build(), MaskedWeightedMSELoss)


def test_replace_top_level_preserves_nested_dataclasses() -> None:
    """Copying through asdict would flatten the sections into dictionaries."""
    original = TrainConfig(run_name="a")

    updated = replace_top_level(original, run_name="b")

    assert updated.run_name == "b"
    assert isinstance(updated.data, DataConfig)
    assert updated.data == original.data


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_sha256_matches_the_reference_digest(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"congestion" * 5000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_git_state_reports_revision_and_cleanliness() -> None:
    state = train_module.git_state()

    assert set(state) == {"revision", "dirty"}
    assert state["revision"] is None or len(state["revision"]) == 40


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_objective_weight_is_present_only_for_the_weighted_loss() -> None:
    target = torch.zeros(1, 1, 4, 4)

    assert objective_weight(MaskedMSELoss(), target) is None
    assert objective_weight(MaskedWeightedMSELoss(), target) is not None


def test_zero_predictor_baseline_covers_the_whole_split(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path)
    dataset = CongestionPatchDataset(gold, "val")
    loader = build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    baseline = zero_predictor_baseline(loader, torch.device("cpu"))

    squares, weights = 0.0, 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        squares += float((item["gt"].square() * item["mask"]).sum())
        weights += float(item["mask"].sum())

    assert baseline == pytest.approx(squares / weights, rel=1e-6)


def test_evaluate_reports_both_metrics_and_the_shrinkage_signal(tmp_path: Path) -> None:
    gold = _write_gold(tmp_path)
    loader = build_dataloader(
        CongestionPatchDataset(gold, "val"),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = UNet(base_channels=4, depth=2)

    result = evaluate(model, loader, torch.device("cpu"), threshold=0.05, positive_weight=10.0)

    assert result.mse > 0.0
    assert result.weighted_mse >= result.mse
    assert result.prediction_max == pytest.approx(0.0)
    assert result.counts.recall == 0.0


def test_eval_result_record_carries_every_reported_field() -> None:
    record = EvalResult(
        mse=1.0,
        weighted_mse=2.0,
        prediction_max=0.3,
        counts=HotspotCounts(true_positive=1, false_positive=2, false_negative=3),
    ).as_record()

    assert record["val_mse"] == 1.0
    assert record["val_prediction_max"] == 0.3
    assert record["val_false_positive"] == 2
    assert record["val_recall"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("include_optimizer", [True, False])
def test_checkpoint_contents_follow_the_optimiser_flag(
    tmp_path: Path, include_optimizer: bool
) -> None:
    """Optimiser moments are twice the size of the weights and are never read
    when a checkpoint is loaded for evaluation."""
    model = UNet(base_channels=4, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    entry = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=7,
        config=TrainConfig(run_name="x"),
        result=EvalResult(1.0, 2.0, 0.1, HotspotCounts()),
        selection={"val_mse": 1.0, "val_objective": 2.0, "val_f1": 0.3},
        include_optimizer=include_optimizer,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert ("optimizer_state" in payload) is include_optimizer
    assert payload["epoch"] == 7
    assert payload["selection"]["val_f1"] == 0.3
    assert entry["sha256"] == sha256_file(path)
    assert entry["bytes"] == path.stat().st_size


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_training_run_writes_a_complete_record(tmp_path: Path, monkeypatch) -> None:
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=2)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])

    train_module.main()

    results = tmp_path / "results" / "tiny"
    history = [json.loads(line) for line in (results / "history.jsonl").read_text().splitlines()]
    record = json.loads((results / "run.json").read_text())

    assert [entry["epoch"] for entry in history] == [1, 2]
    assert all("val_prediction_max" in entry for entry in history)
    assert record["smoke"] is False
    assert record["training"]["epochs_run"] == 2
    assert record["baseline"]["val_zero_predictor_mse"] > 0.0
    assert set(record["notes"]) == {
        "epoch_budget",
        "learning_rate",
        "reproducibility",
        "checkpoints",
        "selection",
    }
    assert set(record["best"]) == {"val_mse", "val_weighted_mse", "val_f1"}

    checkpoints = record["checkpoints"]
    assert "best_val_mse.pt" in checkpoints
    assert "best_val_f1.pt" in checkpoints
    # This assertion is inverted from what it was. It used to require that an
    # unweighted run write no weighted-error checkpoint, which pinned the gate
    # that cost one canonical run its epoch-18 weights permanently.
    assert "best_val_weighted_mse.pt" in checkpoints
    for entry in checkpoints.values():
        assert len(entry["sha256"]) == 64


def test_limit_marks_the_run_as_a_smoke_test(tmp_path: Path, monkeypatch) -> None:
    """Numbers from a truncated split must never reach a results table."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=1, name="smoke")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--limit", "2"])

    train_module.main()
    record = json.loads((tmp_path / "results" / "smoke" / "run.json").read_text())

    assert record["smoke"] is True
    assert record["data"]["train_patches"] == 2


def test_weighted_run_keeps_a_second_checkpoint(tmp_path: Path, monkeypatch) -> None:
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=1, name="weighted")
    config_path.write_text(
        config_path.read_text().replace(
            "  name: masked_mse",
            "  name: masked_weighted_mse\n  threshold: 0.05\n  positive_weight: 10.0",
        )
    )
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])

    train_module.main()
    record = json.loads((tmp_path / "results" / "weighted" / "run.json").read_text())

    assert "best_val_weighted_mse.pt" in record["checkpoints"]


def test_resume_without_a_checkpoint_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=1, name="orphan")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--resume"])

    with pytest.raises(FileNotFoundError, match="does not exist"):
        train_module.main()


def test_resume_continues_the_history(tmp_path: Path, monkeypatch) -> None:
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=1, name="resumed")

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config_path), "--epochs", "3", "--resume"],
    )
    train_module.main()

    history = (tmp_path / "results" / "resumed" / "history.jsonl").read_text().splitlines()
    epochs = [json.loads(line)["epoch"] for line in history]

    assert epochs == [1, 2, 3]


def test_best_f1_checkpoint_is_not_overwritten_by_ties(tmp_path: Path, monkeypatch) -> None:
    """Hotspot F1 is legitimately zero for many early epochs. A non-strict
    comparison would rewrite the checkpoint on each of them and end up holding
    the last epoch rather than the best one."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=3, name="ties")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])

    train_module.main()
    record = json.loads((tmp_path / "results" / "ties" / "run.json").read_text())

    assert record["best"]["val_f1"] == 0.0
    assert record["checkpoints"]["best_val_f1.pt"]["epoch"] == 1


def test_periodic_checkpoints_are_written_and_recorded(tmp_path: Path, monkeypatch) -> None:
    """Three retraining cycles were caused by a selection rule that was only
    recognised as interesting after the run had finished."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=4, name="periodic")
    config_path.write_text(
        config_path.read_text().replace(
            "  tensorboard_dir:", "  checkpoint_every: 2\n  tensorboard_dir:"
        )
    )
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])

    train_module.main()
    record = json.loads((tmp_path / "results" / "periodic" / "run.json").read_text())
    written = tmp_path / "results" / "periodic" / "checkpoints" / "periodic"

    assert {"epoch_002.pt", "epoch_004.pt"} <= set(record["checkpoints"])
    assert sorted(p.name for p in written.glob("*.pt")) == ["epoch_002.pt", "epoch_004.pt"]


def test_resume_preserves_digests_from_earlier_sessions(tmp_path: Path, monkeypatch) -> None:
    """The run record is rewritten at the end of every session. Weights are
    excluded from the repository, so a digest dropped on resume leaves a file
    on disk that no reported number can be traced back to."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=2, name="digests")
    config_path.write_text(
        config_path.read_text().replace(
            "  tensorboard_dir:", "  checkpoint_every: 2\n  tensorboard_dir:"
        )
    )

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()

    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config_path), "--epochs", "3", "--resume"]
    )
    train_module.main()

    record = json.loads((tmp_path / "results" / "digests" / "run.json").read_text())

    assert "epoch_002.pt" in record["checkpoints"]
    assert record["checkpoints"]["last.pt"]["epoch"] == 3


def test_weighted_error_checkpoint_is_written_for_an_unweighted_run(
    tmp_path: Path, monkeypatch
) -> None:
    """The weighted-error selection rule used to write a checkpoint only when
    the training loss was the weighted one. The metric is computed every epoch
    regardless, so the gate did not decide whether the rule had a selection --
    only whether that selection had weights. One canonical run selected epoch
    18 under this rule with no file written and the epoch off the periodic
    grid, which cannot be recovered. The rule now writes for every run."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=2, name="unweighted")
    assert "masked_mse" in config_path.read_text()
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])

    train_module.main()

    record = json.loads((tmp_path / "results" / "unweighted" / "run.json").read_text())
    written = tmp_path / "results" / "unweighted" / "checkpoints"

    assert "best_val_weighted_mse.pt" in record["checkpoints"]
    assert (written / "best_val_weighted_mse.pt").is_file()
    assert "val_weighted_mse" in record["best"]


def test_resume_records_a_second_session_without_rewriting_the_first(
    tmp_path: Path, monkeypatch
) -> None:
    """The record used to report a cumulative epoch count beside one session's
    wall clock, with the start time overwritten by whichever process finished
    last. Read together those three fields described a run that never happened.
    Per-session facts now live in a list and the cumulative fields are derived
    from it."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=1, name="sessions")
    record_path = tmp_path / "results" / "sessions" / "run.json"

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()
    first = json.loads(record_path.read_text())

    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config_path), "--epochs", "3", "--resume"]
    )
    train_module.main()
    second = json.loads(record_path.read_text())

    assert len(first["sessions"]) == 1
    assert len(second["sessions"]) == 2
    assert second["started_utc"] == first["started_utc"]
    assert second["sessions"][0] == first["sessions"][0]
    assert second["sessions"][0]["resumed"] is False
    assert second["sessions"][1]["resumed"] is True
    assert second["sessions"][1]["first_epoch"] == 2
    assert second["sessions"][1]["last_epoch"] == 3
    assert second["training"]["epochs_run"] == 3
    assert second["training"]["sessions_run"] == 2
    assert second["training"]["wall_seconds"] == pytest.approx(
        sum(s["wall_seconds"] for s in second["sessions"])
    )


def test_resume_restores_the_patience_counter(tmp_path: Path, monkeypatch) -> None:
    """Patience is a counter the loop cannot recompute from weights. A resumed
    run that restarts it at zero trains past the point an uninterrupted run
    would have stopped, which makes early stopping depend on where the run was
    interrupted."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=2, name="patience")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()

    state = torch.load(
        tmp_path / "results" / "patience" / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert "epochs_without_improvement" in state["progress"]


def test_crash_mid_loop_does_not_drop_prior_digests(tmp_path: Path, monkeypatch) -> None:
    """The clean-resume test above cannot see this failure: the record is
    rewritten before the epoch loop, and until the fix it was rewritten from a
    fresh dict without the checkpoints key, so a session that crashed inside
    the loop left run.json on disk with every earlier digest gone. Weights are
    excluded from the repository; a file whose digest is dropped is a file no
    reported number can be traced to, permanently."""
    gold = _write_gold(tmp_path)
    config_path = _write_config(tmp_path, gold, epochs=2, name="crashy")
    config_path.write_text(
        config_path.read_text().replace(
            "  tensorboard_dir:", "  checkpoint_every: 2\n  tensorboard_dir:"
        )
    )

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()
    before = json.loads((tmp_path / "results" / "crashy" / "run.json").read_text())
    assert "epoch_002.pt" in before["checkpoints"]

    def explode(*args, **kwargs):
        raise RuntimeError("simulated crash inside the epoch loop")

    monkeypatch.setattr(train_module, "train_one_epoch", explode)
    monkeypatch.setattr(
        sys, "argv", ["train", "--config", str(config_path), "--epochs", "3", "--resume"]
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_module.main()

    after = json.loads((tmp_path / "results" / "crashy" / "run.json").read_text())
    assert "epoch_002.pt" in after.get("checkpoints", {}), (
        "digests from earlier sessions were dropped by the pre-loop record write"
    )
