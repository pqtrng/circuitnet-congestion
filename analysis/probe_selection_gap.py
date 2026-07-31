"""Probe: how far apart the checkpoint selection rules choose.

A training run produces one trajectory but several defensible ways to pick a
model from it. This probe applies each rule to a recorded history and reports
what the rule chose, so the disagreement between them becomes a measurement
rather than an anecdote.

The rules are replayed post hoc over the recorded history, using the same
strict-improvement comparison the training loop uses, so among equal values the
earliest epoch wins in both places. A replayed selection has weights on disk
only when the loop was configured to write a checkpoint covering that epoch;
whether any file covers it, and whether that file still existed when this probe
ran, is recorded per selection. A selection without a surviving file is a
measurement about a model that cannot be loaded.

Superseded runs are included by default. A run that was replaced is still real
measurement, and the disagreement between rules was first observed in one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS = Path("results")
OUTPUT = RESULTS / "probes" / "selection_gap.json"
SMOKE_PREFIX = "smoke_"

# Each rule maps to the history field it reads and whether it is maximised.
RULES: dict[str, tuple[str, bool]] = {
    "val_mse": ("val_mse", False),
    "val_weighted_mse": ("val_weighted_mse", False),
    "val_f1": ("val_f1", True),
}

REPORTED_FIELDS = (
    "val_mse",
    "val_weighted_mse",
    "val_f1",
    "val_recall",
    "val_precision",
    "val_prediction_max",
    "train_objective",
)


def select(history: list[dict[str, Any]], field: str, maximise: bool) -> dict[str, Any]:
    """The epoch a rule selects, matching the training loop's strict comparison.

    The loop overwrites a checkpoint only on strict improvement, so among equal
    values the earliest epoch is the one whose weights survive. Iterating in
    order and replacing only on a strict win reproduces that.
    """
    chosen = history[0]
    for entry in history[1:]:
        better = entry[field] > chosen[field] if maximise else entry[field] < chosen[field]
        if better:
            chosen = entry
    return chosen


def describe(entry: dict[str, Any], baseline: float | None) -> dict[str, Any]:
    described = {"epoch": entry["epoch"]}
    described.update({name: entry[name] for name in REPORTED_FIELDS if name in entry})
    if baseline:
        # Absolute error is not comparable across splits or runs, because the
        # error of predicting zero everywhere differs between them. The ratio is.
        described["val_mse_over_baseline"] = entry["val_mse"] / baseline
    return described


def probe_run(run_dir: Path) -> dict[str, Any] | None:
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return None

    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if not history:
        return None

    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    baseline = record.get("baseline", {}).get("val_zero_predictor_mse")
    recorded_checkpoints = record.get("checkpoints", {})

    def checkpoint_exists(name: str) -> bool:
        """The run record keys checkpoints by bare filename; the loop writes
        best/last under checkpoints/ and periodic files one level deeper."""
        candidates = (
            run_dir / name,
            run_dir / "checkpoints" / name,
            run_dir / "checkpoints" / "periodic" / name,
        )
        return any(c.is_file() for c in candidates)

    def files_covering(epoch: int) -> list[dict[str, Any]]:
        """Checkpoint files whose recorded epoch matches, by any name.

        Matching is by epoch rather than by rule: a periodic checkpoint makes
        an epoch loadable even when the loop never wrote a best-file for the
        rule, and a best-file written for a different quantity (the loop's own
        objective, say) still holds the weights of its epoch. `exists` is the
        state of the file on this machine when the probe ran; a recorded
        digest whose file is gone stays in the record and says so.
        """
        return [
            {
                "file": name,
                "sha256": info.get("sha256"),
                "exists": checkpoint_exists(name),
            }
            for name, info in sorted(recorded_checkpoints.items())
            if info.get("epoch") == epoch
        ]

    selections = {
        name: describe(select(history, field, maximise), baseline)
        for name, (field, maximise) in RULES.items()
    }
    selections["final_epoch"] = describe(history[-1], baseline)
    for chosen in selections.values():
        files = files_covering(chosen["epoch"])
        chosen["checkpoint_files"] = files
        chosen["deployable"] = any(f["exists"] for f in files)

    by_error = selections["val_mse"]
    by_f1 = selections["val_f1"]
    gap = {
        "epochs_apart": by_f1["epoch"] - by_error["epoch"],
        "f1_ratio": (by_f1["val_f1"] / by_error["val_f1"]) if by_error["val_f1"] else None,
        # A ratio above one means the best hotspot detector in the run is rated
        # worse than predicting zero everywhere by the pixel metric.
        "f1_choice_error_over_baseline": by_f1.get("val_mse_over_baseline"),
        "error_choice_error_over_baseline": by_error.get("val_mse_over_baseline"),
    }

    return {
        "epochs_recorded": len(history),
        "superseded": "superseded" in run_dir.parts,
        "loss": record.get("config", {}).get("loss", {}).get("name"),
        "baseline_val_zero_predictor_mse": baseline,
        "git": record.get("git"),
        "checkpoints_recorded": len(recorded_checkpoints),
        "checkpoints_surviving": sum(1 for name in recorded_checkpoints if checkpoint_exists(name)),
        "selections": selections,
        "gap_between_error_and_f1": gap,
    }


def discover(results_dir: Path) -> list[Path]:
    found = {
        path.parent
        for path in results_dir.rglob("history.jsonl")
        if not path.parent.name.startswith(SMOKE_PREFIX)
    }
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoint selection rules.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=None,
        help="probe a specific run directory; repeatable",
    )
    args = parser.parse_args()

    run_dirs = args.run if args.run else discover(args.results_dir)
    runs = {}
    for run_dir in run_dirs:
        probed = probe_run(run_dir)
        if probed:
            runs[str(run_dir.relative_to(args.results_dir))] = probed

    if not runs:
        raise SystemExit(f"no run histories found under {args.results_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"wrote {args.out}\n")

    for name, run in runs.items():
        gap = run["gap_between_error_and_f1"]
        flag = " [superseded]" if run["superseded"] else ""
        print(f"{name}{flag}  {run['epochs_recorded']} epochs  loss={run['loss']}")
        for rule, chosen in run["selections"].items():
            ratio = chosen.get("val_mse_over_baseline")
            ratio_text = f"{ratio:.4f}x base" if ratio else "no baseline"
            print(
                f"    {rule:18s} ep{chosen['epoch']:>3}  "
                f"val_mse={chosen['val_mse']:.4e} ({ratio_text})  "
                f"f1={chosen['val_f1']:.4f}  recall={chosen['val_recall']:.4f}"
            )
        if gap["f1_ratio"]:
            print(
                f"    -> f1 rule beats error rule by {gap['f1_ratio']:.1f}x on F1, "
                f"{gap['epochs_apart']} epochs apart"
            )
        print()


if __name__ == "__main__":
    main()
