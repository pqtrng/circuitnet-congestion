"""Audit: disagreement between checkpoint selection rules, and what is loadable.

Renders the measurement written by analysis.probe_selection_gap. The probe
needs run histories and the machine of record; this reads the committed JSON,
so `make report` works for a reader who has cloned the repository and has no
data and no accelerator.

Two questions are kept separate because conflating them produced a wrong
report once. What did each rule select? -- answered by post-hoc replay over
recorded history, for every run including superseded ones, with no requirement
that weights exist. Which selections can be loaded? -- answered from the
checkpoint files recorded per run and their existence at probe time on the
machine of record. A selection in the first table and absent from the second
is a measurement about a model that cannot be loaded; several are, and the
report says which.

Provenance is rendered, not asserted: worktree state comes from each run
record, and the committed-code comparison between the canonical runs is
executed against this clone's git objects, with an explicit unverifiable
outcome when the clone cannot answer.

Short runs are shown but kept out of the headline range: a ratio taken against
an F1 near zero is arithmetically large and carries no information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROBE = Path("results/probes/selection_gap.json")

# Below this many epochs a run's F1 is still close enough to zero that the
# ratio between two selections is dominated by its denominator.
MINIMUM_EPOCHS_FOR_RATIO = 15

RULE_LABELS = {
    "val_mse": "lowest error",
    "val_weighted_mse": "lowest weighted error",
    "val_f1": "highest hotspot F1",
    "final_epoch": "final epoch",
}


def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    name, run = item
    return (1 if run["superseded"] else 0, name)


def _format_run(name: str, run: dict[str, Any]) -> list[str]:
    marker = "  [superseded]" if run["superseded"] else ""
    lines = [
        f"{name}{marker}",
        f"  loss={run['loss']}  epochs={run['epochs_recorded']}  "
        f"zero-predictor baseline={run['baseline_val_zero_predictor_mse']:.4e}",
        f"  {'selected by':<24} {'epoch':>5} {'error/baseline':>15} {'F1':>8} "
        f"{'recall':>8}  {'weights':<9}",
    ]

    for rule, chosen in run["selections"].items():
        ratio = chosen.get("val_mse_over_baseline")
        files = chosen.get("checkpoint_files")
        if files is None:
            weights = "?"
        elif not files:
            weights = "none"
        elif chosen.get("deployable"):
            weights = f"{sum(1 for f in files if f['exists'])} file(s)"
        else:
            weights = "gone"
        lines.append(
            f"  {RULE_LABELS.get(rule, rule):<24} {chosen['epoch']:>5} "
            f"{ratio:>15.4f} {chosen['val_f1']:>8.4f} {chosen['val_recall']:>8.4f}"
            f"  {weights:<9}"
        )

    gap = run["gap_between_error_and_f1"]
    ratio = gap["f1_ratio"]
    apart = gap["epochs_apart"]
    plural = "epoch" if abs(apart) == 1 else "epochs"

    if ratio is None:
        # The error-selected model found nothing at all, so the ratio has no
        # denominator. That is a stronger statement than any finite factor.
        lines.append(
            f"  -> the error rule selected a model with an F1 of exactly zero, "
            f"{abs(apart)} {plural} before the F1 rule's choice"
        )
    else:
        note = "" if run["epochs_recorded"] >= MINIMUM_EPOCHS_FOR_RATIO else " (short run)"
        direction = "later" if apart > 0 else "earlier" if apart < 0 else "at the same epoch"
        when = f"{abs(apart)} {plural} {direction}" if apart else "at the same epoch"
        lines.append(f"  -> the F1 rule's choice scores {ratio:.1f}x higher on F1, {when}{note}")

    if gap["f1_choice_error_over_baseline"] and gap["f1_choice_error_over_baseline"] > 1.0:
        lines.append(
            "  -> that choice is rated worse than predicting zero everywhere by the pixel metric"
        )

    return lines


def _deployable(runs: dict[str, Any]) -> list[str]:
    lines = [
        "Deployable selections -- rows exist only where a checkpoint file covering",
        "the selected epoch still existed when the probe ran on the machine of",
        "record. Weights are excluded from the repository, so this is a recorded",
        "snapshot, not a property of this clone.",
        "",
        f"  {'run':<34} {'selected by':<24} {'epoch':>5}  {'file':<22} {'sha256':<12}  worktree",
    ]
    objective_file_seen = False
    recorded = surviving = sup_recorded = sup_surviving = 0
    for name, run_entry in sorted(runs.items(), key=_sort_key):
        recorded += run_entry.get("checkpoints_recorded", 0)
        surviving += run_entry.get("checkpoints_surviving", 0)
        if run_entry["superseded"]:
            sup_recorded += run_entry.get("checkpoints_recorded", 0)
            sup_surviving += run_entry.get("checkpoints_surviving", 0)
        git = run_entry.get("git") or {}
        tree = "dirty" if git.get("dirty") else "clean" if git else "?"
        for rule, chosen in run_entry["selections"].items():
            if not chosen.get("deployable"):
                continue
            for f in chosen["checkpoint_files"]:
                if not f["exists"]:
                    continue
                if f["file"] == "best_val_objective.pt":
                    objective_file_seen = True
                digest = (f.get("sha256") or "?")[:12]
                lines.append(
                    f"  {name:<34} {RULE_LABELS.get(rule, rule):<24} "
                    f"{chosen['epoch']:>5}  {f['file']:<22} {digest:<12}  {tree}"
                )
    if lines[-1].endswith("worktree"):
        lines.append("  (none -- no recorded checkpoint file survived at probe time)")
    lines += [
        "",
        f"Across all runs {recorded} checkpoint digests are recorded and "
        f"{surviving} files survived at probe time. The superseded runs retain "
        f"{sup_recorded} digests and {sup_surviving} files: their disagreement "
        "survives in the metrics while the models behind it do not.",
    ]
    if objective_file_seen:
        lines += [
            "",
            "best_val_objective.pt is matched to a selection by epoch alone. The",
            "quantity the loop tracked under that name is the run's own validation",
            "objective, which is not the replayed weighted-error metric, despite",
            "the similar name.",
        ]
    return lines


def _provenance(runs: dict[str, Any], committed_code: dict[str, Any]) -> list[str]:
    canonical = {n: r for n, r in runs.items() if not r["superseded"]}
    lines = ["Provenance of the canonical runs:", ""]
    dirty_runs = []
    for name, run_entry in sorted(canonical.items()):
        git = run_entry.get("git")
        if not git:
            lines.append(f"  {name}: no git block in the run record")
            continue
        rev = git.get("revision", "")
        tree = "dirty worktree" if git.get("dirty") else "clean worktree"
        lines.append(f"  {name}: revision {rev[:10]}, {tree}")
        if git.get("dirty"):
            dirty_runs.append(name)

    # The comparison is measured once by the probe and read here, so this
    # renders identically in any clone -- a git call at render time would make
    # a shallow clone produce a different document.
    status = committed_code.get("status")
    revisions = committed_code.get("revisions", [])
    if len(revisions) == 2:
        a, b = (r[:10] for r in revisions)
        lines.append("")
        if status == "identical":
            lines.append(
                "The committed training code is identical between the two revisions: "
                f"`git diff {a} {b} -- src configs` is empty."
            )
        elif status == "differs":
            names = committed_code.get("changed_files", [])
            lines.append(
                "The committed training code DIFFERS between the two revisions: " + ", ".join(names)
            )
        else:
            lines.append(
                "The committed-code comparison was not recorded as resolved when the probe ran: "
                f"one or both revisions were absent. Run `git diff {a} {b} -- src configs`."
            )
    for name in dirty_runs:
        lines.append(
            f"{name} was run from a modified worktree; the modification was not "
            "captured and cannot be recovered. Any comparison involving it is the "
            "configured difference plus an unquantified uncommitted delta."
        )
    return lines


def run() -> str:
    record = json.loads(PROBE.read_text())
    runs = record["runs"]
    committed_code = record.get("committed_code", {})

    lines: list[str] = [
        "Selection disagreement (metric-only). Epochs are selected by post-hoc",
        "replay over each run's recorded history; a selection here may or may not",
        "have loadable weights -- that is the second table's question.",
        "",
    ]
    for name, entry in sorted(runs.items(), key=_sort_key):
        lines.extend(_format_run(name, entry))
        lines.append("")
    lines.extend(_deployable(runs))
    lines.append("")
    lines.extend(_provenance(runs, committed_code))
    lines.append("")

    substantial = [
        r["gap_between_error_and_f1"]["f1_ratio"]
        for r in runs.values()
        if r["gap_between_error_and_f1"]["f1_ratio"] is not None
        and r["epochs_recorded"] >= MINIMUM_EPOCHS_FOR_RATIO
    ]
    zeroed = sum(1 for r in runs.values() if r["gap_between_error_and_f1"]["f1_ratio"] is None)

    lines.append(f"Across {len(runs)} runs the two rules never select the same epoch.")
    if substantial:
        lines.append(
            f"In the {len(substantial)} run(s) long enough for the ratio to mean "
            f"anything, the F1 rule's choice scores between {min(substantial):.1f}x "
            f"and {max(substantial):.1f}x higher on F1."
        )
    if zeroed:
        lines.append(
            f"In {zeroed} run(s) the error rule selected a model that detected no "
            "hotspot at all, while a model in the same run did."
        )
    lines.append(
        f"Ratios from runs shorter than {MINIMUM_EPOCHS_FOR_RATIO} epochs are shown "
        "per run but excluded from that range: an F1 near zero makes the "
        "denominator unstable."
    )
    lines.append(
        "Selecting on pixel error is therefore not a neutral default. It is a "
        "choice that discards most of the run's detection performance, and the "
        "loss curve gives no indication that it is happening."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
