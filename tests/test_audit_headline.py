"""Tests for the rendered headline table.

A marker gates where an audit's output goes, not what it says, and every audit
writes its own strings. So these parse the rendered table rather than trust the
module that emits it: a swapped column or a mislabelled row is visible at no
other level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import audit_headline

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "predict zero everywhere"
RATIO, PRECISION, RECALL, F1 = 3, 4, 5, 6
# The ratio column is rendered at a coarser precision than the rates, so each
# carries the tolerance its own rounding allows.
RATIO_TOLERANCE = 5e-4
RATE_TOLERANCE = 5e-5


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audits address artefacts relative to the working directory."""
    monkeypatch.chdir(ROOT)


def _body() -> list[list[str]]:
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in audit_headline.run().splitlines()
        if line.startswith("|") and set(line) - set("|- ")
    ]
    return rows[1:]


def _value(cell: str) -> float:
    return float(cell.strip("*"))


def _canonical() -> dict[str, dict]:
    runs = json.loads((ROOT / audit_headline.SELECTION).read_text())["runs"]
    return {name: run for name, run in runs.items() if not run["superseded"]}


def test_the_reference_row_leads_the_table_at_parity() -> None:
    row = _body()[0]

    assert row[0] == REFERENCE
    assert "*" not in row[RATIO]
    assert _value(row[RATIO]) == pytest.approx(1.0, abs=RATIO_TOLERANCE)


def test_the_reference_row_detects_nothing_and_reports_no_precision() -> None:
    """A numeral in that cell would read as a bad detector measured, rather
    than as no detector at all."""
    row = _body()[0]

    assert _value(row[RECALL]) == 0.0
    assert _value(row[F1]) == 0.0
    with pytest.raises(ValueError):
        _value(row[PRECISION])


def test_each_canonical_selection_appears_once_with_its_recorded_figures() -> None:
    body = _body()[1:]
    canonical = _canonical()
    seen = 0

    for run in canonical.values():
        label = audit_headline.LOSS_LABELS[run["loss"]]
        for rule, rule_label in audit_headline.RULES:
            chosen = run["selections"][rule]
            matches = [r for r in body if r[0] == label and r[1] == rule_label]
            assert len(matches) == 1
            row = matches[0]
            seen += 1
            assert row[2] == str(chosen["epoch"])
            ratio = chosen["val_mse_over_baseline"]
            assert _value(row[RATIO]) == pytest.approx(ratio, abs=RATIO_TOLERANCE)
            for column, key in (
                (PRECISION, "val_precision"),
                (RECALL, "val_recall"),
                (F1, "val_f1"),
            ):
                assert _value(row[column]) == pytest.approx(chosen[key], abs=RATE_TOLERANCE)

    assert len(body) == seen


def test_superseded_runs_stay_out_of_the_headline() -> None:
    """Real measurement, and the decision record rests on them -- but not the
    runs this table compares."""
    recorded = json.loads((ROOT / audit_headline.SELECTION).read_text())["runs"]
    canonical = _canonical()

    assert len(canonical) < len(recorded)
    assert len(_body()) == len(canonical) * len(audit_headline.RULES) + 1


def test_bold_marks_the_rows_the_pixel_metric_rates_below_the_reference() -> None:
    marked = {row[0] + row[1]: row[RATIO].startswith("**") for row in _body()}
    worse = {row[0] + row[1]: _value(row[RATIO]) > 1.0 for row in _body()}

    assert marked == worse
    assert any(marked.values()), "the marking rule is never exercised by this evidence"


def test_the_probe_and_the_training_loop_agree_on_the_zero_predictor() -> None:
    """The reference row divides one by the other, so parity is a claim about
    two independent measurements of the same quantity, not an identity."""
    probed = json.loads((ROOT / audit_headline.TARGET_STATS).read_text())
    measured = probed["splits"][audit_headline.SPLIT]["zero_predictor_mse"]

    for run in _canonical().values():
        recorded = run["baseline_val_zero_predictor_mse"]
        assert abs(measured - recorded) / recorded < 1e-6
