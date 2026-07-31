"""Tests for the rendered test-evaluation table.

Parses the emitted rows against the probe artefact, and pins the two facts the
surrounding prose asserts: the F1-selected checkpoint out-detects the
error-selected one in each run, and on this split no trained checkpoint is
rated worse than the zero-predictor -- the inversion seen on validation is
absent here. If a re-measure broke either, the claim should fail at this level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import audit_test_eval

ROOT = Path(__file__).resolve().parents[1]
RATIO, PRECISION, RECALL, F1 = 3, 4, 5, 6


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def _body() -> list[list[str]]:
    rows = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in audit_test_eval.run().splitlines()
        if line.startswith("|") and set(line) - set("|- ")
    ]
    return rows[1:]


def _value(cell: str) -> float:
    return float(cell.strip("*"))


def _runs() -> dict:
    return json.loads((ROOT / audit_test_eval.PROBE).read_text())["runs"]


def test_reference_row_then_one_row_per_published_selection() -> None:
    runs = _runs()
    expected = 1 + sum(len(r["selections"]) for r in runs.values())

    body = _body()
    assert body[0][0] == "predict zero everywhere"
    assert len(body) == expected


def test_each_cell_comes_from_the_probe() -> None:
    runs = _runs()
    body = _body()[1:]
    seen = 0

    for record in runs.values():
        label = audit_test_eval.LOSS_LABELS[record["loss"]]
        for key, rule_label in audit_test_eval.SELECTIONS:
            sel = record["selections"][key]
            row = next(r for r in body if r[0] == label and r[1] == rule_label)
            seen += 1
            assert row[2] == str(sel["epoch"])
            assert _value(row[RATIO]) == pytest.approx(sel["test_mse_over_baseline"], abs=5e-4)
            assert _value(row[PRECISION]) == pytest.approx(sel["test_precision"], abs=5e-5)
            assert _value(row[RECALL]) == pytest.approx(sel["test_recall"], abs=5e-5)
            assert _value(row[F1]) == pytest.approx(sel["test_f1"], abs=5e-5)

    assert seen == len(body)


def test_the_selection_gap_survives_on_test() -> None:
    """In each run the F1-selected checkpoint out-detects the error-selected
    one; this is the result the section is built to report."""
    for record in _runs().values():
        by_error = record["selections"]["best_val_mse"]
        by_f1 = record["selections"]["best_val_f1"]
        assert by_f1["test_recall"] > by_error["test_recall"]
        assert by_f1["test_f1"] > by_error["test_f1"]


def test_no_checkpoint_is_rated_below_the_zero_predictor_on_test() -> None:
    """The validation inversion is absent here, so no row is bold. Pinned so a
    re-measure that reintroduced it would fail rather than quietly contradict
    the prose."""
    for row in _body()[1:]:
        assert "*" not in row[RATIO]
        assert _value(row[RATIO]) < 1.0
