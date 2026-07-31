"""Tests for the rendered blind-spot table.

The table's job is to make one claim legible: predicting zero is almost always
right and never useful. These parse the emitted rows and check the two columns
carry that, per split, from the committed probe -- not from anything a model
produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import audit_blindspot

ROOT = Path(__file__).resolve().parents[1]
CORRECT, HOTSPOT = 2, 3


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def _rows() -> dict[str, list[str]]:
    body = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in audit_blindspot.run().splitlines()
        if line.startswith("|") and set(line) - set("|- ")
    ][1:]
    return {row[0]: row for row in body}


def _percent(cell: str) -> float:
    return float(cell.rstrip("%")) / 100


def test_every_split_appears_once() -> None:
    rows = _rows()

    assert set(rows) == set(audit_blindspot.SPLITS)
    assert len(rows) == len(audit_blindspot.SPLITS)


def test_correct_and_hotspot_come_from_the_probe() -> None:
    splits = json.loads((ROOT / audit_blindspot.PROBE).read_text())["splits"]
    rows = _rows()

    for name, row in rows.items():
        split = splits[name]
        expected_correct = 1 - split["nonzero_fraction_of_valid"]
        expected_hotspot = split["error_share_above"][audit_blindspot.THRESHOLD_KEY][
            "pixel_fraction"
        ]
        assert _percent(row[CORRECT]) == pytest.approx(expected_correct, abs=5e-4)
        assert _percent(row[HOTSPOT]) == pytest.approx(expected_hotspot, abs=5e-5)


def test_zero_is_almost_always_right_and_almost_never_a_hotspot() -> None:
    """The claim the table exists to support, stated as an inequality so a
    future re-measure that broke it would fail here rather than mislead."""
    for row in _rows().values():
        assert _percent(row[CORRECT]) > 0.9
        assert _percent(row[HOTSPOT]) < 0.05
