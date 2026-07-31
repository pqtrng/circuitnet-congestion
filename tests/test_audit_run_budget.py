"""Tests for the run-budget audit.

The sentence it renders is the unit the Open Questions quote costs in, so what
it must not do is invent a number: the epoch budget has to come from the run
records, and the two canonical runs have to agree on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import audit_run_budget

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def test_budget_is_the_recorded_epoch_count() -> None:
    budgets = {
        json.loads((ROOT / path).read_text())["training"]["epochs_run"]
        for path in audit_run_budget.RUNS
    }
    assert len(budgets) == 1
    assert f"{budgets.pop()} epochs" in audit_run_budget.run()


def test_no_absolute_time_or_hardware_is_named() -> None:
    """Cost is quoted in run-budgets precisely so the sentence need not, and
    must not, describe the machine."""
    rendered = audit_run_budget.run().lower()

    for forbidden in ("hour", "minute", "second", "gpu", "cuda", "accelerator"):
        assert forbidden not in rendered
