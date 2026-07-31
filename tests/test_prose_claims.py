"""Prose in source must not carry measured quantities inline.

The repository's argument is that every reported number traces to an artefact.
That rule was enforced for the rendered documents -- render_report.py substitutes
audit output at ``<!--AUDIT:name-->`` markers -- but not for module docstrings,
inline comments, config comments, or the template prose *surrounding* those
markers. Numbers written in those places drift from the artefacts they describe,
and several already have.

This test freezes the known set and fails on anything new.

It is a gate on digits in prose. It is not a proof of correctness. A claim can be
wrong, unsupported, or overstated without containing a single digit -- the
docstring of ``tests/test_losses`` asserts a property of the suite that the suite
does not have, and this test cannot see it. Claims of that kind are reviewed by
hand and recorded in the decision document, not here.

The baseline has two sections, split by one question: can this number drift away
from what it describes without anyone noticing?

  INVARIANT  It cannot. Tensor shapes, patch size, dtype literals, names of hash
             algorithms and other standards, and values a test defines for its
             own use. A figure fixed by code rather than by data belongs here
             only once a test pins it; until then it drifts like any measurement.
  PENDING    It can. Quantities measured from the dataset or the hardware, and
             worked examples whose values are hypothetical but not said to be.
             Each is relocated into an artefact with a pointer left in its place.
             This section must reach zero.

Regenerate with ``UPDATE_PROSE_BASELINE=1 pytest tests/test_prose_claims.py``.
New entries always land in PENDING; promoting one to INVARIANT is a manual edit,
deliberately, because that judgement is the whole point of the exercise.
"""

from __future__ import annotations

import ast
import os
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("prose_claims_baseline.txt")
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

PY_ROOTS = ("src", "analysis", "tests")
TEXT_GLOBS = ("configs/*.yaml", "configs/*.yml", "docs/*.tmpl", "README.md.tmpl")

INVARIANT_HEADER = "# INVARIANT"
PENDING_HEADER = "# PENDING"

DIGIT = re.compile(
    "|".join(
        (
            r"\d+\.\d+(?:[eE][+-]?\d+)?",  # 4.73, 3.0667e-05
            r"\d+[eE][+-]?\d+",  # 1e-6
            r"\d+(?:\.\d+)?\s*%",  # 98.5%
            r"\d+(?:\.\d+)?\s*[x\u00d7](?!\d)",  # 15.9x, 10.5×
            r"\b\d+/\d+\b",  # 1/20, 2/44
            r"\b\d{3,}\b",  # 65504, 16384
        )
    )
)

# Quantities written as words: "seven times slower", "two orders of magnitude".
PHRASE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|ninety|hundred|thousand|million|"
    r"several|many)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?\s+"
    r"(?:times|percent|per\s+cent|orders|hundred|thousand|million|faster|slower)\b",
    re.IGNORECASE,
)
ARTEFACT_REF = re.compile(
    r"(?:results/probes|data/(?:gold|silver|provenance))/[A-Za-z0-9_.-]+\.json"
)


def _emit(text: str, rel_path: str, found: set[str]) -> None:
    """Record every prose line in ``text`` that carries a quantity.

    Keys omit line numbers on purpose: moving a line within a file is not a new
    claim, and a baseline keyed by position would churn on every edit.
    """
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if stripped and (DIGIT.search(stripped) or PHRASE.search(stripped)):
            found.add(f"{rel_path}::{stripped}")


def _scan_python(path: Path, rel_path: str, found: set[str]) -> None:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                _emit(doc, rel_path, found)
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type == tokenize.COMMENT:
                _emit(token.string, rel_path, found)


def collect_claims() -> set[str]:
    found: set[str] = set()
    for root in PY_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            rel_path = path.relative_to(ROOT).as_posix()
            if rel_path == SELF:
                continue  # this file describes the gate; it is not subject to it
            _scan_python(path, rel_path, found)
    for pattern in TEXT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rel_path = path.relative_to(ROOT).as_posix()
            raw = path.read_text(encoding="utf-8")
            if path.suffix in {".yaml", ".yml"}:
                raw = "\n".join(line for line in raw.splitlines() if line.lstrip().startswith("#"))
            _emit(raw, rel_path, found)
    return found


def _read_baseline() -> tuple[set[str], set[str]]:
    if not BASELINE.exists():
        pytest.fail(
            f"{BASELINE.name} is missing. Generate it with:\n"
            f"    UPDATE_PROSE_BASELINE=1 pytest {SELF}"
        )
    invariant: set[str] = set()
    pending: set[str] = set()
    section: set[str] | None = None
    for raw in BASELINE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith(INVARIANT_HEADER):
            section = invariant
        elif line.startswith(PENDING_HEADER):
            section = pending
        elif line and not line.startswith("#") and section is not None:
            section.add(line)
    return invariant, pending


def _write_baseline(claims: set[str]) -> None:
    invariant: set[str] = set()
    if BASELINE.exists():
        invariant, _ = _read_baseline()
    keep = sorted(claim for claim in claims if claim in invariant)
    pending = sorted(claim for claim in claims if claim not in invariant)
    lines = [
        "# Generated by tests/test_prose_claims.py. Do not edit entries by hand;",
        "# move a line between sections to reclassify it, delete it once the",
        "# source line no longer carries a quantity.",
        "",
        f"{INVARIANT_HEADER} -- cannot drift: architecture, format, standards, fixtures.",
        *keep,
        "",
        f"{PENDING_HEADER} -- measurements to relocate. Must reach zero.",
        *pending,
        "",
    ]
    BASELINE.write_text("\n".join(lines), encoding="utf-8")


def test_no_measurement_prose_outside_the_baseline() -> None:
    claims = collect_claims()
    if os.environ.get("UPDATE_PROSE_BASELINE") == "1":
        _write_baseline(claims)
        pytest.skip(f"baseline regenerated with {len(claims)} entries")
    invariant, pending = _read_baseline()
    unexpected = sorted(claims - invariant - pending)
    assert not unexpected, (
        "New quantities in prose. Move each into an artefact and leave a pointer,"
        " or add it to the INVARIANT section with a reason:\n  " + "\n  ".join(unexpected)
    )


def test_baseline_has_no_orphaned_entries() -> None:
    """A fixed line must leave the baseline, so the debt cannot rot in place."""
    invariant, pending = _read_baseline()
    orphans = sorted((invariant | pending) - collect_claims())
    assert not orphans, (
        "These baseline entries no longer match any source line. Delete them:\n  "
        + "\n  ".join(orphans)
    )


def _pointer_sources() -> list[Path]:
    paths: list[Path] = []
    for root in PY_ROOTS:
        paths.extend(sorted((ROOT / root).rglob("*.py")))
    for pattern in TEXT_GLOBS:
        paths.extend(sorted(ROOT.glob(pattern)))
    return paths


def test_artefact_pointers_resolve() -> None:
    """A pointer left in place of a number has to lead somewhere.

    Measurements are removed from prose and replaced by a path into the probe
    output. If the path is wrong, or the artefact is renamed, the sentence
    becomes an unsupported claim wearing a citation, which is worse than the
    number it replaced.
    """
    missing: list[str] = []
    for path in _pointer_sources():
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path == SELF:
            continue
        for match in ARTEFACT_REF.finditer(path.read_text(encoding="utf-8")):
            if not (ROOT / match.group(0)).is_file():
                missing.append(f"{rel_path}: {match.group(0)}")
    assert not missing, "prose points at artefacts that do not exist:\n  " + "\n  ".join(missing)
