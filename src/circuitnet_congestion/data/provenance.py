"""Provenance manifest: pin exactly which dataset revision and files were used.

The manifest is committed to the repo (it is small). Raw data is not. Anyone
who clones the repo can read the manifest and re-download the identical data
by revision hash, making the dataset reproducible without transferring files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class FileRecord:
    path: str  # path within the HF repo
    size: int  # bytes, as reported by HF
    sha256: str | None  # HF LFS sha256 if available, else None


@dataclass
class Provenance:
    hf_repo_id: str
    hf_repo_type: str
    revision: str  # resolved commit hash (never a floating branch)
    downloaded_at: str  # UTC ISO-8601
    designs: dict[str, list[str]]
    files: list[FileRecord]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def utc_now() -> str:
        return datetime.now(UTC).isoformat()


def load_provenance(path: Path) -> Provenance:
    data = json.loads(path.read_text())
    files = [FileRecord(**f) for f in data.pop("files")]
    return Provenance(files=files, **data)
