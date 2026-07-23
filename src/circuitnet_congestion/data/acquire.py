"""Acquire the CircuitNet-N14 routability (congestion) subset from Hugging Face.

Downloads only the designs named in the config, resumes on interruption, pins
the resolved commit revision, and writes a provenance manifest. Idempotent:
re-running with the same revision re-verifies local files instead of refetching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from circuitnet_congestion.data.provenance import FileRecord, Provenance


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def build_allow_patterns(designs: dict[str, list[str]]) -> list[str]:
    """One glob per design tarball under routability_features."""
    all_designs = [d for group in designs.values() for d in group]
    return [f"CircuitNet-N14/routability_features/{d}.tar.gz" for d in all_designs]


def resolve_revision(api: HfApi, repo_id: str, repo_type: str) -> str:
    """Resolve the current head of main to an immutable commit hash."""
    info = api.repo_info(repo_id, repo_type=repo_type, revision="main")
    sha = info.sha
    if sha is None or len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        raise ValueError(f"Expected a 40-char commit hash, got: {sha!r}")
    return sha


def record_files(
    api: HfApi, repo_id: str, repo_type: str, revision: str, patterns: list[str]
) -> list[FileRecord]:
    wanted = set(patterns)
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    records: list[FileRecord] = []
    for sib in info.siblings:
        if sib.rfilename in wanted:
            sha = None
            if sib.lfs is not None:
                sha = sib.lfs.get("sha256") if isinstance(sib.lfs, dict) else None
            records.append(FileRecord(path=sib.rfilename, size=sib.size or 0, sha256=sha))
    return sorted(records, key=lambda r: r.path)


def acquire(config_path: Path) -> Provenance:
    cfg = load_config(config_path)
    repo_id = cfg["dataset"]["hf_repo_id"]
    repo_type = cfg["dataset"]["hf_repo_type"]
    designs = cfg["dataset"]["designs"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    provenance_path = Path(cfg["paths"]["provenance"])

    api = HfApi()
    revision = resolve_revision(api, repo_id, repo_type)
    patterns = build_allow_patterns(designs)

    print(f"Repo:     {repo_id} ({repo_type})")
    print(f"Revision: {revision}")
    print(f"Designs:  {sum(len(v) for v in designs.values())} tarballs")
    print(f"Target:   {raw_dir}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(raw_dir),
    )

    files = record_files(api, repo_id, repo_type, revision, patterns)
    prov = Provenance(
        hf_repo_id=repo_id,
        hf_repo_type=repo_type,
        revision=revision,
        downloaded_at=Provenance.utc_now(),
        designs=designs,
        files=files,
    )
    prov.to_json(provenance_path)
    print(f"Provenance written: {provenance_path} ({len(files)} files)")
    return prov


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire CircuitNet-N14 congestion subset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config YAML.",
    )
    args = parser.parse_args()
    acquire(args.config)


if __name__ == "__main__":
    main()
