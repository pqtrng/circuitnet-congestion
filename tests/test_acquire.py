"""Unit tests for acquisition logic and provenance. No network, CPU-safe."""

from pathlib import Path

from circuitnet_congestion.data.acquire import build_allow_patterns
from circuitnet_congestion.data.provenance import (
    FileRecord,
    Provenance,
    load_provenance,
)


def test_allow_patterns_one_per_design():
    designs = {"train": ["RISCY", "nvdla-large"], "val": ["Vortex-large"], "test": ["openc910-1"]}
    patterns = build_allow_patterns(designs)
    assert len(patterns) == 4
    assert "CircuitNet-N14/routability_features/RISCY.tar.gz" in patterns
    assert all(p.endswith(".tar.gz") for p in patterns)
    assert all("routability_features" in p for p in patterns)


def test_allow_patterns_excludes_other_features():
    designs = {"train": ["RISCY"], "val": [], "test": []}
    patterns = build_allow_patterns(designs)
    joined = " ".join(patterns)
    assert "IR_drop" not in joined
    assert "graph_features" not in joined
    assert "raw_data" not in joined


def test_provenance_roundtrip(tmp_path: Path):
    prov = Provenance(
        hf_repo_id="CircuitNet/CircuitNet",
        hf_repo_type="dataset",
        revision="deadbeef",
        downloaded_at=Provenance.utc_now(),
        designs={"train": ["RISCY"], "val": ["Vortex-large"], "test": ["openc910-1"]},
        files=[
            FileRecord(
                path="CircuitNet-N14/routability_features/RISCY.tar.gz", size=123, sha256="abc"
            )
        ],
    )
    out = tmp_path / "prov.json"
    prov.to_json(out)
    loaded = load_provenance(out)
    assert loaded.revision == "deadbeef"
    assert loaded.files[0].size == 123
    assert loaded.designs["val"] == ["Vortex-large"]
