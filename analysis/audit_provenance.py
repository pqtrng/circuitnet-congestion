"""T2 audit: what was acquired, from which pinned revision."""

from __future__ import annotations

import json
from pathlib import Path

PROV = Path("data/provenance/circuitnet-n14.json")


def run() -> str:
    p = json.loads(PROV.read_text())
    total = sum(f["size"] for f in p["files"])
    lines = [
        f"repo:      {p['hf_repo_id']} ({p['hf_repo_type']})",
        f"revision:  {p['revision']}",
        f"files:     {len(p['files'])} design tarballs",
        f"total raw: {total / 1e9:.1f} GB (compressed)",
        "",
        "design split:",
    ]
    for grp, designs in sorted(p["designs"].items()):
        lines.append(f"  {grp:6s} {', '.join(designs)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
