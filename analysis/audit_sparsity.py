"""T4 audit: GT congestion sparsity — evidence that pixel metrics mislead."""

from __future__ import annotations

import json
from pathlib import Path

SILVER = Path("data/silver/manifest.json")


def run() -> str:
    m = json.loads(SILVER.read_text())
    seen: dict[str, dict] = {}
    for r in m:
        seen.setdefault(r["sha1"], r)
    uniq = list(seen.values())
    fr = [r["gt_frac_nonzero"] for r in uniq]
    n = len(fr)
    mean = sum(fr) / n
    lines = [
        f"unique samples:          {n}",
        f"gt_frac_nonzero mean:    {mean:.4f}  ({mean * 100:.1f}% of pixels are hotspots)",
        f"gt_frac_nonzero min/max: {min(fr):.4f} / {max(fr):.4f}",
        "",
        f"An all-zero predictor is correct on ~{(1 - mean) * 100:.1f}% of pixels,",
        "scoring near-perfect on pixel metrics (SSIM/MSE/NRMS) while detecting",
        "zero hotspots — the only thing that matters for routability.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
