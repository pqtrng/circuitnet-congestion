"""Task 5 audit: patch counts per split and normalization statistics."""

from __future__ import annotations

import json
from pathlib import Path

GOLD = Path("data/gold/summary.json")
STATS = Path("data/gold/norm_stats.json")


def run() -> str:
    m = json.loads(GOLD.read_text())
    stats = json.loads(STATS.read_text())
    padded = m["n_padded_patches"]
    lines = [
        f"patch size:            {m['patch_size']}x{m['patch_size']}",
        f"total patches:         {m['n_patches']}",
        f"  train / val / test:  {m['counts']['train']} / "
        f"{m['counts']['val']} / {m['counts']['test']}",
        f"fully-valid patches:   {m['n_full_patches']}",
        f"edge patches (padded): {padded} ({padded / m['n_patches'] * 100:.0f}%)",
        "",
        "normalization (z-score, fit on TRAIN valid pixels only):",
    ]
    for k, v in sorted(stats.items()):
        lines.append(f"  {k:14s} mean={v['mean']:.6f} std={v['std']:.6f}")
    lines.append("  macro_region:  left binary (not z-scored)")
    lines.append("  gt:            left raw (target, comparable to literature)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
