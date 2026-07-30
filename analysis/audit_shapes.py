"""Task 5 audit: raw shapes cluster by chip, so resizing would confound domain shift."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SILVER = Path("data/silver/manifest.json")


def run() -> str:
    m = json.loads(SILVER.read_text())
    seen: dict[str, dict] = {}
    for r in m:
        seen.setdefault(r["sha1"], r)
    uniq = list(seen.values())

    h = np.array([r["height"] for r in uniq])
    w = np.array([r["width"] for r in uniq])
    ar = w / h
    near_square = ((ar >= 0.9) & (ar <= 1.1)).mean() * 100

    lines = [
        f"unique samples: {len(uniq)}",
        f"H range: {h.min()}-{h.max()} (median {int(np.median(h))})",
        f"W range: {w.min()}-{w.max()} (median {int(np.median(w))})",
        f"aspect W/H median: {np.median(ar):.2f}  (near-square: {near_square:.0f}%)",
        f"smaller dimension under 128: {(np.minimum(h, w) < 128).mean():.0%}"
        f"   under 256: {(np.minimum(h, w) < 256).mean():.0%}",
        "",
        "per-design median shape (size clusters tightly by chip):",
    ]
    by_d = defaultdict(list)
    for r in uniq:
        by_d[r["design"]].append((r["height"], r["width"]))
    for d in sorted(by_d):
        hs = [x[0] for x in by_d[d]]
        ws = [x[1] for x in by_d[d]]
        lines.append(f"  {d:14s} n={len(by_d[d]):5d}  {int(np.median(hs))}x{int(np.median(ws))}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
