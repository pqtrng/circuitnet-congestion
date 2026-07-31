"""Task 4 audit: duplicate detection via Silver SHA-1."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SILVER = Path("data/silver/manifest.json")


def run() -> str:
    m = json.loads(SILVER.read_text())
    total = len(m)
    sha_count = Counter(r["sha1"] for r in m)
    unique = len(sha_count)
    in_dup = sum(n for n in sha_count.values() if n > 1)

    by_sha = defaultdict(list)
    for r in m:
        by_sha[r["sha1"]].append(r["design"])
    cross = sum(1 for ds in by_sha.values() if len(set(ds)) > 1)

    per_total = Counter(r["design"] for r in m)
    per_uniq = defaultdict(set)
    for r in m:
        per_uniq[r["design"]].add(r["sha1"])

    lines = [
        f"total samples:           {total}",
        f"unique (by sha1):        {unique}",
        f"duplicate rate:          {(total - unique) / total * 100:.0f}%",
        f"samples in a dup group:  {in_dup}",
        f"dup groups cross-design: {cross}  (0 = design-wise split is leak-free)",
        "",
        "per design (total -> unique):",
    ]
    for d in sorted(per_total):
        lines.append(f"  {d:14s} {per_total[d]:5d} -> {len(per_uniq[d]):4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
