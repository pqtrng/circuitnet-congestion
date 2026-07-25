"""Render docs/data_decisions.md from a template plus live audit output.

Each <!--AUDIT:name--> marker is replaced by the fenced output of
analysis.audit_<name>.run(). Numbers come from the pipeline manifests and are
never hand-written, so re-running always reflects the current data.
"""

from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path

TMPL = Path("docs/data_decisions.md.tmpl")
OUT = Path("docs/data_decisions.md")
AUDITS = ["provenance", "dedup", "sparsity", "shapes", "patches"]


def render() -> str:
    text = TMPL.read_text()
    for name in AUDITS:
        mod = importlib.import_module(f"analysis.audit_{name}")
        text = text.replace(f"<!--AUDIT:{name}-->", f"```\n{mod.run()}\n```")
    return text.replace("<!--DATE-->", date.today().isoformat())


def main() -> None:
    OUT.write_text(render())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
