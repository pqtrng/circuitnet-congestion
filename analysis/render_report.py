"""Render the decision documents from templates plus live audit output.

Each `<!--AUDIT:name-->` marker is replaced by the fenced output of
`analysis.audit_<name>.run()`. The audit list is discovered from the markers
themselves rather than maintained separately, so adding a section to a template
is enough.

The figures substituted at markers come from committed manifests and probe
records, so re-rendering reflects the current evidence. The sentences around
those figures -- template prose and the audits' own strings -- are written by
hand; tests/test_prose_claims.py gates the hand-written prose it can see, and
its docstring records what it cannot. Audits read only committed JSON, which
means this runs on a fresh clone with no data layer and no accelerator.
"""

from __future__ import annotations

import importlib
import re
from datetime import date
from pathlib import Path

DOCS = Path("docs")
MARKER = re.compile(r"<!--AUDIT:(\w+)-->")


def render(template_path: Path) -> str:
    text = template_path.read_text()

    for name in MARKER.findall(text):
        module = importlib.import_module(f"analysis.audit_{name}")
        text = text.replace(f"<!--AUDIT:{name}-->", f"```\n{module.run()}\n```")

    return text.replace("<!--DATE-->", date.today().isoformat())


def main() -> None:
    templates = sorted(DOCS.glob("*.md.tmpl"))
    if not templates:
        raise SystemExit(f"no templates found under {DOCS}")

    for template_path in templates:
        output_path = template_path.with_suffix("")
        output_path.write_text(render(template_path))
        sections = len(MARKER.findall(template_path.read_text()))
        print(f"wrote {output_path}  ({sections} audit sections)")


if __name__ == "__main__":
    main()
