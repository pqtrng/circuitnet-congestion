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
from pathlib import Path

DOCS = Path("docs")
README = Path("README.md.tmpl")
MARKER = re.compile(r"<!--AUDIT:(\w+)-->")
TABLE = re.compile(r"<!--TABLE:(\w+)-->")


def render(template_path: Path) -> str:
    text = template_path.read_text()

    for name in MARKER.findall(text):
        module = importlib.import_module(f"analysis.audit_{name}")
        text = text.replace(f"<!--AUDIT:{name}-->", f"```\n{module.run()}\n```")

    # TABLE markers insert audit output raw, for markdown tables that must
    # render as tables rather than as fenced text.
    for name in TABLE.findall(text):
        module = importlib.import_module(f"analysis.audit_{name}")
        text = text.replace(f"<!--TABLE:{name}-->", module.run())

    return text


def main() -> None:
    templates = sorted(DOCS.glob("*.md.tmpl"))
    if README.is_file():
        templates.append(README)
    if not templates:
        raise SystemExit(f"no templates found under {DOCS}")

    for template_path in templates:
        output_path = template_path.with_suffix("")
        output_path.write_text(render(template_path))
        source = template_path.read_text()
        sections = len(MARKER.findall(source)) + len(TABLE.findall(source))
        print(f"wrote {output_path}  ({sections} audit sections)")


if __name__ == "__main__":
    main()
