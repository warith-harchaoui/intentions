"""Render every ```mermaid block in a Markdown file to a PNG, for review.

Module summary
--------------
The docs embed Mermaid diagrams (architecture, engine progression). GitHub
renders them, but while *authoring* we want to see the result locally and
iterate on the code until it looks right. This script pulls each ```mermaid
fence out of a Markdown file and renders it to ``<out>/<stem>-<n>.png`` via a
headless Chromium loading mermaid.js from a CDN.

Usage
-----
    python scripts/render_mermaid.py README.md --out /tmp/mermaid

Author
------
Project maintainers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# mermaid.js from a pinned CDN version — only used at authoring time to render
# preview PNGs, never shipped in the product.
_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/+esm"

# One HTML page per diagram: white canvas, Roboto. The diagram source is
# passed to ``mermaid.render`` as a JS string (JSON-encoded) rather than placed
# in the DOM, so ``<br/>`` in node labels is NOT parsed as HTML by the browser.
_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>
  body {{ margin: 0; padding: 24px; background: #ffffff;
          font-family: Roboto, system-ui, sans-serif; }}
  #out {{ font-family: Roboto, system-ui, sans-serif; }}
</style></head><body>
<div id="out"></div>
<script type="module">
  import mermaid from "{cdn}";
  mermaid.initialize({{ startOnLoad: false, theme: "base",
    themeVariables: {{ fontFamily: "Roboto, system-ui, sans-serif" }} }});
  const code = {code_json};
  try {{
    const {{ svg }} = await mermaid.render("g", code);
    document.getElementById("out").innerHTML = svg;
  }} catch (e) {{ document.title = "ERR:" + e.message; }}
</script></body></html>"""

_FENCE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def extract_blocks(markdown: str) -> list[str]:
    """Return the Mermaid source of every ```mermaid fence, in order.

    Parameters
    ----------
    markdown : str
        Full Markdown document text.

    Returns
    -------
    list[str]
        Each fenced diagram's source (without the fences).
    """
    return [m.group(1).strip() for m in _FENCE.finditer(markdown)]


def render_block(code: str, out: Path) -> Path:
    """Render one Mermaid diagram to a PNG at ``out``.

    Parameters
    ----------
    code : str
        Mermaid diagram source.
    out : Path
        Destination PNG path.

    Returns
    -------
    Path
        ``out``.
    """
    import json

    from playwright.sync_api import sync_playwright

    html = _PAGE.format(code_json=json.dumps(code), cdn=_MERMAID_CDN)
    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=2.0)
        page.set_content(html, wait_until="load")
        # Wait until mermaid has injected the rendered SVG into #out.
        page.wait_for_selector("#out svg", timeout=30_000)
        page.locator("#out").screenshot(path=str(out))
        browser.close()
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: render every Mermaid block in a Markdown file to PNGs.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success, 1 if no diagrams were found).
    """
    parser = argparse.ArgumentParser(prog="render_mermaid")
    parser.add_argument("markdown", type=Path, help="Markdown file to scan.")
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/mermaid"), help="Output directory."
    )
    args = parser.parse_args(argv)

    blocks = extract_blocks(args.markdown.read_text(encoding="utf-8"))
    if not blocks:
        print(f"Aucun diagramme mermaid dans {args.markdown}", file=sys.stderr)
        return 1
    stem = args.markdown.stem
    for i, code in enumerate(blocks, start=1):
        out = args.out / f"{stem}-{i}.png"
        render_block(code, out)
        print(f"Rendu : {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
