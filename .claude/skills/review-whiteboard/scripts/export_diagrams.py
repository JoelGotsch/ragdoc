#!/usr/bin/env python3
"""Export every diagram of a review page to a standalone SVG next to it.

    python3 scripts/export_diagrams.py docs/pr-reviews/<slug> [--system-fonts]

Each ``<figure class="diagram" data-diagram="<slug>">`` becomes ``<slug>.svg`` in the review
directory, through the vendored diagram-design exporter (``export_svg.py``): standalone XML,
namespaced ``<defs>`` ids, rgba fills normalised for strict SVG consumers. The page links to
each file from the figure's ``<a class="svg-link">``; this script adds or refreshes that link.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import export_svg

DIAGRAM_RE = re.compile(r'(<figure\b[^>]*\bclass="diagram"[^>]*>)(.*?)(</figure>)', re.DOTALL)
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL)
FONT_LINK_RE = re.compile(r"<link\b[^>]*fonts\.googleapis\.com[^>]*>")
CAPTION_RE = re.compile(r"(<figcaption\b[^>]*>)(.*?)(</figcaption>)", re.DOTALL)
SVG_LINK_RE = re.compile(r'\s*<a class="svg-link" href="[^"]*">[^<]*</a>')

MINI_STYLE = (
    "<style>:root{--color-paper:#f5f5f5;--color-ink:#2d3142;--color-muted:#4f5d75;--color-accent:#bf4520;"
    "--font-sans:'Geist',system-ui,sans-serif;--font-serif:'Instrument Serif',serif;"
    "--font-mono:'Geist Mono',ui-monospace,monospace}"
    "body{font-family:var(--font-sans);color:var(--color-ink)}svg{display:block}</style>"
)


def standalone_size(document: str) -> str:
    """Give the root <svg> an intrinsic size (its viewBox) instead of the page's min-width style."""
    root = re.search(r"<svg\b[^>]*>", document)
    view = re.search(r'viewBox="0 0 (\d+) (\d+)"', root.group(0)) if root else None
    if root is None or view is None:
        return document
    tag = re.sub(r'\s*style="min-width:\s*\d+px;?"', "", root.group(0))
    if " width=" not in tag:
        tag = tag.replace("<svg", f'<svg width="{view[1]}" height="{view[2]}"', 1)
    if "color-scheme" not in tag:
        tag = tag.replace("<svg", '<svg style="color-scheme: only light"', 1)
    return document[: root.start()] + tag + document[root.end() :]


def export(review_dir: Path, system_fonts: bool) -> int:
    review_dir = review_dir.resolve()
    index = review_dir / "index.html"
    page = index.read_text(encoding="utf-8")
    font_link = FONT_LINK_RE.search(page)
    head = (font_link.group(0) if font_link and not system_fonts else "") + MINI_STYLE
    written: list[str] = []
    failures = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal failures
        slug_match = re.search(r'data-diagram="([^"]+)"', match.group(1))
        svg_match = SVG_RE.search(match.group(2))
        if slug_match is None or svg_match is None:
            failures += 1
            print("skip: a diagram figure lacks data-diagram or an <svg>", file=sys.stderr)
            return match.group(0)
        slug = slug_match.group(1)
        mini = f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{slug}</title>{head}</head><body>{svg_match.group(0)}</body></html>'
        try:
            document = export_svg.export_svg_document(mini, Path(f"{slug}.html"), system_fonts=system_fonts)
        except ValueError as exc:
            failures += 1
            print(f"error: {slug}: {exc}", file=sys.stderr)
            return match.group(0)
        (review_dir / f"{slug}.svg").write_text(standalone_size(document), encoding="utf-8")
        written.append(f"{slug}.svg")
        link = f' <a class="svg-link" href="{slug}.svg">↓ {slug}.svg</a>'
        body = match.group(2)
        if CAPTION_RE.search(body):
            body = CAPTION_RE.sub(
                lambda m: m.group(1) + SVG_LINK_RE.sub("", m.group(2)).rstrip() + link + m.group(3), body, count=1
            )
        else:
            body = f"\n  <figcaption>{slug}{link}</figcaption>" + body
        return match.group(1) + body + match.group(3)

    updated = DIAGRAM_RE.sub(replace, page)
    if updated != page:
        index.write_text(updated, encoding="utf-8")
    for name in written:
        print(name)
    if not written and not failures:
        print('no <figure class="diagram"> in the page')
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("--system-fonts", action="store_true", help="omit the Google Fonts @import from the SVGs")
    args = parser.parse_args()
    return export(args.review_dir, args.system_fonts)


if __name__ == "__main__":
    sys.exit(main())
