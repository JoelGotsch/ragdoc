#!/usr/bin/env python3
"""The taste gate for a review page: structure, anchors, accessibility, safety, diagram rules.

    python3 scripts/check_review.py docs/pr-reviews/<slug>

Errors (exit 1): leftover placeholders, unresolved peeks, anchors that do not resolve, changed
files outside every lens, duplicate ids, missing SVG title/desc contract, remote resources other
than the Google Fonts stylesheet, diagonal connectors, a diagram whose inline ``min-width``
disagrees with its ``viewBox``, text below 4.5:1 contrast, a faint dashed stroke, or a
missing light-theme lock. Warnings (printed, exit 0): a diagram over its complexity budget.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import self_check
from _review import Source, changed_files, load_review, repo_root
from contrast import contrast_findings
from resolve import DATA_SOURCE_RE, PEEK_RE, UNCATEGORIZED_RE, verify_anchors

DIAGRAM_RE = re.compile(r'(<figure\b[^>]*\bclass="diagram"[^>]*>)(.*?)(</figure>)', re.DOTALL)
SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.DOTALL)
LINE_RE = re.compile(r"<line\b([^>]*)/?>", re.DOTALL)
ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"')
ID_RE = re.compile(r'\bid="([^"]+)"')
PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\}\}|\[TODO[^\]]*\]")
BUDGETS = {
    "sequence": {"actor": 5, "msg": 12},
    "flow": {"node": 9, "edge": 12},
    "call-stack-diff": {"frame": 20},
}


def attrs_of(tag: str) -> dict[str, str]:
    return dict(ATTR_RE.findall(tag))


def _missing_commit(repo: Path, sha: str) -> bool:

    return (
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode
        != 0
    )


def check_structure(page: str, errors: list[str]) -> None:
    for match in PLACEHOLDER_RE.finditer(page):
        errors.append(f"placeholder left in page: {match.group(0)!r}")
    for section in ("what-why", "implementation", "files"):
        if f'id="{section}"' not in page:
            errors.append(f'missing <section id="{section}">')
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.DOTALL)
    if h1 is None or not re.sub(r"<[^>]+>", "", h1.group(1)).strip():
        errors.append("page needs a non-empty <h1>")
    for match in PEEK_RE.finditer(page):
        if "<pre>" not in match.group(2) or "peek-unresolved" in match.group(2):
            source = DATA_SOURCE_RE.search(match.group(1))
            errors.append(f"unresolved peek {source.group(1) if source else '(no data-source)'}; run resolve.py")
    uncategorized = UNCATEGORIZED_RE.search(page)
    if uncategorized is None:
        errors.append('missing <ul class="uncategorized" data-auto> in the files section')
    ids = Counter(ID_RE.findall(page))
    for value, count in ids.items():
        if count > 1:
            errors.append(f"duplicate id {value!r} ({count}×): prefix diagram-local ids with the diagram slug")


def check_safety(page: str, errors: list[str]) -> None:
    """self_check's single-file safety and accessible-SVG rules; <a href> navigation is allowed."""
    parser = self_check.parsed_document(page)
    errors.extend(parser.unsafe)
    self_check.check_css_references(parser, errors)
    for tag, rel, value in parser.references:
        if tag == "a":
            continue
        finding = self_check.reference_error(tag, rel, value)
        if finding:
            errors.append(finding)
    if parser.svgs:
        self_check.check_svgs(parser, errors)
    self_check.check_scripts(parser, errors)
    if parser.scripts:
        errors.append("review pages are static: remove the <script>")


def check_diagrams(page: str, errors: list[str], warnings: list[str]) -> None:
    for match in DIAGRAM_RE.finditer(page):
        figure = attrs_of(match.group(1))
        slug = figure.get("data-diagram", "")
        kind = figure.get("data-type", "")
        label = f"diagram {slug or '(no data-diagram)'}"
        if not slug:
            errors.append(f'{label}: <figure class="diagram"> needs data-diagram="<slug>"')
        if kind not in BUDGETS:
            errors.append(f"{label}: data-type must be one of {sorted(BUDGETS)}")
        svg_open = SVG_OPEN_RE.search(match.group(2))
        if svg_open is None:
            errors.append(f"{label}: no <svg> inside the figure")
            continue
        svg_attrs = attrs_of(svg_open.group(1))
        if svg_attrs.get("aria-labelledby") != f"{slug}-title {slug}-desc":
            errors.append(f'{label}: svg aria-labelledby must be "{slug}-title {slug}-desc"')
        view = svg_attrs.get("viewBox", "").split()
        if len(view) != 4 or not all(v.lstrip("-").isdigit() for v in view):
            errors.append(f"{label}: viewBox must be four integers")
        else:
            width, height = int(view[2]), int(view[3])
            if width % 4 or height % 4:
                errors.append(f"{label}: viewBox {width}×{height} is not on the 4px grid")
            style = svg_attrs.get("style", "")
            min_width = re.search(r"min-width:\s*(\d+)px", style)
            if min_width is None or int(min_width.group(1)) != width:
                errors.append(f'{label}: svg needs style="min-width:{width}px" (equal to its viewBox width)')
        body = match.group(2)
        for line in LINE_RE.finditer(body):
            a = attrs_of(line.group(1))
            try:
                diagonal = float(a.get("x1", 0)) != float(a.get("x2", 0)) and float(a.get("y1", 0)) != float(
                    a.get("y2", 0)
                )
            except ValueError:
                continue
            if diagonal:
                errors.append(
                    f"{label}: diagonal <line> ({a.get('x1')},{a.get('y1')})→({a.get('x2')},{a.get('y2')}); use an elbow path"
                )
        for role, limit in BUDGETS.get(kind, {}).items():
            count = len(re.findall(rf'data-k="{role}"', body))
            if count > limit:
                warnings.append(f"{label}: {count} {role} elements, budget is {limit}; split into overview + detail")
        if slug and f"{slug}.svg" not in body and f'href="{slug}.svg"' not in match.group(0):
            warnings.append(
                f'{label}: no <a class="svg-link" href="{slug}.svg"> in its figcaption; run export_diagrams.py'
            )


def check(review_dir: Path) -> int:
    review_dir = review_dir.resolve()
    repo = repo_root(review_dir)
    review = load_review(review_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for side in ("base", "head"):
        sha = review[side].get("sha")
        if sha is None:
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            errors.append(f"review.json {side}.sha is not a full sha: {sha!r}")
        elif _missing_commit(repo, sha):
            errors.append(f"review.json {side}.sha {sha[:12]} is not a commit in this repository")
    page = (review_dir / "index.html").read_text(encoding="utf-8")
    check_structure(page, errors)
    check_safety(page, errors)
    check_diagrams(page, errors, warnings)
    errors.extend(contrast_findings(page))
    if 'name="color-scheme" content="only light"' not in page or "darkreader-lock" not in page:
        errors.append(
            'missing light-theme lock: <meta name="color-scheme" content="only light"> and <meta name="darkreader-lock">'
        )
    source = Source(repo, review)
    _report, anchor_errors = verify_anchors(page, source)
    errors.extend(anchor_errors)
    changed = changed_files(repo, review, review_dir)
    uncategorized = UNCATEGORIZED_RE.search(page)
    if uncategorized and "<li>" in uncategorized.group(2):
        errors.append("changed files are listed under 'Not in any lens'; add them to a lens and re-run resolve.py")
    if not changed:
        warnings.append("the pinned range has no changed files")
    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")
    verdict = "FAIL" if errors else "OK"
    print(f"{verdict} {review_dir.relative_to(repo)}: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_dir", type=Path)
    return check(parser.parse_args().review_dir)


if __name__ == "__main__":
    sys.exit(main())
