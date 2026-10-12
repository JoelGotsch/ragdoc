#!/usr/bin/env python3
"""Fill and verify every source-bound element of a review page from git.

    python3 scripts/resolve.py docs/pr-reviews/<slug> [--check-only]

What it does to ``index.html`` (idempotent; re-run after every edit and after a retarget):

* ``<figure class="peek" data-source="head/path#L10-L24" data-caption="…">`` — the body
  (figcaption, numbered code, optional italic note) is regenerated from the pinned commit.
* ``<a class="src" data-source="…">`` — ``href`` is set to the GitHub blob URL at the pinned sha.
* ``<details class="lens">`` with ``<ul class="lens-files" data-files="globs…">`` — the file
  list with ``+added −deleted`` per file is regenerated from ``git diff --numstat``, and the
  ``<span class="lens-stat">`` in its summary is filled.
* ``<ul class="uncategorized" data-auto>`` — lists changed files that no lens claims.

Every ``data-source`` in the file (prose links, peeks, diagram groups) is verified against the
commit it names and its first and last line are printed, so you can check they are the lines
you meant. Exit status 1 when any anchor does not resolve or a changed file is in no lens.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from _review import Anchor, ChangedFile, Source, changed_files, glob_to_regex, load_review, parse_anchor, repo_root

PEEK_RE = re.compile(r'(<figure\b[^>]*\bclass="peek"[^>]*>)(.*?)(</figure>)', re.DOTALL)
LENS_RE = re.compile(r'(<details\b[^>]*\bclass="lens"[^>]*>)(.*?)(</details>)', re.DOTALL)
LENS_FILES_RE = re.compile(r'(<ul\b[^>]*\bclass="lens-files"[^>]*>)(.*?)(</ul>)', re.DOTALL)
LENS_STAT_RE = re.compile(r'(<span\b[^>]*\bclass="lens-stat"[^>]*>)(.*?)(</span>)', re.DOTALL)
UNCATEGORIZED_RE = re.compile(r'(<ul\b[^>]*\bclass="uncategorized"[^>]*>)(.*?)(</ul>)', re.DOTALL)
SRC_LINK_RE = re.compile(r'<a\b([^>]*\bdata-source="[^"]+"[^>]*)>')
DATA_SOURCE_RE = re.compile(r'data-source="([^"]+)"')
ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"')
MAX_PEEK_LINES = 60


def attrs_of(tag: str) -> dict[str, str]:
    return {name: html.unescape(value) for name, value in ATTR_RE.findall(tag)}


def render_peek_body(anchor: Anchor, source: Source, caption: str | None) -> tuple[str, str | None]:
    lines = source.lines(anchor.side, anchor.path)
    label = f'<span class="peek-path">{html.escape(anchor.path)}</span> <span class="peek-lines">{anchor.lines_label}</span>'
    side = f'<span class="peek-side peek-side-{anchor.side}">{anchor.side}</span>'
    link = ""
    url = source.blob_url(anchor)
    if url:
        link = f'<a class="peek-link" href="{html.escape(url)}">open ↗</a>'
    note = f'<p class="peek-note">{html.escape(caption)}</p>' if caption else ""
    if lines is None or anchor.end > len(lines):
        reason = "file not found on this side" if lines is None else f"only {len(lines)} lines"
        body = (
            f"\n  <figcaption>{label} {side}{link}</figcaption>\n"
            f'  <p class="peek-unresolved">unresolved: {anchor} ({reason})</p>\n{note}\n'
        )
        return body, f"MISSING {anchor} ({reason})"
    width = len(str(anchor.end))
    code = "\n".join(
        f'<span class="ln">{str(number).rjust(width)}</span>{html.escape(lines[number - 1])}'
        for number in range(anchor.start, anchor.end + 1)
    )
    warning = None
    if anchor.end - anchor.start + 1 > MAX_PEEK_LINES:
        warning = f"WARN {anchor} is {anchor.end - anchor.start + 1} lines; peeks read best under {MAX_PEEK_LINES}"
    body = f"\n  <figcaption>{label} {side}{link}</figcaption>\n  <pre><code>{code}</code></pre>\n{note}\n"
    return body, warning


def resolve_peeks(page: str, source: Source, messages: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        attrs = attrs_of(match.group(1))
        raw = attrs.get("data-source")
        if raw is None:
            messages.append('ERROR a <figure class="peek"> has no data-source')
            return match.group(0)
        try:
            anchor = parse_anchor(raw)
        except ValueError as exc:
            messages.append(f"ERROR {exc}")
            return match.group(0)
        body, problem = render_peek_body(anchor, source, attrs.get("data-caption"))
        if problem:
            messages.append(problem)
        return match.group(1) + body + match.group(3)

    return PEEK_RE.sub(replace, page)


def resolve_links(page: str, source: Source, messages: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        attrs = attrs_of(inner)
        try:
            anchor = parse_anchor(attrs["data-source"])
        except ValueError as exc:
            messages.append(f"ERROR {exc}")
            return match.group(0)
        url = source.blob_url(anchor)
        inner = re.sub(r'\s*\bhref="[^"]*"', "", inner)
        href = html.escape(url) if url else "#"
        return f'<a href="{href}"{inner}>'

    return SRC_LINK_RE.sub(replace, page)


def verify_anchors(page: str, source: Source) -> tuple[list[str], list[str]]:
    """Return (report lines, error lines) for every distinct data-source in the page."""
    report: list[str] = []
    errors: list[str] = []
    for raw in dict.fromkeys(DATA_SOURCE_RE.findall(page)):
        try:
            anchor = parse_anchor(html.unescape(raw))
        except ValueError as exc:
            errors.append(f"ERROR {exc}")
            continue
        lines = source.lines(anchor.side, anchor.path)
        if lines is None:
            errors.append(f"MISSING {anchor}: no such file on the {anchor.side} side")
            continue
        if anchor.end > len(lines):
            errors.append(f"MISSING {anchor}: file has {len(lines)} lines")
            continue
        first = lines[anchor.start - 1].strip() or "(blank line)"
        last = lines[anchor.end - 1].strip() or "(blank line)"
        report.append(f"OK  {anchor}\n      first: {first[:110]}\n      last:  {last[:110]}")
    return report, errors


def file_row(file: ChangedFile, source: Source) -> str:
    url = source.file_diff_url(file.path)
    name = f'<a href="{html.escape(url)}">{html.escape(file.path)}</a>' if url else html.escape(file.path)
    status = f'<span class="status">{file.status}</span> ' if file.status != "M" else ""
    if file.added is None:
        return f'<li>{status}{name}<span class="status">binary</span><span></span></li>'
    return f'<li>{status}{name}<span class="add">+{file.added}</span><span class="del">−{file.deleted}</span></li>'


def resolve_lenses(page: str, source: Source, files: list[ChangedFile], messages: list[str]) -> str:
    claimed: dict[str, list[str]] = {}

    def replace_lens(match: re.Match[str]) -> str:
        block = match.group(2)
        files_match = LENS_FILES_RE.search(block)
        if files_match is None:
            messages.append('ERROR a <details class="lens"> has no <ul class="lens-files">')
            return match.group(0)
        patterns = attrs_of(files_match.group(1)).get("data-files", "").split()
        title = re.sub(r"<[^>]+>", "", block.split("</summary>")[0]).strip()[:60]
        regexes = [glob_to_regex(p) for p in patterns]
        matched = [f for f in files if any(r.match(f.path) for r in regexes)]
        for f in matched:
            claimed.setdefault(f.path, []).append(title)
        rows = "\n" + "\n".join(f"          {file_row(f, source)}" for f in matched) + "\n        " if matched else ""
        added = sum(f.added or 0 for f in matched)
        deleted = sum(f.deleted or 0 for f in matched)
        stat = f"{len(matched)} files · +{added} −{deleted}" if matched else "0 files"
        if not matched:
            messages.append(f"WARN lens {title!r} matches no changed file")
        block = LENS_FILES_RE.sub(lambda m: m.group(1) + rows + m.group(3), block, count=1)
        block = LENS_STAT_RE.sub(lambda m: m.group(1) + stat + m.group(3), block, count=1)
        return match.group(1) + block + match.group(3)

    page = LENS_RE.sub(replace_lens, page)
    for path, lenses in claimed.items():
        if len(lenses) > 1:
            messages.append(f"WARN {path} is in {len(lenses)} lenses: {lenses}")
    uncategorized = [f for f in files if f.path not in claimed]
    if UNCATEGORIZED_RE.search(page):
        rows = (
            "\n" + "\n".join(f"          {file_row(f, source)}" for f in uncategorized) + "\n        "
            if uncategorized
            else ""
        )
        page = UNCATEGORIZED_RE.sub(lambda m: m.group(1) + rows + m.group(3), page, count=1)
    if uncategorized:
        messages.append(f"ERROR {len(uncategorized)} changed file(s) are in no lens:")
        messages.extend(f"    {f.path}" for f in uncategorized[:40])
        if len(uncategorized) > 40:
            messages.append(f"    … {len(uncategorized) - 40} more")
    else:
        messages.append(f"OK  every one of the {len(files)} changed files is in a lens")
    return page


def resolve(review_dir: Path, check_only: bool = False) -> int:
    review_dir = review_dir.resolve()
    repo = repo_root(review_dir)
    review = load_review(review_dir)
    source = Source(repo, review)
    index = review_dir / "index.html"
    original = index.read_text(encoding="utf-8")
    messages: list[str] = []
    page = resolve_peeks(original, source, messages)
    page = resolve_links(page, source, messages)
    page = resolve_lenses(page, source, changed_files(repo, review, review_dir), messages)
    report, errors = verify_anchors(page, source)
    if page != original and not check_only:
        index.write_text(page, encoding="utf-8")
        print(f"updated {index}")
    print("\n".join(report))
    print()
    for message in messages + errors:
        print(message)
    failed = errors + [m for m in messages if m.startswith(("ERROR", "MISSING"))]
    print(f"\n{len(report)} anchors resolved, {len(failed)} problem(s)")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify without writing index.html")
    args = parser.parse_args()
    return resolve(args.review_dir, args.check_only)


if __name__ == "__main__":
    sys.exit(main())
