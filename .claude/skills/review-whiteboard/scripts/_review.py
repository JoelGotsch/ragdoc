"""Shared helpers for the review-whiteboard scripts (stdlib only).

A review lives in one directory (``docs/pr-reviews/<slug>/``) holding ``index.html`` and
``review.json``. ``review.json`` pins the two sides of the change:

    {"title": ..., "slug": ..., "created": "YYYY-MM-DD",
     "repository": {"remote": "...", "web": "https://github.com/owner/repo" | null},
     "base": {"ref": "main", "sha": "<40 hex>"},
     "head": {"ref": "feature/x", "sha": "<40 hex>"}}        # or {"ref": "worktree", "sha": null}

Anchors name source ranges on one side: ``head/src/pkg/mod.py#L10-L24`` or ``base/...#L7``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ANCHOR_RE = re.compile(r"^(?P<side>head|base)/(?P<path>[^#\s]+)#L(?P<start>\d+)(?:-L(?P<end>\d+))?$")
GITHUB_REMOTE_RE = re.compile(
    r"^(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)(?P<repo>[^/]+/[^/]+?)(?:\.git)?/?$"
)


@dataclass(frozen=True)
class Anchor:
    side: str
    path: str
    start: int
    end: int

    @property
    def lines_label(self) -> str:
        return f"L{self.start}" if self.start == self.end else f"L{self.start}–L{self.end}"

    @property
    def fragment(self) -> str:
        return f"L{self.start}" if self.start == self.end else f"L{self.start}-L{self.end}"

    def __str__(self) -> str:
        return f"{self.side}/{self.path}#{self.fragment}"


def parse_anchor(value: str) -> Anchor:
    match = ANCHOR_RE.match(value.strip())
    if match is None:
        raise ValueError(f"malformed anchor {value!r}: expected head|base/<path>#L<start>[-L<end>]")
    start = int(match["start"])
    end = int(match["end"]) if match["end"] else start
    if start < 1 or end < start:
        raise ValueError(f"bad line range in anchor {value!r}")
    return Anchor(match["side"], match["path"], start, end)


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def repo_root(start: Path) -> Path:
    return Path(git(start if start.is_dir() else start.parent, "rev-parse", "--show-toplevel").strip())


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].rstrip("-") or "review"


def web_url_for_remote(remote: str) -> str | None:
    match = GITHUB_REMOTE_RE.match(remote.strip())
    return f"https://github.com/{match['repo']}" if match else None


def load_review(review_dir: Path) -> dict:
    return json.loads((review_dir / "review.json").read_text(encoding="utf-8"))


def save_review(review_dir: Path, review: dict) -> None:
    (review_dir / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")


def is_worktree(review: dict) -> bool:
    return review["head"].get("sha") is None


class Source:
    """Reads file contents on either side of the pinned change, with a per-call cache."""

    def __init__(self, repo: Path, review: dict) -> None:
        self.repo = repo
        self.review = review
        self._cache: dict[tuple[str, str], list[str] | None] = {}

    def sha(self, side: str) -> str | None:
        return self.review[side].get("sha")

    def lines(self, side: str, path: str) -> list[str] | None:
        key = (side, path)
        if key not in self._cache:
            sha = self.sha(side)
            if sha is None:
                file = self.repo / path
                text = file.read_text(encoding="utf-8", errors="replace") if file.is_file() else None
            else:
                result = subprocess.run(
                    ["git", "-C", str(self.repo), "show", f"{sha}:{path}"], capture_output=True, text=True, check=False
                )
                text = result.stdout if result.returncode == 0 else None
            self._cache[key] = None if text is None else text.split("\n")
        return self._cache[key]

    def web(self) -> str | None:
        return self.review.get("repository", {}).get("web")

    def blob_url(self, anchor: Anchor) -> str | None:
        web, sha = self.web(), self.sha(anchor.side)
        if web is None or sha is None:
            return None
        return f"{web}/blob/{sha}/{anchor.path}#{anchor.fragment}"

    def compare_url(self) -> str | None:
        web, base, head = self.web(), self.sha("base"), self.sha("head")
        if web is None or base is None or head is None:
            return None
        return f"{web}/compare/{base}...{head}"

    def file_diff_url(self, path: str) -> str | None:
        compare = self.compare_url()
        if compare is None:
            return None
        return f"{compare}#diff-{hashlib.sha256(path.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str
    added: int | None
    deleted: int | None


def changed_files(repo: Path, review: dict, review_dir: Path | None = None) -> list[ChangedFile]:
    """Every file the change touches (renames tracked), with numstat counts (None for binary).

    Files under the reviews root (the parent of ``review_dir``) are left out: a review committed on
    the branch it describes is not part of the change under review.
    """
    skip = f"{review_dir.resolve().parent.relative_to(repo.resolve()).as_posix()}/" if review_dir else None
    base = review["base"]["sha"]
    head = review["head"].get("sha")
    range_args = [base] if head is None else [base, head]
    status: dict[str, str] = {}
    for line in git(repo, "diff", "--name-status", "-M", *range_args).splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status[parts[-1]] = parts[0][0]
    files: list[ChangedFile] = []
    for line in git(repo, "diff", "--numstat", "-M", *range_args).splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added = None if parts[0] == "-" else int(parts[0])
        deleted = None if parts[1] == "-" else int(parts[1])
        raw_path = parts[2]
        path = _rename_target(raw_path)
        if skip and path.startswith(skip):
            continue
        files.append(ChangedFile(path, status.get(path, "M"), added, deleted))
    return files


def _rename_target(numstat_path: str) -> str:
    """``a/{old => new}/b`` or ``old => new`` from ``--numstat -M`` becomes the new path."""
    brace = re.match(r"^(.*)\{(.*) => (.*)\}(.*)$", numstat_path)
    if brace:
        return f"{brace[1]}{brace[3]}{brace[4]}".replace("//", "/")
    if " => " in numstat_path:
        return numstat_path.split(" => ", 1)[1]
    return numstat_path


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """``src/pkg/**`` (any depth), ``*.py`` (one segment), ``dir/`` (prefix), or an exact path."""
    pattern = pattern.strip()
    if pattern.endswith("/"):
        return re.compile("^" + re.escape(pattern) + ".*$")
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile("^" + out + "$")


def render_meta(review: dict, source: Source) -> str:
    """The header ``<dl class="meta">`` block, regenerated on create and retarget."""
    base, head = review["base"], review["head"]
    compare = source.compare_url()
    compare_html = f'<a href="{compare}">{html.escape(compare)}</a>' if compare else "—"
    head_sha = head.get("sha") or "working tree (uncommitted changes included)"
    pr = review.get("pr")
    pr_row = f'<dt>PR</dt><dd><a href="{html.escape(pr["url"])}">#{pr["number"]}</a></dd>\n        ' if pr else ""
    return (
        '<dl class="meta">\n'
        f'        <dt>Base</dt><dd><code>{html.escape(base["ref"])}</code> <span class="sha">{base["sha"]}</span></dd>\n'
        f'        <dt>Head</dt><dd><code>{html.escape(head["ref"])}</code> <span class="sha">{head_sha}</span></dd>\n'
        f"        {pr_row}<dt>Compare</dt><dd>{compare_html}</dd>\n"
        f"        <dt>Written</dt><dd>{review.get('created', date.today().isoformat())}"
        + (f" · updated {review['updated']}" if review.get("updated") else "")
        + "</dd>\n      </dl>"
    )


META_BLOCK_RE = re.compile(r"(<!-- meta:start -->\n\s*)(.*?)(\n\s*<!-- meta:end -->)", re.DOTALL)


def replace_meta(index_html: str, review: dict, source: Source) -> str:
    match = META_BLOCK_RE.search(index_html)
    if match is None:
        raise ValueError("index.html has no <!-- meta:start --> … <!-- meta:end --> block")
    return index_html[: match.start(2)] + render_meta(review, source) + index_html[match.end(2) :]
