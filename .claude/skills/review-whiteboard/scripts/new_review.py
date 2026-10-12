#!/usr/bin/env python3
"""Create (or retarget) a review directory pinned to a base and head commit.

    python3 scripts/new_review.py --title "Sync engine rewrite" [--base main] [--head HEAD]
    python3 scripts/new_review.py --title "..." --head worktree          # uncommitted work vs base
    python3 scripts/new_review.py --retarget docs/pr-reviews/<slug> [--head <ref>] [--base <ref>]

Run from anywhere inside the repository. Creates ``docs/pr-reviews/<date>-<slug>/`` with
``review.json`` and an ``index.html`` scaffolded from ``assets/template.html``, then prints the
change summary (commit list and diff stat) to read before writing. ``--retarget`` moves an
existing review to new commits and regenerates its header; run ``resolve.py`` afterwards to
re-verify every anchor.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from _review import (
    Source,
    git,
    load_review,
    replace_meta,
    repo_root,
    save_review,
    slugify,
    web_url_for_remote,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE = SKILL_DIR / "assets" / "template.html"


def resolve_sides(repo: Path, base_ref: str, head_ref: str) -> tuple[dict, dict]:
    if head_ref.lower() == "worktree":
        base_sha = git(repo, "merge-base", base_ref, "HEAD").strip()
        return {"ref": base_ref, "sha": base_sha}, {"ref": "worktree", "sha": None}
    head_sha = git(repo, "rev-parse", "--verify", f"{head_ref}^{{commit}}").strip()
    base_sha = git(repo, "merge-base", base_ref, head_sha).strip()
    if head_ref == "HEAD":
        branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
        head_ref = branch if branch != "HEAD" else head_sha[:12]
    return {"ref": base_ref, "sha": base_sha}, {"ref": head_ref, "sha": head_sha}


def default_base(repo: Path) -> str:
    origin_head = git(repo, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD", check=False).strip()
    if origin_head:
        return origin_head.rsplit("/", 1)[-1]
    for candidate in ("main", "master"):
        if git(repo, "rev-parse", "--verify", "--quiet", candidate, check=False).strip():
            return candidate
    raise SystemExit("cannot guess the default branch; pass --base")


def pr_info(repo: Path, number: int) -> dict | None:
    if shutil.which("gh") is None:
        print("gh is not installed; skipping the PR lookup", file=sys.stderr)
        return None
    result = subprocess.run(
        ["gh", "pr", "view", str(number), "--json", "number,url,title,body"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"gh pr view {number} failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    import json

    return json.loads(result.stdout)


def print_summary(repo: Path, review: dict) -> None:
    base, head = review["base"]["sha"], review["head"].get("sha")
    if head is None:
        print("\n── working tree vs base ──")
        print(git(repo, "diff", "--stat", "--stat-width=110", base).strip() or "(no changes)")
        print("untracked files are excluded; `git add -N <file>` includes a new file")
        return
    commits = git(repo, "log", "--oneline", "--no-merges", f"{base}..{head}").splitlines()
    print(f"\n── {len(commits)} commits {base[:10]}..{head[:10]} ──")
    for line in commits[:40]:
        print(line)
    if len(commits) > 40:
        print(f"… {len(commits) - 40} more")
    stat = git(repo, "diff", "--stat", "--stat-width=110", base, head).splitlines()
    print("\n── diff stat (last 25 rows) ──")
    for line in stat[-25:]:
        print(line)


def create(args: argparse.Namespace) -> int:
    repo = repo_root(Path.cwd())
    base, head = resolve_sides(repo, args.base or default_base(repo), args.head)
    remote = git(repo, "remote", "get-url", "origin", check=False).strip() or None
    today = date.today().isoformat()
    slug = f"{today}-{args.slug or slugify(args.title)}"
    review_dir = repo / args.out / slug
    if review_dir.exists():
        raise SystemExit(f"{review_dir} already exists; pass --slug or --retarget it")
    review = {
        "title": args.title,
        "slug": slug,
        "created": today,
        "repository": {"name": repo.name, "remote": remote, "web": web_url_for_remote(remote) if remote else None},
        "base": base,
        "head": head,
    }
    if args.pr:
        pr = pr_info(repo, args.pr)
        if pr:
            review["pr"] = {"number": pr["number"], "url": pr["url"], "title": pr["title"]}
            print(f"\n── PR #{pr['number']}: {pr['title']} ──\n{pr['body']}\n")
    review_dir.mkdir(parents=True)
    save_review(review_dir, review)
    source = Source(repo, review)
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("{{TITLE}}", html.escape(args.title)).replace("{{REPO_NAME}}", html.escape(repo.name))
    page = page.replace("{{HEAD_SHA}}", head.get("sha") or "worktree")
    page = replace_meta(page, review, source)
    (review_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"created {review_dir.relative_to(repo)}/index.html and review.json")
    print_summary(repo, review)
    return 0


def retarget(args: argparse.Namespace) -> int:
    review_dir = Path(args.retarget).resolve()
    repo = repo_root(review_dir)
    review = load_review(review_dir)
    base_ref = args.base or review["base"]["ref"]
    head_ref = args.head or review["head"]["ref"]
    if head_ref == "worktree" and args.head is None:
        head_ref = "worktree"
    base, head = resolve_sides(repo, base_ref, head_ref)
    review["base"], review["head"] = base, head
    review["updated"] = date.today().isoformat()
    save_review(review_dir, review)
    index = review_dir / "index.html"
    page = replace_meta(index.read_text(encoding="utf-8"), review, Source(repo, review))
    index.write_text(page, encoding="utf-8")
    print(f"retargeted {review_dir.relative_to(repo)} to base {base['sha'][:10]} head {head.get('sha') or 'worktree'}")
    print("now run resolve.py: it regenerates every peek and reports anchors that no longer resolve")
    print_summary(repo, review)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", help="review title (page <h1>); required unless --retarget")
    parser.add_argument("--base", help="base ref (default: origin's default branch, else main)")
    parser.add_argument("--head", default="HEAD", help="head ref, or 'worktree' for uncommitted work (default HEAD)")
    parser.add_argument("--slug", help="directory slug (default: slugified title); the date is prefixed")
    parser.add_argument(
        "--out", default="docs/pr-reviews", help="output root, relative to the repo (default docs/pr-reviews)"
    )
    parser.add_argument("--pr", type=int, help="GitHub PR number: record it and print its description via gh")
    parser.add_argument(
        "--retarget", metavar="DIR", help="move an existing review to new commits instead of creating one"
    )
    args = parser.parse_args()
    if args.retarget:
        return retarget(args)
    if not args.title:
        parser.error("--title is required when creating a review")
    return create(args)


if __name__ == "__main__":
    sys.exit(main())
