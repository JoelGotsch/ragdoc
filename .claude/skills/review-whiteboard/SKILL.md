---
name: review-whiteboard
description: Write a git-committed code-review whiteboard for a branch or PR as a self-contained HTML page with hand-drawn inline SVG diagrams (sequence, flowchart, call-stack diff), verified code peeks and file lenses, under docs/pr-reviews/. Use when asked to "whiteboard" a change, review a branch or PR into a document, or produce a PR review page; replaces the desktop whiteboard app with files.
---

# review-whiteboard

Produce `docs/pr-reviews/<YYYY-MM-DD>-<slug>/` holding `index.html` (the review),
`review.json` (the pinned base/head shas) and one `<diagram>.svg` per diagram. The page reads
like a whiteboard session (what/why → design → implementation → file lenses) and every source
reference in it is verified against the pinned commits by the scripts in this skill. Nothing in
it depends on a desktop app; the directory is meant to be committed with the branch.

All scripts are stdlib Python 3.11+; run them with `python3` from anywhere in the repository.
`SKILL=.claude/skills/review-whiteboard` below.

## Workflow

1. **Scaffold and read the change.**
   ```bash
   python3 $SKILL/scripts/new_review.py --title "<title>" [--base main] [--head <ref>|worktree] [--pr <n>]
   ```
   It pins `merge-base(base, head)` and the head sha, writes the directory, and prints the
   commit list and diff stat. With `--pr` it records the PR and prints its description via `gh`.
   Then read: the PR description or commit messages, the diff stat, and the largest changed
   files. Measure churn before believing line counts (`git diff --ignore-all-space --stat`,
   mode-bit and line-ending commits); say so in the page when counts are inflated.

2. **Write what/why first**, in the `#what-why` section: what the change is, why it was made,
   what breaks. One or two paragraphs plus at most one callout. If you cannot state the why
   from the PR text or commits, say that the page infers it.

3. **Group every changed file into lenses** (`#files`). Non-implementation lenses first
   (tests, docs, notebooks, config/CI, renames, formatting churn), then implementation lenses by
   design thread in reading order. A lens is `<details class="lens">` with a `data-files` glob
   list; see `references/components.md`. Run `resolve.py` and keep going until the
   "Not in any lens" list is empty.

4. **Design** (`#design`): the design threads as a numbered list, one diagram that shows the
   central runtime behaviour (usually a sequence diagram of one request), code peeks for the
   contracts a reader must see verbatim, and a callout for the guarantees and non-guarantees.

5. **Implementation** (`#implementation`): a call-stack diff rooted at the entry point for the
   thread where the most code moved, then one `<details class="sub">` per remaining thread in
   reading order with inline `<a class="src">` links and a peek only for an invariant or guard.

6. **Resolve, read back, check, export.**
   ```bash
   python3 $SKILL/scripts/resolve.py docs/pr-reviews/<dir>          # fills peeks, links, lenses; prints every anchor's first/last line
   python3 $SKILL/scripts/check_review.py docs/pr-reviews/<dir>     # structure, safety, diagram rules, budgets
   python3 $SKILL/scripts/export_diagrams.py docs/pr-reviews/<dir>  # <slug>.svg per diagram + download links
   ```
   Read the anchor report line by line: an anchor that resolves to the wrong function is the
   most common error and no script can catch it. Fix, re-run until `check_review.py` prints `OK`.
   Open `index.html` in a browser once (and one exported `.svg`) before reporting.

7. **Hand over.** Tell the user the directory, that it is uncommitted, and the one-line commit
   recipe. Do not commit unless asked. If the repository's docs site serves `docs/` verbatim,
   mention that the page will be published with the docs.

## Updating a review after new commits

```bash
python3 $SKILL/scripts/new_review.py --retarget docs/pr-reviews/<dir> [--head <ref>]
python3 $SKILL/scripts/resolve.py docs/pr-reviews/<dir>
```
Retarget rewrites `review.json` and the header, then `resolve.py` regenerates every peek and
reports anchors that moved or vanished. Re-read the page: prose that described the old code is
now wrong even where the anchors still resolve.

## Writing rules

- **Verify, do not assert.** Every function you name gets an `<a class="src">`; every claim
  about behaviour points at a line range. If you did not read it on the pinned sha, do not
  write it. Remove claims the read-back does not support rather than hedging them.
- **Anchors are typed from the file you read**, never from memory. `resolve.py` prints the
  first and last line of each range; check both.
- **Budgets are hard**: 5 lifelines / 12 messages, 9 nodes / 12 edges, 10 frames per side,
  2 accents, 2 annotation callouts, peeks under 60 lines. Over budget means a second diagram.
- **The page is static**: no JavaScript, no external resources besides the Google Fonts
  stylesheet, no images except inline SVG, light skin only. `check_review.py` fails otherwise.
- **Numbers come from commands** (`git rev-list --count --no-merges`, `--numstat`) and are
  quoted once, in what/why or a lens note, not sprinkled through prose.
- Write incrementally: scaffold → what/why → lenses → design → implementation, running
  `resolve.py` after each step so an error surfaces next to the edit that caused it.

## Files

- `assets/template.html` — page skeleton, tokens and component CSS.
- `references/components.md` — the markup grammar the scripts resolve.
- `references/style-guide.md` — tokens, type, treatments, connector rules, budgets.
- `references/diagrams.md` — sequence and flowchart drawing rules, SVG skeleton.
- `references/diagram-call-stack-diff.md` — the base|head call-stack diagram.
- `scripts/new_review.py`, `resolve.py`, `check_review.py`, `export_diagrams.py` — see each
  module docstring; `_review.py` holds the shared helpers; `export_svg.py` and `self_check.py`
  are vendored from diagram-design (MIT, `LICENSE-diagram-design.md`).
