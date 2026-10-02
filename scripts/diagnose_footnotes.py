"""
Diagnostic script for footnote resolution.

Runs the full parsing + footnote-resolution pipeline over every document in
test_cases.json and prints a rich report.  The report reasons from ground
truth outward:

  1. Use insertion_point to locate the element that *should* carry the ref.
  2. Check whether that element became a candidate at all.
     - If NOT: show element.html / element.text and why it was skipped
       (page mismatch, wrong element type, number not found by regex).
  3. If it was a candidate, check whether the resolver chose it.
     - If NOT: list every candidate with its score, and highlight which
       one the resolver picked vs. which was correct.

Footnotes with no insertion_point (or a very short one) are skipped because
there is no ground truth to verify against.

Usage (from repo root):
    uv run python scripts/diagnose_footnotes.py
    uv run python scripts/diagnose_footnotes.py --doc 15-november
    uv run python scripts/diagnose_footnotes.py --no-same-page
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import pylatexenc  # noqa: F401
except ImportError:
    print("ERROR: pylatexenc not installed — run: uv sync --all-extras")
    sys.exit(1)

from ragdoc.document import Document, Footnote
from ragdoc.parsing.mineru.base import MinerUMiddleDocument, _latex_to_text
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor as MinerUParser
from ragdoc.processing.footnote import (
    FootnoteCandidate,
    SimpleFootnoteResolver,
    build_footnote_pattern,
    find_footnote_candidates,
    score_footnote_candidates,
)
from ragdoc.utils.helpers import _normalize_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_CASES_FILE = REPO_ROOT / "tests" / "data" / "test_cases.json"
MINERU_DATA_DIR = REPO_ROOT / "tests" / "parsing" / "data" / "mineru"

# ---------------------------------------------------------------------------
# Text normalisation (same as integration tests)
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", _normalize_text(_latex_to_text(text))).strip()


def _contains(haystack: str, needle: str) -> bool:
    hn, nn = _norm(haystack), _norm(needle)
    return nn in hn or hn in nn


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _h1(text: str) -> None:
    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")


def _ok(text: str) -> None:
    print(f"  {GREEN}OK  {RESET} {text}")


def _fail(text: str) -> None:
    print(f"  {RED}FAIL{RESET} {text}")


def _warn(text: str) -> None:
    print(f"  {YELLOW}WARN{RESET} {text}")


def _info(text: str) -> None:
    print(f"         {DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# Core diagnostic logic
# ---------------------------------------------------------------------------


async def diagnose_footnote(
    doc: Document,
    fn_index: int,
    fn_text: str,
    insertion_point: str,
    same_page_only: bool,
) -> str:
    """
    Diagnose one footnote.  Returns a status string:
      'correct' | 'not_found' | 'not_candidate' | 'wrong_pick' | 'no_candidates'
    """
    print(f"\n  fn{fn_index}  text: {fn_text[:80]}")
    print(f"         insertion_point: {insertion_point[:80]}")

    # ------------------------------------------------------------------
    # 1. Find the Footnote element in the parsed document
    # ------------------------------------------------------------------
    fn_norm = _norm(fn_text)
    footnote: Footnote | None = next(
        (f for f in doc.footnotes if fn_norm in _norm(f.text) or _norm(f.text) in fn_norm),
        None,
    )
    if footnote is None:
        _fail("Footnote element not found in parsed document (parsing issue, not resolver issue)")
        return "not_found"

    print(f"         footnote: number={footnote.number}  page={footnote.page}")

    # ------------------------------------------------------------------
    # 2. Find candidates
    # ------------------------------------------------------------------
    candidates = find_footnote_candidates(
        document=doc,
        footnote=footnote,
        context_chars=80,
        same_page_only=same_page_only,
    )

    # ------------------------------------------------------------------
    # 3. Find the expected element (ground truth from insertion_point)
    #
    # Prefer a candidate whose full_context contains the insertion_point,
    # because full_context is the text immediately around the footnote
    # number — exactly what insertion_point describes.  Only fall back to
    # scanning all doc.elements if no candidate matches.
    # ------------------------------------------------------------------
    el_by_id = {el.id: el for el in doc.elements}

    expected_candidate: FootnoteCandidate | None = next(
        (c for c in candidates if _contains(c.full_context, insertion_point)),
        None,
    )
    if expected_candidate is not None:
        expected_el = el_by_id.get(expected_candidate.element_id)
    else:
        expected_el = next(
            (el for el in doc.elements if _contains(el.text, insertion_point)),
            None,
        )

    if expected_el is None:
        _fail("insertion_point not found in any element — fixture may be wrong or text is pre-processed away")
        return "not_found"

    # If the full_context search missed but the element fallback succeeded,
    # still check whether that element happens to be a candidate.
    if expected_candidate is None:
        expected_candidate = next(
            (c for c in candidates if c.element_id == expected_el.id),
            None,
        )

    print(f"         expected element: idx={doc.elements.index(expected_el)}  page={expected_el.page}  type={type(expected_el).__name__}")

    if expected_candidate is None:
        _fail("Expected element is NOT a candidate")
        # Explain why
        if same_page_only and expected_el.page != footnote.page:
            _info(f"page mismatch: element page={expected_el.page}, footnote page={footnote.page}  (try --no-same-page)")
        if isinstance(expected_el, Footnote):
            _info("element is a Footnote — footnote elements are skipped to prevent self-reference")
        else:
            pat = build_footnote_pattern(footnote.number)
            if not pat.search(expected_el.text):
                _info(f"footnote number {footnote.number!r} not matched by build_footnote_pattern in element text")
                _info(f"element.text : {expected_el.text}")
            else:
                _info("number found in text but still not a candidate — investigate further")
        _info(f"element.html : {expected_el.html}")
        return "not_candidate"

    if not candidates:
        _fail("No candidates found at all")
        return "no_candidates"

    # ------------------------------------------------------------------
    # 5. Run resolver — did it pick the expected candidate?
    # ------------------------------------------------------------------
    resolver = SimpleFootnoteResolver()
    best = await resolver.resolve(
        candidates=candidates,
        footnote_number=footnote.number,
        footnote_text=footnote.text,
    )

    if best is not None and best.element_id == expected_el.id:
        _ok(f"Resolver picked correct element ({len(candidates)} candidate(s) total)")
        return "correct"

    # Wrong pick — show all candidates with scores
    scored = score_footnote_candidates(candidates)
    picked_idx = candidates.index(best) if best is not None else -1
    expected_idx = candidates.index(expected_candidate)

    _fail(
        f"Resolver picked wrong candidate  "
        f"(picked=[{picked_idx}], correct=[{expected_idx}], total={len(candidates)})"
    )
    print(f"         {'idx':>3}  {'score':>5}  {'page':>4}  {'label':<10}  context")
    for i, (c, score) in enumerate(scored):
        label = ""
        if i == picked_idx:
            label += f"{RED}PICKED{RESET}"
        if i == expected_idx:
            label += f"{GREEN}CORRECT{RESET}"
        snippet = c.full_context.replace("\n", " ")[:70]
        print(f"         [{i:>2}]  {score:>5}  p{c.page:<3}  {label:<10}  ...{snippet}...")

    if best is not None:
        picked_el = el_by_id.get(best.element_id)
        if picked_el is not None:
            print(f"\n         {RED}PICKED element HTML:{RESET}")
            print(f"         {picked_el.html}")
    print(f"\n         {GREEN}CORRECT element HTML:{RESET}")
    print(f"         {expected_el.html}")

    return "wrong_pick"


async def diagnose_document(
    name: str,
    test_case: dict,
    same_page_only: bool,
) -> dict:
    _h1(f"Document: {name}")

    middle_path = MINERU_DATA_DIR / f"{name}_middle.json"
    if not middle_path.exists():
        _warn(f"Middle JSON not found: {middle_path}")
        return {"name": name, "skipped": True}

    with open(middle_path, encoding="utf-8") as f:
        middle = MinerUMiddleDocument.model_validate(json.load(f))

    parser = MinerUParser(use_default_stages=False)
    parser.use(CoreExtractor())
    doc = await parser.parse(middle)

    print(f"  Parsed: {len(doc.elements)} elements, {len(doc.footnotes)} footnote(s)")

    footnotes = test_case.get("footnotes", [])
    if not footnotes:
        _warn("No footnotes defined in test_cases.json for this document")
        return {"name": name, "skipped": True}

    results = []
    for i, fn in enumerate(footnotes):
        insertion_point = fn.get("insertion_point", "")
        if not insertion_point or len(insertion_point.strip()) < 8:
            _warn(f"fn{i}: no meaningful insertion_point — skipping")
            continue
        try:
            status = await diagnose_footnote(
                doc=copy.deepcopy(doc),
                fn_index=i,
                fn_text=fn["text"],
                insertion_point=insertion_point,
                same_page_only=same_page_only,
            )
        except Exception as exc:
            _fail(f"fn{i}: unexpected error — {exc}")
            status = "error"
        results.append({"fn_index": i, "status": status})

    return {"name": name, "results": results}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(all_results: list[dict]) -> None:
    _h1("SUMMARY")
    counts: dict[str, int] = {}
    for doc_result in all_results:
        if doc_result.get("skipped"):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        for r in doc_result.get("results", []):
            s = r["status"]
            counts[s] = counts.get(s, 0) + 1

    for status, count in sorted(counts.items()):
        color = GREEN if status == "correct" else RED if status in ("wrong_pick", "not_candidate", "no_candidates") else YELLOW
        print(f"  {color}{status:20s}{RESET}: {count}")

    correct = counts.get("correct", 0)
    wrong = sum(counts.get(s, 0) for s in ("wrong_pick", "not_candidate", "no_candidates", "not_found"))
    if correct + wrong > 0:
        pct = 100 * correct / (correct + wrong)
        print(f"\n  Accuracy: {correct}/{correct + wrong}  ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose footnote resolution")
    parser.add_argument("--doc", help="Only diagnose documents whose name contains this string", default=None)
    parser.add_argument("--no-same-page", action="store_true", help="Disable same_page_only restriction")
    args = parser.parse_args()

    if not TEST_CASES_FILE.exists():
        print(f"ERROR: test_cases.json not found at {TEST_CASES_FILE}")
        sys.exit(1)

    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        test_cases: dict = json.load(f)

    selected = {
        name: tc
        for name, tc in test_cases.items()
        if args.doc is None or args.doc.lower() in name.lower()
    }
    if not selected:
        print(f"No documents matched --doc={args.doc!r}")
        sys.exit(1)

    same_page_only = not args.no_same_page
    all_results = []
    for name, tc in selected.items():
        result = await diagnose_document(name, tc, same_page_only=same_page_only)
        all_results.append(result)

    _print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
