"""
Footnote processing.

This module provides processors for resolving footnote references in documents.
It finds potential footnote references in text, matches them with footnote
definitions, and creates InlineRef relationships.

Classes:
    FootnoteCandidate: A candidate location for a footnote reference
    FootnoteResolver: Protocol for resolver injection
    SimpleFootnoteResolver: Heuristic-based resolver (sync)
    LLMFootnoteResolver: LLM-based resolver (async)
    FootnoteProcessor: Main processor for footnote resolution

Functions:
    build_footnote_pattern: Build a regex for a footnote reference number
    find_footnote_candidates: Find candidate locations for a footnote reference
    score_footnote_candidates: Score candidates by structural/positional signals
    apply_ref_patches: Apply collected HTML patches to an element (two-phase design)

Single-reference assumption
----------------------------
This processor assumes that each footnote is referenced **exactly once** in the
document text.  The resolver selects a single best :class:`FootnoteCandidate`
and the processor inserts exactly one ``<ref rel="footnote"/>`` tag per footnote.
Documents where the same footnote number is cited multiple times in the body are
not supported — only the first (highest-scoring) occurrence will be linked.

Two-phase HTML update design
-----------------------------
Naively replacing footnote numbers in element.html one at a time causes two
hard-to-fix problems:

1. **Sequential mutation** — after replacing footnote 1's "1" with a ``<ref>``
   tag, the stored byte offsets for footnote 2's match are stale (the HTML
   string shifted).  In a run like ``Agreement1).2`` the "1" and "2" sit two
   characters apart, so mutating for footnote 1 first would shift the "2"
   offset by ~40 characters (length of the inserted ``<sup><ref …/>`` tag).

2. **Attribute contamination** — a simple ``re.sub(r"\\b1\\b", ref, html)``
   also matches "1" inside HTML attribute values like ``id="footnote-1"``.

The solution implemented here separates *collection* from *application*:

- **Phase 1 (collect)**: For every footnote, run ``find_footnote_candidates``
  over HTMLParser text nodes (which never include attribute values) and
  record ``(text_node_idx, start, end)`` within the matched text node, plus
  the ``ref_html`` to insert.  Do this for **all** footnotes before touching
  the HTML.

- **Phase 2 (apply)**: Call ``apply_ref_patches`` once per element.  It
  locates each text node's absolute position in the raw HTML string using a
  lightweight ``HTMLParser`` pass (``_HtmlTextNodeLocator``), converts node-
  relative offsets to absolute offsets, then applies all patches **right-to-
  left** (descending ``start``).  Applying right-to-left means each earlier
  patch's offset is unaffected by later insertions that shifted characters to
  its right.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ragdoc.document import Footnote
from ragdoc.processing._concurrency import _fan_out
from ragdoc.processing.base import DocumentProcessor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.document import BaseElement, Document


# =============================================================================
# FootnoteCandidate Model
# =============================================================================


class FootnoteCandidate(BaseModel):
    """
    A candidate location for a footnote reference in the document.

    When scanning for footnote references, multiple potential matches may be
    found (e.g., the number "1" appears multiple times). FootnoteCandidate
    captures each potential match with its context for resolution.

    Attributes:
        id: Unique identifier for this candidate
        element_id: ID of the element containing the potential reference
        element_idx: Index of the element in the document's elements list
        text_node_idx: Index of the HTMLParser text node within element.html
            (used by apply_ref_patches to locate the match in the raw HTML)
        page: Page number where the candidate appears
        match_start: Char offset of the reference start within the text node
        match_end: Char offset of the reference end within the text node
        context_before: Text context before the reference (for analysis)
        context_after: Text context after the reference (for analysis)
        full_context: Combined context (before + reference + after)
        reference_number: The footnote number being referenced
        footnote_text: The full text of the footnote definition
        footnote_id: The ID of the Footnote element being referenced
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this candidate",
    )
    element_id: str = Field(
        ...,
        description="ID of the element containing the potential reference",
    )
    element_idx: int = Field(
        ...,
        description="Index of the element in the document's elements list",
    )
    text_node_idx: int = Field(
        default=0,
        description="Index of the BS4 text node within element.html where the match was found",
    )
    page: int = Field(
        ...,
        description="Page number where the candidate appears (1-indexed)",
    )
    match_start: int = Field(
        ...,
        description="Char offset where the reference starts within the text node",
    )
    match_end: int = Field(
        ...,
        description="Char offset where the reference ends within the text node",
    )
    context_before: str = Field(
        ...,
        description="Text context before the reference (typically ~50 chars)",
    )
    context_after: str = Field(
        ...,
        description="Text context after the reference (typically ~50 chars)",
    )
    full_context: str = Field(
        ...,
        description="Combined context: context_before + reference + context_after",
    )
    reference_number: int = Field(
        ...,
        description="The footnote number (e.g., 1, 2, 3) being referenced",
    )
    footnote_text: str = Field(
        ...,
        description="The full text content of the footnote definition",
    )
    footnote_id: str = Field(
        ...,
        description="The ID of the Footnote element being referenced",
    )


# =============================================================================
# FootnoteResolver Protocol
# =============================================================================


@runtime_checkable
class FootnoteResolver(Protocol):
    """
    Protocol for resolving which candidate is the correct footnote reference.
    
    Implementations select the best candidate from a list based on various
    criteria (heuristics, LLM analysis, etc.).
    
    This protocol supports both sync and async implementations:
    - SimpleFootnoteResolver: Sync, heuristic-based
    - LLMFootnoteResolver: Async, LLM-based
    
    The resolve method is async to support LLM calls, but sync implementations
    can simply return an awaitable directly.
    """
    
    async def resolve(
        self,
        candidates: list[FootnoteCandidate],
        footnote_number: int,
        footnote_text: str,
        min_element_idx: int | None = None,
    ) -> FootnoteCandidate | None:
        """
        Select the best candidate for a footnote reference.

        .. note::
            Implementations must select **at most one** candidate.  The
            processor inserts a single ``<ref>`` tag per footnote; returning
            multiple candidates is not supported (see *Single-reference
            assumption* in the module docstring).

        Args:
            candidates: List of candidate locations found in the document
            footnote_number: The footnote number being resolved (e.g., 1, 2, 3)
            footnote_text: The footnote definition text
            min_element_idx: When set, candidates with element_idx below this
                value receive a -10 penalty (sequential ordering constraint).

        Returns:
            The best candidate, or None if no good match found
        """
        ...


# =============================================================================
# Standalone Functions
# =============================================================================


def build_footnote_pattern(number: int) -> re.Pattern[str]:
    """
    Build a regex pattern that matches a footnote reference number in OCR'd text.

    The pattern is intentionally broader than a simple word boundary (``\\b``) to
    handle common OCR artifacts:

    - **Number glued to a word** (OCR missed the superscript space): ``word6``
      matches footnote 6 because the preceding letter is not a digit.
    - **Number after punctuation without space**: ``,6`` or ``.6`` — already
      handled by ``\\b``, but also covered here.
    - **Multi-digit number split by OCR whitespace**: ``1 1`` matches footnote 11
      because digits are joined by ``\\s*`` in the pattern.

    **Digit-preceded matches**: if the footnote number is immediately preceded by
    another digit (e.g. ``January 20046`` where ``6`` is footnote 6 glued to the
    year ``2004`` by OCR), the match is still returned but will receive a
    ``-2`` penalty in :func:`score_footnote_candidates`.  This is preferable to
    silently suppressing the match, which would leave the reference unresolved.

    Args:
        number: The footnote number to build a pattern for.

    Returns:
        Compiled regex that finds the number in OCR-flattened text.
    """
    # Join each digit with \s* to handle OCR spacing (e.g. "1 1" for 11).
    digits_pattern = r"\s*".join(re.escape(d) for d in str(number))
    return re.compile(rf"{digits_pattern}(?!\d)")


def find_footnote_candidates(
    document: "Document",
    footnote: Footnote,
    context_chars: int = 50,
    same_page_only: bool = True,
) -> list[FootnoteCandidate]:
    """
    Find candidate locations for a footnote reference in the document.

    Searches every non-Footnote element's HTML text nodes for occurrences of
    the footnote number.  Text nodes are discovered via :class:`_HtmlTextNodeLocator`
    (HTMLParser) so that attribute values (e.g. ``id="footnote-1"``) are never
    matched — only visible text content is considered.

    Each match records ``text_node_idx`` (the HTMLParser text node index inside
    the element's HTML) together with ``match_start`` / ``match_end`` as offsets
    within that text node's raw string.  Using the same locator as
    :func:`apply_ref_patches` ensures the indices are always consistent.

    .. note::
        Offset arithmetic assumes footnote reference numbers (pure ASCII
        digits) are never HTML-entity-encoded.  In practice this holds for
        all supported document types.

    Args:
        document: The document to search
        footnote: The footnote definition to find references for
        context_chars: Number of characters before/after reference for context
        same_page_only: If True, only search on the same page as the footnote

    Returns:
        List of FootnoteCandidate objects representing potential reference locations
    """
    candidates: list[FootnoteCandidate] = []
    pattern = build_footnote_pattern(footnote.number)

    for idx, element in enumerate(document.elements):
        # Skip footnote definitions — a footnote must not reference itself.
        if isinstance(element, Footnote):
            continue

        # Optionally filter by page
        if same_page_only and element.page != footnote.page:
            continue

        locator = _HtmlTextNodeLocator(element.html)

        for node_idx, (node_start, node_end) in enumerate(locator.spans):
            # Skip text nodes inside existing <ref> tags.  In practice <ref/>
            # is always self-closing so this is defensive, but it mirrors the
            # BS4 parent-check that was here before.
            if locator._in_ref[node_idx]:
                continue

            node_text = element.html[node_start:node_end]

            for match in pattern.finditer(node_text):
                start = match.start()
                end = match.end()

                ctx_start = max(0, start - context_chars)
                ctx_end = min(len(node_text), end + context_chars)

                context_before = node_text[ctx_start:start]
                context_after = node_text[end:ctx_end]

                candidates.append(
                    FootnoteCandidate(
                        element_id=element.id,
                        element_idx=idx,
                        text_node_idx=node_idx,
                        page=element.page or 0,
                        match_start=start,
                        match_end=end,
                        context_before=context_before,
                        context_after=context_after,
                        full_context=context_before + node_text[start:end] + context_after,
                        reference_number=footnote.number,
                        footnote_text=footnote.text,
                        footnote_id=footnote.id,
                    )
                )

    return candidates


# =============================================================================
# Scoring
# =============================================================================


def score_footnote_candidates(
    candidates: list[FootnoteCandidate],
    min_element_idx: int | None = None,
) -> list[tuple[FootnoteCandidate, int]]:
    """
    Score each candidate by structural and positional signals.

    Footnote reference numbers are characterized by their immediate character
    context, not by topical overlap with the footnote content.  The signals
    used (in descending weight) are:

    **Negatives**

    - ``-10`` — candidate's ``element_idx`` is below ``min_element_idx`` (when
      provided): violates sequential footnote ordering.  Footnote references
      must appear in ascending document order; any candidate that precedes the
      element chosen for the previous footnote almost certainly refers to an
      earlier numeral (e.g., a paragraph number) rather than this footnote.
    - ``-5`` — number at the very start of the text node (empty / whitespace-only
      ``context_before``): almost certainly a paragraph or section number.
    - ``-4`` — ``context_before`` consists entirely of digits (non-empty) **and**
      ``context_after`` starts with ``". "`` or ``".) "`` followed by an
      uppercase letter: the matched number is the trailing digit of a
      paragraph/section number (e.g. ``"1" + "6" + ". The Agency…"`` → ``"16."``).
      This is distinct from a year+footnote OCR merge (``"20046"``), where
      ``context_before`` contains letters and spaces.
    - ``-3`` — number followed by ``". "`` / ``") "`` / ``": "`` + non-whitespace:
      numbered list or paragraph item pattern (e.g. ``"2. On 18 September"``).
      *Not applied when the number is preceded by a digit* — in that context the
      trailing period is a sentence-end boundary, not a list-item marker
      (e.g. ``"January 20046. Taking"``).
    - ``-3`` — number immediately preceded by a single uppercase letter with
      optional whitespace (``r'(?<!\\w)[A-Z]\\s*$'``): chemical-formula digit
      such as ``"H 2 O"`` or ``"H2O"``.
    - ``-2`` — number immediately preceded by a digit: OCR artefact where a
      superscript footnote number was merged into an adjacent numeral
      (e.g. ``"20046"`` for year ``2004`` + footnote ``6``).  Still a valid
      candidate but ranked lower.

    **Positives**

    - ``+4`` — number immediately preceded by closing punctuation
      (``.``, ``,``, ``;``, ``:``, ``)``, ``"``): end-of-clause superscript
      pattern (e.g. ``"received.5"`` or ``"Agreement1).2"``).
    - ``+2`` — number immediately preceded by a lowercase letter: digit embedded
      inside a word (e.g. ``"reiterated4"``).
    - ``+1`` — ``len(context_after) < len(context_before)``: match is closer to
      the end of the text node, where superscripts typically appear.

    Args:
        candidates: Candidates to score.
        min_element_idx: When set, any candidate whose ``element_idx`` is
            strictly less than this value receives an additional ``-10`` penalty
            (sequential ordering constraint).  Pass the ``element_idx`` of the
            element chosen for the previous footnote.

    Returns a list of ``(candidate, score)`` pairs in the same order as the
    input.  Higher score → more likely a genuine inline footnote reference.
    """
    scored = []
    for c in candidates:
        score = 0
        before = c.context_before
        after = c.context_after

        # Sequential ordering: candidate precedes the previously resolved element
        if min_element_idx is not None and c.element_idx < min_element_idx:
            score -= 10

        # Strong negative: paragraph / section number at start of text node
        if not before.strip():
            score -= 5

        # Negative: trailing digit of a paragraph/section number.
        # Fires when context_before is entirely digits (e.g. "1" in "16.") AND
        # context_after starts with a period-space-uppercase sequence (e.g.
        # ". The Agency…").  Distinguishes "16. Sentence" from "20046. Taking":
        # in the year+footnote case context_before contains letters/spaces and
        # therefore fails the isdigit() check.
        if before.isdigit() and re.match(r"^[.]\s+[A-Z]", after):
            score -= 4

        # Negative: numbered list / paragraph item  ("2. On 18 Sep…", "6) Item…")
        # Not applied when preceded by a digit: the trailing period is then a
        # sentence-end boundary, not a list-item marker ("January 20046. Taking").
        if re.match(r"^[.)]\s+\S", after) and not (before and before[-1].isdigit()):
            score -= 3

        # Negative: digit-preceded match — OCR artifact where a superscript was
        # merged into an adjacent numeral (e.g. "20046" for year 2004 + footnote 6).
        if before and before[-1].isdigit():
            score -= 2

        # Negative: chemical-formula digit ("H 2 O", "H2O", "C O 2")
        if re.search(r"(?<!\w)[A-Z]\s*$", before):
            score -= 3

        # Strong positive: end-of-clause punctuation before number
        if before and before[-1] in ".,;:)\"'":
            score += 4

        # Positive: lowercase letter directly before number (inline superscript)
        if before and before[-1].islower():
            score += 2

        # Tiebreak: superscripts tend to appear toward the end of a text node
        if len(after) < len(before):
            score += 1

        scored.append((c, score))
    return scored


# =============================================================================
# HTML patch application (two-phase design)
# =============================================================================


class _HtmlTextNodeLocator(HTMLParser):
    """
    Lightweight HTMLParser pass that records the absolute byte span of every
    text node in the raw HTML string.

    ``handle_data`` is called once for each contiguous run of text between
    tags.  ``getpos()`` returns the ``(line, col)`` of the current token's
    start in the source, which we convert to an absolute offset using a
    precomputed line-start table.

    The parser is created with ``convert_charrefs=False`` so that HTML
    entities are left as-is (e.g. ``&amp;`` stays ``&amp;``).  This means
    ``spans[i]`` gives the exact byte range of text node *i* in the original
    HTML string, making it safe to do direct string surgery on that string.

    Note: with ``convert_charrefs=False``, character references (e.g.
    ``&#x00025;``) are handled by ``handle_charref``/``handle_entityref``,
    not ``handle_data``.  This means entity-only text nodes (e.g. a ``<mi>``
    containing only ``&#x00025;``) produce no span here.  ``find_footnote_candidates``
    uses this same locator so that ``text_node_idx`` values are always consistent.

    The ``_tag_stack`` and ``_in_ref`` attributes track the current ancestor
    tags so that callers can skip text nodes inside ``<ref>`` tags.

    Limitation: assumes that text node content never contains raw ``<``
    characters (which would be ``&lt;`` in valid HTML) and that HTML entities
    are not used around pure-ASCII digit sequences (footnote numbers).  Both
    hold for all document types supported by this library.
    """

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self._line_offsets: list[int] = [0]
        for i, ch in enumerate(html):
            if ch == "\n":
                self._line_offsets.append(i + 1)
        self.spans: list[tuple[int, int]] = []  # (start, end) in html
        self._tag_stack: list[str] = []
        self._in_ref: list[bool] = []  # parallel to spans; True if inside a <ref> tag
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[i] == tag:
                del self._tag_stack[i]
                break

    def handle_data(self, data: str) -> None:
        line, col = self.getpos()
        start = self._line_offsets[line - 1] + col
        self.spans.append((start, start + len(data)))
        self._in_ref.append("ref" in self._tag_stack)


def apply_ref_patches(
    element: "BaseElement",
    patches: list[tuple[int, int, int, str]],
) -> None:
    """
    Apply HTML insertion patches to an element, replacing footnote number
    occurrences with ``<ref>`` tags.

    This is the *apply* half of the two-phase footnote HTML update design
    (see module docstring).  It must be called **after all footnotes have
    been resolved** so that every patch for the element is known upfront.

    Args:
        element: The document element whose HTML will be updated in place.
        patches: List of ``(text_node_idx, start, end, ref_html)`` tuples,
            one per resolved footnote reference inside this element.  The
            offsets ``start``/``end`` are character positions within the
            text node (as returned by :func:`find_footnote_candidates`).

    Algorithm:
        1. Run ``_HtmlTextNodeLocator`` over ``element.html`` to obtain the
           absolute byte offset of each text node's start in the raw HTML
           string.
        2. Convert every ``(text_node_idx, start, end)`` to an absolute HTML
           offset pair ``(html_start, html_end)``.
        3. Sort all patches by ``html_start`` in **descending** order and
           apply them as plain string substitutions.  Right-to-left order
           guarantees that inserting ``ref_html`` at a later position does
           not shift the byte offsets of earlier (lower-offset) patches.
        4. Assign the modified string back via ``element.html`` (the
           universal setter defined on ``BaseElement``).

    Note:
        Direct string manipulation is intentional — it preserves the
        self-closing ``<ref id="..."/>`` format expected by the Renderer
        without routing the replacement HTML through BeautifulSoup's
        serializer (which would mangle unknown self-closing tags).
    """
    if not patches:
        return

    html = element.html
    node_spans = _HtmlTextNodeLocator(html).spans

    # Convert node-relative offsets → absolute HTML offsets
    abs_patches: list[tuple[int, int, str]] = []
    for node_idx, start, end, ref_html in patches:
        if node_idx >= len(node_spans):
            continue
        node_start, _ = node_spans[node_idx]
        abs_patches.append((node_start + start, node_start + end, ref_html))

    # Apply right-to-left so earlier offsets stay valid
    for html_start, html_end, ref_html in sorted(abs_patches, key=lambda x: -x[0]):
        html = html[:html_start] + ref_html + html[html_end:]

    element.html = html


# =============================================================================
# SimpleFootnoteResolver (Sync, Heuristic-based)
# =============================================================================


class SimpleFootnoteResolver:
    """
    Simple heuristic-based footnote resolver.

    Uses text matching and position heuristics without LLM.
    Good for testing, batch processing, or when LLM is not available.

    Resolution Strategy:
    1. If only one candidate exists, return it
    2. Score candidates by structural/positional signals (see score_footnote_candidates)
    3. Return the candidate with the highest score
    4. On a tie, return the highest-scoring candidate that appears earliest in document order

    Example:
        >>> resolver = SimpleFootnoteResolver()
        >>> best = await resolver.resolve(candidates, 1, "See Smith et al., 2023")
    """

    async def resolve(
        self,
        candidates: list[FootnoteCandidate],
        footnote_number: int,
        footnote_text: str,
        min_element_idx: int | None = None,
    ) -> FootnoteCandidate | None:
        """Select the best candidate using structural/positional heuristics."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        scored = score_footnote_candidates(candidates, min_element_idx=min_element_idx)
        return max(scored, key=lambda x: (x[1], -x[0].element_idx))[0]


# =============================================================================
# LLMFootnoteResolver (Async, LLM-based)
# =============================================================================


class LLMFootnoteResolver:
    """
    LLM-based footnote reference resolver.
    
    Uses an LLM to select the most likely location for a footnote reference
    from a list of candidates. This is more accurate than heuristics for
    complex documents where context matters.
    
    The resolver uses a system prompt explaining footnote patterns and asks
    the LLM to select the most appropriate candidate.
    
    Attributes:
        client: AsyncOpenAI-compatible client for LLM calls
        model: Model name to use (default: gpt-4o-mini)
    
    Example:
        >>> from openai import AsyncOpenAI
        >>> client = AsyncOpenAI()
        >>> resolver = LLMFootnoteResolver(client)
        >>> best = await resolver.resolve(candidates, 1, "See Smith et al., 2023")
    """
    
    SYSTEM_PROMPT = """You are analyzing document text to find footnote references.
Given a footnote number and its content, plus several candidate locations where 
the reference might appear, select the most appropriate location.

A footnote reference typically appears:
- At the end of a sentence or clause that relates to the footnote content
- Near terms, names, or concepts that the footnote explains or cites
- In a context where the footnote would add relevant supplementary information

Respond with only the candidate number (1, 2, 3, etc.) or "NONE" if no candidate is appropriate."""

    def __init__(
        self,
        client: Any | None = None,  # AsyncOpenAI-compatible client
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the LLM resolver.

        Args:
            client: AsyncOpenAI-compatible client for LLM calls.
                    If None, falls back to ``get_config().openai_client``.
            model: Model name to use for resolution
        """
        from ragdoc.config import get_config

        self.client = client if client is not None else get_config().openai_client
        self.model = model
    
    async def resolve(
        self,
        candidates: list[FootnoteCandidate],
        footnote_number: int,
        footnote_text: str,
        min_element_idx: int | None = None,
    ) -> FootnoteCandidate | None:
        """Use LLM to select the best candidate."""
        if not candidates:
            return None

        # Pre-filter candidates that violate sequential ordering — the LLM
        # should not be distracted by clearly wrong candidates.
        if min_element_idx is not None:
            preferred = [c for c in candidates if c.element_idx >= min_element_idx]
            if preferred:
                candidates = preferred

        if len(candidates) == 1:
            return candidates[0]

        # Build the prompt
        prompt_parts = [
            f"Footnote {footnote_number}: {footnote_text}",
            "",
            "Candidate locations:",
        ]
        
        for i, candidate in enumerate(candidates, 1):
            prompt_parts.append(
                f"{i}. Page {candidate.page}: ...{candidate.context_before}[{footnote_number}]{candidate.context_after}..."
            )
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            if result == "NONE":
                return None
            
            # Try to parse the selection
            try:
                selection = int(result.replace(".", "").strip())
                if 1 <= selection <= len(candidates):
                    return candidates[selection - 1]
            except ValueError:
                pass
            
            return None
            
        except Exception:
            # On error, return the first candidate as fallback
            return candidates[0] if candidates else None


# =============================================================================
# FootnoteProcessor (Main Processor)
# =============================================================================


class FootnoteProcessor(DocumentProcessor):
    """
    Processor that resolves footnote references in documents.

    Finds potential footnote references (numbers matching footnote definitions)
    in text, matches them with extracted footnotes using a resolver, and adds
    InlineRef relationships to the elements.

    The processor modifies elements in place by:
    1. Adding InlineRef entries for resolved footnote references
    2. Optionally updating HTML to include <ref id="..."/> tags

    .. note::
        Each footnote is resolved to **exactly one** reference location (see
        *Single-reference assumption* in the module docstring).

    This is an async processor because resolvers (especially LLM-based) may
    require async I/O.

    Attributes:
        resolver: The FootnoteResolver to use for selecting candidates
        context_chars: Number of context characters for candidate analysis
        same_page_only: Whether to restrict candidate search to same page
        update_html: Whether to update element HTML with ref tags
        only_orphaned: When True, skip footnotes that are already referenced
            in the document (i.e. already have a ``<ref>`` tag somewhere).
            Useful when re-running the processor or when the parser has already
            resolved some footnotes.

    Example:
        >>> resolver = SimpleFootnoteResolver()
        >>> processor = FootnoteProcessor(resolver=resolver)
        >>> doc = await processor.process(doc)
    """
    
    def __init__(
        self,
        resolver: FootnoteResolver | None = None,
        context_chars: int = 50,
        same_page_only: bool = True,
        update_html: bool = True,
        only_orphaned: bool = False,
        concurrency: int | asyncio.Semaphore = 1,
    ):
        """
        Initialize the footnote processor.

        Args:
            resolver: The resolver to use for selecting candidates.
                     If None, uses SimpleFootnoteResolver.
            context_chars: Number of characters before/after reference for context
            same_page_only: Whether to only search on the same page as the footnote
            update_html: Whether to update element HTML with <ref> tags
            only_orphaned: When True, skip footnotes that already have a ``<ref>``
                tag in any element.  Defaults to False (process all footnotes).
            concurrency: Maximum number of footnotes resolved concurrently.  Accepts
                an ``int`` (private semaphore) or a shared ``asyncio.Semaphore``.
                Defaults to ``1`` (sequential), which preserves the sequential
                ordering constraint used by the resolver (``min_element_idx``).
                Values ``> 1`` resolve footnotes in parallel without the ordering
                constraint; use when the resolver is LLM-based and API throughput
                is the bottleneck.
        """
        self.resolver = resolver or SimpleFootnoteResolver()
        self.context_chars = context_chars
        self.same_page_only = same_page_only
        self.update_html = update_html
        self.only_orphaned = only_orphaned
        self._concurrency = concurrency
    
    async def _resolve_footnote(
        self,
        document: "Document",
        footnote: Footnote,
        min_element_idx: int | None,
    ) -> "FootnoteCandidate | None":
        """Find candidates for *footnote* and resolve the best one."""
        candidates = find_footnote_candidates(
            document=document,
            footnote=footnote,
            context_chars=self.context_chars,
            same_page_only=self.same_page_only,
        )
        if not candidates:
            logger.debug(f"FootnoteProcessor: footnote {footnote.number}: no candidates found")
            return None
        logger.debug(
            f"FootnoteProcessor: footnote {footnote.number}: {len(candidates)} candidates found"
        )
        best = await self.resolver.resolve(
            candidates=candidates,
            footnote_number=footnote.number,
            footnote_text=footnote.text,
            min_element_idx=min_element_idx,
        )
        if best is not None:
            logger.debug(
                f"FootnoteProcessor: footnote {footnote.number} resolved to element {best.element_idx}"
            )
        return best

    async def process(self, document: "Document") -> "Document":
        """
        Process the document to resolve footnote references.

        Implements the two-phase collect/apply design (see module docstring):

        **Phase 1 — collect**: For each footnote, find candidates and resolve
        the best one.  Record the InlineRef and, if ``update_html`` is True,
        accumulate ``(text_node_idx, start, end, ref_html)`` patch tuples
        keyed by ``element_idx``.  No HTML is mutated yet.

        **Phase 2 — apply**: For each element that accumulated patches, call
        :func:`apply_ref_patches` once with all its patches.  Patches are
        applied right-to-left inside that function, so multiple footnote refs
        in the same element (e.g. ``Agreement1).2``) are both inserted
        correctly without offset drift.

        When ``concurrency=1`` (default), footnotes are resolved sequentially
        and the ``min_element_idx`` ordering constraint is enforced.  When
        ``concurrency > 1``, footnotes are resolved in parallel without the
        ordering constraint.

        Args:
            document: The document to process

        Returns:
            The processed document (modified in place)
        """
        if not document.footnotes:
            return document

        # element_idx → list of (text_node_idx, start, end, ref_html)
        patches_by_element: dict[int, list[tuple[int, int, int, str]]] = {}

        footnotes_to_process = (
            document.orphaned_footnotes if self.only_orphaned else document.footnotes
        )
        sorted_footnotes = sorted(footnotes_to_process, key=lambda f: f.number)
        logger.debug(f"FootnoteProcessor: {len(sorted_footnotes)} footnotes to resolve")
        resolved_count = 0

        is_sequential = isinstance(self._concurrency, int) and self._concurrency == 1

        if is_sequential:
            # --- Sequential path: ordering constraint via min_element_idx ---
            last_resolved_idx: int | None = None
            for footnote in sorted_footnotes:
                best = await self._resolve_footnote(document, footnote, last_resolved_idx)
                if best is None:
                    continue
                resolved_count += 1
                last_resolved_idx = best.element_idx
                if self.update_html:
                    ref_html = f'<ref id="{footnote.id}" rel="footnote"/>'
                    patches_by_element.setdefault(best.element_idx, []).append(
                        (best.text_node_idx, best.match_start, best.match_end, ref_html)
                    )
        else:
            # --- Parallel path: no ordering constraint ---
            # Pre-allocate results; index matches sorted_footnotes order.
            results: list[FootnoteCandidate | None] = [None] * len(sorted_footnotes)

            async def _resolve_one(idx: int, footnote: Footnote) -> None:
                results[idx] = await self._resolve_footnote(document, footnote, None)

            coros = [_resolve_one(i, f) for i, f in enumerate(sorted_footnotes)]
            await _fan_out(coros, self._concurrency)

            for footnote, best in zip(sorted_footnotes, results):
                if best is None:
                    continue
                resolved_count += 1
                if self.update_html:
                    ref_html = f'<ref id="{footnote.id}" rel="footnote"/>'
                    patches_by_element.setdefault(best.element_idx, []).append(
                        (best.text_node_idx, best.match_start, best.match_end, ref_html)
                    )

        logger.info(
            f"FootnoteProcessor: resolved {resolved_count}/{len(sorted_footnotes)} footnotes"
        )

        # --- Phase 2: apply HTML patches per element ---
        if self.update_html:
            for element_idx, patches in patches_by_element.items():
                apply_ref_patches(document.elements[element_idx], patches)

        return document

class SyncFootnoteProcessor(DocumentProcessor):
    """
    Footnote processor using :class:`SimpleFootnoteResolver` — no LLM required.

    .. note::
        Each footnote is resolved to **exactly one** reference location (see
        *Single-reference assumption* in the module docstring).

    Example:
        >>> processor = SyncFootnoteProcessor()
        >>> doc = await processor.process(doc)
    """

    def __init__(
        self,
        context_chars: int = 50,
        same_page_only: bool = True,
        update_html: bool = True,
        only_orphaned: bool = False,
    ):
        """
        Initialize the sync footnote processor.

        Args:
            context_chars: Number of characters before/after reference for context
            same_page_only: Whether to only search on the same page as the footnote
            update_html: Whether to update element HTML with <ref> tags
            only_orphaned: When True, skip footnotes that already have a ``<ref>``
                tag in any element.  Defaults to False (process all footnotes).
        """
        self.context_chars = context_chars
        self.same_page_only = same_page_only
        self.update_html = update_html
        self.only_orphaned = only_orphaned
    
    async def process(self, document: "Document") -> "Document":
        """
        Process the document to resolve footnote references.

        Uses SimpleFootnoteResolver internally (no I/O).

        Args:
            document: The document to process

        Returns:
            The processed document (modified in place)
        """
        import asyncio
        
        # Create an async processor with SimpleFootnoteResolver
        async_processor = FootnoteProcessor(
            resolver=SimpleFootnoteResolver(),
            context_chars=self.context_chars,
            same_page_only=self.same_page_only,
            update_html=self.update_html,
            only_orphaned=self.only_orphaned,
        )
        
        # Run in a new event loop (or existing if available)
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't use run_until_complete
            # This shouldn't happen for a SyncProcessor, but handle it
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(async_processor.process(document))
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(async_processor.process(document))
