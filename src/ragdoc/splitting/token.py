"""Token-budget splitting: three composable split functions.

Strategy hierarchy (outermost → innermost):
    split_document          — hierarchical by heading structure, recurses on each part
    split_by_elements       — greedy accumulation at group boundaries, ref-aware
    split_oversized_element — three-tier fallback for a single group > max_tokens

Ref-awareness
-------------
Before the greedy loop, elements are partitioned into atomic ElementGroups
(see :mod:`ragdoc.splitting.groups`).  A group = one root element + all
elements it transitively references via ``<ref id="..."/>`` tags.  The group
is the indivisible unit: referenced elements always travel with their root.
An element referenced by multiple roots is duplicated into all their groups.

Oversized-group three-tier fallback
------------------------------------
When a single group exceeds ``max_tokens`` even alone, three strategies are
tried in order:

1. **HTML structural split** (:func:`split_at_html_tags`) — the root element's
   HTML is cut at structural boundaries: table rows (also inside ``<tbody>``),
   list items, block children.  Every fragment is well-formed and keeps its
   wrapper element and element type (a table fragment is a ``Table`` whose
   html is a ``<table>`` with the header repeated; a list fragment is a
   ``DocumentList``), and inherits whichever referenced elements' ``<ref/>``
   tags it contains.  Fragments over budget are halved on unit boundaries;
   only a single row/item that is still too large moves on to tier 2.

2. **Sentence split** (:func:`split_at_sentences`) — renders the oversized
   piece (inline refs already resolved by the renderer) and splits the
   resulting plain text at sentence boundaries.  The sentence splitter is
   pluggable via ``SentenceSplitter = Callable[[str], list[str]]``.

3. **Token slice** — last resort.  Operates on rendered text (refs already
   inlined), slices the token sequence with overlap, and wraps each slice in
   a ``RawText`` element.  Because rendering happens before slicing, no raw
   ``<ref/>`` tags remain in the output.

Shared headings: when a split occurs, the most recently seen Heading at each
level is prepended to the new chunk as context, so every chunk is self-contained.

Reading order is preserved throughout.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from ragdoc.document import (
    BaseElement,
    Document,
    DocumentList,
    ElementType,
    ExternalRef,
    Heading,
    RawText,
    Table,
)
from ragdoc.metadata import BaseMetadata, copy_metadata
from ragdoc.rendering import Renderer
from ragdoc.splitting.groups import ElementGroup, build_element_groups
from ragdoc.splitting.hierarchical import split_hierarchical
from ragdoc.utils import GPTTokenizer, Tokenizer

logger = logging.getLogger(__name__)

_default_tokenizer: GPTTokenizer | None = None
DEFAULT_OVERLAP_TOKENS = 200
DEFAULT_MAX_TOKENS = 7_000
MIN_SLICE_TOKENS = 16
"""Smallest tier-3 window: below this a slice is emitted even if its rendered chunk is over budget."""


def _get_default_tokenizer() -> GPTTokenizer:
    """Memoized default tokenizer.

    tiktoken's BPE load (network on a cold cache) must not run at import time —
    the instance is created on first use and reused afterwards.
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = GPTTokenizer()
    return _default_tokenizer


# TODO: evaluate if split-helpers could operate on a Document
# so its possible for them to be used standalone.


def _overhead_tokens(doc: Document, renderer: Renderer, tokenizer: Tokenizer) -> int:
    """Token count of the rendered overhead (heading context + metadata/title).

    Rendering an empty Document through pypandoc produces ``"\\n"`` even when
    there is no real content, which would cause an off-by-one when slicing
    content tokens.  We guard against this by returning 0 when the rendered
    overhead is entirely whitespace.
    """
    rendered = renderer.render(doc)
    return tokenizer.count(rendered) if rendered.strip() else 0


SentenceSplitter = Callable[[str], list[str]]
"""Callable that splits a plain-text string into sentence strings."""


def _default_sentence_splitter(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# Tier-2 helper: sentence split on rendered content
# ---------------------------------------------------------------------------


def split_at_sentences(
    elements: list[BaseElement],
    heading_ctx: list[Heading],
    renderer: Renderer,
    tokenizer: Tokenizer,
    max_tokens: int,
    doc_title: str | None,
    doc_metadata: BaseMetadata,
    parent_ref: ExternalRef,
    sentence_splitter: SentenceSplitter = _default_sentence_splitter,
    doc_source_path: str = "",
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Document] | None:
    """Split a list of elements at sentence boundaries in the rendered output.

    Renders ``heading_ctx + elements`` to plain text (inline refs are resolved
    by the renderer), measures overhead (headings only), then accumulates
    sentences into chunks up to ``max_tokens``.

    Returns ``None`` when no sentence boundaries are found (caller falls back
    to token-slice tier 3).  Returns ``[original_doc]`` unchanged when the
    whole thing already fits.

    Args:
        elements: Content elements (root + referenced, no headings).
        heading_ctx: Heading context to prepend to every output chunk.
        renderer: Renderer used to convert elements to text.
        tokenizer: Tokenizer for counting and splitting tokens.
        max_tokens: Maximum tokens per output chunk.
        doc_title: Title propagated to each output chunk.
        doc_metadata: Metadata propagated to each output chunk.
        parent_ref: ExternalRef added to every output chunk.
        sentence_splitter: Pluggable sentence boundary detector.
        doc_source_path: Source path propagated to each output chunk.
        overlap_tokens: Overlap used when an over-budget sentence chunk is token-sliced.

    Returns:
        List of Documents or ``None`` if no split was possible.
    """
    full_elements = cast("list[ElementType]", heading_ctx + elements)
    full_doc = Document(elements=full_elements, title=doc_title, metadata=doc_metadata)
    rendered = renderer.render(full_doc)
    if tokenizer.count(rendered) <= max_tokens:
        return [
            Document(
                elements=full_elements,
                title=doc_title,
                source_path=doc_source_path,
                metadata=copy_metadata(doc_metadata),
                external_refs=[parent_ref],
            )
        ]

    overhead_tokens = _overhead_tokens(
        Document(elements=cast("list[ElementType]", heading_ctx), title=doc_title, metadata=doc_metadata),
        renderer,
        tokenizer,
    )
    content_budget = max_tokens - overhead_tokens

    sentences = sentence_splitter(rendered)
    if len(sentences) <= 1:
        return None  # no boundaries found — caller tries tier 3

    def make_chunk(text: str) -> Document:
        return Document(
            elements=[*heading_ctx, RawText(html=text)],
            title=doc_title,
            source_path=doc_source_path,
            metadata=copy_metadata(doc_metadata),
            external_refs=[parent_ref],
        )

    result: list[Document] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = tokenizer.count(sentence)
        if current_sentences and current_tokens + s_tokens > content_budget:
            result.append(make_chunk(" ".join(current_sentences)))
            current_sentences = [sentence]
            current_tokens = s_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += s_tokens

    if current_sentences:
        result.append(make_chunk(" ".join(current_sentences)))

    if len(result) <= 1:
        return None

    # Sentence chunks are sized by the sentences' own token counts, but what callers measure is
    # the chunk *rendered again* (and a single sentence may exceed the budget outright). Any
    # chunk that does not fit is handed to the token slice, which verifies rendered size.
    # (The token slice needs more than 100 tokens of room; below that the chunk is kept as is.)
    sliceable = content_budget > 100
    checked: list[Document] = []
    for chunk in result:
        if not sliceable or tokenizer.count(renderer.render(chunk)) <= max_tokens:
            checked.append(chunk)
        else:
            checked.extend(
                _token_slice(
                    chunk_doc=chunk,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    overlap_tokens=min(overlap_tokens, max(0, content_budget - 101)),
                    overhead_tokens=overhead_tokens,
                    heading_ctx=heading_ctx,
                    doc_title=doc_title,
                    doc_metadata=doc_metadata,
                    parent_ref=parent_ref,
                    doc_source_path=doc_source_path,
                )
            )
    return checked


# ---------------------------------------------------------------------------
# Tier-1 helper: HTML structural split
# ---------------------------------------------------------------------------

_TABLE_HEADER_TAGS = frozenset({"caption", "colgroup", "col", "thead"})
"""Direct ``<table>`` children repeated at the top of every table fragment."""

_TABLE_SECTION_TAGS = frozenset({"tbody", "tfoot"})
"""Direct ``<table>`` children whose rows are the split units (the section tag is re-emitted per fragment)."""

_LIST_TAGS = frozenset({"ul", "ol", "dl"})


@dataclass(frozen=True)
class HtmlSplitUnit:
    """One indivisible piece of a structural split: a table row, a list item, a block child.

    ``section_index`` groups units that share a re-emitted wrapper (``<tbody>``/``<tfoot>``);
    ``-1`` marks a direct child of the outer element.
    """

    html: str
    section_index: int = -1
    section_open: str = ""
    section_close: str = ""


@dataclass(frozen=True)
class HtmlSplitPlan:
    """How to cut one element's HTML into well-formed fragments (see :func:`plan_html_split`)."""

    outer_name: str
    open_tag: str
    close_tag: str
    header_html: str
    units: list[HtmlSplitUnit]
    ordered_start: int | None = None


def _open_and_close_tag(tag: Tag) -> tuple[str, str]:
    """Serialize *tag*'s opening tag (attributes preserved and escaped by bs4) and its closing tag."""
    shell = BeautifulSoup("", "html.parser").new_tag(tag.name)
    shell.attrs = dict(tag.attrs)
    close_tag = f"</{tag.name}>"
    return str(shell).removesuffix(close_tag), close_tag


def _has_bare_text(tag: Tag) -> bool:
    """True when *tag* has a direct non-whitespace text child (splitting at child tags would drop it)."""
    return any(isinstance(child, NavigableString) and child.strip() for child in tag.children)


def plan_html_split(element_html: str) -> HtmlSplitPlan | None:
    """Work out where an element's HTML can be cut into well-formed fragments.

    The outer tag (with its attributes) wraps every fragment. Split units by outer tag:

    * ``<table>`` — the rows (direct ``<tr>`` children and the rows of every
      ``<tbody>``/``<tfoot>``, whose section tag is re-emitted around its rows).
      ``<caption>``/``<colgroup>``/``<col>``/``<thead>`` form the header, repeated in every
      fragment so each one is a self-describing table.
    * ``<ul>`` / ``<ol>`` — the direct ``<li>`` children (``<ol>`` fragments continue the
      numbering via ``start``, unless the list is ``reversed``).
    * ``<dl>`` — each ``<dt>`` group together with its ``<dd>`` definitions.
    * anything else — the direct child tags.

    Returns ``None`` when there is nothing to split on: no outer tag, fewer than two units, or
    non-whitespace text sitting directly inside the outer tag (cutting at the child tags would
    silently drop that text — e.g. a paragraph of prose with inline ``<em>`` tags).

    Args:
        element_html: Full HTML of one element, including its outer tag.

    Returns:
        A plan for :func:`build_fragment_html`, or ``None``.
    """
    outer = BeautifulSoup(element_html, "html.parser").find()
    if not isinstance(outer, Tag) or _has_bare_text(outer):
        return None

    children = [child for child in outer.children if isinstance(child, Tag)]
    header_parts: list[str] = []
    units: list[HtmlSplitUnit] = []

    if outer.name == "table":
        section_index = 0
        for child in children:
            if child.name in _TABLE_HEADER_TAGS:
                header_parts.append(str(child))
            elif child.name in _TABLE_SECTION_TAGS:
                section_open, section_close = _open_and_close_tag(child)
                units.extend(
                    HtmlSplitUnit(str(row), section_index, section_open, section_close)
                    for row in child.children
                    if isinstance(row, Tag)
                )
                section_index += 1
            else:
                units.append(HtmlSplitUnit(str(child)))
    elif outer.name == "dl":
        current: list[Tag] = []
        for child in children:
            if child.name == "dt" and any(member.name != "dt" for member in current):
                units.append(HtmlSplitUnit("".join(str(member) for member in current)))
                current = []
            current.append(child)
        if current:
            units.append(HtmlSplitUnit("".join(str(member) for member in current)))
    else:
        units = [HtmlSplitUnit(str(child)) for child in children]

    if len(units) <= 1:
        return None

    ordered_start: int | None = None
    if outer.name == "ol" and not outer.has_attr("reversed"):
        try:
            ordered_start = int(str(outer.get("start", "1")))
        except ValueError:
            ordered_start = 1

    open_tag, close_tag = _open_and_close_tag(outer)
    return HtmlSplitPlan(
        outer_name=outer.name,
        open_tag=open_tag,
        close_tag=close_tag,
        header_html="".join(header_parts),
        units=units,
        ordered_start=ordered_start,
    )


def build_fragment_html(plan: HtmlSplitPlan, start: int, end: int) -> str:
    """Build the well-formed HTML fragment holding ``plan.units[start:end]``.

    The fragment is wrapped in the original outer tag, starts with the plan's repeated header,
    and re-wraps consecutive units of the same table section in their ``<tbody>``/``<tfoot>``.

    Args:
        plan: Plan from :func:`plan_html_split`.
        start: Index of the first unit (inclusive).
        end: Index after the last unit (exclusive).

    Returns:
        HTML with exactly one root element — the same tag as the element being split.
    """
    open_tag = plan.open_tag
    if plan.ordered_start is not None and start > 0:
        shell = BeautifulSoup(plan.open_tag + plan.close_tag, "html.parser").find()
        if isinstance(shell, Tag):
            shell["start"] = str(plan.ordered_start + start)
            open_tag = str(shell).removesuffix(plan.close_tag)

    parts: list[str] = [open_tag, plan.header_html]
    open_section: HtmlSplitUnit | None = None
    for unit in plan.units[start:end]:
        if open_section is not None and unit.section_index != open_section.section_index:
            parts.append(open_section.section_close)
            open_section = None
        if open_section is None and unit.section_index >= 0:
            parts.append(unit.section_open)
            open_section = unit
        parts.append(unit.html)
    if open_section is not None:
        parts.append(open_section.section_close)
    parts.append(plan.close_tag)
    return "".join(parts)


def build_fragment_element(root: BaseElement, outer_name: str, fragment_html: str) -> BaseElement:
    """Wrap a structural fragment in an element of the right type.

    A fragment of a :class:`Table` whose outer tag is ``<table>`` stays a ``Table``; a fragment
    of a :class:`DocumentList` (``<ul>``/``<ol>``/``<dl>``) stays a ``DocumentList``. Anything
    else becomes a :class:`RawText` (which wraps the fragment in its ``<div>``). ``page``,
    ``bounding_box`` and a copy of ``metadata`` are inherited from *root*.
    """
    metadata = dict(root.metadata)
    if isinstance(root, Table) and outer_name == "table":
        return Table(html=fragment_html, page=root.page, bounding_box=root.bounding_box, metadata=metadata)
    if isinstance(root, DocumentList) and outer_name in _LIST_TAGS:
        return DocumentList(html=fragment_html, page=root.page, bounding_box=root.bounding_box, metadata=metadata)
    return RawText(html=fragment_html, page=root.page, bounding_box=root.bounding_box, metadata=metadata)


def collect_ref_ids(html_str: str) -> list[str]:
    """IDs of all ``<ref id="..."/>`` tags in *html_str*, in document order, de-duplicated."""
    ids: dict[str, None] = {}
    for tag in BeautifulSoup(html_str, "html.parser").find_all("ref"):
        rid = tag.get("id")
        if isinstance(rid, str) and rid:
            ids[rid] = None
    return list(ids)


def split_at_html_tags(
    group: ElementGroup,
    heading_ctx: list[Heading],
    renderer: Renderer,
    tokenizer: Tokenizer,
    max_tokens: int,
    doc_title: str | None,
    doc_metadata: BaseMetadata,
    parent_ref: ExternalRef,
    sentence_splitter: SentenceSplitter = _default_sentence_splitter,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    doc_source_path: str = "",
) -> list[Document] | None:
    """Split a group's root element at structural HTML boundaries (rows, list items, blocks).

    The cut points come from :func:`plan_html_split`. Every fragment is **well-formed and keeps
    its wrapper element**: a table fragment is a ``<table>`` (header repeated) carried by a
    :class:`Table`, a list fragment is a ``<ul>``/``<ol>``/``<dl>`` carried by a
    :class:`DocumentList`; other containers yield :class:`RawText` fragments wrapped in the
    original outer tag. Fragments are never cut mid-tag. Each fragment inherits the referenced
    elements whose ``<ref id="..."/>`` tags it contains.

    Units are packed greedily using a token estimate, then every fragment is measured through
    the renderer: a fragment over ``max_tokens`` is halved until it fits. Only a *single* unit
    (one row / one item) that still exceeds the budget leaves the structural path — it goes to
    :func:`split_at_sentences` (tier 2) and then to the token slice (tier 3).

    Returns ``None`` when :func:`plan_html_split` finds nothing to split on — caller falls back
    to tier 2.
    """
    root = group.root
    plan = plan_html_split(root.html)
    if plan is None:
        return None

    ref_by_id: dict[str, BaseElement] = {el.id: el for el in group.referenced}

    # Measure overhead once
    overhead_tokens = _overhead_tokens(
        Document(elements=cast("list[ElementType]", heading_ctx), title=doc_title, metadata=doc_metadata),
        renderer,
        tokenizer,
    )
    content_budget = max_tokens - overhead_tokens

    # Unit sizes are estimated from raw HTML, scaled by how the whole element renders in the
    # target format (markdown is far smaller than HTML). The estimate only steers the greedy
    # packing — the rendered size of every fragment is verified in _emit below.
    raw_root_tokens = max(1, tokenizer.count(root.html))
    rendered_root_tokens = tokenizer.count(renderer.render(Document(elements=cast("list[ElementType]", [root]))))
    scale = rendered_root_tokens / raw_root_tokens

    def _estimate(html_str: str) -> int:
        return math.ceil(tokenizer.count(html_str) * scale) if html_str else 0

    unit_budget = content_budget - _estimate(plan.open_tag + plan.header_html + plan.close_tag)

    result: list[Document] = []

    def _emit(start: int, end: int) -> None:
        fragment_html = build_fragment_html(plan, start, end)
        extra = [ref_by_id[rid] for rid in collect_ref_ids(fragment_html) if rid in ref_by_id]
        fragment = build_fragment_element(root, plan.outer_name, fragment_html)
        chunk_doc = Document(
            elements=cast("list[ElementType]", [*heading_ctx, fragment, *extra]),
            title=doc_title,
            source_path=doc_source_path,
            metadata=copy_metadata(doc_metadata),
            external_refs=[parent_ref],
        )

        if tokenizer.count(renderer.render(chunk_doc)) <= max_tokens:
            result.append(chunk_doc)
            return

        if end - start > 1:
            # Estimate was too optimistic: halve on a unit boundary, stay structural.
            mid = (start + end) // 2
            _emit(start, mid)
            _emit(mid, end)
            return

        # A single unit is too big on its own: try tier 2 (sentence split on rendered)
        sub = split_at_sentences(
            elements=[fragment, *extra],
            heading_ctx=heading_ctx,
            renderer=renderer,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            doc_title=doc_title,
            doc_metadata=doc_metadata,
            parent_ref=parent_ref,
            sentence_splitter=sentence_splitter,
            doc_source_path=doc_source_path,
            overlap_tokens=overlap_tokens,
        )
        if sub:
            result.extend(sub)
            return

        # Tier 3: token slice on rendered content
        result.extend(
            _token_slice(
                chunk_doc=chunk_doc,
                renderer=renderer,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                overhead_tokens=overhead_tokens,
                heading_ctx=heading_ctx,
                doc_title=doc_title,
                doc_metadata=doc_metadata,
                parent_ref=parent_ref,
                doc_source_path=doc_source_path,
            )
        )

    start = 0
    current_tokens = 0
    for index, unit in enumerate(plan.units):
        unit_tokens = _estimate(unit.html)
        if index > start and current_tokens + unit_tokens > unit_budget:
            _emit(start, index)
            start = index
            current_tokens = 0
        current_tokens += unit_tokens
    _emit(start, len(plan.units))

    return result if len(result) > 1 else None


# ---------------------------------------------------------------------------
# Tier-3 helper: token slice (unchanged logic, now a private function)
# ---------------------------------------------------------------------------


def _token_slice(
    chunk_doc: Document,
    renderer: Renderer,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    overhead_tokens: int,
    heading_ctx: list[Heading],
    doc_title: str | None,
    doc_metadata: BaseMetadata,
    parent_ref: ExternalRef,
    doc_source_path: str = "",
) -> list[Document]:
    """Split ``chunk_doc`` by slicing its rendered token sequence with overlap.

    Renders and tokenizes ONLY the content elements (inline refs are resolved —
    no raw ``<ref/>`` tags remain). The heading context is deliberately excluded
    from the tokenized text: slicing a combined render by the token *count* of a
    separately rendered overhead document is unsound (BPE merges across the seam
    drop or duplicate boundary text). ``overhead_tokens`` still sizes the budget,
    because ``make_chunk`` re-attaches ``heading_ctx`` to every output chunk.

    Every chunk is measured through the renderer and its window shrunk until the rendered
    chunk fits ``max_tokens`` (down to a floor of :data:`MIN_SLICE_TOKENS` content tokens).
    """
    content_budget = max_tokens - overhead_tokens

    if content_budget - overlap_tokens <= 100:
        raise ValueError(
            f"content_budget={content_budget} with overlap_tokens={overlap_tokens} leaves "
            "too little room for content. Reduce overlap_tokens or increase max_tokens."
        )

    heading_ids = {h.id for h in heading_ctx}
    content_elements = [el for el in chunk_doc.elements if el.id not in heading_ids]
    rendered = renderer.render(Document(elements=content_elements))
    content_tokens = tokenizer(rendered)

    def make_chunk(text: str) -> Document:
        return Document(
            elements=[*heading_ctx, RawText(html=text)],
            title=doc_title,
            source_path=doc_source_path,
            metadata=copy_metadata(doc_metadata),
            external_refs=[parent_ref],
        )

    result: list[Document] = []
    total = len(content_tokens)
    start = 0
    while start < total:
        # A slice is re-rendered when its chunk is rendered (the text becomes RawText html), and
        # that render can be larger than the slice itself (escaping, wrapper syntax). Shrink the
        # window until the *rendered chunk* fits, so the budget holds for what callers measure.
        window = content_budget
        while True:
            end = min(start + window, total)
            chunk = make_chunk(tokenizer.decode(content_tokens[start:end]))
            if window <= MIN_SLICE_TOKENS:
                break
            rendered_tokens = tokenizer.count(renderer.render(chunk))
            if rendered_tokens <= max_tokens:
                break
            window = max(MIN_SLICE_TOKENS, min(window - 1, window * max_tokens // rendered_tokens))
        result.append(chunk)
        if end >= total:
            break
        start = max(start + 1, end - min(overlap_tokens, (end - start) // 2))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_oversized_element(
    document: Document,
    renderer: Renderer,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    parent_id: str | None = None,
) -> list[Document]:
    """Split a document whose content exceeds max_tokens using a three-tier strategy.

    The document is expected to contain optional heading context plus one or more
    content elements (root + its referenced elements form one group).

    Strategy (applied in order until a split is achieved):

    1. **HTML structural split** — :func:`split_at_html_tags`: splits the root
       element's HTML at top-level block children.  Referenced elements follow
       their ``<ref/>`` tags into the appropriate sub-chunk.
    2. **Sentence split** — :func:`split_at_sentences`: renders the content
       (resolving refs inline) and splits at sentence boundaries.
    3. **Token slice** — last resort: rendered text (refs inlined) sliced into
       overlapping windows of ``content_budget`` tokens.

    Because tiers 2 and 3 operate on *rendered* text, no raw ``<ref/>`` tags
    are present in the output — refs are resolved before any text-level split.

    Args:
        document: Document to split.  May contain heading context elements plus
            the content group (root + referenced elements).
        renderer: Renderer used to convert elements to text for token counting.
        tokenizer: Tokenizer for counting and splitting.  Defaults to GPTTokenizer.
        max_tokens: Maximum tokens per output chunk.
        overlap_tokens: Tokens of overlap for tier-3 token-slice chunks.
        parent_id: Override for the ``external-parent`` target ID.  Pass the
            originating document's id when calling from ``split_by_elements``.

    Returns:
        List of Documents in reading order.  Returns ``[document]`` when no
        split is needed.
    """
    tokenizer = tokenizer or _get_default_tokenizer()

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        return [document]

    heading_ctx: list[Heading] = [e for e in document.elements if isinstance(e, Heading)]
    groups = build_element_groups(cast("list[BaseElement]", document.elements))
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]

    assert len(content_groups) == 1, (
        "split_oversized_element expects a single content group (root + optional refs) "
        f"plus optional heading context. Got {len(content_groups)} content groups. "
        f"Document id: {document.id}"
    )

    group = content_groups[0]

    overhead_tokens = _overhead_tokens(
        Document(
            elements=cast("list[ElementType]", heading_ctx),
            title=document.title,
            metadata=copy_metadata(document.metadata),
        ),
        renderer,
        tokenizer,
    )
    content_budget = max_tokens - overhead_tokens

    if content_budget - overlap_tokens <= 100:
        raise ValueError(
            f"content_budget={content_budget} with overlap_tokens={overlap_tokens} leaves "
            "too little room for actual content tokens. Reduce overlap_tokens or increase max_tokens."
        )

    parent_ref = ExternalRef(target_id=parent_id or document.id, rel_type="external-parent")

    # Tier 1: HTML structural split
    tier1 = split_at_html_tags(
        group=group,
        heading_ctx=heading_ctx,
        renderer=renderer,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        doc_title=document.title,
        doc_metadata=document.metadata,
        parent_ref=parent_ref,
        overlap_tokens=overlap_tokens,
        doc_source_path=document.source_path,
    )
    if tier1:
        return tier1

    # Tier 2: Sentence split on rendered content
    tier2 = split_at_sentences(
        elements=group.all_elements,
        heading_ctx=heading_ctx,
        renderer=renderer,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        doc_title=document.title,
        doc_metadata=document.metadata,
        parent_ref=parent_ref,
        doc_source_path=document.source_path,
        overlap_tokens=overlap_tokens,
    )
    if tier2:
        return tier2

    # Tier 3: Token slice on rendered content
    full_doc = Document(
        elements=cast("list[ElementType]", heading_ctx + group.all_elements),
        title=document.title,
        source_path=document.source_path,
        metadata=copy_metadata(document.metadata),
    )
    return _token_slice(
        chunk_doc=full_doc,
        renderer=renderer,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        overhead_tokens=overhead_tokens,
        heading_ctx=heading_ctx,
        doc_title=document.title,
        doc_metadata=document.metadata,
        parent_ref=parent_ref,
        doc_source_path=document.source_path,
    )


def split_by_elements(
    document: Document,
    renderer: Renderer,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Document]:
    """Split a document greedily at element-group boundaries by token count.

    Elements are first partitioned into atomic :class:`~ragdoc.splitting.groups.ElementGroup`
    objects via :func:`~ragdoc.splitting.groups.build_element_groups`.  A group
    (root + its transitively-referenced elements) is the minimum indivisible unit:
    referenced elements always stay with their root.

    Accumulates groups into chunks until adding the next group would exceed
    ``max_tokens``, then starts a new chunk.  Heading elements update a
    running context that is prepended to every new chunk.

    Oversized single groups (token count > ``max_tokens`` even alone) are
    handled by :func:`split_oversized_element` (three-tier: HTML structural →
    sentence → token-slice).

    Args:
        document: The document to split.
        renderer: Renderer for token counting.
        tokenizer: Tokenizer.  Defaults to GPTTokenizer.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Passed through to :func:`split_oversized_element`.

    Returns:
        List of Documents in reading order.  Returns ``[document]`` when no
        split is needed.

    Note:
        Does not set ``split_sequence`` / ``split_total`` in metadata. Call
        :func:`split_document` for reading-order numbering.
    """
    tokenizer = tokenizer or _get_default_tokenizer()
    doc_label = f"{Path(document.source_path).name} ({document.id})" if document.source_path else document.id

    if not document.elements:
        return [document]

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        return [document]

    groups = build_element_groups(cast("list[BaseElement]", document.elements))

    # Pre-compute token counts per element (for heading context tracking)
    # and per group (root rendered together with its referenced elements).
    el_tokens: dict[str, int] = {
        el.id: tokenizer.count(renderer.render(Document(elements=[el])))
        for el in document.elements
        if isinstance(el, Heading)
    }
    group_tokens: dict[int, int] = {}  # keyed by index into groups list
    for i, item in enumerate(groups):
        if isinstance(item, ElementGroup):
            group_tokens[i] = tokenizer.count(
                renderer.render(Document(elements=cast("list[ElementType]", item.all_elements)))
            )

    parent_ref = ExternalRef(target_id=document.id, rel_type="external-parent")

    heading_ctx: dict[int, Heading] = {}  # level → most recent heading
    ctx_level_order: list[int] = []

    def get_ctx() -> list[Heading]:
        return [heading_ctx[lvl] for lvl in ctx_level_order]

    def get_ctx_tokens() -> int:
        return sum(el_tokens.get(heading_ctx[lvl].id, 0) for lvl in ctx_level_order)

    result: list[Document] = []
    current: list[BaseElement] = []
    current_tokens: int = 0
    # True while `current` holds only the heading context re-seeded after an oversized-group
    # delegation. Those headings already lead every delegated sub-split, so flushing them as
    # a *final* split would emit a content-free heading-only chunk. Any subsequent append
    # (a new heading or a group's elements) clears the flag — new material must survive.
    reseeded_ctx_only = False

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            result.append(
                Document(
                    elements=cast("list[ElementType]", list(current)),
                    title=document.title,
                    source_path=document.source_path,
                    metadata=copy_metadata(document.metadata),
                    external_refs=[parent_ref],
                )
            )
        current = []
        current_tokens = 0

    for i, item in enumerate(groups):
        if isinstance(item, Heading):
            current.append(item)
            current_tokens += el_tokens.get(item.id, 0)
            reseeded_ctx_only = False
            if item.level not in ctx_level_order:
                ctx_level_order.append(item.level)
            heading_ctx[item.level] = item
            continue

        # ElementGroup
        tokens = group_tokens[i]
        has_content = any(not isinstance(e, Heading) for e in current)

        if tokens > max_tokens:
            # Oversized group: flush current if has content, then delegate
            if has_content:
                flush()
            ctx = get_ctx()
            ctx_tokens = get_ctx_tokens()
            oversized_doc = Document(
                elements=cast("list[ElementType]", ctx + item.all_elements),
                title=document.title,
                source_path=document.source_path,
                metadata=copy_metadata(document.metadata),
            )
            sub_splits = split_oversized_element(
                oversized_doc,
                renderer,
                tokenizer,
                max_tokens,
                overlap_tokens,
                parent_id=document.id,
            )
            result.extend(sub_splits)
            current = list(ctx)
            current_tokens = ctx_tokens
            reseeded_ctx_only = True

        elif has_content and current_tokens + tokens > max_tokens:
            flush()
            ctx = get_ctx()
            ctx_tokens = get_ctx_tokens()
            current = list(ctx) + item.all_elements
            current_tokens = ctx_tokens + tokens
            reseeded_ctx_only = False

        else:
            current.extend(item.all_elements)
            current_tokens += tokens
            reseeded_ctx_only = False

    if not reseeded_ctx_only:
        flush()
    final = result if result else [document]
    logger.debug(f"split_by_elements: {doc_label} produced {len(final)} sub-documents")
    return final


def split_document(
    document: Document,
    renderer: Renderer,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Document]:
    """Split a document into token-bounded chunks using a three-tier strategy.

    Strategy (applied in order):

    1. **Fits** — if the whole document is within ``max_tokens``, return as-is.
    2. **Hierarchical** — attempt :func:`~ragdoc.splitting.hierarchical.split_hierarchical`.
       If that yields ≥ 2 parts, recurse on each part.
    3. **Element-level fallback** — if hierarchical cannot split (single heading
       level or no headings), delegate to :func:`split_by_elements`, which itself
       falls back to :func:`split_oversized_element` for any group that exceeds
       the budget.

    Args:
        document: The document to split.
        renderer: Renderer for token counting.
        tokenizer: Tokenizer.  Defaults to GPTTokenizer.
        max_tokens: Maximum tokens per output chunk.
        overlap_tokens: Tokens of overlap for text-level splits.

    Returns:
        A list of ``Document`` objects in reading order. Every document carries
        ``metadata["split_sequence"]`` (1-based position; ``1`` for a document that
        was not split, or ``1..N`` for a split document) and ``metadata["split_total"]``
        (total split count from this source; ``1`` for a non-split document).
        Useful for sorting after retrieval:
        ``sorted(splits, key=lambda d: d.metadata["split_sequence"])``.
    """
    tokenizer = tokenizer or _get_default_tokenizer()
    doc_label = f"{Path(document.source_path).name} ({document.id})" if document.source_path else document.id

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        logger.debug(f"split_document: {doc_label} fits within {max_tokens} tokens, no split needed")
        # Never mutate the input document: return a shallow copy owning a fresh metadata dict.
        # Elements stay shared by reference — consistent with split behavior generally.
        doc = document.model_copy(update={"metadata": {**document.metadata, "split_sequence": 1, "split_total": 1}})
        return [doc]

    splits = split_hierarchical(document)

    if len(splits) == 1:
        logger.debug(f"split_document: {doc_label} falling back to element-level split")
        result = split_by_elements(document, renderer, tokenizer, max_tokens, overlap_tokens)
    else:
        logger.debug(f"split_document: {doc_label} hierarchical split produced {len(splits)} parts, recursing")
        result: list[Document] = []
        for split in splits:
            result.extend(split_document(split, renderer, tokenizer, max_tokens, overlap_tokens))
    logger.info(f"split_document: {doc_label} -> {len(result)} splits")

    # Write reading-order provenance into metadata. Only split_document does this —
    # individual splitters leave split_sequence/split_total absent so the outermost
    # call owns the numbering. Each split owns its metadata dict (copied at construction),
    # so in-place stamping is safe.
    if len(result) > 1:
        total = len(result)
        for i, doc in enumerate(result):
            doc.metadata["split_sequence"] = i + 1
            doc.metadata["split_total"] = total

    return result
