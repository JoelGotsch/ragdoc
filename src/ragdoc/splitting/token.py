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

1. **HTML structural split** (:func:`split_at_html_tags`) — BS4 parses the
   root element's HTML and splits at top-level block children (``<tr>``,
   ``<li>``, ``<p>``, etc.).  Each piece keeps the block tags intact and
   inherits whichever referenced elements' ``<ref/>`` tags it contains.

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
import re
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup

from ragdoc.document import BaseElement, Document, ExternalRef, Heading, RawText
from ragdoc.rendering import Renderer
from ragdoc.splitting.groups import ElementGroup, build_element_groups
from ragdoc.splitting.hierarchical import split_hierarchical
from ragdoc.utils import GPTTokenizer, Tokenizer

logger = logging.getLogger(__name__)

_default_tokenizer = GPTTokenizer()
DEFAULT_OVERLAP_TOKENS = 200
DEFAULT_MAX_TOKENS = 7_000

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
    doc_metadata: dict,
    parent_ref: ExternalRef,
    sentence_splitter: SentenceSplitter = _default_sentence_splitter,
    doc_source_path: str = "",
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

    Returns:
        List of Documents or ``None`` if no split was possible.
    """
    full_doc = Document(
        elements=heading_ctx + elements, title=doc_title, metadata=doc_metadata
    )
    rendered = renderer.render(full_doc)
    if tokenizer.count(rendered) <= max_tokens:
        return [Document(elements=heading_ctx + elements, title=doc_title, source_path=doc_source_path, metadata=doc_metadata, external_refs=[parent_ref])]

    overhead_tokens = _overhead_tokens(
        Document(elements=heading_ctx, title=doc_title, metadata=doc_metadata), renderer, tokenizer
    )
    content_budget = max_tokens - overhead_tokens

    sentences = sentence_splitter(rendered)
    if len(sentences) <= 1:
        return None  # no boundaries found — caller tries tier 3

    def make_chunk(text: str) -> Document:
        return Document(
            elements=heading_ctx + [RawText(innerhtml=text)],
            title=doc_title,
            source_path=doc_source_path,
            metadata=doc_metadata,
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

    return result if len(result) > 1 else None


# ---------------------------------------------------------------------------
# Tier-1 helper: HTML structural split
# ---------------------------------------------------------------------------


def split_at_html_tags(
    group: ElementGroup,
    heading_ctx: list[Heading],
    renderer: Renderer,
    tokenizer: Tokenizer,
    max_tokens: int,
    doc_title: str | None,
    doc_metadata: dict,
    parent_ref: ExternalRef,
    sentence_splitter: SentenceSplitter = _default_sentence_splitter,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    doc_source_path: str = "",
) -> list[Document] | None:
    """Split a group at top-level HTML block-child boundaries using BS4.

    Parses ``group.root.html``, extracts the outer tag's direct children, and
    accumulates them into sub-chunks.  Each sub-chunk inherits the referenced
    elements whose ``<ref id="..."/>`` tags appear in that sub-chunk's HTML.

    Each sub-chunk that still exceeds ``max_tokens`` is passed to
    :func:`split_at_sentences` (tier 2) and then to token-slice (tier 3).

    Returns ``None`` when the root has only one structural child (e.g. a plain
    ``<p>`` with no block sub-children) — caller falls back to tier 2.
    """
    root_html = group.root.html
    soup = BeautifulSoup(root_html, "html.parser")
    outer = soup.find()
    if outer is None:
        return None

    # Direct children that are actual tags (skip bare text nodes and whitespace)
    children = [c for c in outer.children if hasattr(c, "name") and c.name]
    if len(children) <= 1:
        return None  # nothing useful to split on

    outer_tag = outer.name  # e.g. "table", "ul", "p"
    ref_by_id: dict[str, BaseElement] = {el.id: el for el in group.referenced}

    def _referenced_ids_in(html_str: str) -> set[str]:
        s = BeautifulSoup(html_str, "html.parser")
        return {tag.get("id") for tag in s.find_all("ref") if tag.get("id")}

    def _wrap(child_htmls: list[str]) -> str:
        inner = "".join(child_htmls)
        return f"<{outer_tag}>{inner}</{outer_tag}>"

    def _make_doc(child_htmls: list[str], extra_els: list[BaseElement]) -> Document:
        raw = RawText(innerhtml=_wrap(child_htmls))
        return Document(
            elements=heading_ctx + [raw] + extra_els,
            title=doc_title,
            source_path=doc_source_path,
            metadata=doc_metadata,
            external_refs=[parent_ref],
        )

    # Measure overhead once
    overhead_tokens = _overhead_tokens(
        Document(elements=heading_ctx, title=doc_title, metadata=doc_metadata), renderer, tokenizer
    )
    content_budget = max_tokens - overhead_tokens

    result: list[Document] = []
    current_children: list[str] = []
    current_tokens = 0

    def _flush(children_html: list[str]) -> None:
        ref_ids = _referenced_ids_in("".join(children_html))
        extra = [ref_by_id[rid] for rid in ref_ids if rid in ref_by_id]
        chunk_doc = _make_doc(children_html, extra)

        if tokenizer.count(renderer.render(chunk_doc)) <= max_tokens:
            result.append(chunk_doc)
            return

        # Sub-chunk still too big: try tier 2 (sentence split on rendered)
        sub = split_at_sentences(
            elements=[RawText(innerhtml=_wrap(children_html))] + extra,
            heading_ctx=heading_ctx,
            renderer=renderer,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            doc_title=doc_title,
            doc_metadata=doc_metadata,
            parent_ref=parent_ref,
            sentence_splitter=sentence_splitter,
            doc_source_path=doc_source_path,
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

    for child in children:
        child_html = str(child)
        child_tokens = tokenizer.count(child_html)

        if current_children and current_tokens + child_tokens > content_budget:
            _flush(current_children)
            current_children = [child_html]
            current_tokens = child_tokens
        else:
            current_children.append(child_html)
            current_tokens += child_tokens

    if current_children:
        _flush(current_children)

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
    doc_metadata: dict,
    parent_ref: ExternalRef,
    doc_source_path: str = "",
) -> list[Document]:
    """Split ``chunk_doc`` by slicing its rendered token sequence with overlap.

    Renders the full document first (inline refs are resolved — no raw
    ``<ref/>`` tags remain), then slices the content-token window.
    """
    rendered = renderer.render(chunk_doc)
    content_budget = max_tokens - overhead_tokens

    if content_budget - overlap_tokens <= 100:
        raise ValueError(
            f"content_budget={content_budget} with overlap_tokens={overlap_tokens} leaves "
            "too little room for content. Reduce overlap_tokens or increase max_tokens."
        )

    token_ids = tokenizer(rendered)
    content_tokens = token_ids[overhead_tokens:]

    def make_chunk(text: str) -> Document:
        return Document(
            elements=heading_ctx + [RawText(innerhtml=text)],
            title=doc_title,
            source_path=doc_source_path,
            metadata=doc_metadata,
            external_refs=[parent_ref],
        )

    result: list[Document] = []
    start = 0
    while start < len(content_tokens):
        end = min(start + content_budget, len(content_tokens))
        chunk_text = tokenizer.decode(content_tokens[start:end])
        result.append(make_chunk(chunk_text))
        if end >= len(content_tokens):
            break
        start = max(start + 1, end - overlap_tokens)
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
    tokenizer = tokenizer or _default_tokenizer

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        return [document]

    heading_ctx: list[Heading] = [e for e in document.elements if isinstance(e, Heading)]
    groups = build_element_groups(document.elements)
    content_groups = [g for g in groups if isinstance(g, ElementGroup)]

    assert len(content_groups) == 1, (
        "split_oversized_element expects a single content group (root + optional refs) "
        f"plus optional heading context. Got {len(content_groups)} content groups. "
        f"Document id: {document.id}"
    )

    group = content_groups[0]

    overhead_tokens = _overhead_tokens(
        Document(elements=heading_ctx, title=document.title, metadata=document.metadata),
        renderer, tokenizer,
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
    )
    if tier2:
        return tier2

    # Tier 3: Token slice on rendered content
    full_doc = Document(
        elements=heading_ctx + group.all_elements,
        title=document.title,
        source_path=document.source_path,
        metadata=document.metadata,
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
    tokenizer = tokenizer or _default_tokenizer
    doc_label = f"{Path(document.source_path).name} ({document.id})" if document.source_path else document.id

    if not document.elements:
        return [document]

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        return [document]

    groups = build_element_groups(document.elements)

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
                renderer.render(Document(elements=item.all_elements))
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

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            result.append(Document(
                elements=list(current),
                title=document.title,
                source_path=document.source_path,
                metadata=document.metadata,
                external_refs=[parent_ref],
            ))
        current = []
        current_tokens = 0

    for i, item in enumerate(groups):
        if isinstance(item, Heading):
            current.append(item)
            current_tokens += el_tokens.get(item.id, 0)
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
                elements=ctx + item.all_elements,
                title=document.title,
                source_path=document.source_path,
                metadata=document.metadata,
            )
            sub_splits = split_oversized_element(
                oversized_doc, renderer, tokenizer, max_tokens, overlap_tokens,
                parent_id=document.id,
            )
            result.extend(sub_splits)
            current = list(ctx)
            current_tokens = ctx_tokens

        elif has_content and current_tokens + tokens > max_tokens:
            flush()
            ctx = get_ctx()
            ctx_tokens = get_ctx_tokens()
            current = list(ctx) + item.all_elements
            current_tokens = ctx_tokens + tokens

        else:
            current.extend(item.all_elements)
            current_tokens += tokens

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
    tokenizer = tokenizer or _default_tokenizer
    doc_label = f"{Path(document.source_path).name} ({document.id})" if document.source_path else document.id

    if tokenizer.count(renderer.render(document)) <= max_tokens:
        logger.debug(f"split_document: {doc_label} fits within {max_tokens} tokens, no split needed")
        document.metadata["split_sequence"] = 1
        document.metadata["split_total"] = 1
        return [document]

    splits = split_hierarchical(document)

    if len(splits) == 1:
        logger.debug(f"split_document: {doc_label} falling back to element-level split")
        result = split_by_elements(document, renderer, tokenizer, max_tokens, overlap_tokens)
    else: 
        logger.debug(
            f"split_document: {doc_label} hierarchical split produced {len(splits)} parts, recursing"
        )
        result: list[Document] = []
        for split in splits:
            result.extend(split_document(split, renderer, tokenizer, max_tokens, overlap_tokens))
    logger.info(f"split_document: {doc_label} -> {len(result)} splits")

    # Write reading-order provenance into metadata. Only split_document does this —
    # individual splitters leave split_sequence/split_total absent so the outermost
    # call owns the numbering.  Create a new metadata dict per doc because splitters
    # propagate metadata by reference (all splits share the parent's dict object); in-place
    # mutation would leave every split with the last iteration's value.
    if len(result) > 1:
        total = len(result)
        for i, doc in enumerate(result):
            doc.metadata = {**doc.metadata, "split_sequence": i + 1, "split_total": total}

    return result
