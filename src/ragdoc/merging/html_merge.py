"""Approach B: render-merge-reparse merge (merge_documents_html).

Renders both documents to HTML via ``render_raw``, aligns the top-level HTML
elements using text similarity, selects the richer element for each aligned
pair, then reparses the merged HTML into a new Document.

Trade-offs vs. Approach A:
- Simpler — reuses the existing rendering + HTML parsing pipeline.
- ``Footnote`` element structure is preserved: ``Footnote.html`` renders as
  ``<aside class="footnote">`` which ``generate_document`` parses back directly.
- Image IDs are regenerated (base64 data is preserved).
- Page / bounding-box metadata is lost.
"""

from __future__ import annotations

import difflib
from typing import Literal

from bs4 import BeautifulSoup, Tag

from ragdoc.document import Document, ElementTypeEnum
from ragdoc.merging.heuristics import has_heading_hierarchy, markup_richness_score
from ragdoc.parsing.html.load import HTML, generate_document
from ragdoc.rendering import OutputFormat, Renderer, render_raw
from ragdoc.utils.helpers import normalize_text

# Mapping from HTML tag names to ElementTypeEnum values.
_TAG_TO_TYPE: dict[str, ElementTypeEnum] = {
    "h1": ElementTypeEnum.HEADING,
    "h2": ElementTypeEnum.HEADING,
    "h3": ElementTypeEnum.HEADING,
    "h4": ElementTypeEnum.HEADING,
    "h5": ElementTypeEnum.HEADING,
    "h6": ElementTypeEnum.HEADING,
    "p": ElementTypeEnum.PARAGRAPH,
    "div": ElementTypeEnum.PARAGRAPH,
    "table": ElementTypeEnum.TABLE,
    "ul": ElementTypeEnum.DOCUMENT_LIST,
    "ol": ElementTypeEnum.DOCUMENT_LIST,
    "dl": ElementTypeEnum.DOCUMENT_LIST,
    "aside": ElementTypeEnum.FOOTNOTE,
    "img": ElementTypeEnum.IMAGE,
}

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _tag_to_element_type(tag: Tag) -> ElementTypeEnum | None:
    """Map a BeautifulSoup tag to an ElementTypeEnum, or None if not a content tag."""
    return _TAG_TO_TYPE.get(tag.name)


def _extract_top_level_tags(html_str: str) -> list[Tag]:
    """Parse HTML and return a flat list of top-level content tags.

    Looks inside a ``<body>`` if present; otherwise uses the root.
    Only returns tags whose name maps to a known ElementTypeEnum.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    root = soup.find("body") or soup
    tags: list[Tag] = []
    for child in root.children:
        if not isinstance(child, Tag):
            continue
        if _tag_to_element_type(child) is not None:
            tags.append(child)
    return tags


def _alignment_key(tag: Tag) -> str:
    """Normalized text key for aligning HTML tags.

    For ``<img>`` tags ``get_text()`` always returns ``""`` so we use the src
    or alt attribute instead.  An unidentifiable image gets a unique sentinel
    so it is never auto-matched (safest behaviour).
    """
    if tag.name == "img":
        src = tag.get("src", "")
        if isinstance(src, str) and src.startswith("data:image/"):
            _, payload = src.split(",", maxsplit=1)
            return f"__image__{payload[:40]}"
        if isinstance(src, str) and src.startswith("http"):
            return f"__image__{src}"
        alt = tag.get("alt", "")
        if isinstance(alt, str) and alt.strip():
            return f"__image__{alt.strip()}"
        return f"__image__{id(tag)}"
    return normalize_text(tag.get_text())


def _block_ratio(keys_a: list[str], keys_b: list[str]) -> float:
    """Compute text similarity ratio for a replace block, ignoring image keys.

    Image alignment keys (``__image__…``) share text with paragraph keys
    (e.g. alt text overlaps with caption text), which would artificially
    inflate the ratio and cause the block to enter the merge path when it
    should be treated as "too different".  By filtering image keys out before
    the ratio computation we keep the merge/replace decision purely
    text-driven.

    If both sides contain only image keys, the ratio is 1.0 (treat as a match
    between the image groups).  If only one side has text keys, 0.0.
    """
    text_a = [k for k in keys_a if not k.startswith("__image__")]
    text_b = [k for k in keys_b if not k.startswith("__image__")]
    if not text_a and not text_b:
        return 1.0  # both sides are images only — treat as matched
    if not text_a or not text_b:
        return 0.0  # one side text, the other images — too different
    return difflib.SequenceMatcher(None, " ".join(text_a), " ".join(text_b)).ratio()


def _is_insertion_allowed(
    tag: Tag,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum],
) -> bool:
    """Return True if inserting this tag from doc_b is permitted."""
    if isinstance(allow_insertions_from_b, bool):
        return allow_insertions_from_b
    etype = _tag_to_element_type(tag)
    return etype in allow_insertions_from_b if etype is not None else False


def _select_tag(
    tag_a: Tag,
    tag_b: Tag,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None,
    hier_a: bool = False,
    hier_b: bool = False,
) -> Tag:
    """Select the better of two aligned HTML tags.

    Priority:
    1. ``prefer_source`` per element type, if set.
    2. Heading hierarchy heuristic: when exactly one document has multi-level
       heading hierarchy, that document's heading level wins; the richer
       content is used (may come from the other document).
    3. Markup richness fallback.

    Args:
        tag_a: HTML tag from document A.
        tag_b: HTML tag from document B.
        prefer_source: Per-type source preference.
        hier_a: Whether document A has multi-level heading hierarchy.
        hier_b: Whether document B has multi-level heading hierarchy.

    Returns:
        The winning tag (may be a newly constructed tag for the hierarchy case).
    """
    from bs4 import BeautifulSoup as _BS

    etype = _tag_to_element_type(tag_a)
    if etype is not None and prefer_source is not None:
        pref = prefer_source.get(etype)
        if pref == "a":
            return tag_a
        if pref == "b":
            return tag_b

    # Heading hierarchy heuristic (only when exactly one side has hierarchy)
    if tag_a.name in _HEADING_TAGS and tag_b.name in _HEADING_TAGS:
        if hier_a and not hier_b:
            # doc_a has reliable levels; use tag_a's tag name (level)
            score_a = markup_richness_score(str(tag_a))
            score_b = markup_richness_score(str(tag_b))
            if score_b > score_a:
                # tag_b has richer content — splice it into tag_a's level
                new_html = f"<{tag_a.name}>{tag_b.decode_contents()}</{tag_a.name}>"
                merged = _BS(new_html, "html.parser").find(tag_a.name)
                return merged if merged is not None else tag_a
            return tag_a
        if hier_b and not hier_a:
            # doc_b has reliable levels; use tag_b's tag name (level)
            score_a = markup_richness_score(str(tag_a))
            score_b = markup_richness_score(str(tag_b))
            if score_a > score_b:
                # tag_a has richer content — splice it into tag_b's level
                new_html = f"<{tag_b.name}>{tag_a.decode_contents()}</{tag_b.name}>"
                merged = _BS(new_html, "html.parser").find(tag_b.name)
                return merged if merged is not None else tag_b
            return tag_b

    score_a = markup_richness_score(str(tag_a))
    score_b = markup_richness_score(str(tag_b))
    return tag_b if score_b > score_a else tag_a


def merge_documents_html(
    doc_a: Document,
    doc_b: Document,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None = None,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum] = True,
    similarity_threshold: float = 0.6,
) -> Document:
    """Merge two Documents via render-merge-reparse (Approach B).

    Pipeline:
    1. Render both documents to HTML using ``render_raw``.
    2. Parse both HTML strings to flat lists of top-level content tags.
    3. Align tags using ``difflib.SequenceMatcher`` on normalised text keys.
    4. For each aligned pair, select the richer tag (or honour ``prefer_source``).
    5. Reassemble merged HTML and reparse via ``generate_document()``.

    Args:
        doc_a: Base document.
        doc_b: Overlay document.
        prefer_source: Per-type source preference.  Example::

                {ElementTypeEnum.HEADING: "b"}  # always take headings from doc_b

        allow_insertions_from_b: Whether to include elements that appear only in
            doc_b.  Pass ``False`` to suppress all insertions.
        similarity_threshold: Minimum ``SequenceMatcher`` ratio to treat a
            ``replace`` opcode as a merge rather than DELETE + INSERT.

    Returns:
        A new merged ``Document`` with ``parser="merged"``.

    """
    hier_a = has_heading_hierarchy(doc_a)
    hier_b = has_heading_hierarchy(doc_b)

    renderer = Renderer(format=OutputFormat.HTML, element_renderer=render_raw)
    html_a = renderer.render(doc_a)
    html_b = renderer.render(doc_b)

    tags_a = _extract_top_level_tags(html_a)
    tags_b = _extract_top_level_tags(html_b)

    keys_a = [_alignment_key(t) for t in tags_a]
    keys_b = [_alignment_key(t) for t in tags_b]

    matcher = difflib.SequenceMatcher(None, keys_a, keys_b, autojunk=False)
    merged_tags: list[Tag] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            for idx_a, idx_b in zip(range(i1, i2), range(j1, j2), strict=True):
                merged_tags.append(_select_tag(tags_a[idx_a], tags_b[idx_b], prefer_source, hier_a, hier_b))

        elif opcode == "replace":
            group_a = tags_a[i1:i2]
            group_b = tags_b[j1:j2]
            ratio = _block_ratio(keys_a[i1:i2], keys_b[j1:j2])

            # The single-element force-merge only applies when both sides have
            # text content (ratio > 0).  A ratio of 0.0 means one side is
            # image-only; forcing a merge would silently drop the image.
            if ratio >= similarity_threshold or (ratio > 0 and i2 - i1 == j2 - j1 == 1):
                # Similar enough — keep whichever is richer
                if len(group_a) == len(group_b):
                    for ta, tb in zip(group_a, group_b, strict=True):
                        merged_tags.append(_select_tag(ta, tb, prefer_source, hier_a, hier_b))
                else:
                    # Different lengths: pick the side with higher total richness
                    score_a = sum(markup_richness_score(str(t)) for t in group_a)
                    score_b = sum(markup_richness_score(str(t)) for t in group_b)

                    # Honour prefer_source for the first tag's type
                    etype = _tag_to_element_type(group_a[0]) if group_a else None
                    if etype is not None and prefer_source is not None:
                        pref = prefer_source.get(etype)
                        if pref == "a":
                            merged_tags.extend(group_a)
                            continue
                        if pref == "b":
                            merged_tags.extend(group_b)
                            continue

                    merged_tags.extend(group_b if score_b >= score_a else group_a)
            else:
                # Too different: drop doc_a group, add doc_b group if allowed
                for tb in group_b:
                    if _is_insertion_allowed(tb, allow_insertions_from_b):
                        merged_tags.append(tb)

        elif opcode == "insert":
            for tb in tags_b[j1:j2]:
                if _is_insertion_allowed(tb, allow_insertions_from_b):
                    merged_tags.append(tb)

        elif opcode == "delete":
            # Preserve doc_a-only tags (no counterpart in doc_b)
            merged_tags.extend(tags_a[i1:i2])

    # Reassemble merged HTML
    merged_html = "".join(str(tag) for tag in merged_tags)

    doc = generate_document(HTML(content=merged_html))
    doc.parser = "merged"
    doc.metadata = {
        "source_parser_a": doc_a.parser,
        "source_parser_b": doc_b.parser,
    }
    return doc
