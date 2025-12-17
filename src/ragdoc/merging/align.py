"""Element-alignment logic for Approach A (element-alignment with DocumentPatch).

Uses ``difflib.SequenceMatcher`` on normalized text keys to align primary
elements from two Documents.  Secondary elements (Image, Footnote) trail their
preceding primary anchor and travel with it through all operations.
"""

from __future__ import annotations

import difflib
from typing import Literal

from ragdoc.document import (
    Document,
    ElementType,
    ElementTypeEnum,
    Footnote,
    Heading,
    Image,
    Paragraph,
    RawText,
    Table,
)
from ragdoc.merging.heuristics import (
    inject_inline_markup,
    markup_richness_score,
    select_footnote,
    select_heading,
    select_paragraph,
    select_table,
)
from ragdoc.merging.patch import PatchOperation, PatchOperationType
from ragdoc.utils.helpers import _normalize_text

# Primary element types that drive alignment (text-bearing elements)
_PRIMARY_TYPES = frozenset(
    {
        ElementTypeEnum.HEADING,
        ElementTypeEnum.PARAGRAPH,
        ElementTypeEnum.TABLE,
        ElementTypeEnum.DOCUMENT_LIST,
        ElementTypeEnum.RAW_TEXT,
    }
)


def alignment_key(element: ElementType) -> str:
    """Return a canonical text key for sequence alignment.

    - ``Image`` → unique per-element sentinel (never auto-aligned)
    - ``Footnote`` → ``"__footnote__{number}"`` (aligned by footnote number)
    - All others → ``_normalize_text(element.text)``

    Args:
        element: Element to produce a key for.

    Returns:
        A string key for use with ``difflib.SequenceMatcher``.
    """
    if isinstance(element, Image):
        return f"__image__{element.id}"
    if isinstance(element, Footnote):
        return f"__footnote__{element.number}"
    return _normalize_text(element.text)


def _is_insertion_allowed(
    element: ElementType,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum],
) -> bool:
    """Return True if inserting this element from doc_b is permitted."""
    if isinstance(allow_insertions_from_b, bool):
        return allow_insertions_from_b
    return ElementTypeEnum(element.element_type) in allow_insertions_from_b


def _effective_prefer(
    element_type: ElementTypeEnum,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None,
) -> Literal["a", "b"] | None:
    """Return the preferred source for this element type, or None."""
    if prefer_source is None:
        return None
    return prefer_source.get(element_type)


def _group_with_trailing(
    elements: list[ElementType],
) -> list[tuple[ElementType, list[ElementType]]]:
    """Group primary elements with their trailing secondary elements.

    Secondary elements (Image, Footnote) that follow a primary element are
    grouped as its "trailing companions".  Any leading secondary elements
    (before the first primary) are attached to a synthetic RawText sentinel
    with empty text.

    Returns:
        List of ``(anchor, trailing)`` tuples where ``anchor`` is a primary
        element and ``trailing`` is the list of secondary elements that follow
        it before the next primary element.
    """
    groups: list[tuple[ElementType, list[ElementType]]] = []
    current_anchor: ElementType | None = None
    current_trailing: list[ElementType] = []

    for element in elements:
        if ElementTypeEnum(element.element_type) in _PRIMARY_TYPES:
            if current_anchor is not None:
                groups.append((current_anchor, current_trailing))
            current_anchor = element
            current_trailing = []
        else:
            # Secondary element
            if current_anchor is None:
                # Before first primary — create a dummy anchor
                current_anchor = RawText(innerhtml="")
            current_trailing.append(element)

    if current_anchor is not None:
        groups.append((current_anchor, current_trailing))

    return groups


def _with_injected_markup(winner: ElementType, loser: ElementType) -> ElementType:
    """Return *winner* enhanced with inline markup carried over from *loser*.

    When doc_b's element wins but doc_a's element has additional inline markup
    (e.g. ``<strong>`` around a word), the markup is injected into the winner's
    HTML so neither the structure (from doc_b) nor the formatting (from doc_a)
    is lost.  If no injection is possible the original winner is returned.
    """
    new_html = inject_inline_markup(winner.html, loser.html)
    if new_html == winner.html:
        return winner
    if isinstance(winner, (Paragraph, Heading)):
        # Both types store their full outer HTML in ``html_content``; ``innerhtml`` is a
        # read-only property, so a model_copy update against it would be a silent no-op.
        return winner.model_copy(update={"html_content": new_html})
    return winner  # unsupported type — return as-is


def _apply_heuristics(
    elements_a: list[ElementType],
    elements_b: list[ElementType],
    parser_a: str | None,
    parser_b: str | None,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None,
) -> tuple[list[ElementType], str]:
    """Apply element-type-specific heuristics to select the winning elements.

    For 1:1 pairs, uses ``select_heading``, ``select_paragraph``, etc.
    For M:N mismatches, keeps whichever side has higher total markup richness
    (or honours ``prefer_source`` as a tiebreaker).

    Args:
        elements_a: Elements from document A.
        elements_b: Elements from document B.
        parser_a: Parser name for document A.
        parser_b: Parser name for document B.
        prefer_source: Per-type source preference override.

    Returns:
        A tuple of ``(resolved_elements, reason_string)``.
    """
    if not elements_a:
        return elements_b, "no elements_a; using elements_b"
    if not elements_b:
        return elements_a, "no elements_b; using elements_a"

    # For 1:1 pairs of the same type, use type-specific heuristics
    if len(elements_a) == 1 and len(elements_b) == 1:
        ea, eb = elements_a[0], elements_b[0]
        etype = ElementTypeEnum(ea.element_type)

        # Check per-type preference first
        pref = _effective_prefer(etype, prefer_source)
        if pref == "a":
            return [ea], f"prefer_source='a' for {etype}"
        if pref == "b":
            return [eb], f"prefer_source='b' for {etype}"

        if isinstance(ea, Heading) and isinstance(eb, Heading):
            winner = select_heading(ea, eb, parser_a=parser_a, parser_b=parser_b)
            return [winner], "select_heading heuristic"
        if isinstance(ea, Paragraph) and isinstance(eb, Paragraph):
            winner = select_paragraph(ea, eb)
            return [winner], "select_paragraph heuristic (markup richness)"
        if isinstance(ea, Table) and isinstance(eb, Table):
            winner = select_table(ea, eb)
            return [winner], "select_table heuristic (header count)"
        if isinstance(ea, Footnote) and isinstance(eb, Footnote):
            winner = select_footnote(ea, eb)
            return [winner], "select_footnote heuristic (text length)"

    # M:N mismatch or mixed types: compare total markup richness.
    # Secondary elements (Image, Footnote) are structural signals — each counts as 2
    # additional richness points so that a properly structured doc_b (with explicit
    # Footnote/Image elements) beats a flat doc_a that only has inline markup tags.
    _SECONDARY_STRUCTURE_BONUS = 2
    score_a = sum(markup_richness_score(e.html) for e in elements_a) + sum(
        _SECONDARY_STRUCTURE_BONUS for e in elements_a if isinstance(e, (Image, Footnote))
    )
    score_b = sum(markup_richness_score(e.html) for e in elements_b) + sum(
        _SECONDARY_STRUCTURE_BONUS for e in elements_b if isinstance(e, (Image, Footnote))
    )

    # Check prefer_source for the first element's type (best effort for M:N)
    if elements_a:
        etype = ElementTypeEnum(elements_a[0].element_type)
        pref = _effective_prefer(etype, prefer_source)
        if pref == "a":
            return elements_a, f"prefer_source='a' for {etype} (M:N group)"
        if pref == "b":
            return elements_b, f"prefer_source='b' for {etype} (M:N group)"

    if score_b > score_a:
        return elements_b, f"higher markup richness in B (score_b={score_b} > score_a={score_a})"
    return elements_a, f"higher or equal markup richness in A (score_a={score_a} >= score_b={score_b})"


def align_elements(  # noqa: C901  (inherently branchy alignment/merge algorithm)
    doc_a: Document,
    doc_b: Document,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None = None,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum] = True,
    similarity_threshold: float = 0.6,
) -> list[PatchOperation]:
    """Align elements from two documents and produce a list of PatchOperations.

    Uses ``difflib.SequenceMatcher`` on normalized text keys of primary elements.
    Secondary elements (Image, Footnote) trail their preceding primary anchor.
    All ``resolved_elements`` fields are populated by heuristics.

    Args:
        doc_a: Base document (typically the first parser's output).
        doc_b: Overlay document (typically the second parser's output).
        prefer_source: Per-type source preference.  ``{"heading": "b"}`` forces
            all headings to come from doc_b.
        allow_insertions_from_b: Whether to include elements that appear only in
            doc_b.  Pass a ``frozenset`` to allow only specific types.
        similarity_threshold: Minimum ``SequenceMatcher`` ratio to treat a
            ``replace`` opcode as MERGE rather than DELETE_A + INSERT_B.

    Returns:
        Ordered list of ``PatchOperation`` objects with ``resolved_elements``
        already populated.
    """
    groups_a = _group_with_trailing(doc_a.elements)
    groups_b = _group_with_trailing(doc_b.elements)

    keys_a = [alignment_key(anchor) for anchor, _ in groups_a]
    keys_b = [alignment_key(anchor) for anchor, _ in groups_b]

    parser_a = doc_a.parser
    parser_b = doc_b.parser

    matcher = difflib.SequenceMatcher(None, keys_a, keys_b, autojunk=False)
    operations: list[PatchOperation] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            for idx_a, idx_b in zip(range(i1, i2), range(j1, j2)):
                anchor_a, trailing_a = groups_a[idx_a]
                anchor_b, trailing_b = groups_b[idx_b]
                etype = ElementTypeEnum(anchor_a.element_type)
                pref = _effective_prefer(etype, prefer_source)

                elems_a = [anchor_a, *trailing_a]
                elems_b = [anchor_b, *trailing_b]

                if pref == "b":
                    op_type = PatchOperationType.KEEP_B
                    resolved = elems_b
                    reason = f"prefer_source='b' for {etype}"
                elif pref == "a":
                    op_type = PatchOperationType.KEEP_A
                    resolved = elems_a
                    reason = f"prefer_source='a' for {etype}"
                else:
                    # Same text but HTML may differ — apply heuristics to pick
                    # the richer/more accurate version
                    resolved, reason = _apply_heuristics(elems_a, elems_b, parser_a, parser_b, prefer_source)
                    if resolved == elems_a:
                        op_type = PatchOperationType.KEEP_A
                    elif resolved == elems_b:
                        op_type = PatchOperationType.KEEP_B
                    else:
                        op_type = PatchOperationType.MERGE

                operations.append(
                    PatchOperation(
                        op=op_type,
                        elements_a=elems_a,
                        elements_b=elems_b,
                        resolved_elements=resolved,
                        reason=reason,
                    )
                )

        elif opcode == "replace":
            # Check if the groups are similar enough to MERGE rather than split
            group_keys_a = keys_a[i1:i2]
            group_keys_b = keys_b[j1:j2]
            ratio = difflib.SequenceMatcher(None, " ".join(group_keys_a), " ".join(group_keys_b)).ratio()

            elems_a = [e for idx in range(i1, i2) for e in [groups_a[idx][0]] + groups_a[idx][1]]
            elems_b = [e for idx in range(j1, j2) for e in [groups_b[idx][0]] + groups_b[idx][1]]

            len_a = i2 - i1
            len_b = j2 - j1

            if ratio >= similarity_threshold or (len_a == len_b == 1):
                if len_a == len_b:
                    # Equal-length groups: 1:1 element-wise merge
                    for idx_a, idx_b in zip(range(i1, i2), range(j1, j2)):
                        anchor_a, trailing_a = groups_a[idx_a]
                        anchor_b, trailing_b = groups_b[idx_b]
                        ea = [anchor_a, *trailing_a]
                        eb = [anchor_b, *trailing_b]
                        resolved_pair, reason_pair = _apply_heuristics(ea, eb, parser_a, parser_b, prefer_source)
                        if resolved_pair == ea:
                            op_type = PatchOperationType.KEEP_A if ratio > 0.95 else PatchOperationType.MERGE
                        elif resolved_pair == eb:
                            op_type = PatchOperationType.KEEP_B if ratio > 0.95 else PatchOperationType.MERGE
                        else:
                            op_type = PatchOperationType.MERGE
                        operations.append(
                            PatchOperation(
                                op=op_type,
                                elements_a=ea,
                                elements_b=eb,
                                resolved_elements=resolved_pair,
                                reason=reason_pair,
                            )
                        )
                elif len_a > len_b:
                    # More doc_a elements than doc_b: pairwise match each a-element
                    # to its closest b-element. Unmatched a-elements are preserved
                    # (they represent unique doc_a content within the aligned region).
                    matched_b: set[int] = set()
                    a_to_b: dict[int, int] = {}
                    for idx_a in range(i1, i2):
                        best_ratio = 0.0
                        best_b_idx: int | None = None
                        for idx_b in range(j1, j2):
                            if idx_b in matched_b:
                                continue
                            r = difflib.SequenceMatcher(None, keys_a[idx_a], keys_b[idx_b]).ratio()
                            if r > best_ratio:
                                best_ratio = r
                                best_b_idx = idx_b
                        if best_b_idx is not None and best_ratio >= similarity_threshold:
                            a_to_b[idx_a] = best_b_idx
                            matched_b.add(best_b_idx)

                    for idx_a in range(i1, i2):
                        anchor_a, trailing_a = groups_a[idx_a]
                        ea = [anchor_a, *trailing_a]
                        if idx_a in a_to_b:
                            idx_b = a_to_b[idx_a]
                            anchor_b, trailing_b = groups_b[idx_b]
                            eb = [anchor_b, *trailing_b]
                            resolved_pair, reason_pair = _apply_heuristics(ea, eb, parser_a, parser_b, prefer_source)
                            # When doc_b wins a 1:1 primary pair, carry inline markup
                            # from doc_a's element into the winner so formatting from
                            # doc_a (e.g. <strong>) is not silently discarded.
                            if resolved_pair == eb and len(ea) == 1 and len(eb) == 1:
                                enhanced = _with_injected_markup(eb[0], ea[0])
                                if enhanced is not eb[0]:
                                    resolved_pair = [enhanced, *eb[1:]]
                            if resolved_pair == ea:
                                op_type = PatchOperationType.KEEP_A
                            elif resolved_pair == eb:
                                op_type = PatchOperationType.KEEP_B
                            else:
                                op_type = PatchOperationType.MERGE
                            operations.append(
                                PatchOperation(
                                    op=op_type,
                                    elements_a=ea,
                                    elements_b=eb,
                                    resolved_elements=resolved_pair,
                                    reason=reason_pair,
                                )
                            )
                        else:
                            # No matching b-element: preserve this unique doc_a element
                            operations.append(
                                PatchOperation(
                                    op=PatchOperationType.DELETE_A,
                                    elements_a=ea,
                                    resolved_elements=ea,
                                    reason="doc_a element in replace group with no close doc_b counterpart; preserving",
                                )
                            )
                    # Insert unmatched b-elements if allowed
                    for idx_b in range(j1, j2):
                        if idx_b not in matched_b:
                            anchor_b, trailing_b = groups_b[idx_b]
                            elems_b_single = [anchor_b, *trailing_b]
                            if _is_insertion_allowed(anchor_b, allow_insertions_from_b):
                                operations.append(
                                    PatchOperation(
                                        op=PatchOperationType.INSERT_B,
                                        elements_b=elems_b_single,
                                        resolved_elements=elems_b_single,
                                        reason="doc_b element in replace group with no doc_a counterpart; inserting",
                                    )
                                )
                else:
                    # More doc_b elements than doc_a: whole-block M:N merge
                    resolved, reason = _apply_heuristics(elems_a, elems_b, parser_a, parser_b, prefer_source)
                    if resolved == elems_a:
                        op_type = PatchOperationType.KEEP_A if ratio > 0.95 else PatchOperationType.MERGE
                    elif resolved == elems_b:
                        op_type = PatchOperationType.KEEP_B if ratio > 0.95 else PatchOperationType.MERGE
                    else:
                        op_type = PatchOperationType.MERGE
                    operations.append(
                        PatchOperation(
                            op=op_type,
                            elements_a=elems_a,
                            elements_b=elems_b,
                            resolved_elements=resolved,
                            reason=reason,
                        )
                    )
            else:
                # Too different → DELETE_A + INSERT_B
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.DELETE_A,
                        elements_a=elems_a,
                        resolved_elements=[],
                        reason=f"low similarity ratio={ratio:.2f}; dropping doc_a group",
                    )
                )
                for idx in range(j1, j2):
                    anchor_b, trailing_b = groups_b[idx]
                    elems_b_single = [anchor_b, *trailing_b]
                    if all(_is_insertion_allowed(e, allow_insertions_from_b) for e in elems_b_single):
                        operations.append(
                            PatchOperation(
                                op=PatchOperationType.INSERT_B,
                                elements_b=elems_b_single,
                                resolved_elements=elems_b_single,
                                reason="low similarity; inserting doc_b group",
                            )
                        )

        elif opcode == "insert":
            # Elements only in doc_b
            for idx in range(j1, j2):
                anchor_b, trailing_b = groups_b[idx]
                if _is_insertion_allowed(anchor_b, allow_insertions_from_b):
                    # Anchor is allowed → insert anchor + all trailing elements
                    elems_b = [anchor_b, *trailing_b]
                    operations.append(
                        PatchOperation(
                            op=PatchOperationType.INSERT_B,
                            elements_b=elems_b,
                            resolved_elements=elems_b,
                            reason="element only in doc_b",
                        )
                    )
                else:
                    # Anchor blocked → check each trailing element individually
                    for trailing_elem in trailing_b:
                        if _is_insertion_allowed(trailing_elem, allow_insertions_from_b):
                            operations.append(
                                PatchOperation(
                                    op=PatchOperationType.INSERT_B,
                                    elements_b=[trailing_elem],
                                    resolved_elements=[trailing_elem],
                                    reason="trailing secondary element only in doc_b (anchor blocked)",
                                )
                            )

        elif opcode == "delete":
            # Elements only in doc_a
            for idx in range(i1, i2):
                anchor_a, trailing_a = groups_a[idx]
                elems_a = [anchor_a, *trailing_a]
                operations.append(
                    PatchOperation(
                        op=PatchOperationType.DELETE_A,
                        elements_a=elems_a,
                        resolved_elements=[],
                        reason="element only in doc_a; no counterpart in doc_b",
                    )
                )

    return operations
