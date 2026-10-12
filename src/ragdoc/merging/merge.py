"""Approach A: element-alignment merge (compute_patch / merge_documents).

This module provides the high-level API for the element-alignment approach.
``compute_patch()`` returns a ``DocumentPatch`` whose ``resolved_elements``
are already populated by heuristics.  Callers may inspect and override
individual operations before calling ``patch.apply()``.
"""

from __future__ import annotations

from typing import Literal

from ragdoc.document import Document, ElementTypeEnum
from ragdoc.merging.align import align_elements
from ragdoc.merging.patch import DocumentPatch


def compute_patch(
    doc_a: Document,
    doc_b: Document,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None = None,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum] = True,
    similarity_threshold: float = 0.6,
) -> DocumentPatch:
    """Align two documents and return a ``DocumentPatch`` without applying it.

    All ``PatchOperation.resolved_elements`` fields are populated by heuristics.
    Callers may inspect ``patch.operations``, override ``resolved_elements`` on
    any operation, then call ``patch.apply()`` to obtain the merged Document.

    Args:
        doc_a: Base document (typically the first parser's output).
        doc_b: Overlay document (typically the second parser's output).
        prefer_source: Per-type source preference.  Keys are ``ElementTypeEnum``
            values; values are ``"a"`` or ``"b"``.  Example::

                {ElementTypeEnum.HEADING: "b"}  # always take headings from doc_b

        allow_insertions_from_b: Whether to include elements that appear only in
            doc_b.  Pass ``False`` to suppress all insertions, or a
            ``frozenset`` to allow only specific element types.
        similarity_threshold: Minimum ``SequenceMatcher`` ratio to treat a
            mismatched group as MERGE rather than DELETE_A + INSERT_B.

    Returns:
        A ``DocumentPatch`` with ``source_parser_a/b`` set and all operations
        populated.
    """
    operations = align_elements(
        doc_a,
        doc_b,
        prefer_source=prefer_source,
        allow_insertions_from_b=allow_insertions_from_b,
        similarity_threshold=similarity_threshold,
    )
    return DocumentPatch(
        source_parser_a=doc_a.parser,
        source_parser_b=doc_b.parser,
        operations=operations,
    )


def merge_documents(
    doc_a: Document,
    doc_b: Document,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None = None,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum] = True,
    similarity_threshold: float = 0.6,
) -> Document:
    """Merge two Documents using element-alignment heuristics.

    Convenience wrapper around ``compute_patch(...).apply()``.  Use
    ``compute_patch()`` directly if you need to inspect or override the patch
    before applying it.

    Args:
        doc_a: Base document.
        doc_b: Overlay document.
        prefer_source: Per-type source preference (see ``compute_patch``).
        allow_insertions_from_b: Insertion control (see ``compute_patch``).
        similarity_threshold: Similarity threshold (see ``compute_patch``).

    Returns:
        A new merged ``Document`` with ``parser="merged"``.
    """
    return compute_patch(
        doc_a,
        doc_b,
        prefer_source=prefer_source,
        allow_insertions_from_b=allow_insertions_from_b,
        similarity_threshold=similarity_threshold,
    ).apply()
