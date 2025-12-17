from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ragdoc.document import Document, ExternalRef, Heading

logger = logging.getLogger(__name__)


@runtime_checkable
class Splitter(Protocol):
    """Protocol for document splitters."""

    def __call__(self, document: Document) -> list[Document]:
        pass


def split_by_headings(document: Document) -> list[Document]:
    """Split a document at each heading boundary into separate Documents.

    A new document begins whenever a Heading is encountered and the current
    accumulation is not *only* headings (to avoid empty heading-only fragments).

    Parent-child relationships: every resulting document gets an ExternalRef
    pointing to the source document as its parent.

    Args:
        document: The document to split.

    Returns:
        A list of Documents. If the document has no headings, a single-element
        list containing the original document is returned unchanged.

    Note:
        Does not set ``split_sequence`` / ``split_total`` in metadata. Call
        :func:`ragdoc.splitting.token.split_document` for reading-order numbering.
    """
    split_groups: list[list] = []
    current: list = []

    for element in document.elements:
        if isinstance(element, Heading) and current and not all(isinstance(e, Heading) for e in current):
            split_groups.append(current)
            current = [element]
        else:
            current.append(element)

    if current:
        split_groups.append(current)

    if len(split_groups) <= 1:
        return [document]

    logger.debug(f"split_by_headings: split into {len(split_groups)} sections")

    parent_ref = ExternalRef(target_id=document.id, rel_type="external-parent")
    return [
        Document(
            elements=group,
            source_path=document.source_path,
            metadata=document.metadata,
            title=document.title,
            external_refs=[parent_ref],
        )
        for group in split_groups
    ]
