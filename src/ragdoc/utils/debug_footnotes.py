"""Utility for inspecting unreferenced (orphan) footnotes in a Document."""

from pydantic import BaseModel, Field

from ragdoc.document import Document, Footnote, Heading


class OrphanFootnoteInfo(BaseModel):
    number: int = Field(description="Footnote number as it appears in the document.")
    text: str = Field(description="Plain text content of the footnote.")
    page: int | None = Field(description="Page number, or None if not set.")
    preceding_heading: str | None = Field(
        description="Plain text of the nearest heading before the footnote, or None."
    )


def _preceding_heading(document: Document, footnote: Footnote) -> str | None:
    """Return the plain text of the nearest Heading before *footnote* in element order."""
    idx: int | None = None
    for i, el in enumerate(document.elements):
        if el.id == footnote.id:
            idx = i
            break
    if idx is None:
        return None
    for el in reversed(document.elements[:idx]):
        if isinstance(el, Heading):
            return el.text
    return None


def describe_orphan_footnotes(document: Document) -> list[OrphanFootnoteInfo]:
    """Return info for each orphan footnote in *document*, sorted by number."""
    orphans = document.orphaned_footnotes
    infos = [
        OrphanFootnoteInfo(
            number=orphan.number,
            text=orphan.text,
            page=orphan.page if orphan.page else None,
            preceding_heading=_preceding_heading(document, orphan),
        )
        for orphan in orphans
    ]
    return sorted(infos, key=lambda info: info.number)


def describe_orphan_footnotes_batch(
    documents: list[Document],
) -> list[tuple[Document, list[OrphanFootnoteInfo]]]:
    """Return ``(document, orphan_infos)`` pairs for documents that have orphan footnotes."""
    results: list[tuple[Document, list[OrphanFootnoteInfo]]] = []
    for doc in documents:
        infos = describe_orphan_footnotes(doc)
        if infos:
            results.append((doc, infos))
    return results
