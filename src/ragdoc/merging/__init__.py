"""Document merging — intelligent merge of two Document objects.

Two independent approaches are provided:

**Approach A — Element-alignment (with DocumentPatch)**
    Uses ``difflib.SequenceMatcher`` to align primary elements by text
    similarity, applies heuristics to select the richer/more accurate element
    per aligned pair, and returns a ``DocumentPatch`` that the caller can
    inspect and override before applying.

    - Preserves all element structure (Footnote, Image, page metadata).
    - Produces a JSON-serializable ``DocumentPatch`` for external inspection.

**Approach B — Render-merge-reparse**
    Renders both documents to HTML via ``render_raw``, aligns top-level HTML
    elements, selects the richer element for each aligned pair, then reparses
    the merged HTML into a new Document.

    - Simpler; reuses the existing rendering + HTML parsing pipeline.
    - ``Footnote`` structure is lost (recoverable by running ``FootnoteProcessor``
      afterward).  Page / bounding-box metadata is also lost.

Both approaches accept the same per-type control parameters:

- ``prefer_source``: ``dict[ElementTypeEnum, Literal["a", "b"]]`` — choose which
  document's version to prefer for each element type.
- ``allow_insertions_from_b``: ``bool | frozenset[ElementTypeEnum]`` — control
  whether elements that appear only in doc_b are included.

Example::

    from ragdoc.merging import compute_patch, merge_documents, merge_documents_html
    from ragdoc.document import ElementTypeEnum

    # Approach A — inspect the patch before applying
    patch = compute_patch(
        doc_mineru, doc_html,
        prefer_source={ElementTypeEnum.HEADING: "b"},
    )
    for op in patch.operations:
        print(op.op, op.reason)
    merged = patch.apply()

    # Approach A — convenience wrapper
    merged = merge_documents(doc_mineru, doc_html)

    # Approach B — render-merge-reparse
    merged = merge_documents_html(doc_mineru, doc_html)
"""
from ragdoc.merging.html_merge import merge_documents_html
from ragdoc.merging.html_patch import DocumentHtmlPatch, HtmlMergeOperation, compute_html_patch
from ragdoc.merging.merge import compute_patch, merge_documents
from ragdoc.merging.patch import (
    DocumentPatch,
    PatchOperation,
    PatchOperationType,
    validate_inline_refs,
)

__all__ = [
    # Approach A
    "compute_patch",
    "merge_documents",
    "DocumentPatch",
    "PatchOperation",
    "PatchOperationType",
    "validate_inline_refs",
    # Approach B
    "merge_documents_html",
    "compute_html_patch",
    "DocumentHtmlPatch",
    "HtmlMergeOperation",
]
