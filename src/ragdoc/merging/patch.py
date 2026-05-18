"""DocumentPatch — structured merge result for Approach A (element-alignment).

A DocumentPatch records the ordered list of PatchOperation objects produced by
compute_patch(). Each operation holds the elements from both sides plus the
heuristically-resolved result. Users may inspect and override
``operation.resolved_elements`` on any operation before calling ``apply()``.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ragdoc.document import Document, ElementType


class PatchOperationType(str, Enum):
    """Type of merge operation for an aligned element group."""

    KEEP_A = "keep_a"
    """Use elements from document A unchanged (identical text or prefer_source="a")."""

    KEEP_B = "keep_b"
    """Use elements from document B unchanged (prefer_source="b" for this type)."""

    MERGE = "merge"
    """Heuristics selected the winning element(s); resolved_elements is populated."""

    INSERT_B = "insert_b"
    """Elements only in document B; included per allow_insertions_from_b."""

    DELETE_A = "delete_a"
    """Elements only in document A with no counterpart in B; dropped from output."""


class PatchOperation(BaseModel):
    """A single merge decision for one aligned group of elements."""

    op: PatchOperationType = Field(
        ...,
        description="Type of merge operation for this aligned group.",
    )
    elements_a: list[ElementType] = Field(
        default_factory=list,
        description="Elements from document A participating in this operation "
        "(empty for INSERT_B).",
    )
    elements_b: list[ElementType] = Field(
        default_factory=list,
        description="Elements from document B participating in this operation "
        "(empty for DELETE_A).",
    )
    resolved_elements: list[ElementType] = Field(
        default_factory=list,
        description="Final elements produced after applying heuristics. "
        "Populated by compute_patch(); override before apply() to customise the output. "
        "Must be non-empty for MERGE operations — use an empty list only for DELETE_A.",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of why this operation was chosen.",
    )


class DocumentPatch(BaseModel):
    """Structured merge result between two Document objects.

    Produced by ``compute_patch()``; all ``resolved_elements`` fields are
    populated by heuristics. Users may override individual operations before
    calling ``apply()``.

    The model is JSON-serializable via ``.model_dump_json()``.
    """

    source_parser_a: str | None = Field(
        default=None,
        description="Parser name that produced document A (e.g. 'mineru').",
    )
    source_parser_b: str | None = Field(
        default=None,
        description="Parser name that produced document B (e.g. 'azure_di').",
    )
    operations: list[PatchOperation] = Field(
        default_factory=list,
        description="Ordered list of merge operations. Applying them in order "
        "reconstructs the merged document.",
    )

    def apply(self) -> Document:
        """Materialise the patch into a new Document.

        Concatenates ``resolved_elements`` from every operation in order.
        Duplicate element IDs are silently deduplicated (first occurrence wins).

        Raises:
            ValueError: If any MERGE operation has an empty ``resolved_elements``
                list (indicates heuristics have not been applied or the caller
                forgot to set a resolved result).
        """
        for op in self.operations:
            if op.op == PatchOperationType.MERGE and not op.resolved_elements:
                raise ValueError(
                    f"MERGE operation has empty resolved_elements. "
                    f"Call compute_patch() to populate heuristics or set "
                    f"resolved_elements manually before calling apply(). "
                    f"elements_a={[e.id for e in op.elements_a]}, "
                    f"elements_b={[e.id for e in op.elements_b]}"
                )

        seen_ids: set[str] = set()
        merged: list[ElementType] = []
        for op in self.operations:
            for element in op.resolved_elements:
                if element.id not in seen_ids:
                    seen_ids.add(element.id)
                    merged.append(element)

        return Document(
            elements=merged,
            parser="merged",
            metadata={
                "source_parser_a": self.source_parser_a,
                "source_parser_b": self.source_parser_b,
            },
        )


def validate_inline_refs(doc: Document) -> list[str]:
    """Return a list of broken inline-ref IDs in the document.

    Broken means a ``<ref id="..."/>`` tag in some element's HTML points to an
    element ID that does not exist in ``doc.elements``.

    Args:
        doc: Document to validate.

    Returns:
        List of ref target IDs that have no matching element.  Empty list means
        all inline refs are resolved.
    """
    element_ids: set[str] = {e.id for e in doc.elements}
    broken: list[str] = []
    for element in doc.elements:
        for ref in element.inline_refs:
            if ref.target_id not in element_ids:
                broken.append(ref.target_id)
    return broken
