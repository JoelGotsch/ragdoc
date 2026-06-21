"""DocumentHTMLPatch — inspectable intermediate for Approach B (render-merge-reparse).

Analogous to ``DocumentPatch`` in Approach A, but operates at the HTML tag level.
Callers can inspect and override the ``selected`` list on any operation before
calling ``apply()`` to obtain the merged ``Document``.

Example::

    from ragdoc.merging.html_patch import compute_html_patch

    patch = compute_html_patch(doc_mineru, doc_html)
    for op in patch.operations:
        print(op.opcode, op.selected)

    # Override a specific block before applying
    patch.operations[2].selected = ["<h2>Custom heading</h2>"]
    merged = patch.apply()
"""

from __future__ import annotations

import difflib
from typing import Literal

from pydantic import BaseModel, Field

from ragdoc.document import Document, ElementTypeEnum
from ragdoc.merging.heuristics import has_heading_hierarchy, markup_richness_score
from ragdoc.merging.html_merge import (
    _alignment_key,
    _block_ratio,
    _extract_top_level_tags,
    _is_insertion_allowed,
    _select_tag,
    _tag_to_element_type,
)
from ragdoc.parsing.html.load import HTML, generate_document
from ragdoc.rendering import OutputFormat, Renderer, render_raw


class HtmlMergeOperation(BaseModel):
    """A single HTML-level merge operation.

    Attributes:
        opcode: The difflib opcode for this block.
        tags_a: HTML strings from doc_a's side of the alignment.
        tags_b: HTML strings from doc_b's side of the alignment.
        selected: The HTML strings chosen for the merged output.
            Mutable — override before calling ``DocumentHtmlPatch.apply()``.
    """

    opcode: Literal["equal", "replace", "insert", "delete"] = Field(
        ..., description="difflib opcode for this alignment block."
    )
    tags_a: list[str] = Field(
        default_factory=list,
        description="HTML strings from doc_a participating in this block.",
    )
    tags_b: list[str] = Field(
        default_factory=list,
        description="HTML strings from doc_b participating in this block.",
    )
    selected: list[str] = Field(
        default_factory=list,
        description="HTML strings chosen for the merged output. Override to customise.",
    )


class DocumentHtmlPatch(BaseModel):
    """HTML-level patch produced by ``compute_html_patch()``.

    All ``selected`` fields are pre-populated by the merge heuristics.
    Callers may override individual ``op.selected`` lists before calling
    ``apply()``.

    The model is JSON-serializable via ``.model_dump_json()``.

    Limitations vs ``DocumentPatch`` (Approach A):
    - Operations are HTML strings, not typed element objects.
    - No per-operation ``reason`` string.
    - Page / bounding-box metadata is not preserved (architectural limitation
      of the render-reparse pipeline).  Use Approach A when metadata
      preservation is required.
    """

    operations: list[HtmlMergeOperation] = Field(
        default_factory=list,
        description="Ordered list of HTML merge operations.",
    )
    source_parser_a: str | None = Field(
        default=None,
        description="Parser name that produced document A.",
    )
    source_parser_b: str | None = Field(
        default=None,
        description="Parser name that produced document B.",
    )

    def apply(self) -> Document:
        """Materialise the patch into a merged ``Document``.

        Concatenates all ``op.selected`` HTML strings in order, reparses via
        ``generate_document()``, and sets ``parser="merged"`` plus
        ``metadata`` with the source parser names.

        Returns:
            A new ``Document`` with ``parser="merged"``.
        """
        merged_html = "".join(s for op in self.operations for s in op.selected)
        doc = generate_document(HTML(content=merged_html))
        doc.parser = "merged"
        doc.metadata = {
            "source_parser_a": self.source_parser_a,
            "source_parser_b": self.source_parser_b,
        }
        return doc


def compute_html_patch(
    doc_a: Document,
    doc_b: Document,
    prefer_source: dict[ElementTypeEnum, Literal["a", "b"]] | None = None,
    allow_insertions_from_b: bool | frozenset[ElementTypeEnum] = True,
    similarity_threshold: float = 0.6,
) -> DocumentHtmlPatch:
    """Align two Documents and return a ``DocumentHtmlPatch`` without applying it.

    The patch records every HTML-level alignment decision.  All ``selected``
    fields are pre-populated by heuristics.  Callers may override
    ``op.selected`` on any operation before calling ``patch.apply()``.

    Args:
        doc_a: Base document.
        doc_b: Overlay document.
        prefer_source: Per-type source preference (same as ``merge_documents_html``).
        allow_insertions_from_b: Insertion control (same as ``merge_documents_html``).
        similarity_threshold: Minimum ratio for treating a replace block as a
            merge rather than delete+insert.

    Returns:
        A populated ``DocumentHtmlPatch``.
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
    operations: list[HtmlMergeOperation] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        ta_strs = [str(t) for t in tags_a[i1:i2]]
        tb_strs = [str(t) for t in tags_b[j1:j2]]

        if opcode == "equal":
            selected: list[str] = [
                str(_select_tag(tags_a[idx_a], tags_b[idx_b], prefer_source, hier_a, hier_b))
                for idx_a, idx_b in zip(range(i1, i2), range(j1, j2))
            ]
            operations.append(HtmlMergeOperation(opcode="equal", tags_a=ta_strs, tags_b=tb_strs, selected=selected))

        elif opcode == "replace":
            group_a = tags_a[i1:i2]
            group_b = tags_b[j1:j2]
            ratio = _block_ratio(keys_a[i1:i2], keys_b[j1:j2])

            if ratio >= similarity_threshold or (ratio > 0 and i2 - i1 == j2 - j1 == 1):
                if len(group_a) == len(group_b):
                    selected = [
                        str(_select_tag(ta, tb, prefer_source, hier_a, hier_b)) for ta, tb in zip(group_a, group_b)
                    ]
                else:
                    score_a = sum(markup_richness_score(str(t)) for t in group_a)
                    score_b = sum(markup_richness_score(str(t)) for t in group_b)
                    etype = _tag_to_element_type(group_a[0]) if group_a else None
                    if etype is not None and prefer_source is not None:
                        pref = prefer_source.get(etype)
                        if pref == "a":
                            selected = ta_strs
                        elif pref == "b":
                            selected = tb_strs
                        else:
                            selected = tb_strs if score_b >= score_a else ta_strs
                    else:
                        selected = tb_strs if score_b >= score_a else ta_strs
            else:
                selected = [str(tb) for tb in group_b if _is_insertion_allowed(tb, allow_insertions_from_b)]
            operations.append(HtmlMergeOperation(opcode="replace", tags_a=ta_strs, tags_b=tb_strs, selected=selected))

        elif opcode == "insert":
            selected = [str(tb) for tb in tags_b[j1:j2] if _is_insertion_allowed(tb, allow_insertions_from_b)]
            operations.append(HtmlMergeOperation(opcode="insert", tags_a=[], tags_b=tb_strs, selected=selected))

        elif opcode == "delete":
            operations.append(HtmlMergeOperation(opcode="delete", tags_a=ta_strs, tags_b=[], selected=ta_strs))

    return DocumentHtmlPatch(
        operations=operations,
        source_parser_a=doc_a.parser,
        source_parser_b=doc_b.parser,
    )
