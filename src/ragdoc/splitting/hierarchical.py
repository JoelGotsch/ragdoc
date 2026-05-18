
from ragdoc.document import Document, ExternalRef, Heading
from ragdoc.splitting.groups import ElementGroup, build_element_groups


def split_hierarchical(document: Document) -> list[Document]:
    """Split a document at a chosen heading level, propagating parent context into each split.

    Finds the lowest heading level that produces at least 2 splits, then splits the
    document at every heading at exactly that level. Parent headings (levels numerically
    lower than the split level) are prepended to each resulting chunk as context, so
    every split document is self-contained.

    Context propagation rules:
    - A parent heading encountered *before* the first split-level heading becomes
      context for all chunks (it is prepended but not included in the raw chunk body).
    - A parent heading encountered *inside* a chunk's body is kept in that chunk's
      body AND updates the context for all subsequent chunks (using the most recent
      heading seen at each level).
    - Context headings are prepended in their original document order.

    If no split can be found, a list with the original document is returned.

    ExternalRef: each split document receives an "external-parent" ref pointing to
    the original document's id.

    Ref-awareness: elements are partitioned into ElementGroups before splitting.
    A referenced element (e.g. a Footnote) always travels with the root that references
    it, even when the referenced element is physically positioned after a split-level
    heading in the original document.

    Examples (heading levels in document order):
        [2, 1, 3, 2, 3, 4, 4, 3] → split_level=2 → [[2,1,3], [1,2,3,4,4,3]]
        [2, 1, 3, 3, 3] → split_level=3 → [[2,1,3], [2,1,3], [2,1,3]]  (h2 and h1 duplicated as context)
        [1, 2, 3, 3, 2, 3] → split_level=2 → [[1,2,3,3], [1,2,3]]
        [1, 2, 3] → no split (only one heading at each level)
        [2, 1, 3, 3, 2] → split_level=2 → [[2,1,3,3], [1,2]]

    Args:
        document: The document to split.

    Returns:
        A list of Documents with ExternalRef relationships set.
        Returns a single-element list if no meaningful split is possible.

    Note:
        Does not set ``split_sequence`` / ``split_total`` in metadata. Call
        :func:`ragdoc.splitting.token.split_document` for reading-order numbering.
    """
    heading_levels = [h.level for h in document.headings]
    if not heading_levels:
        return [document]

    split_level = min(heading_levels)
    while heading_levels.count(split_level) < 2:
        split_level += 1
        if split_level > max(heading_levels):
            return [document]

    # Scan element groups in document order:
    # - Headings at level < split_level are "parent context" headings.
    #   Before the first split-level heading they only update context (preamble).
    #   Inside a chunk they are included in that chunk's body AND update context.
    # - Headings at level == split_level are split points: they close the current
    #   chunk and start a new one.
    # - Everything else (non-headings, headings at level > split_level) goes into
    #   the current chunk, or into preamble_extras if no split has been seen yet.
    # - ElementGroups carry root + all transitively-referenced elements together,
    #   so a footnote physically past a heading boundary follows its referencing paragraph.

    parent_context: dict[int, Heading] = {}   # level → most recent heading
    context_level_order: list[int] = []        # levels in document order of first appearance

    preamble_extras: list = []  # non-context elements before first split-level heading
    current_chunk: list | None = None
    current_ctx_snapshot: list[Heading] = []
    raw_chunks: list[tuple[list, list[Heading]]] = []

    for item in build_element_groups(document.elements):
        if isinstance(item, Heading):
            if item.level == split_level:
                if current_chunk is not None:
                    raw_chunks.append((current_chunk, current_ctx_snapshot))
                current_ctx_snapshot = [parent_context[lvl] for lvl in context_level_order]
                current_chunk = [item]
            elif item.level < split_level:
                if item.level not in context_level_order:
                    context_level_order.append(item.level)
                parent_context[item.level] = item
                if current_chunk is not None:
                    current_chunk.append(item)
                # else: preamble parent heading — context only, not added to any chunk body
            else:
                # Heading at level > split_level: child heading, treated as content
                if current_chunk is not None:
                    current_chunk.append(item)
                else:
                    preamble_extras.append(item)
        else:
            # ElementGroup: root + all referenced elements travel together
            assert isinstance(item, ElementGroup)
            if current_chunk is not None:
                current_chunk.extend(item.all_elements)
            else:
                preamble_extras.extend(item.all_elements)

    if current_chunk is not None:
        raw_chunks.append((current_chunk, current_ctx_snapshot))

    if len(raw_chunks) <= 1:
        return [document]

    split_docs: list[Document] = []
    for i, (elements, ctx) in enumerate(raw_chunks):
        prefix = preamble_extras if i == 0 else []
        all_elements = ctx + prefix + elements
        doc = Document(elements=all_elements, title=document.title, source_path=document.source_path, metadata=document.metadata)
        split_docs.append(doc)

    parent_ref = ExternalRef(target_id=document.id, rel_type="external-parent")
    for doc in split_docs:
        doc.external_refs.append(parent_ref)

    return split_docs
