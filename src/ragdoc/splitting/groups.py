"""Element group construction for ref-aware splitting.

An ElementGroup is a root element plus every element it transitively references
via inline <ref> tags.  The group is the atomic unit for splitting: referenced
elements must never end up in a different chunk from the root that references them.

If two roots reference the same element it is duplicated into both groups —
downstream consumers work with Chunks (not Documents) so duplicate
element IDs in intermediate Documents are harmless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragdoc.document import BaseElement, Heading


@dataclass
class ElementGroup:
    """A root element plus all elements it transitively references via inline_refs.

    Attributes:
        root: The primary (non-heading, non-referenced) element.
        referenced: All elements reachable from root through inline_refs,
            in document order.  An element may appear in multiple groups when
            it is referenced by more than one root.
    """

    root: BaseElement
    referenced: list[BaseElement] = field(default_factory=list)

    @property
    def all_elements(self) -> list[BaseElement]:
        """root followed by referenced elements in document order."""
        return [self.root] + self.referenced


def build_element_groups(elements: list[BaseElement]) -> list[Heading | ElementGroup]:
    """Partition a flat element list into headings and atomic ElementGroups.

    Rules:
    - Headings are returned as-is (standalone context elements, not groups).
    - Non-heading elements that are NOT the target of any other element's
      inline_refs become *roots* — each root heads one ElementGroup.
    - Non-heading elements that ARE targeted by another element's inline_refs
      are *referenced-only*; they appear inside their referencing root's group
      and are NOT emitted as standalone items in the result.
    - Transitive references are followed (BFS): if A → B → C, then C ends up
      in A's group even if B is itself referenced-only.
    - Document order is preserved throughout: groups appear in root order,
      referenced elements within a group are sorted by original document position.
    - An element referenced by multiple roots is duplicated into all their groups.

    Args:
        elements: Flat list of document elements in reading order.

    Returns:
        List of Headings and ElementGroups in reading order.
    """
    el_by_id: dict[str, BaseElement] = {el.id: el for el in elements}
    refs_to: dict[str, list[str]] = {
        el.id: [ref.target_id for ref in el.inline_refs] for el in elements
    }
    all_referenced_ids: set[str] = {
        target_id for targets in refs_to.values() for target_id in targets
    }
    doc_order: dict[str, int] = {el.id: i for i, el in enumerate(elements)}

    def _collect_refs(root_id: str) -> list[BaseElement]:
        """BFS transitive closure of elements reachable from root_id."""
        visited: set[str] = set()
        queue = list(refs_to.get(root_id, []))
        result: list[BaseElement] = []
        while queue:
            ref_id = queue.pop(0)
            if ref_id in visited:
                continue
            visited.add(ref_id)
            ref_el = el_by_id.get(ref_id)
            if ref_el is not None:
                result.append(ref_el)
                queue.extend(refs_to.get(ref_id, []))
        result.sort(key=lambda e: doc_order.get(e.id, 0))
        return result

    result: list[Heading | ElementGroup] = []
    for el in elements:
        if isinstance(el, Heading):
            result.append(el)
        elif el.id not in all_referenced_ids:
            result.append(ElementGroup(root=el, referenced=_collect_refs(el.id)))
        # else: referenced-only — handled inside its owner's group, skip here

    return result
