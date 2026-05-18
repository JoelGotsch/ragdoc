"""
Document splitting utilities.

Pipeline stage: SPLITTING (stage 4)

Splits a Document into multiple Documents based on heading structure or token counts.
After splitting, each resulting document carries ExternalRef entries that record the
parent-child relationship back to the original (or within the split set).

Public API:
    split_by_headings(document)       — split at every heading boundary (flat)
    split_hierarchical(document)      — split recursively by the lowest heading level
                                        that yields ≥ 2 parts
    split_document(...)               — main entry point: hierarchical → element-level →
                                        text-level, three-tier token-budget splitting
    split_by_elements(...)            — greedy group-boundary split with heading context
    split_oversized_element(...)      — three-tier split for a single oversized group:
                                        HTML structural → sentence → token-slice
    split_at_html_tags(...)           — tier-1 helper: BS4 block-child split
    split_at_sentences(...)           — tier-2 helper: sentence-boundary split (pluggable)
    build_element_groups(...)         — partition elements into atomic groups for splitting
"""

from ragdoc.splitting.base import split_by_headings, Splitter
from ragdoc.splitting.groups import ElementGroup, build_element_groups
from ragdoc.splitting.hierarchical import split_hierarchical
from ragdoc.splitting.token import (
    SentenceSplitter,
    split_at_html_tags,
    split_at_sentences,
    split_by_elements,
    split_document,
    split_oversized_element,
)

__all__ = [
    "split_by_headings",
    "split_hierarchical",
    "split_document",
    "split_by_elements",
    "split_oversized_element",
    "split_at_html_tags",
    "split_at_sentences",
    "build_element_groups",
    "ElementGroup",
    "SentenceSplitter",
    "Splitter",
]