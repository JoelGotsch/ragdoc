"""Structured entity extraction: typed mentions, sync, and resolution.

Two layers (see ``designs/DESIGN-structured-extraction.md``):

* **Mentions** — raw, provenance-tagged extraction occurrences
  (:class:`~ragdoc.extraction.mention.Mention`), returned directly by an
  :class:`~ragdoc.extraction.extractor.Extractor` (``StructuredExtractor`` /
  ``KnowledgeGraphExtractor``) and synced per source by ``MentionStorePipeline``. Extraction is a
  typed channel: ``extract()`` returns ``Mention`` objects and never touches
  ``document.metadata`` (the opt-in :func:`~ragdoc.extraction.extractor.as_processor` adapter
  exists for callers who want a document dump).
* **Entities** — canonical, deduplicated records produced by the resolution pass.

:class:`~ragdoc.extraction.dates.FuzzyDate` is a reusable date type embeddable in any
payload model. The extractors/settings require the ``extraction`` extra (``edtf`` +
``pydantic-settings``); the data models do not, so they import unconditionally.
"""

from ragdoc.extraction.dates import FuzzyDate, Precision
from ragdoc.extraction.entity import Entity, EntityStore, LocalEntityStore, mint_entity_id
from ragdoc.extraction.extractor import Extractor, as_processor
from ragdoc.extraction.graph_store import GraphStore, LocalGraphStore
from ragdoc.extraction.kg_resolution import (
    KGResolutionResult,
    KnowledgeGraphResolutionPipeline,
)
from ragdoc.extraction.mention import Mention, finalize_mention, mint_mention_id, rewrite_edge_refs
from ragdoc.extraction.pipeline import MentionStorePipeline
from ragdoc.extraction.query import DateRange, EntityQuery, entity_matches, filter_entities
from ragdoc.extraction.resolution import (
    EntityResolutionPipeline,
    ResolutionResult,
    ReviewGroup,
    ReviewResult,
    build_entity_embedder,
    default_identity_text,
    make_llm_reviewer,
)
from ragdoc.extraction.schema import (
    EdgeRef,
    GraphSchema,
    allowed_pattern_kinds,
    build_edge_union,
    build_node_union,
    kind_of,
    render_patterns_prompt,
)
from ragdoc.extraction.stores import LocalMentionStore, MentionStore

__all__ = [
    "DateRange",
    "EdgeRef",
    # Layer 2 — entities + resolution
    "Entity",
    "EntityQuery",
    "EntityResolutionPipeline",
    "EntityStore",
    # Layer 1 — the extraction stage
    "Extractor",
    "FuzzyDate",
    # Layer 3 — typed knowledge graph schema + resolution
    "GraphSchema",
    "GraphStore",
    "KGResolutionResult",
    "KnowledgeGraphResolutionPipeline",
    "LocalEntityStore",
    "LocalGraphStore",
    "LocalMentionStore",
    "Mention",
    "MentionStore",
    "MentionStorePipeline",
    "Precision",
    "ResolutionResult",
    "ReviewGroup",
    "ReviewResult",
    "allowed_pattern_kinds",
    "as_processor",
    "build_edge_union",
    "build_entity_embedder",
    "build_node_union",
    "default_identity_text",
    "entity_matches",
    "filter_entities",
    "finalize_mention",
    "kind_of",
    "make_llm_reviewer",
    "mint_entity_id",
    "mint_mention_id",
    "render_patterns_prompt",
    "rewrite_edge_refs",
]

try:
    from ragdoc.extraction.kg import (
        KG_EXTRACTION_SYSTEM_PROMPT,
        KnowledgeGraphExtractor,
        build_graph_batch_model,
        build_kg_messages,
    )
    from ragdoc.extraction.structured import (
        EXTRACTION_SYSTEM_PROMPT,
        ExtractionSettings,
        StructuredExtractor,
        build_extraction_messages,
    )

    __all__ += [
        "EXTRACTION_SYSTEM_PROMPT",
        "KG_EXTRACTION_SYSTEM_PROMPT",
        "ExtractionSettings",
        # KG layer
        "KnowledgeGraphExtractor",
        "StructuredExtractor",
        "build_extraction_messages",
        "build_graph_batch_model",
        "build_kg_messages",
    ]
except ImportError:  # pragma: no cover - pydantic-settings not installed (extraction extra absent)
    pass
