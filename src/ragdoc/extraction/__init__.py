"""Structured entity extraction: typed mentions, sync, and resolution.

Two layers (see ``designs/DESIGN-structured-extraction.md``):

* **Mentions** — raw, provenance-tagged extraction occurrences
  (:class:`~ragdoc.extraction.mention.Mention`), produced by
  :class:`~ragdoc.extraction.processor.StructuredExtractionProcessor` and synced per source.
* **Entities** — canonical, deduplicated records produced by the resolution pass.

:class:`~ragdoc.extraction.dates.FuzzyDate` is a reusable date type embeddable in any
payload model. The processor/settings require the ``extraction`` extra (``edtf`` +
``pydantic-settings``); the data models do not, so they import unconditionally.
"""

from ragdoc.extraction.changeset import MentionChangeSet, MentionSourceChange
from ragdoc.extraction.dates import FuzzyDate, Precision
from ragdoc.extraction.entity import Entity, EntityStore, LocalEntityStore
from ragdoc.extraction.graph_store import GraphStore, LocalGraphStore
from ragdoc.extraction.kg_resolution import (
    KGResolutionResult,
    KnowledgeGraphResolutionPipeline,
)
from ragdoc.extraction.mention import Mention, finalize_mention, mint_mention_id
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
    build_edge_union,
    build_node_union,
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
    "MentionChangeSet",
    "MentionSourceChange",
    "MentionStore",
    "MentionStorePipeline",
    "Precision",
    "ResolutionResult",
    "ReviewGroup",
    "ReviewResult",
    "build_edge_union",
    "build_entity_embedder",
    "build_node_union",
    "default_identity_text",
    "entity_matches",
    "filter_entities",
    "finalize_mention",
    "make_llm_reviewer",
    "mint_mention_id",
]

try:
    from ragdoc.extraction.kg_processor import (
        KG_EXTRACTION_SYSTEM_PROMPT,
        KnowledgeGraphProcessor,
        build_graph_batch_model,
        build_kg_messages,
    )
    from ragdoc.extraction.processor import (
        EXTRACTION_SYSTEM_PROMPT,
        ExtractionSettings,
        StructuredExtractionProcessor,
        build_extraction_messages,
    )

    __all__ += [
        "EXTRACTION_SYSTEM_PROMPT",
        "KG_EXTRACTION_SYSTEM_PROMPT",
        "ExtractionSettings",
        # KG layer
        "KnowledgeGraphProcessor",
        "StructuredExtractionProcessor",
        "build_extraction_messages",
        "build_graph_batch_model",
        "build_kg_messages",
    ]
except ImportError:  # pragma: no cover - pydantic-settings not installed (extraction extra absent)
    pass
