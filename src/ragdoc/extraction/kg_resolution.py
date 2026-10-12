"""KnowledgeGraphResolutionPipeline: per-type node resolution + edge endpoint rewrite.

Orchestrates the typed-KG resolution pass on top of the existing
:class:`~ragdoc.extraction.resolution.EntityResolutionPipeline`:

1. **Node resolution per type.** For each registered node type ``T``, fetch mentions via
   ``mention_store.list_mentions(T)`` and run an :class:`EntityResolutionPipeline` against a
   per-type identity function (user-supplied or :func:`default_identity_text`). Collects the
   ``mention_id → entity_id`` map across all node types.
2. **Edge endpoint rewrite.** Walk every edge mention via ``list_mentions(EdgeT)``; rewrite
   each ``refs.source_mention_id`` / ``refs.target_mention_id`` from the chunk-local-then-
   mention-id stage to the resolved node ``entity_id``. Edges whose endpoint mention is
   missing from the node-entity map are surfaced as ``orphan_edges`` — visible, never silently
   dropped.
3. **Edge "resolution".** By default a no-op promotion: each rewritten edge mention becomes one
   ``Entity[EdgeT]`` 1:1 with its own provenance (two documents claiming the same edge stay as
   two distinct attestations). Per-edge-type opt-in dedup: passing
   ``edge_identity_text_fns={EdgeT: fn}`` routes that edge type through a full
   :class:`EntityResolutionPipeline` pass, clustering across documents.

The result is a :class:`KGResolutionResult` carrying node entities, edge entities, orphan edges,
and aggregate pending/iteration metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragdoc.extraction.entity import Entity, mint_entity_id
from ragdoc.extraction.mention import Mention
from ragdoc.extraction.query import EntityQuery, filter_entities
from ragdoc.extraction.resolution import (
    EntityResolutionPipeline,
    IdentityTextFn,
    default_identity_text,
)
from ragdoc.extraction.schema import GraphSchema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.extraction.entity import EntityStore
    from ragdoc.extraction.resolution import Embed, Reviewer
    from ragdoc.extraction.stores import MentionStore
    from ragdoc.extraction.structured import ExtractionSettings


@dataclass
class KGResolutionResult:
    """Outcome of a typed-KG resolution run.

    Attributes:
        node_entities: all canonical node entities, across every node type.
        edge_entities: all canonical edge entities, across every edge type (1:1 with mentions
            by default, or clustered when an ``edge_identity_text_fns`` entry was provided).
        orphan_edges: edge mentions whose endpoint mention did not resolve to a known node
            entity — surfaced so the caller can audit / re-extract, not silently dropped.
        pending: union of pending tiers from per-type node resolution.
        iterations: max iteration count across all per-type node resolutions.
    """

    node_entities: list[Entity] = field(default_factory=list)
    edge_entities: list[Entity] = field(default_factory=list)
    orphan_edges: list[Mention] = field(default_factory=list)
    pending: list[tuple[str, ...]] = field(default_factory=list)
    iterations: int = 0


class KnowledgeGraphResolutionPipeline:
    """Resolve a typed-KG ``MentionStore`` into node + edge ``EntityStore``s.

    Args:
        schema: the :class:`~ragdoc.extraction.schema.GraphSchema` declaring node types,
            edge types, and patterns.
        mention_store: source of node + edge mentions (filtered by ``list_mentions(type=...)``).
        node_entity_store: sink for resolved node entities.
        edge_entity_store: sink for resolved edge entities.
        embed: text embedder (``list[str] -> vectors``) used for blocking during node resolution.
        reviewer: LLM cluster reviewer (see :func:`make_llm_reviewer`); inject a fake in tests.
        identity_text_fns: per node-type identity text functions. Missing types use
            :func:`default_identity_text`.
        edge_identity_text_fns: per edge-type identity text functions. **Edges without an entry
            are not deduplicated** — each mention becomes one entity 1:1. Edges with an entry
            run through the same :class:`EntityResolutionPipeline` as nodes.
        settings: :class:`ExtractionSettings` (cosine threshold, auto-merge bar, max iterations,
            top-k). ``None`` reads env (``EXTRACTION_*``).
    """

    def __init__(
        self,
        schema: GraphSchema,
        mention_store: MentionStore,
        node_entity_store: EntityStore,
        edge_entity_store: EntityStore,
        embed: Embed,
        reviewer: Reviewer,
        identity_text_fns: dict[type[BaseModel], IdentityTextFn] | None = None,
        edge_identity_text_fns: dict[type[BaseModel], IdentityTextFn] | None = None,
        settings: ExtractionSettings | None = None,
    ) -> None:
        from ragdoc.extraction.structured import ExtractionSettings

        self._schema = schema
        self._mention_store = mention_store
        self._node_entity_store = node_entity_store
        self._edge_entity_store = edge_entity_store
        self._embed = embed
        self._reviewer = reviewer
        self._identity_text_fns = identity_text_fns or {}
        self._edge_identity_text_fns = edge_identity_text_fns or {}
        self.settings = settings or ExtractionSettings()

    async def resolve(self) -> KGResolutionResult:
        """Run per-type node resolution, rewrite edges, then per-type edge resolution; write stores."""
        # ---- Phase 1: node resolution per type ----
        node_entities: list[Entity] = []
        node_entity_map: dict[str, str] = {}  # mention_id → entity_id
        pending: list[tuple[str, ...]] = []
        iterations = 0

        for node_type in self._schema.node_types:
            mentions = await self._mention_store.list_mentions(node_type)
            if not mentions:
                continue
            identity_fn = self._identity_text_fns.get(node_type, default_identity_text)
            entities, sub_pending, sub_iters = await self._resolve_type(mentions, identity_fn)
            node_entities.extend(entities)
            pending.extend(sub_pending)
            iterations = max(iterations, sub_iters)
            for e in entities:
                for mid in e.member_mention_ids:
                    node_entity_map[mid] = e.entity_id

        # ---- Phase 2: edge endpoint rewrite + per-type promotion or dedup ----
        edge_entities: list[Entity] = []
        orphan_edges: list[Mention] = []

        for edge_type in self._schema.edge_types:
            edge_mentions = await self._mention_store.list_mentions(edge_type)
            if not edge_mentions:
                continue
            rewritten, orphans = _rewrite_edge_endpoints(edge_mentions, node_entity_map)
            orphan_edges.extend(orphans)
            if not rewritten:
                continue

            if edge_type in self._edge_identity_text_fns:
                # Opt-in dedup: cluster edges across documents.
                identity_fn = self._edge_identity_text_fns[edge_type]
                entities, sub_pending, sub_iters = await self._resolve_type(rewritten, identity_fn)
                edge_entities.extend(entities)
                pending.extend(sub_pending)
                iterations = max(iterations, sub_iters)
            else:
                # Default: 1:1 promotion, each mention is its own edge entity.
                edge_entities.extend(_promote_mentions_to_entities(rewritten))

        # ---- Phase 3: write entity stores (delete stale, then upsert) ----
        await _write_entity_store(self._node_entity_store, node_entities)
        await _write_entity_store(self._edge_entity_store, edge_entities)

        return KGResolutionResult(
            node_entities=node_entities,
            edge_entities=edge_entities,
            orphan_edges=orphan_edges,
            pending=pending,
            iterations=iterations,
        )

    async def _resolve_type(
        self,
        mentions: list[Mention],
        identity_text_fn: IdentityTextFn,
    ) -> tuple[list[Entity], list[tuple[str, ...]], int]:
        """Run one EntityResolutionPipeline pass against *mentions* (delegated, no store writes)."""
        sub_mention_store = _ExplicitMentionStore(mentions)
        sub_entity_store = _InMemoryEntityStore()
        sub_pipeline = EntityResolutionPipeline(
            mention_store=sub_mention_store,
            entity_store=sub_entity_store,
            embed=self._embed,
            reviewer=self._reviewer,
            identity_text_fn=identity_text_fn,
            settings=self.settings,
        )
        result = await sub_pipeline.resolve()
        return list(result.entities), list(result.pending), result.iterations


# ---------------------------------------------------------------------------
# Edge endpoint rewrite
# ---------------------------------------------------------------------------


def _rewrite_edge_endpoints(
    edge_mentions: list[Mention], node_entity_map: dict[str, str]
) -> tuple[list[Mention], list[Mention]]:
    """Rewrite each edge's ``refs.source_mention_id`` / ``target_mention_id`` to entity ids.

    Returns ``(rewritten, orphans)``: edges whose endpoint mention isn't in *node_entity_map*
    are returned as orphans (mutated payload left intact) so the caller can surface them.
    """
    rewritten: list[Mention] = []
    orphans: list[Mention] = []
    for m in edge_mentions:
        refs = m.payload.refs
        src = refs.source_mention_id
        tgt = refs.target_mention_id
        if src not in node_entity_map or tgt not in node_entity_map:
            logger.warning(
                "Edge mention %s has orphan endpoint(s): source=%s (resolved=%s), target=%s (resolved=%s)",
                m.mention_id,
                src,
                src in node_entity_map,
                tgt,
                tgt in node_entity_map,
            )
            orphans.append(m)
            continue
        refs.source_mention_id = node_entity_map[src]
        refs.target_mention_id = node_entity_map[tgt]
        rewritten.append(m)
    return rewritten, orphans


def _promote_mentions_to_entities(mentions: list[Mention]) -> list[Entity]:
    """Build one ``Entity[EdgeT]`` per mention (no clustering — the default for edges)."""
    out: list[Entity] = []
    for m in mentions:
        out.append(
            Entity(
                entity_id=mint_entity_id([m.mention_id]),
                payload=m.payload,
                aliases=[],
                member_mention_ids=[m.mention_id],
                source_ids=[m.source_id],
                confidence=1.0,
                date=None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Internal helpers — wrapper stores
# ---------------------------------------------------------------------------


class _ExplicitMentionStore:
    """A read-only MentionStore that holds an explicit mention list. Used to feed
    EntityResolutionPipeline a pre-filtered slice (one node type) without touching the
    real store."""

    def __init__(self, mentions: list[Mention]) -> None:
        self._mentions = mentions

    async def upsert(self, mentions: list[Mention]) -> list[str]:  # pragma: no cover
        del mentions  # part of the MentionStore protocol; this store is read-only
        raise NotImplementedError("_ExplicitMentionStore is read-only.")

    async def delete_by_source(self, source_id: str) -> None:  # pragma: no cover
        del source_id  # part of the MentionStore protocol; this store is read-only
        raise NotImplementedError("_ExplicitMentionStore is read-only.")

    async def list_source_ids(self) -> set[str]:
        return {m.source_id for m in self._mentions}

    async def list_source_state(self):
        from ragdoc.pipeline.stores import SourceState

        return {
            m.source_id: SourceState(source_hash=m.source_hash or "", content_hash=m.content_hash)
            for m in self._mentions
        }

    async def list_mentions(self, payload_type: type[BaseModel] | None = None) -> list[Mention]:
        if payload_type is None:
            return list(self._mentions)
        return [m for m in self._mentions if isinstance(m.payload, payload_type)]


class _InMemoryEntityStore:
    """A throwaway EntityStore the per-type EntityResolutionPipeline writes into; we collect
    the entities back out and combine across types before writing the real store(s)."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}

    async def upsert(self, entities: list[Entity]) -> list[str]:
        for e in entities:
            self._entities[e.entity_id] = e
        return [e.entity_id for e in entities]

    async def delete(self, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self._entities.pop(eid, None)

    async def delete_by_source(self, source_id: str) -> None:  # pragma: no cover
        del source_id  # part of the EntityStore protocol; this throwaway store never deletes by source

    async def list_source_ids(self) -> set[str]:
        return {s for e in self._entities.values() for s in e.source_ids}

    async def list_entities(self) -> list[Entity]:
        return list(self._entities.values())

    async def query(self, query: EntityQuery) -> list[Entity]:
        return filter_entities(list(self._entities.values()), query)


async def _write_entity_store(store: EntityStore, entities: list[Entity]) -> None:
    """Replace *store* contents with *entities*: upsert all, delete stale.

    Mirrors :class:`EntityResolutionPipeline._write` — the run's output is the new ground truth.
    """
    existing = {e.entity_id for e in await store.list_entities()}
    new_ids = {e.entity_id for e in entities}
    if entities:
        await store.upsert(entities)
    stale = list(existing - new_ids)
    if stale:
        await store.delete(stale)
