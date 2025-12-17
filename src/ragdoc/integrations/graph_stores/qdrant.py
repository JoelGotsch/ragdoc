"""Qdrant implementation of ragdoc's GraphStore protocol.

Requires the ``ragdoc[qdrant]`` extra::

    uv add --optional qdrant qdrant-client

:class:`QdrantGraphStore` is the shared-access counterpart to
:class:`~ragdoc.extraction.graph_store.LocalGraphStore`. Per the design it uses **two Qdrant
collections** — ``<name>_nodes`` and ``<name>_edges`` — each a key-value store keyed on
``entity_id`` (a throwaway 1-dim vector, like
:class:`~ragdoc.integrations.entity_stores.qdrant.QdrantEntityStore`; graph entities are not
embedded). Node/edge payloads are rehydrated into their typed Pydantic models via the
:class:`~ragdoc.extraction.schema.GraphSchema`.

Unlike ``LocalGraphStore`` (which walks the whole edge set in memory), :meth:`neighbors` filters
**server-side**: edge payloads index ``payload.refs.source_mention_id`` /
``payload.refs.target_mention_id`` (the endpoint ``entity_id``\\ s after resolution) and
``payload.kind``, so an adjacency lookup is one filtered scroll plus a node retrieve.

The point id is a deterministic ``UUID5(entity_id)`` so re-resolution replaces entities in place.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError as _e:
    raise ImportError(
        "qdrant-client is not installed. "
        "Install it with: uv add --optional qdrant qdrant-client  "
        "or: pip install 'ragdoc[qdrant]'"
    ) from _e

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.schema import GraphSchema, build_edge_union, build_node_union

if TYPE_CHECKING:
    from pydantic import BaseModel

_NAMESPACE = uuid.UUID("e2f3a4b5-6c7d-8e9f-0a1b-2c3d4e5f6a7b")
_SCROLL_BATCH = 1000
_UPSERT_BATCH = 100
_SRC_KEY = "payload.refs.source_mention_id"
_TGT_KEY = "payload.refs.target_mention_id"
_KIND_KEY = "payload.kind"


def _point_id(entity_id: str) -> str:
    """Deterministic Qdrant point id for *entity_id* (so upsert replaces in place)."""
    return str(uuid.uuid5(_NAMESPACE, entity_id))


def _kind_value(payload_type: type[BaseModel]) -> str | None:
    """Return the ``kind`` discriminator literal of *payload_type*, or ``None`` if absent.

    Lets ``list_nodes``/``list_edges`` push a type filter to Qdrant (``payload.kind == kind``)
    instead of fetching everything and filtering by ``isinstance``.
    """
    field = payload_type.model_fields.get("kind")
    default = getattr(field, "default", None) if field is not None else None
    return default if isinstance(default, str) else None


async def _create_index_ignore_conflict(
    client: AsyncQdrantClient, collection_name: str, field_name: str, field_schema: models.PayloadSchemaType
) -> None:
    """Create a payload index, ignoring 409 (index already exists).

    Runs per index call so that an already-existing collection still gains any
    missing payload indexes (a collection-level 409 must not skip indexing).
    """
    try:
        await client.create_payload_index(
            collection_name=collection_name, field_name=field_name, field_schema=field_schema
        )
    except UnexpectedResponse as exc:
        if exc.status_code != 409:
            raise


class QdrantGraphStore:
    """Async :class:`~ragdoc.extraction.graph_store.GraphStore` backed by Qdrant.

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient` (caller owns its lifecycle).
        collection_name: Base name; nodes live in ``<collection_name>_nodes`` and edges in
            ``<collection_name>_edges``.
        schema: :class:`~ragdoc.extraction.schema.GraphSchema` used to rehydrate node + edge
            payloads into their typed Pydantic models.
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str, schema: GraphSchema) -> None:
        self._client = client
        self._nodes = f"{collection_name}_nodes"
        self._edges = f"{collection_name}_edges"
        self._schema = schema

    def _node_type(self) -> type[Entity]:
        node_union = build_node_union(self._schema.node_types)
        return Entity[node_union]

    def _edge_type(self) -> type[Entity]:
        edge_union = build_edge_union(self._schema.edge_types)
        return Entity[edge_union]

    @classmethod
    async def create(cls, client: AsyncQdrantClient, collection_name: str, schema: GraphSchema) -> QdrantGraphStore:
        """Create both collections (idempotent) with the indexes :meth:`neighbors` / type-filtering need."""
        nodes, edges = f"{collection_name}_nodes", f"{collection_name}_edges"
        indexes = {
            nodes: {"entity_id": models.PayloadSchemaType.KEYWORD, _KIND_KEY: models.PayloadSchemaType.KEYWORD},
            edges: {
                "entity_id": models.PayloadSchemaType.KEYWORD,
                _KIND_KEY: models.PayloadSchemaType.KEYWORD,
                _SRC_KEY: models.PayloadSchemaType.KEYWORD,
                _TGT_KEY: models.PayloadSchemaType.KEYWORD,
            },
        }
        for coll, fields in indexes.items():
            try:
                await client.create_collection(
                    collection_name=coll,
                    vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
                )
            except UnexpectedResponse as exc:
                if exc.status_code != 409:
                    raise
            for field_name, schema_type in fields.items():
                await _create_index_ignore_conflict(client, coll, field_name, schema_type)
        return cls(client, collection_name, schema)

    # ---- upsert ----

    async def _upsert(self, collection: str, entities: list[Entity]) -> list[str]:
        if not entities:
            return []
        points = [
            models.PointStruct(id=_point_id(e.entity_id), vector=[0.0], payload=e.model_dump(mode="json"))
            for e in entities
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(collection_name=collection, points=points[i : i + _UPSERT_BATCH])
        return [e.entity_id for e in entities]

    async def upsert_nodes(self, entities: list[Entity]) -> list[str]:
        """Add or replace node *entities*; return their ids."""
        return await self._upsert(self._nodes, entities)

    async def upsert_edges(self, entities: list[Entity]) -> list[str]:
        """Add or replace edge *entities*; return their ids."""
        return await self._upsert(self._edges, entities)

    # ---- get ----

    async def _get(self, collection: str, etype: type[Entity], entity_id: str) -> Entity | None:
        records = await self._client.retrieve(
            collection_name=collection, ids=[_point_id(entity_id)], with_payload=True, with_vectors=False
        )
        return etype.model_validate(records[0].payload) if records else None

    async def get_node(self, entity_id: str) -> Entity | None:
        """Return the node entity for *entity_id*, or ``None``."""
        return await self._get(self._nodes, self._node_type(), entity_id)

    async def get_edge(self, entity_id: str) -> Entity | None:
        """Return the edge entity for *entity_id*, or ``None``."""
        return await self._get(self._edges, self._edge_type(), entity_id)

    # ---- list ----

    async def _list(self, collection: str, etype: type[Entity], payload_type: type[BaseModel] | None) -> list[Entity]:
        kind = _kind_value(payload_type) if payload_type is not None else None
        flt = (
            models.Filter(must=[models.FieldCondition(key=_KIND_KEY, match=models.MatchValue(value=kind))])
            if kind is not None
            else None
        )
        out = [etype.model_validate(p) async for p in self._scroll(collection, True, flt)]
        # Couldn't derive a kind discriminator → fall back to an in-memory isinstance filter.
        if payload_type is not None and kind is None:
            out = [e for e in out if isinstance(e.payload, payload_type)]
        return out

    async def list_nodes(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        """Return node entities, optionally filtered by type (server-side on ``payload.kind``)."""
        return await self._list(self._nodes, self._node_type(), payload_type)

    async def list_edges(self, payload_type: type[BaseModel] | None = None) -> list[Entity]:
        """Return edge entities, optionally filtered by type (server-side on ``payload.kind``)."""
        return await self._list(self._edges, self._edge_type(), payload_type)

    # ---- delete ----

    async def delete(self, entity_ids: list[str]) -> None:
        """Remove entities (nodes or edges) by id from both collections (silent on unknown)."""
        if not entity_ids:
            return
        selector = models.PointIdsList(points=[_point_id(eid) for eid in entity_ids])
        await self._client.delete(collection_name=self._nodes, points_selector=selector)
        await self._client.delete(collection_name=self._edges, points_selector=selector)

    async def delete_by_source(self, source_id: str) -> None:
        """Delete every node and edge whose ``source_ids == [source_id]`` (originated solely there)."""
        for collection, etype in ((self._nodes, self._node_type()), (self._edges, self._edge_type())):
            stale = [e.entity_id async for e in self._entities(collection, etype) if e.source_ids == [source_id]]
            if stale:
                await self._client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=[_point_id(eid) for eid in stale]),
                )

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values contributing to stored nodes + edges."""
        ids: set[str] = set()
        for e in await self.list_nodes():
            ids.update(e.source_ids)
        for e in await self.list_edges():
            ids.update(e.source_ids)
        return ids

    # ---- neighbors (server-side adjacency) ----

    async def neighbors(self, entity_id: str, edge_types: list[str] | None = None) -> list[Entity]:
        """Return node entities reachable from *entity_id* via any (or *edge_types*-filtered) edge.

        Filters edges server-side: ``(refs.source == id OR refs.target == id)`` and, when given,
        ``payload.kind IN edge_types``. Both edge directions are followed.
        """
        endpoint_or = models.Filter(
            should=[
                models.FieldCondition(key=_SRC_KEY, match=models.MatchValue(value=entity_id)),
                models.FieldCondition(key=_TGT_KEY, match=models.MatchValue(value=entity_id)),
            ]
        )
        must: list[models.Condition] = [endpoint_or]
        if edge_types is not None:
            must.append(models.FieldCondition(key=_KIND_KEY, match=models.MatchAny(any=edge_types)))
        edge_filter = models.Filter(must=must)

        etype = self._edge_type()
        other_ids: set[str] = set()
        async for payload in self._scroll(self._edges, True, edge_filter):
            edge = etype.model_validate(payload)
            src, tgt = edge.payload.refs.source_mention_id, edge.payload.refs.target_mention_id
            if src == entity_id:
                other_ids.add(tgt)
            elif tgt == entity_id:
                other_ids.add(src)

        if not other_ids:
            return []
        records = await self._client.retrieve(
            collection_name=self._nodes,
            ids=[_point_id(eid) for eid in other_ids],
            with_payload=True,
            with_vectors=False,
        )
        node_type = self._node_type()
        return [node_type.model_validate(r.payload) for r in records]

    # ---- scroll helpers ----

    async def _entities(self, collection, etype):
        async for payload in self._scroll(collection, True):
            yield etype.model_validate(payload)

    async def _scroll(self, collection, with_payload, scroll_filter: models.Filter | None = None):
        offset: str | int | None = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=False,
                limit=_SCROLL_BATCH,
                offset=offset,
            )
            for point in results:
                yield point.payload or {}
            if next_offset is None:
                break
            offset = next_offset  # type: ignore[assignment]
