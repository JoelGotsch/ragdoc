"""Qdrant implementation of ragdoc's EntityStore protocol.

Requires the ``ragdoc[qdrant]`` extra::

    uv add --optional qdrant qdrant-client

:class:`QdrantEntityStore` is the shared-access counterpart to
:class:`~ragdoc.extraction.entity.LocalEntityStore`. Canonical entities have no natural
embedding at write time, so — like :class:`~ragdoc.integrations.document_stores.qdrant.QdrantDocumentStore`
— each entity is a **single point with a throwaway 1-dimensional vector** that is never queried;
the collection is a key-value store keyed on ``entity_id``. The full entity is serialized into the
payload, plus flat datetime-indexed ``date_start`` / ``date_end`` keys so :meth:`QdrantEntityStore.query`
can range-filter server-side (mirroring the in-memory filter over ``LocalEntityStore``).

The point id is a deterministic ``UUID5(entity_id)`` so re-resolution replaces entities in place.
"""

from __future__ import annotations

import datetime as dt
import uuid

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError as _e:
    raise ImportError(
        "qdrant-client is not installed. "
        "Install it with: uv add --optional qdrant qdrant-client  "
        "or: pip install 'ragdoc[qdrant]'"
    ) from _e

from pydantic import BaseModel

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.query import EntityQuery

_NAMESPACE = uuid.UUID("c7d8e9f0-1a2b-3c4d-5e6f-7a8b9c0d1e2f")
_SCROLL_BATCH = 1000
_UPSERT_BATCH = 100
# Flat, datetime-indexed payload keys carrying the entity's date interval, for range filtering.
_DATE_START = "date_start"
_DATE_END = "date_end"


def _point_id(entity_id: str) -> str:
    """Deterministic Qdrant point id for *entity_id* (so upsert replaces in place)."""
    return str(uuid.uuid5(_NAMESPACE, entity_id))


def _rfc3339(d: dt.date) -> str:
    """Render a date as a UTC-midnight RFC3339 timestamp for a Qdrant datetime index/range."""
    return dt.datetime.combine(d, dt.time.min, tzinfo=dt.timezone.utc).isoformat()


def _entity_to_payload(entity: Entity) -> dict:
    """Serialize *entity*, adding flat ``date_start`` / ``date_end`` keys for range filtering."""
    payload = entity.model_dump(mode="json")
    if entity.date is not None and entity.date.start is not None:
        payload[_DATE_START] = _rfc3339(entity.date.start)
    if entity.date is not None and entity.date.end is not None:
        payload[_DATE_END] = _rfc3339(entity.date.end)
    return payload


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


class QdrantEntityStore:
    """Async :class:`~ragdoc.extraction.entity.EntityStore` backed by Qdrant.

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient` (caller owns its lifecycle).
        collection_name: Qdrant collection holding the entities.
        payload_model: The payload model to rehydrate entities into (e.g. ``Event``).
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str, payload_model: type[BaseModel]) -> None:
        self._client = client
        self._collection_name = collection_name
        self._payload_model = payload_model

    def _entity_type(self) -> type[Entity]:
        payload_model = self._payload_model
        return Entity[payload_model]

    @classmethod
    async def create(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        payload_model: type[BaseModel],
        indexed_fields: dict[str, models.PayloadSchemaType] | None = None,
    ) -> QdrantEntityStore:
        """Create the collection (idempotent) with a throwaway 1-dim vector; return a store.

        Always indexes ``entity_id`` (keyword) and ``date_start`` / ``date_end`` (datetime, for
        :meth:`query` range filtering). *indexed_fields* maps **canonical-payload** field names to a
        ``PayloadSchemaType`` so they can be filtered server-side (indexes are created on the nested
        ``payload.<field>`` path), e.g. ``{"facility": models.PayloadSchemaType.KEYWORD}``.
        """
        indexes: dict[str, models.PayloadSchemaType] = {
            "entity_id": models.PayloadSchemaType.KEYWORD,
            _DATE_START: models.PayloadSchemaType.DATETIME,
            _DATE_END: models.PayloadSchemaType.DATETIME,
        }
        for field_name, schema in (indexed_fields or {}).items():
            indexes[f"payload.{field_name}"] = schema
        try:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise
        for field_name, schema in indexes.items():
            await _create_index_ignore_conflict(client, collection_name, field_name, schema)
        return cls(client, collection_name, payload_model)

    async def upsert(self, entities: list[Entity]) -> list[str]:
        """Store or replace *entities* (keyed on ``entity_id``); return their ids."""
        if not entities:
            return []
        points = [
            models.PointStruct(id=_point_id(e.entity_id), vector=[0.0], payload=_entity_to_payload(e)) for e in entities
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(collection_name=self._collection_name, points=points[i : i + _UPSERT_BATCH])
        return [e.entity_id for e in entities]

    async def delete(self, entity_ids: list[str]) -> None:
        """Remove entities by id (no-op for ids that do not exist)."""
        if not entity_ids:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[_point_id(eid) for eid in entity_ids]),
        )

    async def delete_by_source(self, source_id: str) -> None:
        """Delete entities that originated solely from *source_id*."""
        stale = [e.entity_id for e in await self.list_entities() if e.source_ids == [source_id]]
        await self.delete(stale)

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values contributing to stored entities."""
        ids: set[str] = set()
        async for payload in self._scroll(["source_ids"]):
            for sid in payload.get("source_ids") or []:
                ids.add(sid)
        return ids

    async def list_entities(self) -> list[Entity]:
        """Return every stored canonical entity (rehydrated into ``Entity[payload_model]``)."""
        etype = self._entity_type()
        return [etype.model_validate(payload) async for payload in self._scroll(True)]

    async def query(self, query: EntityQuery) -> list[Entity]:
        """Filter entities **server-side**: a date-interval overlap plus exact payload-field matches.

        Date overlap uses the half-open rule ``date_start < query.end AND date_end > query.start``
        (entities with no date are excluded by a date filter). ``where`` keys match the nested
        ``payload.<field>`` path. For efficiency, index the fields you filter on via
        :meth:`create`'s ``indexed_fields`` (``date_start`` / ``date_end`` are always indexed).
        """
        conditions: list[models.Condition] = []
        if query.date is not None:
            if query.date.end is not None:
                conditions.append(models.FieldCondition(key=_DATE_START, range=models.DatetimeRange(lt=query.date.end)))
            if query.date.start is not None:
                conditions.append(models.FieldCondition(key=_DATE_END, range=models.DatetimeRange(gt=query.date.start)))
        for key, value in query.where.items():
            conditions.append(models.FieldCondition(key=f"payload.{key}", match=models.MatchValue(value=value)))

        scroll_filter = models.Filter(must=conditions) if conditions else None
        etype = self._entity_type()
        out: list[Entity] = []
        async for payload in self._scroll(True, scroll_filter=scroll_filter):
            out.append(etype.model_validate(payload))
            if query.limit is not None and len(out) >= query.limit:
                break
        return out

    async def _scroll(self, with_payload, scroll_filter: models.Filter | None = None):
        offset: str | int | None = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
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
