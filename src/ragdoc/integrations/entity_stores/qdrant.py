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

from pydantic import BaseModel

from ragdoc.extraction.entity import Entity
from ragdoc.extraction.query import EntityQuery
from ragdoc.integrations._qdrant_base import AsyncQdrantClient, _QdrantCollectionStore, models

_NAMESPACE = uuid.UUID("c7d8e9f0-1a2b-3c4d-5e6f-7a8b9c0d1e2f")
# Flat, datetime-indexed payload keys carrying the entity's date interval, for range filtering.
_DATE_START = "date_start"
_DATE_END = "date_end"


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


def _bound_condition(key: str, date_range: models.DatetimeRange) -> models.Filter:
    """One date bound as a nested *should*: in *date_range* OR *key* absent (= unbounded).

    A half-bounded entity stores no payload key for its open side (see
    :func:`_entity_to_payload`); a bare ``must`` range condition could never match it,
    diverging from :meth:`~ragdoc.extraction.dates.FuzzyDate.overlaps` where a ``None``
    bound is unbounded.
    """
    return models.Filter(
        should=[
            models.FieldCondition(key=key, range=date_range),
            models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
        ]
    )


class QdrantEntityStore(_QdrantCollectionStore):
    """Async :class:`~ragdoc.extraction.entity.EntityStore` backed by Qdrant.

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient` (caller owns its lifecycle).
        collection_name: Qdrant collection holding the entities.
        payload_model: The payload model to rehydrate entities into (e.g. ``Event``).
    """

    _ID_NAMESPACE = _NAMESPACE

    def __init__(self, client: AsyncQdrantClient, collection_name: str, payload_model: type[BaseModel]) -> None:
        super().__init__(client, collection_name)
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
        await cls._ensure_collection(
            client,
            collection_name,
            models.VectorParams(size=1, distance=models.Distance.DOT),
            payload_indexes=indexes,
        )
        return cls(client, collection_name, payload_model)

    async def upsert(self, entities: list[Entity]) -> list[str]:
        """Store or replace *entities* (keyed on ``entity_id``); return their ids."""
        if not entities:
            return []
        points = [
            models.PointStruct(id=self._point_id(e.entity_id), vector=[0.0], payload=_entity_to_payload(e))
            for e in entities
        ]
        await self._batched_upsert(points)
        return [e.entity_id for e in entities]

    async def delete(self, entity_ids: list[str]) -> None:
        """Remove entities by id (no-op for ids that do not exist)."""
        if not entity_ids:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(eid) for eid in entity_ids]),
        )

    async def delete_by_source(self, source_id: str) -> None:
        """Delete entities that originated solely from *source_id*."""
        stale = [e.entity_id for e in await self.list_entities() if e.source_ids == [source_id]]
        await self.delete(stale)

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values contributing to stored entities."""
        ids: set[str] = set()
        async for payload in self._scroll_payloads(["source_ids"]):
            sids = payload.get("source_ids")
            if isinstance(sids, list):
                ids.update(sid for sid in sids if isinstance(sid, str))
        return ids

    async def list_entities(self) -> list[Entity]:
        """Return every stored canonical entity (rehydrated into ``Entity[payload_model]``)."""
        etype = self._entity_type()
        return [etype.model_validate(payload) async for payload in self._scroll_payloads(True)]

    async def query(self, query: EntityQuery) -> list[Entity]:
        """Filter entities **server-side**: a date-interval overlap plus exact payload-field matches.

        Date overlap uses the half-open rule ``date_start < query.end AND date_end > query.start``,
        with **missing-bound = unbounded**, mirroring :func:`~ragdoc.extraction.query.entity_matches`
        over :meth:`~ragdoc.extraction.dates.FuzzyDate.overlaps`: a half-bounded entity (e.g. EDTF
        ``"1994/.."``) stores no ``date_end`` payload key, so each bound is a nested *should* —
        in-range OR key-absent. Entities with no date at all (``date`` null) are excluded by any
        date filter, again matching the local semantics. ``where`` keys match the nested
        ``payload.<field>`` path. For efficiency, index the fields you filter on via
        :meth:`create`'s ``indexed_fields`` (``date_start`` / ``date_end`` are always indexed).
        """
        conditions: list[models.Condition] = []
        if query.date is not None:
            # Parity with entity_matches: an entity without any date never satisfies a date filter.
            conditions.append(
                models.Filter(must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key="date"))])
            )
            if query.date.end is not None:
                conditions.append(_bound_condition(_DATE_START, models.DatetimeRange(lt=query.date.end)))
            if query.date.start is not None:
                conditions.append(_bound_condition(_DATE_END, models.DatetimeRange(gt=query.date.start)))
        for key, value in query.where.items():
            conditions.append(models.FieldCondition(key=f"payload.{key}", match=models.MatchValue(value=value)))

        scroll_filter = models.Filter(must=conditions) if conditions else None
        etype = self._entity_type()
        out: list[Entity] = []
        async for payload in self._scroll_payloads(True, scroll_filter=scroll_filter):
            out.append(etype.model_validate(payload))
            if query.limit is not None and len(out) >= query.limit:
                break
        return out
