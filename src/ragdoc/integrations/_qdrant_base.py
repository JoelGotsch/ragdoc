"""Shared mechanics for the Qdrant-backed store integrations (private module).

All five Qdrant stores (vector / document / mention / entity / graph) subclass
:class:`_QdrantCollectionStore`. The base owns the payload-agnostic plumbing —
idempotent collection creation with **per-index** 409 handling, paginated payload
scrolls, batched upserts, ``source_id`` filtering, and deterministic namespaced
point ids — while each subclass owns only its payload marshalling and
store-specific queries.

Requires the ``ragdoc[qdrant]`` extra (the import guard lives here, stated once).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import ClassVar

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.conversions import common_types
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError as _e:
    raise ImportError(
        "qdrant-client is not installed. "
        "Install it with: uv add --optional qdrant qdrant-client  "
        "or: pip install 'ragdoc[qdrant]'"
    ) from _e

from ragdoc.pipeline.stores import SourceState

logger = logging.getLogger(__name__)

# Payload keys read by the bulk change-detection path (must stay cheap — no full bodies).
_STATE_KEYS = ["source_id", "source_hash", "content_hash"]


def _source_id_filter(source_id: str) -> models.Filter:
    """Build a Qdrant Filter matching points whose ``source_id`` payload key equals *source_id*."""
    return models.Filter(must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))])


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


class _QdrantCollectionStore:
    """Shared mechanics for one Qdrant collection: idempotent creation with per-index 409
    handling, paginated payload scroll, batched upsert, deterministic namespaced point ids.

    Subclasses own payload marshalling only. Multi-collection stores (e.g. the graph store's
    ``_nodes`` / ``_edges`` pair) pass ``collection=`` on every base call.
    """

    _SCROLL_BATCH: ClassVar[int] = 1000
    _UPSERT_BATCH: ClassVar[int] = 100
    #: UUID5 namespace for :meth:`_point_id`; set by subclasses that key points on an id string.
    _ID_NAMESPACE: ClassVar[uuid.UUID | None] = None

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    @staticmethod
    async def _ensure_collection(
        client: AsyncQdrantClient,
        collection_name: str,
        vectors_config: models.VectorParams | dict[str, models.VectorParams],
        sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None,
        payload_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ) -> None:
        """Create *collection_name* if absent, then ensure every payload index exists.

        ``create_collection`` runs in its own ``try`` (409 ⇒ collection exists, fine); each
        ``create_payload_index`` then runs in its own 409-tolerant call — so an
        already-existing collection still gains any missing payload indexes.
        Safe to call on every application start.
        """
        try:
            if sparse_vectors_config is not None:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
            else:
                await client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
        except UnexpectedResponse as exc:
            if exc.status_code != 409:  # 409: collection already exists — fine
                raise
        for field_name, field_schema in (payload_indexes or {}).items():
            await _create_index_ignore_conflict(client, collection_name, field_name, field_schema)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def _scroll_payloads(
        self,
        with_payload: bool | list[str],
        scroll_filter: models.Filter | None = None,
        collection: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Yield every matching point's payload, paginating with ``with_vectors=False``.

        Args:
            with_payload: ``True`` for the full payload, or an explicit key list for cheap reads.
            scroll_filter: Optional server-side filter.
            collection: Override for multi-collection stores; defaults to ``self._collection_name``.
        """
        offset: common_types.PointId | None = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=collection or self._collection_name,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=False,
                limit=self._SCROLL_BATCH,
                offset=offset,
            )
            for point in results:
                yield point.payload or {}
            if next_offset is None:
                break
            offset = next_offset

    async def _list_source_state(self, collection: str | None = None) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every source, in one scan.

        First seen wins per ``source_id`` (all points of a source share the hashes). A **cheap**
        read: scrolls only the three hash payload keys with ``with_vectors=False``, so it never
        pulls full point bodies (keeps ``plan()`` fast).
        """
        state: dict[str, SourceState] = {}
        async for payload in self._scroll_payloads(_STATE_KEYS, collection=collection):
            sid = payload.get("source_id")
            if isinstance(sid, str) and sid not in state:
                source_hash = payload.get("source_hash")
                content_hash = payload.get("content_hash")
                state[sid] = SourceState(
                    source_hash=source_hash if isinstance(source_hash, str) else "",
                    content_hash=content_hash if isinstance(content_hash, str) else None,
                )
        return state

    async def _collect_source_ids(self, key: str = "source_id", collection: str | None = None) -> set[str]:
        """Return the distinct string values stored under payload *key* (scroll-paginated)."""
        source_ids: set[str] = set()
        async for payload in self._scroll_payloads([key], collection=collection):
            value = payload.get(key)
            if isinstance(value, str):
                source_ids.add(value)
        return source_ids

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def _batched_upsert(self, points: list[models.PointStruct], collection: str | None = None) -> None:
        """Upsert *points* in ``_UPSERT_BATCH``-sized batches (avoids message-size limits).

        On a mid-run batch failure, earlier batches are already persisted carrying the NEW
        ``source_hash``/``content_hash`` — the next sync would read that state as "unchanged"
        and skip the half-written source forever. So a failure triggers a **best-effort
        rollback**: every distinct ``source_id`` present in *points* is deleted from the
        collection, then the original exception is re-raised. The next run sees those sources
        missing and re-processes them. Cancellation is never caught (no rollback on
        ``asyncio.CancelledError`` — consistent with the sync engine's error discipline).
        """
        try:
            for i in range(0, len(points), self._UPSERT_BATCH):
                await self._client.upsert(
                    collection_name=collection or self._collection_name,
                    points=points[i : i + self._UPSERT_BATCH],
                )
        except Exception as exc:
            source_ids = sorted({sid for p in points if isinstance(sid := (p.payload or {}).get("source_id"), str)})
            if source_ids:
                logger.warning(
                    f"batched upsert failed mid-run ({exc!r}); rolling back sources {source_ids} "
                    "so the next sync re-processes them instead of seeing a half-written 'unchanged' state"
                )
            for sid in source_ids:
                try:
                    await self._delete_by_source_filter(sid, collection=collection)
                except Exception:
                    logger.error(f"rollback delete_by_source({sid!r}) failed", exc_info=True)
            raise

    async def _delete_by_source_filter(self, source_id: str, collection: str | None = None) -> None:
        """Delete every point whose ``source_id`` payload key equals *source_id*."""
        await self._client.delete(
            collection_name=collection or self._collection_name,
            points_selector=_source_id_filter(source_id),
        )

    # ------------------------------------------------------------------
    # Ids
    # ------------------------------------------------------------------

    def _point_id(self, key: str) -> str:
        """Deterministic namespaced point id for *key* (UUID5, so upserts replace in place).

        Raises:
            TypeError: if the subclass does not set ``_ID_NAMESPACE`` (stores whose point ids
                come from elsewhere, e.g. ``chunk.id``, must not call this).
        """
        namespace = type(self)._ID_NAMESPACE
        if namespace is None:
            raise TypeError(f"{type(self).__name__} sets no _ID_NAMESPACE; provide point ids explicitly instead.")
        return str(uuid.uuid5(namespace, key))
