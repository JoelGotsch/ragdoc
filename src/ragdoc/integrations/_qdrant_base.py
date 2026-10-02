"""Shared mechanics for the Qdrant-backed store integrations (private module).

All five Qdrant stores (vector / document / mention / entity / graph) subclass
:class:`_QdrantCollectionStore`. The base owns the payload-agnostic plumbing —
idempotent collection creation with **per-index** 409 handling, paginated payload
scrolls, interruption-safe batched upserts, ``source_id`` filtering, and deterministic
namespaced point ids — while each subclass owns only its payload marshalling and
store-specific queries.

Requires the ``ragdoc[qdrant]`` extra (the import guard lives here, stated once).
"""

from __future__ import annotations

import asyncio
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
# The change-detection hashes: blanked while a multi-batch write is in flight, stamped last.
_HASH_KEYS = ("source_hash", "content_hash")
# What change detection sees for a partially written or mixed source: "always changed".
_BLANK_STATE = SourceState(source_hash="", content_hash=None)


def _blank_hashes(point: models.PointStruct) -> models.PointStruct:
    """Return a copy of *point* whose change-detection hashes are ``None``.

    A point that carries neither hash key (entity / graph payloads) is returned unchanged.
    """
    payload = point.payload
    if not payload or not any(key in payload for key in _HASH_KEYS):
        return point
    blanked = {key: (None if key in _HASH_KEYS else value) for key, value in payload.items()}
    return models.PointStruct(id=point.id, vector=point.vector, payload=blanked)


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

        A source reports its stored hashes only when **every** one of its points carries the
        same ``(source_hash, content_hash)`` pair. If its points disagree — a half-replaced
        source mixing old and new points, or an interrupted write whose points still carry
        blank hashes next to stamped ones (see :meth:`_batched_upsert`) — the source is
        reported with blank hashes (``source_hash=""``, ``content_hash=None``), which every
        change-detection path reads as "always changed". A partially written source therefore
        never looks "unchanged".

        A **cheap** read: scrolls only the three hash payload keys with ``with_vectors=False``,
        so it never pulls full point bodies (keeps ``plan()`` fast).
        """
        state: dict[str, SourceState] = {}
        async for payload in self._scroll_payloads(_STATE_KEYS, collection=collection):
            sid = payload.get("source_id")
            if not isinstance(sid, str):
                continue
            source_hash = payload.get("source_hash")
            content_hash = payload.get("content_hash")
            seen = SourceState(
                source_hash=source_hash if isinstance(source_hash, str) else "",
                content_hash=content_hash if isinstance(content_hash, str) else None,
            )
            known = state.get(sid)
            if known is None:
                state[sid] = seen
            elif known != seen:
                state[sid] = _BLANK_STATE  # mixed / partially written ⇒ always changed
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

        **Single batch** — one request, applied by Qdrant as a whole or not at all. There is
        nothing partial to undo, so a failure simply re-raises and **nothing is deleted**.

        **Multiple batches** — a write that stops part-way (error, cancellation, process
        crash) must never look "unchanged" to change detection, so the write is two-phase:

        1. every batch is written with its change-detection hashes (``source_hash`` /
           ``content_hash``) **blanked**; a blank hash reads as "always changed";
        2. once all batches have landed, the real hashes are stamped with one ``set_payload``
           request per distinct hash pair (the commit marker).

        An interruption at any point before the stamp leaves only blank-hash points, so the
        next sync re-processes the source — this holds even after a hard crash, where no
        cleanup code runs. Points that carry neither hash key (entity / graph stores) are
        written as-is and skip the stamp.

        **Rollback** (multi-batch only, best effort). When a batch fails with an ``Exception``
        or the call is cancelled (``asyncio.CancelledError`` — the cleanup then runs shielded
        and the cancellation is re-raised), the points *this call created* are deleted by
        point id. Rollback never deletes by source and never deletes a point id that existed
        before the call.

        Limitations (stated plainly):

        * A pre-existing point that this call already **overwrote** is not restored: it keeps
          the new content with blank hashes (so its source is re-processed); it is not
          reverted to the old version.
        * "Existed before the call" is one ``retrieve`` taken before the first batch; a
          concurrent writer creating the same ids in between is not detected.
        * No rollback after a failed stamp (all points landed; they stay, unstamped) nor on
          non-``Exception`` aborts such as ``KeyboardInterrupt`` or a process crash — the
          blank hashes already guarantee re-processing, but the partial points remain
          retrievable until then.
        * When several sources share one call and the process dies between two stamp
          requests, the already-stamped sources are complete and the rest read as changed.
        """
        coll = collection or self._collection_name
        if len(points) <= self._UPSERT_BATCH:
            if points:
                await self._client.upsert(collection_name=coll, points=points)
            return
        await self._write_blanked(points, coll)
        await self._stamp_hashes(points, coll)

    async def _replace_source(
        self, source_id: str, points: list[models.PointStruct], collection: str | None = None
    ) -> None:
        """Replace everything stored for *source_id* with *points*, new-before-old.

        Order: write the new points with blank hashes → delete the source's **stale** points
        (those whose id is not among *points*) → stamp the real hashes. The old points stay
        in place until every new point has landed, so a failed write leaves the previously
        stored data in place (subject to the overwrite limitation documented on
        :meth:`_batched_upsert`), and the source reads as "changed" until the final stamp —
        an interruption anywhere is re-processed by the next sync. Applies to single-batch
        writes too: the stamp is the commit marker for the whole replacement.

        With no *points* this is a plain delete of the source.
        """
        coll = collection or self._collection_name
        if not points:
            await self._delete_by_source_filter(source_id, collection=coll)
            return
        await self._write_blanked(points, coll)
        await self._client.delete(
            collection_name=coll,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))],
                must_not=[models.HasIdCondition(has_id=[p.id for p in points])],
            ),
        )
        await self._stamp_hashes(points, coll)

    async def _existing_ids(self, ids: list[models.ExtendedPointId], collection: str) -> set[models.ExtendedPointId]:
        """Return the subset of *ids* already present in *collection* (ids only, no bodies)."""
        existing: set[models.ExtendedPointId] = set()
        for i in range(0, len(ids), self._SCROLL_BATCH):
            records = await self._client.retrieve(
                collection_name=collection,
                ids=ids[i : i + self._SCROLL_BATCH],
                with_payload=False,
                with_vectors=False,
            )
            existing.update(record.id for record in records)
        return existing

    async def _write_blanked(self, points: list[models.PointStruct], collection: str) -> None:
        """Phase 1 of a two-phase write: upsert *points* in batches with blank hashes.

        On an ``Exception`` or a cancellation of a multi-batch write, the points this call
        created (ids absent before the call) are deleted — see :meth:`_batched_upsert` for
        the full contract.
        """
        multi_batch = len(points) > self._UPSERT_BATCH
        # Only a multi-batch write can be left partial, so only it needs the pre-image.
        preexisting = await self._existing_ids([p.id for p in points], collection) if multi_batch else set()
        sent: list[models.ExtendedPointId] = []
        try:
            for i in range(0, len(points), self._UPSERT_BATCH):
                batch = points[i : i + self._UPSERT_BATCH]
                # Recorded BEFORE the request: a failed/cancelled request may still have landed.
                sent.extend(p.id for p in batch)
                await self._client.upsert(collection_name=collection, points=[_blank_hashes(p) for p in batch])
        except asyncio.CancelledError:
            if multi_batch:
                # Shielded: the cleanup completes even though this task is being cancelled.
                await asyncio.shield(self._rollback_created(sent, preexisting, collection))
            raise
        except Exception as exc:
            if multi_batch:
                logger.warning(f"batched upsert failed part-way ({exc!r}); removing the points this call created")
                await self._rollback_created(sent, preexisting, collection)
            raise

    async def _rollback_created(
        self,
        sent: list[models.ExtendedPointId],
        preexisting: set[models.ExtendedPointId],
        collection: str,
    ) -> None:
        """Best-effort delete of the point ids this call created (never a pre-existing id)."""
        created = [pid for pid in sent if pid not in preexisting]
        if not created:
            return
        try:
            await self._client.delete(collection_name=collection, points_selector=models.PointIdsList(points=created))
        except Exception:
            logger.error(
                f"rollback of {len(created)} partially written point(s) failed; they keep blank hashes, "
                "so the next sync still re-processes their source",
                exc_info=True,
            )

    async def _stamp_hashes(self, points: list[models.PointStruct], collection: str) -> None:
        """Phase 2 of a two-phase write: set the real hashes on *points* (the commit marker).

        One ``set_payload`` request per distinct hash pair, addressed by point id — never by
        a ``source_id`` filter, which would also stamp stale points of the same source.
        """
        groups: dict[tuple[tuple[str, str | None], ...], list[models.ExtendedPointId]] = {}
        for point in points:
            payload = point.payload or {}
            hashes = tuple(
                (key, value if isinstance(value := payload[key], str) else None) for key in _HASH_KEYS if key in payload
            )
            if hashes:
                groups.setdefault(hashes, []).append(point.id)
        for hashes, ids in groups.items():
            await self._client.set_payload(
                collection_name=collection, payload=dict(hashes), points=models.PointIdsList(points=ids)
            )

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
