"""Qdrant implementation of ragdoc's DocumentStore protocol.

Requires the ``ragdoc[qdrant]`` extra::

    uv add --optional qdrant qdrant-client

:class:`QdrantDocumentStore` is the production / shared-access counterpart to
:class:`~ragdoc.pipeline.local_document_store.LocalDocumentStore`: parsed+processed
:class:`~ragdoc.document.Document` objects live in a Qdrant collection that multiple
workers can read and write, so a corpus can be parsed once and re-chunked many times without
re-parsing.

**Storage model.** Qdrant requires every point to carry a vector, but a Document at this stage
has no embedding. So each Document is stored as a **single point** with a throwaway
1-dimensional vector ``[0.0]`` that is never queried — this collection is intentionally a
key-value store keyed on ``source_id``, **not** semantically searchable. The point id is a
deterministic ``UUID5(source_id)`` so ``upsert`` replaces in place (one Document per
``source_id``, no versioning). The full document is serialized into the point payload,
alongside top-level ``source_id`` / ``source_hash`` / ``content_hash`` for filtering and cheap
bulk change-detection reads.

The 1-dim vector config is effectively permanent: adding real Document-level embeddings later
would require re-creating the collection.

Usage::

    from qdrant_client import AsyncQdrantClient
    from ragdoc.integrations.document_stores import QdrantDocumentStore

    client = AsyncQdrantClient("http://localhost:6333")
    store = await QdrantDocumentStore.create(client, "my_documents")
    await store.upsert([document])
    doc = await store.get_document("report.pdf")
"""

from __future__ import annotations

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

from ragdoc.document import Document
from ragdoc.pipeline.stores import SourceState

# Fixed namespace so UUID5 point ids are stable across processes and versions.
_NAMESPACE = uuid.UUID("b9c1f4e2-3a7d-5e8c-9f01-2d3c4b5a6e7f")
_SCROLL_BATCH = 1000
# Payload keys read by the bulk change-detection path (must stay cheap — no full bodies).
_STATE_KEYS = ["source_id", "source_hash", "content_hash"]


class DocumentTooLargeError(Exception):
    """Raised when a Document's serialized payload exceeds Qdrant's accepted size.

    Carries the offending ``source_id`` and the serialized ``size_bytes`` so callers can act
    (raise Qdrant's limit, externalize images, split upstream) instead of seeing a raw 4xx.
    """

    def __init__(self, source_id: str, size_bytes: int) -> None:
        self.source_id = source_id
        self.size_bytes = size_bytes
        super().__init__(
            f"Document {source_id!r} payload is {size_bytes} bytes, which Qdrant rejected as too "
            "large. Increase the Qdrant payload/body limit, or reduce the document size "
            "(e.g. externalize image data)."
        )


def _point_id(source_id: str) -> str:
    """Deterministic Qdrant point id for *source_id* (so upsert replaces in place)."""
    return str(uuid.uuid5(_NAMESPACE, source_id))


def _document_to_point(document: Document) -> models.PointStruct:
    """Convert a Document to a single Qdrant point (dummy vector + JSON payload).

    Raises:
        ValueError: if ``document.source_id`` is not set (it is the store key).
    """
    if not document.source_id:
        raise ValueError(
            "QdrantDocumentStore requires document.source_id to be set "
            "(normally done by DocumentStorePipeline before upsert)."
        )
    payload = document.model_dump(mode="json")
    # Provenance/hash fields are already top-level keys of the Document dump; ensure presence.
    payload["source_id"] = document.source_id
    payload["source_hash"] = document.source_hash or ""
    payload["content_hash"] = document.content_hash()
    return models.PointStruct(id=_point_id(document.source_id), vector=[0.0], payload=payload)


def _is_too_large(exc: UnexpectedResponse) -> bool:
    """Heuristically detect Qdrant rejecting a payload as too large."""
    if exc.status_code in (413,):
        return True
    raw = getattr(exc, "content", b"") or b""
    text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return exc.status_code == 400 and ("too large" in text.lower() or "payload" in text.lower())


class QdrantDocumentStore:
    """Async DocumentStore backed by Qdrant (one point per ``source_id``).

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient`. The caller owns its lifecycle.
        collection_name: Qdrant collection holding the Documents (separate from any vector
            collection).
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    async def create(cls, client: AsyncQdrantClient, collection_name: str) -> QdrantDocumentStore:
        """Create the collection (idempotent) and return a store instance.

        The collection uses a throwaway 1-dimensional vector (Documents are not embedded) and a
        KEYWORD payload index on ``source_id`` for O(log n) filtering. Unlike
        :meth:`QdrantVectorStore.create`, it takes **no** vector-size/distance parameters —
        there is no embedding to configure.

        Safe to call on every startup: a 409 (collection exists) is ignored.
        """
        try:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="source_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise
        return cls(client, collection_name)

    # ------------------------------------------------------------------
    # DocumentStore protocol
    # ------------------------------------------------------------------

    async def upsert(self, documents: list[Document]) -> list[str]:
        """Store or replace *documents* (keyed on ``source_id``); return their source_ids.

        Raises:
            ValueError: if any document has no ``source_id``.
            DocumentTooLargeError: if Qdrant rejects a document's payload as too large.
        """
        if not documents:
            return []
        points = [_document_to_point(doc) for doc in documents]
        try:
            await self._client.upsert(collection_name=self._collection_name, points=points)
        except UnexpectedResponse as exc:
            if _is_too_large(exc):
                # Attribute the failure to the largest document in the batch.
                worst = max(documents, key=lambda d: len(d.model_dump_json()))
                raise DocumentTooLargeError(worst.source_id or "", len(worst.model_dump_json())) from exc
            raise
        return [doc.source_id for doc in documents if doc.source_id]

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document stored for *source_id* (no-op if absent)."""
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[_point_id(source_id)]),
        )

    async def get_document(self, source_id: str) -> Document | None:
        """Return the single Document for *source_id* (O(1) by point id), or ``None``."""
        records = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[_point_id(source_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return Document.model_validate(records[0].payload)

    async def list_source_ids(self) -> set[str]:
        """Return all stored ``source_id`` values (scroll-paginated)."""
        source_ids: set[str] = set()
        offset: str | int | None = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=None,
                with_payload=["source_id"],
                with_vectors=False,
                limit=_SCROLL_BATCH,
                offset=offset,
            )
            for point in results:
                sid = (point.payload or {}).get("source_id")
                if sid is not None:
                    source_ids.add(sid)
            if next_offset is None:
                break
            offset = next_offset  # type: ignore[reportAssignmentType]  # qdrant PointId is str | int at runtime
        return source_ids

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every stored document.

        A **cheap** read: scrolls only the three hash payload keys with ``with_vectors=False``,
        so it never pulls multi-MB document bodies (keeps ``plan()`` fast).
        """
        state: dict[str, SourceState] = {}
        offset: str | int | None = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=None,
                with_payload=_STATE_KEYS,
                with_vectors=False,
                limit=_SCROLL_BATCH,
                offset=offset,
            )
            for point in results:
                payload = point.payload or {}
                sid = payload.get("source_id")
                if sid is not None:
                    state[sid] = SourceState(
                        source_hash=payload.get("source_hash") or "",
                        content_hash=payload.get("content_hash"),
                    )
            if next_offset is None:
                break
            offset = next_offset  # type: ignore[reportAssignmentType]  # qdrant PointId is str | int at runtime
        return state
