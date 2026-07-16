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

from ragdoc.document import Document
from ragdoc.integrations._qdrant_base import (
    AsyncQdrantClient,
    UnexpectedResponse,
    _QdrantCollectionStore,
    models,
)
from ragdoc.pipeline.stores import SourceState

# Fixed namespace so UUID5 point ids are stable across processes and versions.
_NAMESPACE = uuid.UUID("b9c1f4e2-3a7d-5e8c-9f01-2d3c4b5a6e7f")


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


_SIZE_REJECTION_PHRASES = ("too large", "larger than allowed")


def _is_too_large(exc: UnexpectedResponse) -> bool:
    """Heuristically detect Qdrant rejecting a payload as too large.

    Intentionally approximate. 413 (Payload Too Large) is unambiguous; for 400 we require an
    explicit size phrase (Qdrant emits ``"larger than allowed"``; ``"too large"`` covers proxies
    and older versions). Matching bare ``"payload"`` is too broad — Qdrant uses it for validation
    errors unrelated to size.
    """
    if exc.status_code == 413:
        return True
    raw = getattr(exc, "content", b"") or b""
    text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return exc.status_code == 400 and any(phrase in text.lower() for phrase in _SIZE_REJECTION_PHRASES)


class QdrantDocumentStore(_QdrantCollectionStore):
    """Async DocumentStore backed by Qdrant (one point per ``source_id``).

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient`. The caller owns its lifecycle.
        collection_name: Qdrant collection holding the Documents (separate from any vector
            collection).
    """

    _ID_NAMESPACE = _NAMESPACE

    def _document_to_point(self, document: Document) -> models.PointStruct:
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
        return models.PointStruct(id=self._point_id(document.source_id), vector=[0.0], payload=payload)

    @classmethod
    async def create(cls, client: AsyncQdrantClient, collection_name: str) -> QdrantDocumentStore:
        """Create the collection (idempotent) and return a store instance.

        The collection uses a throwaway 1-dimensional vector (Documents are not embedded) and a
        KEYWORD payload index on ``source_id`` for O(log n) filtering. Unlike
        :meth:`QdrantVectorStore.create`, it takes **no** vector-size/distance parameters —
        there is no embedding to configure.

        Safe to call on every startup: a 409 (collection exists) is ignored.
        """
        await cls._ensure_collection(
            client,
            collection_name,
            models.VectorParams(size=1, distance=models.Distance.DOT),
            payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
        )
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
        points = [self._document_to_point(doc) for doc in documents]
        try:
            await self._client.upsert(collection_name=self._collection_name, points=points)
        except UnexpectedResponse as exc:
            if _is_too_large(exc):
                # Attribute the failure to the largest document in the batch (serialize once each).
                sized = [(len(doc.model_dump_json()), doc) for doc in documents]
                worst_size, worst = max(sized, key=lambda t: t[0])
                raise DocumentTooLargeError(worst.source_id or "", worst_size) from exc
            raise
        return [doc.source_id for doc in documents if doc.source_id]

    async def delete_by_source(self, source_id: str) -> None:
        """Delete the document stored for *source_id* (no-op if absent).

        Overrides the base's filter delete: documents are keyed 1:1 on ``source_id``, so a
        direct point-id delete is cheaper.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(source_id)]),
        )

    async def get_document(self, source_id: str) -> Document | None:
        """Return the single Document for *source_id* (O(1) by point id), or ``None``."""
        records = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[self._point_id(source_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return Document.model_validate(records[0].payload)

    async def list_source_ids(self) -> set[str]:
        """Return all stored ``source_id`` values (scroll-paginated)."""
        return await self._collect_source_ids()

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every stored document.

        A **cheap** read: scrolls only the three hash payload keys with ``with_vectors=False``,
        so it never pulls multi-MB document bodies (keeps ``plan()`` fast).
        """
        return await self._list_source_state()
