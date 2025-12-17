"""Qdrant implementation of ragdoc's VectorStore protocol.

Requires the ``ragdoc[qdrant]`` extra::

    uv add --optional qdrant qdrant-client

Usage — dense only::

    from qdrant_client import AsyncQdrantClient
    from ragdoc.integrations.vector_stores import QdrantVectorStore

    client = AsyncQdrantClient("http://localhost:6333")
    store = await QdrantVectorStore.create(client, "my_collection", vector_size=1536)

Usage — dense + BM25 server-side sparse::

    from ragdoc.integrations.vector_stores import QdrantVectorStore, ServerSideVector
    from ragdoc.pipeline import prompt_content_text

    store = await QdrantVectorStore.create(
        client,
        "my_collection",
        vector_size=1536,
        sparse_vectors={"sparse": ServerSideVector(model="Qdrant/bm25", text_fn=prompt_content_text)},
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, get_args, get_origin

from ragdoc.chunking import Chunk
from ragdoc.integrations._qdrant_base import (
    AsyncQdrantClient,
    _create_index_ignore_conflict,
    _QdrantCollectionStore,
    models,
)
from ragdoc.pipeline.stores import SourceState

# ---------------------------------------------------------------------------
# QdrantIndex annotation marker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QdrantIndex:
    """Annotation marker for TypedDict fields that need a Qdrant payload index.

    Attach via ``Annotated`` to a field in a ``BaseMetadata`` subclass.
    Pass the desired ``PayloadSchemaType`` (e.g. ``KEYWORD``, ``INTEGER``,
    ``DATETIME``, ``FLOAT``, ``BOOL``, ``GEO``)::

        from typing import Annotated
        from qdrant_client.http import models
        from ragdoc.integrations.vector_stores import QdrantIndex

        class MyMetadata(ragdoc.BaseMetadata, total=False):
            document_name: Annotated[str, QdrantIndex(PayloadSchemaType.KEYWORD)]
            document_date: Annotated[str | None, QdrantIndex(PayloadSchemaType.DATETIME)]

    Pass ``metadata_type=MyMetadata`` to :meth:`QdrantVectorStore.create` or
    call :func:`register_indexes_from_type` directly to create the indexes.
    """

    schema_type: models.PayloadSchemaType


# ---------------------------------------------------------------------------
# register_indexes_from_type
# ---------------------------------------------------------------------------


async def register_indexes_from_type(
    metadata_type: type,
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """Register Qdrant payload indexes for all :class:`QdrantIndex`-annotated fields.

    Walks ``get_type_hints(metadata_type, include_extras=True)`` and calls
    ``create_payload_index`` for each field annotated with :class:`QdrantIndex`.
    TypedDict inheritance is handled automatically by ``get_type_hints``.

    ``409 Conflict`` responses (index already exists) are silently ignored so
    this function is safe to call on every application start.

    Args:
        metadata_type:   A TypedDict class (typically a subclass of
                         :class:`~ragdoc.metadata.BaseMetadata`) whose
                         ``Annotated`` fields carry :class:`QdrantIndex` markers.
        client:          An :class:`~qdrant_client.AsyncQdrantClient` instance.
        collection_name: Name of the Qdrant collection to index.

    Example:
        ```python
        from typing import Annotated
        from qdrant_client.http import models
        import ragdoc
        from ragdoc.integrations.vector_stores import QdrantIndex, register_indexes_from_type

        class MyMetadata(ragdoc.BaseMetadata, total=False):
            document_name: Annotated[str, QdrantIndex(models.PayloadSchemaType.KEYWORD)]
            document_date: Annotated[str | None, QdrantIndex(models.PayloadSchemaType.DATETIME)]

        await register_indexes_from_type(MyMetadata, client, "my_collection")
        ```
    """
    from typing import get_type_hints

    hints = get_type_hints(metadata_type, include_extras=True)
    for field_name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue
        for meta in get_args(hint)[1:]:
            if isinstance(meta, QdrantIndex):
                await _create_index_ignore_conflict(client, collection_name, field_name, meta.schema_type)


# ---------------------------------------------------------------------------
# ServerSideVector
# ---------------------------------------------------------------------------


def _default_bm25_text(chunk: Chunk) -> str:
    return chunk.prompt_content


@dataclass
class ServerSideVector:
    """Configuration for a Qdrant server-side inference sparse vector (e.g. BM25).

    Qdrant computes the sparse vector on the server from raw text, so no
    client-side embedding call is required.  The model must be available on
    the Qdrant instance (BM25 is built-in; other models require enabling
    server-side inference).

    Args:
        model:   Qdrant model identifier, e.g. ``"Qdrant/bm25"``.
        text_fn: Callable that extracts the text to pass to the server.
                 Defaults to ``chunk.prompt_content`` (full document text —
                 recommended for keyword-based retrieval).

    Example::

        from ragdoc.integrations.vector_stores import ServerSideVector
        from ragdoc.pipeline import prompt_content_text

        bm25 = ServerSideVector(model="Qdrant/bm25", text_fn=prompt_content_text)
    """

    model: str
    text_fn: Callable[[Chunk], str] = field(default=_default_bm25_text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk_to_point(
    chunk: Chunk,
    sparse_vectors: dict[str, ServerSideVector] | None = None,
) -> models.PointStruct:
    """Convert a Chunk to a Qdrant PointStruct.

    Args:
        chunk:          The chunk to convert.
        sparse_vectors: Server-side sparse vector configurations.  For each
                        entry a ``models.Document`` is added to the point's
                        vector dict so Qdrant computes the sparse embedding
                        server-side on ingest.

    Returns:
        A ``PointStruct`` ready for upsert.

    Raises:
        ValueError: If ``chunk.named_embeddings`` is empty and no
                    ``sparse_vectors`` are configured.
    """
    if not chunk.named_embeddings and not sparse_vectors:
        raise ValueError(
            f"Chunk '{chunk.id}' has no embeddings. "
            "Populate chunk.named_embeddings via VectorStorePipeline embedders "
            "or configure sparse_vectors on QdrantVectorStore."
        )
    payload = chunk.model_dump(mode="json", exclude={"id", "named_embeddings"})

    # Build the vector payload: merge client-side dense vectors with server-side documents.
    if sparse_vectors:
        vectors: dict = dict(chunk.named_embeddings)
        for name, config in sparse_vectors.items():
            vectors[name] = models.Document(text=config.text_fn(chunk), model=config.model)
        return models.PointStruct(id=chunk.id, vector=vectors, payload=payload)

    # No server-side vectors: use named dict if multiple dense vectors, else single unnamed.
    if len(chunk.named_embeddings) > 1:
        return models.PointStruct(id=chunk.id, vector=dict(chunk.named_embeddings), payload=payload)
    return models.PointStruct(
        id=chunk.id,
        vector=next(iter(chunk.named_embeddings.values())),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# QdrantVectorStore
# ---------------------------------------------------------------------------


class QdrantVectorStore(_QdrantCollectionStore):
    """Async VectorStore backed by Qdrant, implementing ragdoc's VectorStore protocol.

    Args:
        client:          An :class:`~qdrant_client.AsyncQdrantClient` instance.
                         The caller owns the client lifecycle (connect / close).
        collection_name: Qdrant collection to read/write.
        sparse_vectors:  Optional server-side sparse vector configurations.
                         When set, each entry produces a ``models.Document``
                         in the upserted point so Qdrant computes the sparse
                         embedding server-side (e.g. BM25).

    Requires:
        ``ragdoc[qdrant]`` extra (``qdrant-client>=1.9``).

    Example — dense + BM25::

        from ragdoc.integrations.vector_stores import QdrantVectorStore, ServerSideVector
        from ragdoc.pipeline import prompt_content_text

        store = await QdrantVectorStore.create(
            client,
            "docs",
            vector_size=1536,
            sparse_vectors={"sparse": ServerSideVector("Qdrant/bm25", text_fn=prompt_content_text)},
        )
        pipeline = VectorStorePipeline(document_pipeline, store, embedders={"dense": EmbedderConfig(my_embedder)})
        await pipeline.sync(paths)
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        sparse_vectors: dict[str, ServerSideVector] | None = None,
    ) -> None:
        super().__init__(client, collection_name)
        self._sparse_vectors = sparse_vectors or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
        sparse_vectors: dict[str, ServerSideVector] | None = None,
        dense_vector_name: str = "dense",
        metadata_type: type | None = None,
    ) -> QdrantVectorStore:
        """Create the collection (if it does not exist) and return a store instance.

        A payload index on ``source_id`` is created so that
        :meth:`delete_by_source` and :meth:`list_source_state` filter in
        O(log n) rather than O(n).

        When *sparse_vectors* are provided the collection is created with
        **named** dense and sparse vector fields, enabling hybrid retrieval:

        * Dense vector field: *dense_vector_name* (default ``"dense"``).
        * Sparse vector fields: one per entry in *sparse_vectors*.

        Qdrant's built-in ``"Qdrant/bm25"`` model is available on all
        deployments (Cloud and self-hosted).

        Args:
            client:            An :class:`~qdrant_client.AsyncQdrantClient`.
            collection_name:   Name of the Qdrant collection.
            vector_size:       Dimensionality of the dense embedding vectors.
            distance:          Distance metric (default: COSINE).
            sparse_vectors:    Server-side sparse vector configurations.
                               Triggers named-vector collection layout.
            dense_vector_name: Name for the dense vector field when
                               *sparse_vectors* are also configured.
                               Ignored when *sparse_vectors* is ``None``.
            metadata_type:     Optional TypedDict subclass of
                               :class:`~ragdoc.metadata.BaseMetadata`.
                               When provided, :func:`register_indexes_from_type`
                               is called to create Qdrant payload indexes for
                               all :class:`QdrantIndex`-annotated fields.

        Returns:
            A ready-to-use :class:`QdrantVectorStore`.
        """
        if sparse_vectors:
            # Named-vector layout: separate dense and sparse fields.
            vectors_cfg: models.VectorParams | dict[str, models.VectorParams] = {
                dense_vector_name: models.VectorParams(size=vector_size, distance=distance)
            }
            sparse_cfg: dict[str, models.SparseVectorParams] | None = {
                name: models.SparseVectorParams() for name in sparse_vectors
            }
        else:
            # Single unnamed vector — backward-compatible layout.
            vectors_cfg = models.VectorParams(size=vector_size, distance=distance)
            sparse_cfg = None
        await cls._ensure_collection(
            client,
            collection_name,
            vectors_cfg,
            sparse_vectors_config=sparse_cfg,
            payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
        )

        if metadata_type is not None:
            await register_indexes_from_type(metadata_type, client, collection_name)

        return cls(client, collection_name, sparse_vectors=sparse_vectors)

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk]) -> list[str]:
        """Upload or update *chunks* and return their stored IDs.

        Args:
            chunks: Chunks to upsert.  Each chunk must have ``named_embeddings``
                    populated, or the store must be configured with
                    ``sparse_vectors`` for server-side inference.

        Returns:
            IDs of the upserted chunks in the same order as *chunks*.

        Raises:
            ValueError: If any chunk has empty ``named_embeddings`` and no
                        ``sparse_vectors`` are configured.
        """
        if not chunks:
            return []
        points = [_chunk_to_point(c, self._sparse_vectors or None) for c in chunks]
        await self._batched_upsert(points)
        return [c.id for c in chunks]

    async def delete(self, ids: list[str]) -> None:
        """Remove chunks by ID.  IDs that do not exist are silently ignored.

        Args:
            ids: Chunk IDs to delete.
        """
        if not ids:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=list(ids)),
        )

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks whose ``source_id`` equals *source_id*.

        Args:
            source_id: The source identity key identifying chunks to remove.
        """
        await self._delete_by_source_filter(source_id)

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values stored in this collection.

        Paginates via Qdrant's scroll API so it works correctly at any scale.

        Returns:
            Set of source identity strings.
        """
        return await self._collect_source_ids()

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` for every source, in one scan.

        Scrolls the whole collection once, recording the first ``source_hash`` /
        ``content_hash`` seen per ``source_id`` (all chunks of a source share them).
        """
        return await self._list_source_state()
