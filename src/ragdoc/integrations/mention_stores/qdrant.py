"""Qdrant implementation of ragdoc's MentionStore protocol.

Requires the ``ragdoc[qdrant]`` extra::

    uv add --optional qdrant qdrant-client

:class:`QdrantMentionStore` is the shared-access counterpart to
:class:`~ragdoc.extraction.stores.LocalMentionStore`: each
:class:`~ragdoc.extraction.mention.Mention` is one Qdrant point whose **vector is the
mention's identity embedding** (so the same collection doubles as the blocking index for entity
resolution) and whose payload is the serialized mention (with top-level ``source_id`` /
``source_hash`` / ``content_hash`` for filtering and cheap bulk change detection).

The point id is a deterministic ``UUID5(mention_id)`` so re-upserting the same mention replaces it
in place. ``delete_by_source`` filters on the ``source_id`` payload index.

Usage::

    from qdrant_client import AsyncQdrantClient
    from ragdoc.extraction import build_entity_embedder
    from ragdoc.integrations.mention_stores import QdrantMentionStore

    embedder = build_entity_embedder(my_embed, identity_text_fn)
    client = AsyncQdrantClient("http://localhost:6333")
    store = await QdrantMentionStore.create(client, "mentions", Event, embedder, vector_size=1536)
    await store.upsert(mentions)
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ragdoc.extraction.mention import Mention
from ragdoc.integrations._qdrant_base import AsyncQdrantClient, _QdrantCollectionStore, models
from ragdoc.pipeline.stores import SourceState

_NAMESPACE = uuid.UUID("a4e6c0d1-2b3f-4a5e-8c7d-9e0f1a2b3c4d")
PayloadEmbedder = Callable[[list[BaseModel]], Awaitable[list[list[float]]]]


class QdrantMentionStore(_QdrantCollectionStore):
    """Async :class:`~ragdoc.extraction.stores.MentionStore` backed by Qdrant.

    Args:
        client: An :class:`~qdrant_client.AsyncQdrantClient` (caller owns its lifecycle).
        collection_name: Qdrant collection holding the mentions.
        payload_model: The payload model to rehydrate mentions into (e.g. ``Event``).
        embedder: A payload embedder (``list[payload] -> vectors``), e.g. from
            :func:`~ragdoc.extraction.build_entity_embedder`. Used to embed a mention's
            identity at upsert so the vector doubles as the resolution blocking index.
    """

    _ID_NAMESPACE = _NAMESPACE

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        payload_model: type[BaseModel],
        embedder: PayloadEmbedder,
    ) -> None:
        super().__init__(client, collection_name)
        self._payload_model = payload_model
        self._embedder = embedder

    def _mention_type(self) -> type[Mention]:
        payload_model = self._payload_model
        return Mention[payload_model]

    @classmethod
    async def create(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        payload_model: type[BaseModel],
        embedder: PayloadEmbedder,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> QdrantMentionStore:
        """Create the collection (idempotent) with a ``source_id`` payload index; return a store."""
        await cls._ensure_collection(
            client,
            collection_name,
            models.VectorParams(size=vector_size, distance=distance),
            payload_indexes={"source_id": models.PayloadSchemaType.KEYWORD},
        )
        return cls(client, collection_name, payload_model, embedder)

    async def upsert(self, mentions: list[Mention]) -> list[str]:
        """Embed each mention's identity and upsert one point per mention; return their ids."""
        if not mentions:
            return []
        vectors = await self._embedder([m.payload for m in mentions])
        points = [
            models.PointStruct(id=self._point_id(m.mention_id), vector=vec, payload=m.model_dump(mode="json"))
            for m, vec in zip(mentions, vectors, strict=True)
        ]
        await self._batched_upsert(points)
        return [m.mention_id for m in mentions]

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all mentions whose ``source_id`` equals *source_id*."""
        await self._delete_by_source_filter(source_id)

    async def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values with mentions present (scroll-paginated)."""
        return await self._collect_source_ids()

    async def list_source_state(self) -> dict[str, SourceState]:
        """Return ``source_id`` -> :class:`SourceState` (cheap: only the hash payload keys)."""
        return await self._list_source_state()

    async def list_mentions(self, payload_type: type[BaseModel] | None = None) -> list[Mention]:
        """Return stored mentions, optionally filtered to those whose payload is *payload_type*.

        Filtering uses ``isinstance(mention.payload, payload_type)`` after rehydration, mirroring
        the :class:`~ragdoc.extraction.stores.MentionStore` Protocol contract and the
        in-memory :class:`LocalMentionStore` implementation. A heterogeneous store (e.g. one
        carrying the merged ``KnowledgeGraphProcessor`` node + edge union) returns only the
        node or edge mentions of the requested concrete type; a single-type store returns
        everything when the type matches and ``[]`` otherwise.
        """
        mtype = self._mention_type()
        mentions = [mtype.model_validate(payload) async for payload in self._scroll_payloads(True)]
        if payload_type is None:
            return mentions
        return [m for m in mentions if isinstance(m.payload, payload_type)]
