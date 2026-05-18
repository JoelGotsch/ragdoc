"""Vector store adapters for ragdoc's VectorStore protocol.

Each adapter is an optional integration — install the corresponding extra to use it:

* ``ragdoc[qdrant]`` — :class:`~ragdoc.integrations.vector_stores.qdrant.QdrantVectorStore`
"""
from __future__ import annotations

try:
    from ragdoc.integrations.vector_stores.qdrant import (
        QdrantIndex,
        QdrantVectorStore,
        ServerSideVector,
        register_indexes_from_type,
    )

    __all__ = ["QdrantIndex", "QdrantVectorStore", "ServerSideVector", "register_indexes_from_type"]
except ImportError:
    __all__ = []
