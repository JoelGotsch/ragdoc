"""Graph store adapters for ragdoc's GraphStore protocol.

* ``ragdoc[qdrant]`` — :class:`~ragdoc.integrations.graph_stores.qdrant.QdrantGraphStore`

For local development, see :class:`~ragdoc.extraction.graph_store.LocalGraphStore` (no extra).
"""

from __future__ import annotations

try:
    from ragdoc.integrations.graph_stores.qdrant import QdrantGraphStore

    __all__ = ["QdrantGraphStore"]
except ImportError:
    __all__ = []
