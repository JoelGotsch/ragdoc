"""Entity store adapters for ragdoc's EntityStore protocol.

* ``ragdoc[qdrant]`` — :class:`~ragdoc.integrations.entity_stores.qdrant.QdrantEntityStore`

For local development, see :class:`~ragdoc.extraction.entity.LocalEntityStore` (no extra).
"""

from __future__ import annotations

try:
    from ragdoc.integrations.entity_stores.qdrant import QdrantEntityStore

    __all__ = ["QdrantEntityStore"]
except ImportError:
    __all__ = []
