"""ChangeSet: the serializable, reviewable artifact produced by ``plan()``.

A sync pipeline's ``plan()`` returns a :class:`ChangeSet` describing what would change; the
user may inspect or edit it (drop items, move source_ids into ``to_delete``) before
``apply()`` writes it to the store.  ``run()`` consumes the same event stream ``plan()``
buffers; it produces the same end state, differing only in scheduling (per-source
pre-write-and-write versus whole-corpus pre-write-first — see
:class:`~ragdoc.pipeline.sync.SyncEngine`).

The type parameter is the payload carried per source (any Pydantic model):

* ``ChangeSet[Document]`` — produced by ``DocumentStorePipeline`` (Boundary 1).
* ``ChangeSet[Chunk]`` — produced by ``VectorStorePipeline`` (direct path / Boundary 2).
* ``ChangeSet[Mention[P]]`` — produced by ``MentionStorePipeline``.

``ChangeSet`` is a Pydantic generic model.  Call :meth:`load` on the **concrete**
parametrization so Pydantic can resolve ``list[T]``::

    cs = ChangeSet[Chunk].load(path)      # ✓
    cs = ChangeSet.load(path)             # ✗ — T is unbound, items won't validate
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class SourceChange(BaseModel, Generic[T]):
    """The change for a single source: its identity and the items it produces.

    Attributes:
        source_id: The source identity key.
        source_hash: File-byte hash recorded for this source.
        content_hash: Document content hash (``None`` on the direct path when unknown).
        items: The payload for this source (e.g. ``list[Document]``, ``list[Chunk]``,
            ``list[Mention[P]]``). May be empty — "this source now yields nothing".
    """

    source_id: str
    source_hash: str
    content_hash: str | None = None
    items: list[T]


class ChangeSet(BaseModel, Generic[T]):
    """A planned set of changes, serializable for human review between plan and apply.

    Attributes:
        to_add: Sources new to the target store.
        to_update: Sources whose change-token differs (their stale entries are replaced).
        to_delete: source_ids to remove (orphans); empty unless ``delete_orphans=True``.
    """

    to_add: list[SourceChange[T]] = Field(default_factory=list)
    to_update: list[SourceChange[T]] = Field(default_factory=list)
    to_delete: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write this ChangeSet to *path* as indented JSON."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChangeSet[T]:
        """Load a ChangeSet from *path*. Call on the concrete type, e.g. ``ChangeSet[Chunk]``."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
