"""Serializable change set produced by ``MentionStorePipeline.plan()``.

Mirrors :class:`~ragdoc.pipeline.changeset.ChangeSet` but carries mentions. The shared
``ChangeSet`` is constrained to ``Document``/``Chunk``, so extraction defines its own generic
variant over the payload model.

``MentionChangeSet`` is a Pydantic generic; call :meth:`load` on the **concrete** parametrization so
``list[Mention[T]]`` resolves::

    cs = MentionChangeSet[Event].load(path)   # ✓
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic

from pydantic import BaseModel, Field

from ragdoc.extraction.mention import Mention, PayloadT


class MentionSourceChange(BaseModel, Generic[PayloadT]):
    """The mentions extracted for a single source, with its change tokens."""

    source_id: str
    source_hash: str
    content_hash: str | None = None
    items: list[Mention[PayloadT]] = Field(default_factory=list)


class MentionChangeSet(BaseModel, Generic[PayloadT]):
    """A planned set of mention changes, serializable for review between plan and apply."""

    to_add: list[MentionSourceChange[PayloadT]] = Field(default_factory=list)
    to_update: list[MentionSourceChange[PayloadT]] = Field(default_factory=list)
    to_delete: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write this change set to *path* as indented JSON."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MentionChangeSet[PayloadT]:
        """Load a change set from *path*. Call on the concrete type, e.g. ``MentionChangeSet[Event]``."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
