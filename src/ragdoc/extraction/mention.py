""":class:`Mention` — one raw, provenance-tagged extraction occurrence.

A :class:`Mention` wraps a caller-supplied payload model (e.g. ``Event``) in an envelope of
provenance the payload never sees. Mentions are the **Layer-1** product of structured extraction:
one per occurrence, synced per source like ``Chunk``\\ s, and later clustered into canonical
``Entity`` objects by the resolution pass.

Provenance comes from two places, by the library's metadata rule:

* the first-class change tokens (``source_id``, ``source_hash``, ``content_hash``) are read from
  the source :class:`~ragdoc.document.Document`'s first-class fields, never from metadata;
* the **locator** rides in ``metadata`` — a verbatim copy of the (post-split) Document's metadata,
  carrying ``split_sequence``/``split_total`` plus any user-defined keys. ``(source_id,
  split_sequence)`` is a stable, re-runnable locator back to the split's text.

``mention_id`` is a **minted primary key**, not a reproducible identity: the payload is
LLM-generated, so a re-extraction of unchanged content may differ. Idempotency lives one level up,
in the source-level hash gate of :class:`~ragdoc.extraction.pipeline.MentionStorePipeline`
(unchanged sources are never re-extracted). The id hashes ``source_id`` + ``content_hash`` +
``ordinal`` (disambiguating near-identical payloads in one source) + the payload, so a byte-identical
re-run merely coincides.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from ragdoc.metadata import MetadataDict

PayloadT = TypeVar("PayloadT", bound=BaseModel)
"""The caller-supplied extraction payload model (e.g. ``Event``)."""


def mint_mention_id(source_id: str, content_hash: str | None, ordinal: int, payload: BaseModel) -> str:
    """Return the deterministic primary-key hash for a mention.

    Content-addressed over ``(source_id, content_hash, ordinal, payload)`` so two near-identical
    payloads in one source still get distinct ids (via *ordinal*) and a byte-identical re-extraction
    coincides. Not relied upon for cross-run stability — see the module docstring.
    """
    canonical = payload.model_dump_json()
    raw = "\x00".join([source_id, content_hash or "", str(ordinal), canonical])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Mention(BaseModel, Generic[PayloadT]):
    """A single extracted occurrence of a payload entity, with provenance.

    The optional type parameter binds :attr:`payload` to the caller's model::

        mentions: list[Mention[Event]] = ...
    """

    mention_id: str = Field(description="Minted primary key (see mint_mention_id); not a stable identity.")
    source_id: str = Field(description="Sync identity key of the source this mention came from.")
    source_path: str | None = Field(default=None, description="Full path to the source file, if known.")
    source_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the original source file bytes (Boundary-1 change token). Honestly "
            "optional: None when no file-byte hash exists — never faked from the content hash."
        ),
    )
    content_hash: str | None = Field(
        default=None,
        description="Renderer-stable hash of the source Document (Boundary-2 change token; None ⇒ always changed).",
    )
    ordinal: int = Field(default=0, description="Position of this mention within its source's extraction.")
    metadata: MetadataDict = Field(
        default_factory=dict,
        description="Verbatim copy of the (post-split) Document's metadata: split_sequence/split_total + custom keys.",
    )
    payload: PayloadT = Field(description="The caller-supplied payload instance (e.g. an Event).")
    created_at: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc),
        description="Timestamp of mention creation.",
    )


def finalize_mention(
    mention: Mention[PayloadT],
    *,
    content_hash: str | None,
    source_id: str | None = None,
    source_hash: str | None = None,
) -> Mention[PayloadT]:
    """Stamp authoritative provenance onto *mention* and re-mint its ``mention_id``.

    Used by :class:`~ragdoc.extraction.pipeline.MentionStorePipeline` so a source's
    mentions share one ``content_hash`` (uniform per source, as required for change detection),
    overriding the best-effort values the extractor computed. ``source_hash`` is stamped as
    given — ``None`` means "no file-byte hash exists" (honestly optional, never faked from the
    content hash). ``ordinal`` and ``payload`` are unchanged, so re-minting is deterministic.
    """
    if source_id is not None:
        mention.source_id = source_id
    mention.source_hash = source_hash
    mention.content_hash = content_hash
    mention.mention_id = mint_mention_id(mention.source_id, content_hash, mention.ordinal, mention.payload)
    return mention


def rewrite_edge_refs(payload: BaseModel, id_map: Mapping[str, str]) -> None:
    """Rewrite an edge-carrying payload's endpoint refs through *id_map* (old -> new mention id).

    Generic over any payload carrying a ``refs`` model with ``source_mention_id`` /
    ``target_mention_id`` string fields (the :class:`~ragdoc.extraction.schema.EdgeRef` shape),
    checked structurally so no single schema is special-cased. Payloads without such refs are
    left untouched; ref values absent from *id_map* are preserved (the caller decides whether a
    dangling ref is an error — e.g. kg_resolution surfaces them as orphans).
    """
    refs = getattr(payload, "refs", None)
    if refs is None:
        return
    src = getattr(refs, "source_mention_id", None)
    tgt = getattr(refs, "target_mention_id", None)
    if isinstance(src, str) and src in id_map:
        refs.source_mention_id = id_map[src]
    if isinstance(tgt, str) and tgt in id_map:
        refs.target_mention_id = id_map[tgt]
