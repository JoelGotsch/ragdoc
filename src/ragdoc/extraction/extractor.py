"""Extractor protocol — the typed output channel for structured extraction (D3-A).

Extraction is a first-class pipeline stage, not a document processor: an
:class:`Extractor` *returns* typed :class:`~ragdoc.extraction.mention.Mention` objects instead of
smuggling serialized dicts through ``document.metadata``. The one legitimate document-dump use
case (mentions serialized inside a stored Document) survives via the explicit
:func:`as_processor` adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, runtime_checkable

from ragdoc.extraction.mention import Mention, PayloadT
from ragdoc.processing.base import DocumentProcessor

if TYPE_CHECKING:
    from ragdoc.document import Document


@runtime_checkable
class Extractor(Protocol[PayloadT]):
    """Anything that extracts typed, provenance-tagged mentions from a Document.

    ``extract`` is side-effect-free with respect to the document: it never reads or writes
    ``document.metadata`` keys of its own (it *copies* metadata into each mention's locator).

    ``payload_model`` is deliberately not part of the protocol — consumers of the returned
    mentions no longer need it, and the KG extractor's payload "model" is a discriminated-union
    alias, not a ``type[BaseModel]``. Both concrete extractors keep it as a documented public
    attribute for store queries and ``ChangeSet[Mention[...]]`` parametrization.
    """

    async def extract(self, document: Document) -> list[Mention[PayloadT]]:
        """Extract every mention of the payload type from *document*."""
        ...


class _ExtractorProcessor(DocumentProcessor, Generic[PayloadT]):
    """The :func:`as_processor` adapter — see its docstring for semantics."""

    def __init__(self, extractor: Extractor[PayloadT], metadata_key: str, overwrite: bool) -> None:
        self._extractor = extractor
        self._metadata_key = metadata_key
        self._overwrite = overwrite

    async def process(self, document: Document) -> Document:
        existing = document.metadata.get(self._metadata_key)
        if existing and not self._overwrite:
            return document
        # Strip a stale list BEFORE extracting so a mention's metadata locator never embeds a
        # previous mention list (the recursion guarantee the extractors' old snapshot provided).
        document.metadata.pop(self._metadata_key, None)
        mentions = await self._extractor.extract(document)
        document.metadata[self._metadata_key] = [m.model_dump(mode="json") for m in mentions]
        return document


def as_processor(
    extractor: Extractor[PayloadT],
    *,
    metadata_key: str = "mentions",
    overwrite: bool = False,
) -> DocumentProcessor:
    """Adapt *extractor* into a DocumentProcessor that dumps mentions into ``document.metadata``.

    Escape hatch for callers who genuinely want mentions serialized inside a Document (e.g. a
    DocumentStore dump). The write violates the library metadata rule by design and is therefore
    opt-in and explicit; ``MentionStorePipeline`` never uses it. Placed in a chunking pipeline,
    the key propagates into chunk metadata — that is the documented metadata contract ("any key
    present arrives on final chunks"), not a bug.

    Semantics of the returned processor:

    * ``metadata[metadata_key]`` already set (non-empty) and not *overwrite* → returned unchanged,
      no LLM call (idempotency guard).
    * Otherwise the key is popped before extraction (a mention's locator never embeds a previous
      mention list), then written as ``[m.model_dump(mode="json") for m in mentions]`` — an empty
      list when nothing was extracted, so the key is always present after processing.
    """
    return _ExtractorProcessor(extractor, metadata_key, overwrite)
