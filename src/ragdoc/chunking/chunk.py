from __future__ import annotations

import datetime
import uuid
from typing import Generic

from pydantic import BaseModel, Field, SkipValidation, model_validator
from typing_extensions import Self

from ragdoc.metadata import TMetadata, validate_metadata_dict


class Chunk(BaseModel, Generic[TMetadata]):
    """Represents a Document that is uploaded to a vector store.

    The optional type parameter ``TMetadata`` binds the ``metadata`` field to a
    user-declared :class:`~ragdoc.metadata.BaseMetadata` subclass, enabling
    typed access to metadata keys at retrieval time::

        chunks: list[Chunk[CustomChunkMetadata]] = await pipeline.run(path)
        reveal_type(chunks[0].metadata["document_name"])  # str

    Bare usage (``Chunk``, no type argument) is valid and treats ``metadata``
    as ``BaseMetadata``, so ``chunk.metadata["filename"]`` is always safe
    without a cast.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID of this chunk")
    source_path: str | None = Field(default=None, description="Full path to the source file")
    source_id: str = Field(
        ...,
        description=(
            "Sync identity key for this chunk's source document. REQUIRED. "
            "Derived from the source Path by DocumentPipeline.source_id_fn and propagated "
            "through chunking. Chunkers synthesize a fallback "
            "(doc.source_id or doc.source_path or doc.id) for documents produced outside a "
            "sync pipeline, so this field is never None."
        ),
    )
    source_hash: str = Field(
        ...,
        description=(
            "SHA-256 hex digest of the ORIGINAL SOURCE FILE bytes. REQUIRED. True file "
            "provenance — always reflects the bytes on disk, never the rendered Document "
            "content. Set by DocumentPipeline.hash_fn; chunkers fall back to "
            "doc.content_hash() when no file hash is available. For document-content change "
            "detection (Boundary 2), use `content_hash` instead."
        ),
    )
    content_hash: str | None = Field(
        default=None,
        description=(
            "Renderer-stable hash of the Document this chunk derived from "
            "(Document.content_hash()). Drives DocumentStore -> VectorStore change detection: "
            "a stored chunk is re-generated when its source Document's content_hash changes "
            "(e.g. a manual edit in the DocumentStore). Populated on the direct path too. "
            "None is treated as 'always changed' by Boundary-2 detection."
        ),
    )
    prompt_content: str = Field(..., description="Content used as prompt context")
    embedding_content: str = Field(..., description="Content used for vector Embedding")
    metadata: SkipValidation[TMetadata] = Field(  # type: ignore[assignment]
        default_factory=dict,
        description="Metadata attached to this chunk",
    )
    named_embeddings: dict[str, list[float]] = Field(
        default_factory=dict,
        description=(
            "Named embedding vectors keyed by embedder name (e.g. 'dense', 'sparse'). "
            "Populated by VectorStorePipeline before upsert. "
            "VectorStore implementations map these to named vector fields."
        ),
    )
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Timestamp of chunk creation",
    )

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        validate_metadata_dict(self.metadata)  # type: ignore[arg-type]
        return self
