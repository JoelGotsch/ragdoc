"""KnowledgeGraphProcessor: multi-type extraction → typed node + edge mentions.

Mirrors :class:`~ragdoc.extraction.processor.StructuredExtractionProcessor` and **satisfies
the same interface** (``metadata_key`` + ``payload_model`` + ``process()``) so it composes with the
existing :class:`~ragdoc.extraction.pipeline.MentionStorePipeline` unchanged. One
``.parse()`` call per chunk returns both node mentions (discriminated union over the schema's node
types, each carrying a chunk-local ``local_id``) and edge mentions (discriminated union over the
schema's edge types, each carrying ``refs: EdgeRef`` whose ``source_mention_id`` /
``target_mention_id`` are also chunk-local ids).

The processor then performs the **chunk-local-id → mention-id rewrite**: mints a real
``mention_id`` for each extracted node, builds a ``local_id → mention_id`` map, and rewrites each
edge's two endpoint references through it. Edges that reference a non-existent ``local_id`` raise
``ValueError`` — silent drops would corrupt the graph.

Final output is **one merged list** written to ``document.metadata[metadata_key]`` — nodes first,
then edges, with contiguous ordinals. The downstream :class:`MentionStorePipeline` validates each
entry against ``payload_model`` (the discriminated union over node + edge types), so node and
edge mentions live in one heterogeneous store and the resolution pipeline slices them by type via
``mention_store.list_mentions(Person)`` / ``list_mentions(Employment)``.

Optional **gleaning** runs one additional ``.parse()`` call asking the LLM for any entities it
missed; results are merged with offset chunk-local ids to avoid collision. Gleaning is gated by
``EXTRACTION_GLEANING`` env var (default off) or the constructor's ``gleaning=`` flag.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from ragdoc.extraction.mention import Mention, mint_mention_id
from ragdoc.extraction.processor import ExtractionSettings
from ragdoc.extraction.schema import (
    GraphSchema,
    build_edge_union,
    build_node_union,
)
from ragdoc.processing.base import DocumentProcessor
from ragdoc.utils import Tokenizer

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from ragdoc.document import Document
    from ragdoc.rendering import Renderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


KG_EXTRACTION_SYSTEM_PROMPT = """
You are an expert at extracting structured knowledge graphs from documents.

You will be given a piece of a document. Identify every distinct entity (node) and every typed
relationship (edge) the text describes, following the registered schema. Rules:

1. Extract only what the text actually supports — never invent facts.
2. Each node mention has a `local_id` (e.g. "n0", "n1", "n2", ...) you assign in order; use these
   ids consistently from edges in this same response. The ids are chunk-local — only meaningful
   inside this one extraction call.
3. Each edge mention's `refs.source_mention_id` and `refs.target_mention_id` MUST be `local_id`
   values you assigned to nodes in the same response. Do NOT invent ids that don't match any node.
4. Return one record per distinct occurrence; if the same entity or relationship is described
   twice, return it twice.
5. Fill each field as completely as the text allows; leave a field null when the text does not say.
6. For dates, copy the wording verbatim into the date's original text and give the canonical form.

If the text contains no entities of the registered kinds, return empty lists.
""".strip()


KG_USER_MESSAGE = "Extract all matching nodes and edges from the following content."


KG_GLEANING_MESSAGE = (
    "Are there any additional entities or relationships in the text above that you missed? "
    "If yes, return them — assigning NEW chunk-local ids (continue numbering: if you previously "
    "used n0..n{last}, start the new ones at n{next}). If no, return empty lists."
)


# ---------------------------------------------------------------------------
# Runtime schema (lru-cached per GraphSchema)
# ---------------------------------------------------------------------------


def build_graph_batch_model(schema: GraphSchema) -> tuple[type[BaseModel], type[BaseModel]]:
    """Build the ``GraphBatch`` and ``NodeWithLocalId`` Pydantic models for a *schema*.

    ``GraphBatch`` is the ``response_format`` for the LLM call: ``node_mentions`` is a list of
    ``NodeWithLocalId`` (a thin wrapper carrying the chunk-local id alongside the validated
    node), and ``edge_mentions`` is a list of the schema's edge discriminated union. Built fresh
    per ``schema`` and cached so the LLM call's ``response_format`` is referentially stable.
    """
    return _build_graph_batch_model(_SchemaKey(schema))


class _SchemaKey:
    """Hashable wrapper for lru_cache (GraphSchema itself is a BaseModel — hashable but the
    cache key wants something simple). Identity on ``(node_types, edge_types)``."""

    __slots__ = ("_key", "_schema")

    def __init__(self, schema: GraphSchema) -> None:
        self._schema = schema
        self._key = (schema.node_types, schema.edge_types)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SchemaKey) and self._key == other._key


@cache
def _build_graph_batch_model(key: _SchemaKey) -> tuple[type[BaseModel], type[BaseModel]]:
    schema = key._schema
    node_union = build_node_union(schema.node_types)
    edge_union = build_edge_union(schema.edge_types)

    node_with_local_id = create_model(
        "NodeWithLocalId",
        local_id=(
            str,
            Field(description="Chunk-local id (e.g. 'n0', 'n1'). Edges reference nodes by this id."),
        ),
        node=(node_union, Field(description="The extracted node payload.")),
    )
    graph_batch = create_model(
        "GraphBatch",
        node_mentions=(
            list[node_with_local_id],
            Field(default_factory=list, description="Node mentions with chunk-local ids."),
        ),
        edge_mentions=(
            list[edge_union],
            Field(
                default_factory=list,
                description=(
                    "Edge mentions. Each refs.source_mention_id / refs.target_mention_id is a "
                    "chunk-local id from the node list."
                ),
            ),
        ),
    )
    return graph_batch, node_with_local_id


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_kg_messages(
    text: str,
    *,
    system_prompt: str = KG_EXTRACTION_SYSTEM_PROMPT,
    user_message: str = KG_USER_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for a KG extraction LLM call."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_message}\n\n{text}"},
    ]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class KnowledgeGraphProcessor(DocumentProcessor):
    """Extracts typed nodes and typed edges into a single merged metadata key.

    Satisfies the same interface as
    :class:`~ragdoc.extraction.processor.StructuredExtractionProcessor`
    (``metadata_key`` + ``payload_model`` + ``process()``) so it drops into the existing
    :class:`~ragdoc.extraction.pipeline.MentionStorePipeline` unchanged. The merged list
    holds node mentions first (in extraction order), then edge mentions, both with the same
    ``Mention`` envelope shape; downstream code distinguishes them by ``payload.kind`` (the
    discriminator on every node and edge model).

    Idempotent by default: if ``metadata_key`` is already set (non-empty) and ``overwrite`` is
    False, :meth:`process` returns unchanged without an LLM call.

    Args:
        schema: :class:`~ragdoc.extraction.schema.GraphSchema` declaring node types,
            edge types, and legal patterns.
        client: Async OpenAI-compatible client; ``None`` falls back to ``get_config().openai_client``.
        model: Model name; ``None`` falls back to ``settings.model_name`` then config default.
        settings: :class:`~ragdoc.extraction.processor.ExtractionSettings`; ``None`` reads
            env (``EXTRACTION_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        tokenizer: Tokenizer for the ``min_tokens`` decision. ``None`` uses ``GPTTokenizer``.
        gleaning: When True, runs one additional ``.parse()`` after the first call asking the
            LLM for any missed entities. ``None`` reads ``EXTRACTION_GLEANING`` (default off).
        metadata_key: Single metadata key to write (default ``"kg_mentions"``); the merged
            node + edge mention list.
        overwrite: When False (default), skip documents that already have the key set.
    """

    def __init__(
        self,
        schema: GraphSchema,
        client: Any | None = None,
        model: str | None = None,
        settings: ExtractionSettings | None = None,
        renderer: Renderer | None = None,
        tokenizer: Tokenizer | None = None,
        gleaning: bool | None = None,
        metadata_key: str = "kg_mentions",
        overwrite: bool = False,
    ) -> None:
        self._schema = schema
        self.settings = settings or ExtractionSettings()
        self._client = client
        self._model = model
        self._renderer = renderer
        self._tokenizer = tokenizer
        self._gleaning = gleaning if gleaning is not None else _read_gleaning_env()
        self.metadata_key = metadata_key
        self.overwrite = overwrite

    @property
    def schema(self) -> GraphSchema:
        return self._schema

    @property
    def payload_model(self) -> Any:
        """The discriminated union over node + edge types — the merged payload model.

        Read by :class:`~ragdoc.extraction.pipeline.MentionStorePipeline` to build the
        ``Mention[payload_model]`` validator. Cached on first read because building a
        discriminated union over the schema's types is pure and the schema is immutable.
        """
        if not hasattr(self, "_payload_model_cache"):
            self._payload_model_cache = build_node_union(
                tuple(self._schema.node_types) + tuple(self._schema.edge_types)
            )
        return self._payload_model_cache

    # ------------------------------------------------------------------
    # fallbacks (mirror StructuredExtractionProcessor)
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from ragdoc.config import get_config

        client = get_config().openai_client
        if client is None:
            raise ValueError("No client provided and get_config().openai_client is None.")
        return client

    def _get_model(self) -> str:
        if self._model is not None:
            return self._model
        if self.settings.model_name:
            return self.settings.model_name
        from ragdoc.config import get_config

        return get_config().default_llm_model

    def _get_renderer(self) -> Renderer:
        if self._renderer is not None:
            return self._renderer
        from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

        return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        from ragdoc.utils import GPTTokenizer

        return GPTTokenizer()

    # ------------------------------------------------------------------
    # extraction
    # ------------------------------------------------------------------

    async def _extract_batch(
        self,
        text: str,
        client: Any,
        model: str,
        graph_batch_cls: type[BaseModel],
        *,
        system_prompt: str = KG_EXTRACTION_SYSTEM_PROMPT,
        user_message: str = KG_USER_MESSAGE,
    ) -> BaseModel:
        """One structured ``.parse()`` call with retries; returns the GraphBatch instance."""
        settings = self.settings
        messages = build_kg_messages(text, system_prompt=system_prompt, user_message=user_message)
        parse_kwargs: dict[str, Any] = {}
        if settings.request_timeout is not None:
            parse_kwargs["timeout"] = settings.request_timeout
        for attempt in range(settings.max_retries + 1):
            try:
                response = await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=settings.temperature,
                    response_format=graph_batch_cls,
                    **parse_kwargs,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("LLM returned no parsed extraction")
                return parsed
            except Exception as exc:
                logger.error(
                    "KnowledgeGraphProcessor: API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    settings.max_retries + 1,
                    exc,
                )
                if attempt == settings.max_retries:
                    raise
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _extract_with_halving(
        self,
        text: str,
        client: Any,
        model: str,
        graph_batch_cls: type[BaseModel],
        *,
        depth: int = 0,
        max_depth: int = 3,
        min_chars: int = 1000,
    ) -> BaseModel:
        """Extract from *text* with reactive halving on failure.

        Mirrors :class:`DocumentSummarizerProcessor`'s recursive fold for the extractor:
        when ``_extract_batch`` raises (timeout, validation failure, etc.) we halve *text*
        at the closest paragraph boundary to its midpoint, recurse on each half, and merge
        the resulting batches using the same chunk-local-id offset trick as gleaning. The
        halving is **internal to the LLM call**: it never touches ``document.metadata``, so
        the caller's ``split_sequence`` propagation is preserved — every resulting mention
        still attributes to the parent split that the agent can retrieve via
        ``get_document_part(name, split_sequence)``.

        Loss-of-information caveat: an edge whose two endpoints fall on opposite sides of a
        halving boundary cannot be emitted (each half-call sees only its own nodes). In typical
        prose most short-range ties live within the same paragraph, so the loss is generally
        deemed acceptable below ``max_depth``. Above ``max_depth`` or below ``min_chars`` we
        re-raise so the source goes to ``result.errors`` honestly rather than producing a
        bad-quality extraction silently.
        """
        try:
            return await self._extract_batch(text, client, model, graph_batch_cls)
        except Exception as exc:
            if depth >= max_depth or len(text) <= min_chars:
                logger.error(
                    "KnowledgeGraphProcessor: halving exhausted at depth=%d len=%d (%r); giving up",
                    depth,
                    len(text),
                    exc,
                )
                raise
            left, right = _halve_text(text)
            if not left.strip() or not right.strip():
                logger.error(
                    "KnowledgeGraphProcessor: could not halve text at depth=%d len=%d (%r); giving up",
                    depth,
                    len(text),
                    exc,
                )
                raise
            logger.warning(
                "KnowledgeGraphProcessor: halving on failure at depth=%d (len %d -> %d + %d): %s",
                depth,
                len(text),
                len(left),
                len(right),
                exc,
            )
            primary = await self._extract_with_halving(
                left,
                client,
                model,
                graph_batch_cls,
                depth=depth + 1,
                max_depth=max_depth,
                min_chars=min_chars,
            )
            secondary = await self._extract_with_halving(
                right,
                client,
                model,
                graph_batch_cls,
                depth=depth + 1,
                max_depth=max_depth,
                min_chars=min_chars,
            )
            offset = len(primary.node_mentions)  # type: ignore[attr-defined]
            merged_nodes, merged_edges = _merge_gleaning(
                list(primary.node_mentions),  # type: ignore[attr-defined]
                list(primary.edge_mentions),  # type: ignore[attr-defined]
                list(secondary.node_mentions),  # type: ignore[attr-defined]
                list(secondary.edge_mentions),  # type: ignore[attr-defined]
                offset,
            )
            return graph_batch_cls(node_mentions=merged_nodes, edge_mentions=merged_edges)

    async def process(self, document: Document) -> Document:
        existing = document.metadata.get(self.metadata_key)
        if not self.overwrite and existing:
            logger.debug("KnowledgeGraphProcessor: %s already set, skipping", self.metadata_key)
            return document

        renderer = self._get_renderer()
        rendered = renderer.render(document)
        if not rendered.strip():
            document.metadata[self.metadata_key] = []
            return document
        if self.settings.min_tokens > 0 and self._get_tokenizer().count(rendered) < self.settings.min_tokens:
            logger.debug("KnowledgeGraphProcessor: below min_tokens, skipping")
            document.metadata[self.metadata_key] = []
            return document

        graph_batch_cls, _ = build_graph_batch_model(self._schema)
        client = self._get_client()
        model = self._get_model()

        primary = await self._extract_with_halving(rendered, client, model, graph_batch_cls)
        wrapped_nodes = list(primary.node_mentions)  # type: ignore[attr-defined]
        edge_payloads = list(primary.edge_mentions)  # type: ignore[attr-defined]

        if self._gleaning:
            # Re-number local ids in the second batch so they don't collide with the first.
            offset = len(wrapped_nodes)
            gleaning_message = (
                f"{KG_GLEANING_MESSAGE.format(last=offset - 1, next=offset)}"
                if offset > 0
                else KG_GLEANING_MESSAGE.format(last=-1, next=0)
            )
            secondary = await self._extract_batch(
                rendered,
                client,
                model,
                graph_batch_cls,
                user_message=gleaning_message,
            )
            wrapped_nodes, edge_payloads = _merge_gleaning(
                wrapped_nodes,
                edge_payloads,
                # ``graph_batch_cls`` is a Pydantic model built at runtime, so its
                # mention fields are invisible to static analysis.
                list(secondary.node_mentions),  # pyright: ignore[reportAttributeAccessIssue]
                list(secondary.edge_mentions),  # pyright: ignore[reportAttributeAccessIssue]
                offset,
            )

        content_hash = document.content_hash()
        source_id = document.source_id or document.source_path or document.id
        source_hash = document.source_hash or content_hash
        # Snapshot metadata BEFORE writing our key.
        meta_snapshot = {k: v for k, v in document.metadata.items() if k != self.metadata_key}

        # Build the merged mention list (nodes first so the local_id map is ready for edges).
        # Ordinals are contiguous across nodes + edges — single position-within-source counter.
        all_mentions: list[Mention] = []
        local_to_mention: dict[str, str] = {}
        ordinal = 0

        for wrapped in wrapped_nodes:
            local_id: str = wrapped.local_id
            payload = wrapped.node
            if local_id in local_to_mention:
                raise ValueError(
                    f"Duplicate node local_id {local_id!r} in extraction output; the LLM must "
                    f"assign each node a unique chunk-local id."
                )
            mention = Mention(
                mention_id=mint_mention_id(source_id, content_hash, ordinal, payload),
                source_id=source_id,
                source_path=document.source_path or None,
                source_hash=source_hash,
                content_hash=content_hash,
                ordinal=ordinal,
                metadata=meta_snapshot,
                payload=payload,
            )
            local_to_mention[local_id] = mention.mention_id
            all_mentions.append(mention)
            ordinal += 1

        for edge_payload in edge_payloads:
            refs = edge_payload.refs
            src_local = refs.source_mention_id
            tgt_local = refs.target_mention_id
            if src_local not in local_to_mention:
                raise ValueError(
                    f"Edge {edge_payload.kind!r} references unknown source local_id {src_local!r}; "
                    f"known local_ids: {sorted(local_to_mention.keys())}"
                )
            if tgt_local not in local_to_mention:
                raise ValueError(
                    f"Edge {edge_payload.kind!r} references unknown target local_id {tgt_local!r}; "
                    f"known local_ids: {sorted(local_to_mention.keys())}"
                )
            refs.source_mention_id = local_to_mention[src_local]
            refs.target_mention_id = local_to_mention[tgt_local]

            mention = Mention(
                mention_id=mint_mention_id(source_id, content_hash, ordinal, edge_payload),
                source_id=source_id,
                source_path=document.source_path or None,
                source_hash=source_hash,
                content_hash=content_hash,
                ordinal=ordinal,
                metadata=meta_snapshot,
                payload=edge_payload,
            )
            all_mentions.append(mention)
            ordinal += 1

        document.metadata[self.metadata_key] = [m.model_dump(mode="json") for m in all_mentions]
        return document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_gleaning_env() -> bool:
    import os

    raw = os.environ.get("EXTRACTION_GLEANING", "").lower()
    return raw in ("1", "true", "yes", "on")


def _halve_text(text: str) -> tuple[str, str]:
    """Split *text* at the closest paragraph (double-newline) boundary to the midpoint.

    Falls back to single-newline, then to the literal character midpoint. Used by
    :meth:`KnowledgeGraphProcessor._extract_with_halving` to recursively shrink an LLM
    input that triggered a pathological extraction (timeout, validation failure).
    """
    if not text:
        return "", ""
    mid = len(text) // 2
    for sep in ("\n\n", "\n"):
        idx_before = text.rfind(sep, 0, mid)
        idx_after = text.find(sep, mid)
        if idx_before == -1 and idx_after == -1:
            continue
        if idx_before == -1:
            boundary = idx_after
        elif idx_after == -1:
            boundary = idx_before
        else:
            # Prefer whichever boundary is closer to the midpoint.
            boundary = idx_before if (mid - idx_before <= idx_after - mid) else idx_after
        return text[:boundary].rstrip(), text[boundary:].lstrip()
    return text[:mid], text[mid:]


def _merge_gleaning(
    primary_nodes: list,
    primary_edges: list,
    secondary_nodes: list,
    secondary_edges: list,
    offset: int,
) -> tuple[list, list]:
    """Renumber the secondary batch's local ids to avoid collision with the primary, then concat.

    The secondary nodes' ``local_id`` strings are remapped through ``"n{offset + i}"``; each
    secondary edge's endpoint refs are rewritten through the same remap so the connections stay
    intact across the offset.
    """
    # Build remap: secondary local_id -> renumbered local_id
    remap: dict[str, str] = {}
    renumbered_secondary_nodes = []
    for i, wrapped in enumerate(secondary_nodes):
        old = wrapped.local_id
        new = f"n{offset + i}"
        remap[old] = new
        # mutate in place — the wrapper is locally owned
        wrapped.local_id = new
        renumbered_secondary_nodes.append(wrapped)

    renumbered_secondary_edges = []
    for edge in secondary_edges:
        # Edge endpoint refs may reference primary nodes (already n0..n{offset-1}) or
        # secondary nodes (which got renumbered). Only rewrite the latter.
        if edge.refs.source_mention_id in remap:
            edge.refs.source_mention_id = remap[edge.refs.source_mention_id]
        if edge.refs.target_mention_id in remap:
            edge.refs.target_mention_id = remap[edge.refs.target_mention_id]
        renumbered_secondary_edges.append(edge)

    return primary_nodes + renumbered_secondary_nodes, primary_edges + renumbered_secondary_edges
