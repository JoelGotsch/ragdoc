"""KnowledgeGraphExtractor: multi-type extraction → typed node + edge mentions.

Implements the same :class:`~ragdoc.extraction.extractor.Extractor` protocol as
:class:`~ragdoc.extraction.structured.StructuredExtractor`, so it drops into
:class:`~ragdoc.extraction.pipeline.MentionStorePipeline` unchanged. One ``.parse()`` call per
chunk returns both node mentions (discriminated union over the schema's node types, each carrying
a chunk-local ``local_id``) and edge mentions (discriminated union over the schema's edge types,
each carrying ``refs: EdgeRef`` whose ``source_mention_id`` / ``target_mention_id`` are also
chunk-local ids).

The extractor then performs the **chunk-local-id → mention-id rewrite**: mints a real
``mention_id`` for each extracted node, builds a ``local_id → mention_id`` map, and rewrites each
edge's two endpoint references through it. Edges that reference a non-existent ``local_id`` raise
``ValueError`` — silent drops would corrupt the graph.

``GraphSchema.patterns`` are **enforced** in two layers: the legal
``(source)-[edge]->(target)`` triples are appended to the system prompt
(:func:`~ragdoc.extraction.schema.render_patterns_prompt`), and every extracted edge's
``(source_kind, edge_kind, target_kind)`` is validated against
:func:`~ragdoc.extraction.schema.allowed_pattern_kinds`. A violating edge is dropped (counted,
one WARNING per document) or raises, per ``ExtractionSettings.on_pattern_violation``.

The result is **one merged, returned list** — nodes first, then edges, with contiguous ordinals —
so node and edge mentions live in one heterogeneous store and the resolution pipeline slices them
by type via ``mention_store.list_mentions(Person)`` / ``list_mentions(Employment)``.

Optional **gleaning** runs one additional ``.parse()`` call asking the LLM for any entities it
missed; results are merged with offset chunk-local ids to avoid collision. Gleaning is gated by
``ExtractionSettings.gleaning`` (env ``EXTRACTION_GLEANING``, default off) or the constructor's
``gleaning=`` override.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from ragdoc.extraction._llm import (
    build_messages,
    parse_with_retry,
    resolve_client,
    resolve_model,
    resolve_renderer,
    resolve_tokenizer,
)
from ragdoc.extraction.mention import Mention, mint_mention_id
from ragdoc.extraction.schema import (
    GraphSchema,
    allowed_pattern_kinds,
    build_edge_union,
    build_node_union,
    render_patterns_prompt,
)
from ragdoc.extraction.structured import ExtractionSettings
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
# Runtime schema (cached per GraphSchema)
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
    """Hashable wrapper for the caches (GraphSchema itself is a BaseModel — hashable but the
    cache key wants something simple). Identity on ``(node_types, edge_types, patterns)``."""

    __slots__ = ("_key", "_schema")

    def __init__(self, schema: GraphSchema) -> None:
        self._schema = schema
        self._key = (schema.node_types, schema.edge_types, schema.patterns)

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


@cache
def _allowed_pattern_kinds(key: _SchemaKey) -> frozenset[tuple[str, str, str]]:
    """Cached :func:`~ragdoc.extraction.schema.allowed_pattern_kinds` per schema."""
    return allowed_pattern_kinds(key._schema)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_kg_messages(
    text: str,
    *,
    system_prompt: str = KG_EXTRACTION_SYSTEM_PROMPT,
    user_message: str = KG_USER_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for a KG extraction LLM call (documents the per-extractor defaults)."""
    return build_messages(text, system_prompt=system_prompt, user_message=user_message)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class KnowledgeGraphExtractor:
    """Extracts typed nodes and typed edges as a single merged, returned mention list.

    Implements the :class:`~ragdoc.extraction.extractor.Extractor` protocol
    (``Extractor[BaseModel]``): :meth:`extract` returns node mentions first (in extraction
    order), then edge mentions, both with the same ``Mention`` envelope shape; downstream code
    distinguishes them by ``payload.kind`` (the discriminator on every node and edge model). It
    never reads or writes ``document.metadata`` keys of its own.

    Args:
        schema: :class:`~ragdoc.extraction.schema.GraphSchema` declaring node types,
            edge types, and legal patterns (enforced — see the module docstring).
        client: Async OpenAI-compatible client; ``None`` falls back to ``get_config().openai_client``.
        model: Model name; ``None`` falls back to ``settings.model_name`` then config default.
        settings: :class:`~ragdoc.extraction.structured.ExtractionSettings`; ``None`` reads
            env (``EXTRACTION_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        tokenizer: Tokenizer for the ``min_tokens`` decision. ``None`` uses ``GPTTokenizer``.
        gleaning: When True, runs one additional ``.parse()`` after the first call asking the
            LLM for any missed entities. ``None`` reads ``settings.gleaning`` (default off).

    Raises:
        ValueError: when the schema's node or edge union exceeds ``settings.max_union_size``.
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
    ) -> None:
        self.settings = settings or ExtractionSettings()
        max_size = self.settings.max_union_size
        if len(schema.node_types) > max_size:
            raise ValueError(
                f"node_types has {len(schema.node_types)} entries, above "
                f"ExtractionSettings.max_union_size={max_size}. OpenAI structured-output "
                f"reliability drops with large discriminated unions; split this schema across "
                f"multiple KnowledgeGraphExtractors, or raise the limit."
            )
        if len(schema.edge_types) > max_size:
            raise ValueError(
                f"edge_types has {len(schema.edge_types)} entries, above "
                f"ExtractionSettings.max_union_size={max_size}. OpenAI structured-output "
                f"reliability drops with large discriminated unions; split this schema across "
                f"multiple KnowledgeGraphExtractors, or raise the limit."
            )
        self._schema = schema
        self._client = client
        self._model = model
        self._renderer = renderer
        self._tokenizer = tokenizer
        self._gleaning = gleaning if gleaning is not None else self.settings.gleaning

    @property
    def schema(self) -> GraphSchema:
        return self._schema

    @property
    def payload_model(self) -> Any:
        """The discriminated union over node + edge types — the merged payload model.

        Kept as a documented public attribute (parametrizes store queries and
        ``ChangeSet[Mention[...]]``); the union alias is not a ``type[BaseModel]``, hence ``Any``.
        Cached on first read because building a discriminated union over the schema's types is
        pure and the schema is immutable.
        """
        if not hasattr(self, "_payload_model_cache"):
            self._payload_model_cache = build_node_union(
                tuple(self._schema.node_types) + tuple(self._schema.edge_types)
            )
        return self._payload_model_cache

    # ------------------------------------------------------------------
    # extraction internals
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        """The resolved system prompt: user setting or KG default, plus the patterns section."""
        base = self.settings.system_prompt or KG_EXTRACTION_SYSTEM_PROMPT
        patterns_section = render_patterns_prompt(self._schema)
        return f"{base}\n\n{patterns_section}" if patterns_section else base

    async def _extract_batch(
        self,
        text: str,
        client: Any,
        model: str,
        graph_batch_cls: type[BaseModel],
        *,
        user_message: str = KG_USER_MESSAGE,
    ) -> BaseModel:
        """One structured ``.parse()`` call (shared retry loop); returns the GraphBatch instance."""
        messages = build_kg_messages(text, system_prompt=self._system_prompt(), user_message=user_message)
        return await parse_with_retry(
            client,
            model=model,
            messages=messages,
            response_format=graph_batch_cls,
            settings=self.settings,
            log_prefix="KnowledgeGraphExtractor",
        )

    async def _extract_with_halving(
        self,
        text: str,
        client: Any,
        model: str,
        graph_batch_cls: type[BaseModel],
        *,
        depth: int = 0,
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

        ``settings.halving_max_depth`` / ``settings.halving_min_chars`` bound the recursion
        (``depth`` is the internal recursion counter).

        Loss-of-information caveat: an edge whose two endpoints fall on opposite sides of a
        halving boundary cannot be emitted (each half-call sees only its own nodes). In typical
        prose most short-range ties live within the same paragraph, so the loss is generally
        deemed acceptable below ``halving_max_depth``. Above ``halving_max_depth`` or below
        ``halving_min_chars`` we re-raise so the source goes to ``result.errors`` honestly
        rather than producing a bad-quality extraction silently.
        """
        max_depth = self.settings.halving_max_depth
        min_chars = self.settings.halving_min_chars
        try:
            return await self._extract_batch(text, client, model, graph_batch_cls)
        except Exception as exc:
            if depth >= max_depth or len(text) <= min_chars:
                logger.error(
                    "KnowledgeGraphExtractor: halving exhausted at depth=%d len=%d (%r); giving up",
                    depth,
                    len(text),
                    exc,
                )
                raise
            left, right = _halve_text(text)
            if not left.strip() or not right.strip():
                logger.error(
                    "KnowledgeGraphExtractor: could not halve text at depth=%d len=%d (%r); giving up",
                    depth,
                    len(text),
                    exc,
                )
                raise
            logger.warning(
                "KnowledgeGraphExtractor: halving on failure at depth=%d (len %d -> %d + %d): %s",
                depth,
                len(text),
                len(left),
                len(right),
                exc,
            )
            primary = await self._extract_with_halving(left, client, model, graph_batch_cls, depth=depth + 1)
            secondary = await self._extract_with_halving(right, client, model, graph_batch_cls, depth=depth + 1)
            offset = len(primary.node_mentions)  # type: ignore[attr-defined]
            merged_nodes, merged_edges = _merge_gleaning(
                list(primary.node_mentions),  # type: ignore[attr-defined]
                list(primary.edge_mentions),  # type: ignore[attr-defined]
                list(secondary.node_mentions),  # type: ignore[attr-defined]
                list(secondary.edge_mentions),  # type: ignore[attr-defined]
                offset,
            )
            return graph_batch_cls(node_mentions=merged_nodes, edge_mentions=merged_edges)

    async def extract(self, document: Document) -> list[Mention[BaseModel]]:
        """Extract nodes + edges from *document*; returns the merged mention list (nodes first)."""
        renderer = resolve_renderer(self._renderer)
        rendered = renderer.render(document)
        if not rendered.strip():
            return []
        min_tokens = self.settings.min_tokens
        if min_tokens > 0 and resolve_tokenizer(self._tokenizer).count(rendered) < min_tokens:
            logger.debug("KnowledgeGraphExtractor: below min_tokens, skipping")
            return []

        graph_batch_cls, _ = build_graph_batch_model(self._schema)
        client = resolve_client(self._client)
        model = resolve_model(self._model, self.settings)

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
        metadata_copy = dict(document.metadata)

        # Build the merged mention list (nodes first so the local_id map is ready for edges).
        # Ordinals are contiguous across nodes + edges — single position-within-source counter.
        all_mentions: list[Mention[BaseModel]] = []
        local_to_mention: dict[str, str] = {}
        local_to_kind: dict[str, str] = {}
        ordinal = 0

        for wrapped in wrapped_nodes:
            local_id: str = wrapped.local_id
            payload = wrapped.node
            if local_id in local_to_mention:
                raise ValueError(
                    f"Duplicate node local_id {local_id!r} in extraction output; the LLM must "
                    f"assign each node a unique chunk-local id."
                )
            mention = Mention[BaseModel](
                mention_id=mint_mention_id(source_id, content_hash, ordinal, payload),
                source_id=source_id,
                source_path=document.source_path or None,
                source_hash=source_hash,
                content_hash=content_hash,
                ordinal=ordinal,
                metadata=metadata_copy,
                payload=payload,
            )
            local_to_mention[local_id] = mention.mention_id
            local_to_kind[local_id] = payload.kind
            all_mentions.append(mention)
            ordinal += 1

        allowed = _allowed_pattern_kinds(_SchemaKey(self._schema))
        dropped = 0
        dropped_triples: set[tuple[str, str, str]] = set()

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

            # Pattern enforcement (Layer B): validated on local ids so errors can cite them.
            triple = (local_to_kind[src_local], edge_payload.kind, local_to_kind[tgt_local])
            if triple not in allowed:
                if self.settings.on_pattern_violation == "error":
                    raise ValueError(
                        f"Edge {edge_payload.kind!r} violates the schema patterns: "
                        f"({triple[0]})-[{triple[1]}]->({triple[2]}) (endpoints {src_local!r} -> "
                        f"{tgt_local!r}) is not a legal triple. Legal patterns: {sorted(allowed)}."
                    )
                dropped += 1
                dropped_triples.add(triple)
                continue

            refs.source_mention_id = local_to_mention[src_local]
            refs.target_mention_id = local_to_mention[tgt_local]

            mention = Mention[BaseModel](
                mention_id=mint_mention_id(source_id, content_hash, ordinal, edge_payload),
                source_id=source_id,
                source_path=document.source_path or None,
                source_hash=source_hash,
                content_hash=content_hash,
                ordinal=ordinal,
                metadata=metadata_copy,
                payload=edge_payload,
            )
            all_mentions.append(mention)
            ordinal += 1

        if dropped:
            logger.warning(
                "KnowledgeGraphExtractor: dropped %d edge(s) violating schema patterns in source %r; "
                "illegal triples: %s. Legal patterns: %s.",
                dropped,
                source_id,
                sorted(dropped_triples),
                sorted(allowed),
            )

        return all_mentions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _halve_text(text: str) -> tuple[str, str]:
    """Split *text* at the closest paragraph (double-newline) boundary to the midpoint.

    Falls back to single-newline, then to the literal character midpoint. Used by
    :meth:`KnowledgeGraphExtractor._extract_with_halving` to recursively shrink an LLM
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
