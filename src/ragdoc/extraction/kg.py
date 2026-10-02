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
edge's two endpoint references through it.

**Dangling edge references are dropped, not fatal.** An edge whose endpoint names a ``local_id``
that the *same LLM response* did not define (for a gleaning response: nor the previous answer) is
withheld and reported in one WARNING per response (:func:`drop_dangling_edges`). The check runs
per response, before any merge, so a dangling id can never accidentally resolve to a node from
another response (the other half of a halved input, a renumbered gleaned node). One bad edge
costs that edge — the nodes and the remaining edges of the source are kept. Duplicate node
``local_id``\\ s within one response still raise ``ValueError`` (edges through a duplicated id
are ambiguous; attaching them to either node could corrupt the graph).

``GraphSchema.patterns`` are **enforced** in two layers: the legal
``(source)-[edge]->(target)`` triples are appended to the system prompt
(:func:`~ragdoc.extraction.schema.render_patterns_prompt`), and every extracted edge's
``(source_kind, edge_kind, target_kind)`` is validated against
:func:`~ragdoc.extraction.schema.allowed_pattern_kinds`. A violating edge is dropped (counted,
one WARNING per document) or raises, per ``ExtractionSettings.on_pattern_violation``.

The result is **one merged, returned list** — nodes first, then edges, with contiguous ordinals —
so node and edge mentions live in one heterogeneous store and the resolution pipeline slices them
by type via ``mention_store.list_mentions(Person)`` / ``list_mentions(Employment)``.

**Halving** reacts to *size-related* failures only (:func:`ragdoc.llm.is_size_related_failure`:
context-length errors, truncated output, malformed structured output, and timeouts that outlived
the shared retry policy): the text is halved and each half extracted separately. Every other
failure — authentication, rate limit, connection, 5xx, refusal, bugs — propagates after the
shared retry policy without a single extra call.

Optional **gleaning** runs one additional ``.parse()`` call that *continues the conversation*:
the model is shown its own first answer (as an assistant turn) and asked only for what is
missing; the merge de-duplicates anything it repeats anyway. Gleaning is gated by
``ExtractionSettings.gleaning`` (env ``EXTRACTION_GLEANING``, default off) or the constructor's
``gleaning=`` override.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Set as AbstractSet
from functools import cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from ragdoc.extraction._llm import (
    build_messages,
    parse_with_retry,
    resolve_model,
    resolve_renderer,
    resolve_tokenizer,
)
from ragdoc.extraction.mention import Mention, mint_mention_id, rewrite_edge_refs
from ragdoc.extraction.schema import (
    EdgeRef,
    GraphSchema,
    allowed_pattern_kinds,
    build_edge_union,
    build_node_union,
    render_patterns_prompt,
)
from ragdoc.extraction.structured import ExtractionSettings
from ragdoc.llm import ChatClient, is_size_related_failure, resolve_openai_client
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
    "Check the content again against your previous answer above. Return ONLY nodes and edges "
    "that the content supports and that are MISSING from your previous answer — do not repeat "
    "anything you already returned. Give every new node a NEW local_id that your previous "
    "answer did not use, starting at n{next}. A new edge may reference new nodes and nodes "
    "from your previous answer alike, by their local_id. If nothing is missing, return empty lists."
)


# ---------------------------------------------------------------------------
# Runtime schema (cached per GraphSchema)
# ---------------------------------------------------------------------------


class LocalNode(BaseModel):
    """Static shape of one LLM-emitted node: the node payload plus its chunk-local id.

    The per-schema ``NodeWithLocalId`` model built by :func:`build_graph_batch_model` subclasses
    this and narrows ``node`` to the schema's discriminated node union. This docstring is not
    part of the LLM-facing JSON schema (a subclass does not inherit ``__doc__``).
    """

    local_id: str = Field(description="Chunk-local id (e.g. 'n0', 'n1'). Edges reference nodes by this id.")
    node: BaseModel = Field(description="The extracted node payload.")


class GraphBatchBase(BaseModel):
    """Static shape of one LLM response: wrapped nodes plus edge payloads.

    The per-schema ``GraphBatch`` model built by :func:`build_graph_batch_model` subclasses this
    and narrows both lists to the schema's node wrapper / discriminated edge union.
    """

    node_mentions: list[LocalNode] = Field(default_factory=list, description="Node mentions with chunk-local ids.")
    edge_mentions: list[BaseModel] = Field(
        default_factory=list,
        description=(
            "Edge mentions. Each refs.source_mention_id / refs.target_mention_id is a "
            "chunk-local id from the node list."
        ),
    )


def build_graph_batch_model(schema: GraphSchema) -> tuple[type[GraphBatchBase], type[LocalNode]]:
    """Build the ``GraphBatch`` and ``NodeWithLocalId`` Pydantic models for a *schema*.

    ``GraphBatch`` is the ``response_format`` for the LLM call: ``node_mentions`` is a list of
    ``NodeWithLocalId`` (a thin wrapper carrying the chunk-local id alongside the validated
    node), and ``edge_mentions`` is a list of the schema's edge discriminated union. Built fresh
    per ``schema`` and cached so the LLM call's ``response_format`` is referentially stable.

    The returned classes subclass :class:`GraphBatchBase` / :class:`LocalNode`, which give the
    dynamically built models a static shape (``node_mentions`` / ``edge_mentions`` / ``local_id``
    / ``node``); their JSON schema is exactly that of the per-schema fields.
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
def _build_graph_batch_model(key: _SchemaKey) -> tuple[type[GraphBatchBase], type[LocalNode]]:
    schema = key._schema
    node_union = build_node_union(schema.node_types)
    edge_union = build_edge_union(schema.edge_types)

    node_with_local_id = create_model(
        "NodeWithLocalId",
        __base__=LocalNode,
        local_id=(
            str,
            Field(description="Chunk-local id (e.g. 'n0', 'n1'). Edges reference nodes by this id."),
        ),
        node=(node_union, Field(description="The extracted node payload.")),
    )
    graph_batch = create_model(
        "GraphBatch",
        __base__=GraphBatchBase,
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


def build_kg_gleaning_messages(
    text: str,
    previous_answer: GraphBatchBase,
    *,
    system_prompt: str = KG_EXTRACTION_SYSTEM_PROMPT,
    user_message: str = KG_USER_MESSAGE,
    gleaning_message: str = KG_GLEANING_MESSAGE,
) -> list[ChatCompletionMessageParam]:
    """Build the ``messages`` list for the gleaning call: a continuation of the first call.

    ``[system, user (the original request), assistant (the first answer as JSON), user (the
    gleaning instruction)]``. The model sees exactly what it answered before, so "what did you
    miss?" is answerable; without the assistant turn the call is an independent second
    extraction that mostly re-emits the same nodes.

    Args:
        text: The rendered content, exactly as sent in the first call.
        previous_answer: The first call's (possibly halving-merged) batch.
        system_prompt: The same system prompt as the first call.
        user_message: The same user instruction as the first call.
        gleaning_message: The follow-up instruction; ``{next}`` is replaced by the first unused
            node index (see :func:`next_local_index`).
    """
    return [
        *build_kg_messages(text, system_prompt=system_prompt, user_message=user_message),
        {"role": "assistant", "content": previous_answer.model_dump_json()},
        {
            "role": "user",
            "content": gleaning_message.format(next=next_local_index(previous_answer.node_mentions)),
        },
    ]


_LOCAL_ID_PATTERN = re.compile(r"n(\d+)")


def next_local_index(nodes: list[LocalNode]) -> int:
    """The first ``k`` such that no node in *nodes* uses an id ``n<j>`` with ``j >= k``.

    Falls back to ``len(nodes)`` when no id follows the ``n<digits>`` convention.
    """
    indices = [int(m.group(1)) for wrapped in nodes if (m := _LOCAL_ID_PATTERN.fullmatch(wrapped.local_id))]
    return max(indices) + 1 if indices else len(nodes)


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

    Failure products (see the module docstring for detail): size-related LLM failures halve
    the input (bounded by ``settings.halving_max_depth`` / ``halving_min_chars``); every other
    LLM failure **raises** after the shared retry policy; an edge with a dangling endpoint
    reference is dropped with a WARNING; an illegal-pattern edge is dropped or raises per
    ``settings.on_pattern_violation``; duplicate node ``local_id``\\ s raise ``ValueError``.

    Args:
        schema: :class:`~ragdoc.extraction.schema.GraphSchema` declaring node types,
            edge types, and legal patterns (enforced — see the module docstring).
        client: Async OpenAI-compatible client; ``None`` falls back to ``get_config().openai_client``,
            resolved fail-loud at construction.
        model: Model name; ``None`` falls back to ``settings.model_name`` then config default.
        settings: :class:`~ragdoc.extraction.structured.ExtractionSettings`; ``None`` reads
            env (``EXTRACTION_*``).
        renderer: Renderer for document → text. ``None`` uses ``Renderer(MARKDOWN, render_for_prompt)``.
        tokenizer: Tokenizer for the ``min_tokens`` decision. ``None`` uses ``GPTTokenizer``.
        gleaning: When True, runs one additional ``.parse()`` after the first call that shows
            the LLM its first answer and asks for anything it missed; repeated nodes/edges are
            de-duplicated in the merge. ``None`` reads ``settings.gleaning`` (default off).

    Raises:
        ValueError: when the schema's node or edge union exceeds ``settings.max_union_size``.
        LLMNotConfiguredError: when no client is given and the active config has none.
    """

    def __init__(
        self,
        schema: GraphSchema,
        client: ChatClient | None = None,
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
        self._client: ChatClient = resolve_openai_client(client)
        self._model = model
        self._renderer = renderer
        self._tokenizer = tokenizer
        self._gleaning = gleaning if gleaning is not None else self.settings.gleaning
        self._payload_model: Any = None

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
        if self._payload_model is None:
            self._payload_model = build_node_union(tuple(self._schema.node_types) + tuple(self._schema.edge_types))
        return self._payload_model

    # ------------------------------------------------------------------
    # extraction internals
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        """The resolved system prompt: user setting or KG default, plus the patterns section."""
        base = self.settings.system_prompt or KG_EXTRACTION_SYSTEM_PROMPT
        patterns_section = render_patterns_prompt(self._schema)
        return f"{base}\n\n{patterns_section}" if patterns_section else base

    async def _parse_batch(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        graph_batch_cls: type[GraphBatchBase],
    ) -> GraphBatchBase:
        """One structured ``.parse()`` call (shared retry policy); returns the GraphBatch instance."""
        return await parse_with_retry(
            self._client,
            model=model,
            messages=messages,
            response_format=graph_batch_cls,
            settings=self.settings,
            log_prefix="KnowledgeGraphExtractor",
        )

    async def _extract_with_halving(
        self,
        text: str,
        model: str,
        graph_batch_cls: type[GraphBatchBase],
        *,
        source_label: str,
        depth: int = 0,
    ) -> GraphBatchBase:
        """Extract from *text*, halving it when — and only when — the failure is size-related.

        The LLM call runs under the shared retry policy first. If it still fails,
        :func:`ragdoc.llm.is_size_related_failure` decides:

        - **size-related** (context-length error, truncated output, malformed structured
          output, timeout after retries): halve *text* at the paragraph boundary closest to
          its midpoint, recurse on each half concurrently, and merge the two batches with
          collision-free chunk-local ids (:func:`merge_graph_batches`).
        - **anything else** (401/403, 429 after retries, connection errors, 5xx, refusals,
          non-LLM errors): re-raise immediately. A smaller input does not fix a bad API key or
          an exhausted quota — halving there only multiplies the failing calls (a 401 on one
          ~9.5k-character split used to cost 15 calls instead of 1).

        The halving is **internal to the LLM call**: it never touches ``document.metadata``, so
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
            batch = await self._parse_batch(
                build_kg_messages(text, system_prompt=self._system_prompt()), model, graph_batch_cls
            )
        except Exception as exc:
            if not is_size_related_failure(exc):
                raise
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
                "KnowledgeGraphExtractor: halving on size-related failure at depth=%d (len %d -> %d + %d): %s",
                depth,
                len(text),
                len(left),
                len(right),
                exc,
            )
            # Extract both halves concurrently; the merge consumes them positionally after both
            # complete. return_exceptions=True keeps the losing half awaited (no orphan task /
            # unretrieved-exception warning) — the first failure still propagates as before.
            halved = await asyncio.gather(
                self._extract_with_halving(left, model, graph_batch_cls, source_label=source_label, depth=depth + 1),
                self._extract_with_halving(right, model, graph_batch_cls, source_label=source_label, depth=depth + 1),
                return_exceptions=True,
            )
            batches: list[GraphBatchBase] = []
            for res in halved:
                if isinstance(res, BaseException):
                    raise res from exc  # first failing half propagates; *exc* triggered the halving
                batches.append(res)
            primary, secondary = batches
            # The halves are different text: identical payloads are distinct occurrences, so no
            # de-duplication here (that is a gleaning-only concern).
            merged_nodes, merged_edges = merge_graph_batches(
                list(primary.node_mentions),
                list(primary.edge_mentions),
                list(secondary.node_mentions),
                list(secondary.edge_mentions),
                deduplicate=False,
            )
            return graph_batch_cls(node_mentions=merged_nodes, edge_mentions=merged_edges)

        # Per-response check, before any merge can renumber ids (see drop_dangling_edges).
        drop_dangling_edges(batch, source_label=source_label)
        return batch

    async def extract(self, document: Document) -> list[Mention[BaseModel]]:
        """Extract nodes + edges from *document*; returns the merged mention list (nodes first)."""
        renderer = resolve_renderer(self._renderer)
        # Rendering is sync CPU/pandoc work — hop off the event loop (chunker precedent).
        rendered = await asyncio.to_thread(renderer.render, document)
        if not rendered.strip():
            return []
        min_tokens = self.settings.min_tokens
        if min_tokens > 0 and resolve_tokenizer(self._tokenizer).count(rendered) < min_tokens:
            logger.debug("KnowledgeGraphExtractor: below min_tokens, skipping")
            return []

        graph_batch_cls, _ = build_graph_batch_model(self._schema)
        model = resolve_model(self._model, self.settings)
        source_id = document.source_id or document.source_path or document.id

        primary = await self._extract_with_halving(rendered, model, graph_batch_cls, source_label=source_id)
        wrapped_nodes = list(primary.node_mentions)
        edge_payloads = list(primary.edge_mentions)

        if self._gleaning:
            # A continuation of the first call: the model sees its own first answer and is asked
            # only for what is missing. Failures here propagate (no halving — the first call
            # already proved the input fits).
            gleaned = await self._parse_batch(
                build_kg_gleaning_messages(rendered, primary, system_prompt=self._system_prompt()),
                model,
                graph_batch_cls,
            )
            # Gleaned edges may legitimately reference nodes of the previous answer.
            drop_dangling_edges(
                gleaned,
                source_label=source_id,
                extra_known_ids=frozenset(wrapped.local_id for wrapped in wrapped_nodes),
            )
            wrapped_nodes, edge_payloads = merge_graph_batches(
                wrapped_nodes,
                edge_payloads,
                list(gleaned.node_mentions),
                list(gleaned.edge_mentions),
                deduplicate=True,
            )

        content_hash = document.content_hash()
        # Honestly optional: the file-byte hash, or None — never faked from the content hash.
        source_hash = document.source_hash
        metadata_copy = dict(document.metadata)

        # Build the merged mention list (nodes first so the local_id map is ready for edges).
        # Ordinals are contiguous across nodes + edges — single position-within-source counter.
        all_mentions: list[Mention[BaseModel]] = []
        local_to_mention: dict[str, str] = {}
        local_to_kind: dict[str, str] = {}
        ordinal = 0

        for wrapped in wrapped_nodes:
            local_id = wrapped.local_id
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
            local_to_kind[local_id] = payload_kind(payload)
            all_mentions.append(mention)
            ordinal += 1

        allowed = _allowed_pattern_kinds(_SchemaKey(self._schema))
        dropped = 0
        dropped_triples: set[tuple[str, str, str]] = set()

        for edge_payload in edge_payloads:
            refs = edge_refs(edge_payload)
            # Both endpoints are known here: dangling refs were dropped per response
            # (drop_dangling_edges) and the merges only produce ids of merged nodes.
            src_local = refs.source_mention_id
            tgt_local = refs.target_mention_id
            edge_kind = payload_kind(edge_payload)

            # Pattern enforcement (Layer B): validated on local ids so errors can cite them.
            triple = (local_to_kind[src_local], edge_kind, local_to_kind[tgt_local])
            if triple not in allowed:
                if self.settings.on_pattern_violation == "error":
                    raise ValueError(
                        f"Edge {edge_kind!r} violates the schema patterns: "
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


def edge_refs(edge: BaseModel) -> EdgeRef:
    """Return the ``refs: EdgeRef`` of an edge payload.

    Edge payloads are instances of caller-defined models, so the field is not statically
    visible; :class:`~ragdoc.extraction.schema.GraphSchema` guarantees every registered edge
    type declares it.

    Raises:
        TypeError: *edge* has no ``refs`` of type :class:`~ragdoc.extraction.schema.EdgeRef`
            (i.e. it is not an instance of a registered edge type).
    """
    refs = getattr(edge, "refs", None)
    if not isinstance(refs, EdgeRef):
        raise TypeError(f"{type(edge).__name__} is not an edge payload: it has no `refs: EdgeRef` field.")
    return refs


def payload_kind(payload: BaseModel) -> str:
    """Return the ``kind`` discriminator value of a node or edge payload instance.

    Raises:
        TypeError: *payload* has no string ``kind`` (i.e. it is not an instance of a registered
            node/edge type — :class:`~ragdoc.extraction.schema.GraphSchema` requires ``kind``).
    """
    kind = getattr(payload, "kind", None)
    if not isinstance(kind, str):
        raise TypeError(f"{type(payload).__name__} has no `kind: Literal[...]` discriminator.")
    return kind


def drop_dangling_edges(
    batch: GraphBatchBase,
    *,
    source_label: str,
    extra_known_ids: AbstractSet[str] = frozenset(),
) -> int:
    """Remove (in place) every edge of *batch* whose endpoint names an unknown ``local_id``.

    "Known" ids are the ``local_id``\\ s of *batch*'s own nodes plus *extra_known_ids* (for a
    gleaning response: the ids of the previous answer, which the model was shown). Must run
    **per LLM response, before merging**: merges renumber ids, and a dangling id could
    otherwise coincide with — and silently attach the edge to — a node from another response.

    A dangling reference is a model slip on one edge; dropping it loses that edge only, whereas
    raising would discard every node and edge of the source. Logs one WARNING per call that
    drops anything.

    Args:
        batch: One LLM response; its ``edge_mentions`` list is replaced when edges are dropped.
        source_label: Source identifier for the log message.
        extra_known_ids: Additional ids an edge may legitimately reference.

    Returns:
        The number of edges dropped.
    """
    known = {wrapped.local_id for wrapped in batch.node_mentions} | extra_known_ids
    kept: list[BaseModel] = []
    dangling: list[tuple[str, str, str]] = []
    for edge in batch.edge_mentions:
        refs = edge_refs(edge)
        if refs.source_mention_id in known and refs.target_mention_id in known:
            kept.append(edge)
        else:
            dangling.append((payload_kind(edge), refs.source_mention_id, refs.target_mention_id))
    if dangling:
        batch.edge_mentions = kept
        logger.warning(
            "KnowledgeGraphExtractor: dropped %d edge(s) with dangling endpoint references in source %r "
            "(kind, source local_id, target local_id): %s; known local_ids: %s.",
            len(dangling),
            source_label,
            dangling,
            sorted(known),
        )
    return len(dangling)


def _halve_text(text: str) -> tuple[str, str]:
    """Split *text* at the closest paragraph (double-newline) boundary to the midpoint.

    Falls back to single-newline, then to the literal character midpoint. Used by
    :meth:`KnowledgeGraphExtractor._extract_with_halving` to recursively shrink an LLM
    input that triggered a size-related failure (context length, truncated or malformed
    output, timeout).
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


def merge_graph_batches(
    primary_nodes: list[LocalNode],
    primary_edges: list[BaseModel],
    secondary_nodes: list[LocalNode],
    secondary_edges: list[BaseModel],
    *,
    deduplicate: bool,
) -> tuple[list[LocalNode], list[BaseModel]]:
    """Append a secondary batch to a primary one under fresh, collision-free local ids.

    Every kept secondary node gets a fresh id ``"n{k}"`` (``k`` counting up from
    ``len(primary_nodes)``, skipping ids a primary node already uses), so non-contiguous
    primary ids (e.g. ``n0, n2``) can never collide with a renumbered secondary id. Each
    secondary edge's endpoint refs are rewritten through the same remap so the connections
    stay intact; refs naming a primary id that no secondary node shadows pass through
    untouched. Secondary nodes and edges are mutated in place (they are locally owned).

    Callers must have removed dangling edges from each batch first
    (:func:`drop_dangling_edges`) — this function assumes every ref is meaningful.

    Args:
        primary_nodes: Nodes of the first batch (kept as-is, ids unchanged).
        primary_edges: Edges of the first batch (kept as-is).
        secondary_nodes: Nodes of the batch to append.
        secondary_edges: Edges of the batch to append.
        deduplicate: ``True`` for **gleaning** (the secondary batch re-reads the *same* text):
            a secondary node whose payload equals a primary node's payload is not added — its
            id is mapped onto that primary node, so edges through it re-point at the original
            — and a secondary edge that (after remapping) equals a primary edge is not added.
            ``False`` for **halving** (the batches cover *different* text): equal payloads are
            distinct occurrences and are all kept.

    Returns:
        ``(nodes, edges)`` — primary followed by the kept secondary items.

    Raises:
        ValueError: when the secondary batch itself contains duplicate ``local_id``\\ s — the
            same fail-loud product as the primary-batch duplicate check in ``extract()``
            (silent last-wins remapping would attach edges to the wrong node).
    """
    secondary_ids = [wrapped.local_id for wrapped in secondary_nodes]
    duplicates = {lid for lid in secondary_ids if secondary_ids.count(lid) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate node local_id(s) {sorted(duplicates)!r} in the secondary (gleaning/halving) "
            f"extraction output; the LLM must assign each node a unique chunk-local id."
        )

    # Build remap: secondary local_id -> the id that node carries in the merged batch (a fresh
    # id guaranteed absent from the primary set, or — when de-duplicating — its primary twin's).
    taken = {wrapped.local_id for wrapped in primary_nodes}
    remap: dict[str, str] = {}
    kept_secondary_nodes: list[LocalNode] = []
    counter = len(primary_nodes)
    for wrapped in secondary_nodes:
        if deduplicate:
            twin = next((p for p in primary_nodes if p.node == wrapped.node), None)
            if twin is not None:
                remap[wrapped.local_id] = twin.local_id
                continue
        while (new := f"n{counter}") in taken:
            counter += 1
        counter += 1
        taken.add(new)
        remap[wrapped.local_id] = new
        # mutate in place — the wrapper is locally owned
        wrapped.local_id = new
        kept_secondary_nodes.append(wrapped)

    kept_secondary_edges: list[BaseModel] = []
    for edge in secondary_edges:
        # Endpoint refs may name primary nodes (ids unchanged) or secondary nodes (remapped).
        rewrite_edge_refs(edge, remap)
        if deduplicate and any(edge == existing for existing in primary_edges):
            continue
        kept_secondary_edges.append(edge)

    return primary_nodes + kept_secondary_nodes, primary_edges + kept_secondary_edges
