"""Entity resolution: cluster mentions into canonical entities via an iterative loop.

The :class:`EntityResolutionPipeline` reads a ``MentionStore`` and writes an ``EntityStore`` by
running a fixpoint loop (the chosen Option B):

#. **Exact-key seed** — group mentions whose normalised identity text is identical (zero LLM calls).
#. **Block** — embed each current cluster's representative text and find near neighbours
   (cosine ≥ ``cosine_threshold``, top ``top_k``).
#. **Review** — ask an injected LLM reviewer which neighbours are the same entity; groups whose
   confidence ≥ ``auto_merge_bar`` become merges, the rest are recorded as *pending*.
#. **Merge + re-represent** — union-find the merge edges, combine each component into one cluster,
   and (crucially) recompute the merged cluster's representative text so the next round's embedding
   can surface neighbours that no single member was close to.
#. Repeat until a round produces no merges, or ``max_iterations`` is reached.

Finally each cluster is canonicalised into an :class:`~ragdoc.extraction.entity.Entity`.
The pass is **derived and re-creatable**: ``entity_id`` is a content hash of the sorted member
mention ids, so the same mention set always yields the same entities.

The embedding function and LLM reviewer are injected (real implementations call out; tests pass
deterministic fakes), keeping the loop itself pure and testable.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, cast

from pydantic import BaseModel, Field

from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.entity import Entity, mint_entity_id
from ragdoc.extraction.mention import Mention, PayloadT
from ragdoc.llm import ChatClient, LLMRefusalError, call_structured

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ragdoc.extraction.entity import EntityStore
    from ragdoc.extraction.stores import MentionStore
    from ragdoc.extraction.structured import ExtractionSettings

# Injected collaborators.
Embed = Callable[[list[str]], Awaitable[list[list[float]]]]
"""Embed a batch of texts into vectors."""
IdentityTextFn = Callable[[BaseModel], str]
"""Render a payload to the text used for blocking + review."""
DateFn = Callable[[BaseModel], "FuzzyDate | None"]
"""Extract the date from a payload, if it carries one."""


# ---------------------------------------------------------------------------
# Identity text + embedder helper
# ---------------------------------------------------------------------------


def default_identity_text(payload: BaseModel) -> str:
    """Render *payload* to readable ``key: value`` lines (dates as their original text).

    The default text used for both embedding (blocking) and LLM review. Callers override to focus
    it (e.g. ``lambda e: f"{e.title} {e.date.original_text}"``).
    """
    parts: list[str] = []
    for name, value in payload.model_dump(mode="json").items():
        if value is None:
            continue
        if isinstance(value, dict) and "original_text" in value:  # a FuzzyDate-like field
            value = value.get("original_text")
        parts.append(f"{name}: {value}")
    return "\n".join(parts)


def build_entity_embedder(embed: Embed, identity_text_fn: IdentityTextFn = default_identity_text):
    """Compose *embed* with *identity_text_fn* into a payload embedder (``list[payload] -> vectors``).

    Used by stores (e.g. ``QdrantMentionStore``) to embed a mention's identity for blocking without
    re-deriving the text at query time.
    """

    async def _embed_payloads(payloads: list[BaseModel]) -> list[list[float]]:
        return await embed([identity_text_fn(p) for p in payloads])

    return _embed_payloads


# ---------------------------------------------------------------------------
# Review contract (LLM in production; fakes in tests)
# ---------------------------------------------------------------------------


class ReviewGroup(BaseModel):
    """A subset of the reviewed items judged to refer to the same real-world entity."""

    members: list[int] = Field(description="Indices (into the reviewed items) that are the SAME entity.")
    confidence: float = Field(default=1.0, description="Confidence in this grouping, 0..1.")


class ReviewResult(BaseModel):
    """The reviewer's partition of the candidate items into same-entity groups."""

    groups: list[ReviewGroup] = Field(default_factory=list)


Reviewer = Callable[[list[str]], Awaitable[ReviewResult]]
"""Given representative texts of candidate clusters, return same-entity groups (by index)."""


# ---------------------------------------------------------------------------
# Internal cluster state
# ---------------------------------------------------------------------------


@dataclass
class _Cluster(Generic[PayloadT]):
    mention_ids: list[str]
    payloads: list[BaseModel]
    identity_texts: list[str]
    source_ids: list[str]

    @property
    def representative_text(self) -> str:
        """Deterministic text representing the whole cluster (changes as members are merged)."""
        return " | ".join(sorted(set(self.identity_texts)))


@dataclass
class ResolutionResult(Generic[PayloadT]):
    """Outcome of a resolution run.

    Attributes:
        entities: the canonical entities produced.
        pending: candidate groups the reviewer was not confident enough to merge (each a tuple of
            the representative texts involved), surfaced rather than silently dropped.
        iterations: how many refinement rounds ran before convergence / the cap.
    """

    entities: list[Entity] = field(default_factory=list)
    pending: list[tuple[str, ...]] = field(default_factory=list)
    iterations: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class EntityResolutionPipeline(Generic[PayloadT]):
    """Cluster a ``MentionStore`` into an ``EntityStore`` via the iterative loop.

    Args:
        mention_store: source of mentions to resolve.
        entity_store: sink for canonical entities (replaced to reflect the current mention set).
        embed: text embedder (``list[str] -> vectors``) used for blocking.
        reviewer: LLM cluster reviewer (see :func:`make_llm_reviewer`); inject a fake in tests.
        identity_text_fn: payload → blocking/review text (default :func:`default_identity_text`).
        date_fn: payload → its :class:`FuzzyDate`, when the payload carries one (enables date union).
        canonicalizer: optional async ``list[payload] -> payload`` LLM merge; defaults to a
            deterministic pick (first member by sorted identity text).
        settings: :class:`ExtractionSettings` (resolution knobs: ``cosine_threshold``,
            ``auto_merge_bar``, ``max_iterations``, ``top_k``).
    """

    def __init__(
        self,
        mention_store: MentionStore,
        entity_store: EntityStore,
        embed: Embed,
        reviewer: Reviewer,
        identity_text_fn: IdentityTextFn = default_identity_text,
        date_fn: DateFn | None = None,
        canonicalizer: Callable[[list[BaseModel]], Awaitable[BaseModel]] | None = None,
        settings: ExtractionSettings | None = None,
    ) -> None:
        from ragdoc.extraction.structured import ExtractionSettings

        self._mention_store = mention_store
        self._entity_store = entity_store
        self._embed = embed
        self._reviewer = reviewer
        self._identity_text_fn = identity_text_fn
        self._date_fn = date_fn
        self._canonicalizer = canonicalizer
        self.settings = settings or ExtractionSettings()

    async def resolve(self) -> ResolutionResult[PayloadT]:
        """Run the full loop, write the entity store, and return the result."""
        mentions = await self._mention_store.list_mentions()
        clusters = self._seed_exact(mentions)
        pending: list[tuple[str, ...]] = []

        iterations = 0
        for iterations in range(1, self.settings.max_iterations + 1):  # noqa: B007 -- read after the loop (ResolutionResult.iterations)
            edges, round_pending = await self._propose_and_review(clusters)
            pending.extend(round_pending)
            if not edges:
                break
            clusters = _apply_merges(clusters, edges)

        entities = [await self._canonicalize(c) for c in clusters]
        await self._write(entities)
        return ResolutionResult(entities=entities, pending=pending, iterations=iterations)

    # -- seeding -------------------------------------------------------------

    def _seed_exact(self, mentions: list[Mention]) -> list[_Cluster[PayloadT]]:
        groups: dict[str, list[Mention]] = {}
        for m in mentions:
            key = self._identity_text_fn(m.payload).strip().lower()
            groups.setdefault(key, []).append(m)
        return [self._cluster_from(ms) for ms in groups.values()]

    def _cluster_from(self, mentions: list[Mention]) -> _Cluster[PayloadT]:
        return _Cluster(
            mention_ids=[m.mention_id for m in mentions],
            payloads=[m.payload for m in mentions],
            identity_texts=[self._identity_text_fn(m.payload) for m in mentions],
            source_ids=[m.source_id for m in mentions],
        )

    # -- propose + review ----------------------------------------------------

    async def _propose_and_review(
        self, clusters: list[_Cluster[PayloadT]]
    ) -> tuple[list[tuple[int, int]], list[tuple[str, ...]]]:
        if len(clusters) < 2:
            return [], []
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "EntityResolutionPipeline requires numpy (installed with the 'extraction' extra)"
            ) from exc

        texts = [c.representative_text for c in clusters]
        arr = np.asarray(await self._embed(texts), dtype=float)
        normed = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        sims = normed @ normed.T

        threshold = self.settings.cosine_threshold
        top_k = self.settings.top_k
        edges: list[tuple[int, int]] = []
        pending: list[tuple[str, ...]] = []
        reviewed: set[frozenset[int]] = set()

        for i in range(len(clusters)):
            order = [int(j) for j in np.argsort(-sims[i])]
            neighbours = [j for j in order if j != i and sims[i][j] >= threshold][:top_k]
            if not neighbours:
                continue
            group_idx = [i, *neighbours]
            key = frozenset(group_idx)
            if key in reviewed:
                continue
            reviewed.add(key)

            result = await self._reviewer([clusters[g].representative_text for g in group_idx])
            for grp in result.groups:
                members = [group_idx[m] for m in grp.members if 0 <= m < len(group_idx)]
                if len(members) < 2:
                    continue
                if grp.confidence >= self.settings.auto_merge_bar:
                    for other in members[1:]:
                        edges.append((members[0], other))
                else:
                    pending.append(tuple(clusters[m].representative_text for m in members))
        return edges, pending

    # -- canonicalize + write -----------------------------------------------

    async def _canonicalize(self, cluster: _Cluster[PayloadT]) -> Entity[PayloadT]:
        member_ids = sorted(cluster.mention_ids)
        entity_id = mint_entity_id(member_ids)

        if self._canonicalizer is not None:
            canonical_payload = await self._canonicalizer(cluster.payloads)
        else:
            # deterministic: the payload whose identity text sorts first
            first = min(range(len(cluster.payloads)), key=lambda i: cluster.identity_texts[i])
            canonical_payload = cluster.payloads[first]

        date: FuzzyDate | None = None
        if self._date_fn is not None:
            dates = [d for d in (self._date_fn(p) for p in cluster.payloads) if d is not None]
            date = _union_dates(dates)

        return Entity(
            entity_id=entity_id,
            payload=cast(PayloadT, canonical_payload),
            aliases=sorted(set(cluster.identity_texts)),
            member_mention_ids=member_ids,
            source_ids=sorted(set(cluster.source_ids)),
            date=date,
        )

    async def _write(self, entities: list[Entity[PayloadT]]) -> None:
        """Replace the entity store's contents with *entities* (delete stale, then upsert)."""
        existing = {e.entity_id for e in await self._entity_store.list_entities()}
        new_ids = {e.entity_id for e in entities}
        await self._entity_store.upsert(entities)
        stale = list(existing - new_ids)
        if stale:
            await self._entity_store.delete(stale)


# ---------------------------------------------------------------------------
# Merge (union-find over merge edges)
# ---------------------------------------------------------------------------


def _apply_merges(clusters: list[_Cluster[PayloadT]], edges: list[tuple[int, int]]) -> list[_Cluster[PayloadT]]:
    parent = list(range(len(clusters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)

    components: dict[int, list[int]] = {}
    for i in range(len(clusters)):
        components.setdefault(find(i), []).append(i)

    merged: list[_Cluster[PayloadT]] = []
    for idxs in components.values():
        group = [clusters[i] for i in idxs]
        merged.append(
            _Cluster(
                mention_ids=[mid for c in group for mid in c.mention_ids],
                payloads=[p for c in group for p in c.payloads],
                identity_texts=[t for c in group for t in c.identity_texts],
                source_ids=[s for c in group for s in c.source_ids],
            )
        )
    return merged


# ---------------------------------------------------------------------------
# Date union
# ---------------------------------------------------------------------------


def _union_dates(dates: list[FuzzyDate]) -> FuzzyDate | None:
    """Return a FuzzyDate spanning the union of *dates*' intervals, or ``None`` if none are bounded."""
    if not dates:
        return None
    starts = [d.start for d in dates if d.start is not None]
    ends = [d.end for d in dates if d.end is not None]  # exclusive
    if not starts or not ends:
        # fall back to the first date that has any signal
        return next((d for d in dates if d.edtf), dates[0])
    import datetime as dt

    min_start = min(starts)
    last_inclusive = max(ends) - dt.timedelta(days=1)
    original = "; ".join(d.original_text for d in dates)
    return FuzzyDate(
        original_text=original,
        edtf=f"{min_start.isoformat()}/{last_inclusive.isoformat()}",
        precision=dates[0].precision,
    )


# ---------------------------------------------------------------------------
# Real LLM reviewer
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = """
You are an expert at entity resolution. You will be given a numbered list of candidate entity
descriptions. Decide which of them refer to the SAME real-world entity.

Return groups of indices. Put indices that refer to the same entity in one group; give each
distinct entity its own group (a single-index group is fine). Assign each group a confidence in
[0, 1] reflecting how sure you are that its members are truly the same.
""".strip()


def make_llm_reviewer(client: ChatClient, model: str, *, temperature: float = 0.0) -> Reviewer:
    """Build a :data:`Reviewer` backed by an LLM ``.parse()`` call returning a :class:`ReviewResult`.

    Transport errors are retried by the shared LLM layer; a refusal degrades to an empty
    :class:`ReviewResult` (no merge groups), preserving the reviewer's safe-degrade contract.
    """

    async def _review(texts: list[str]) -> ReviewResult:
        numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
        try:
            return await call_structured(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Candidates:\n{numbered}"},
                ],
                response_format=ReviewResult,
                temperature=temperature,
                log_prefix="EntityReviewer",
            )
        except LLMRefusalError:
            return ReviewResult()

    return _review
