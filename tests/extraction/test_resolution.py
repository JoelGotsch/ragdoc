"""Tests for EntityResolutionPipeline — the iterative cluster→review→merge loop.

Embeddings and the LLM reviewer are injected fakes, so these tests pin the loop's behaviour:
exact-key seeding, semantic merge, the re-embed-after-merge effect, convergence + max_iterations,
the pending tier, date union, re-creatability, and the identity-text helpers.
"""

from __future__ import annotations

import datetime as dt

import pytest

from ragdoc.extraction.dates import FuzzyDate
from ragdoc.extraction.resolution import (
    EntityResolutionPipeline,
    build_entity_embedder,
    default_identity_text,
)
from ragdoc.extraction.structured import ExtractionSettings

from .conftest import (
    Event,
    MemoryEntityStore,
    MemoryMentionStore,
    dict_embed,
    low_conf_reviewer,
    make_mention,
    merge_all_reviewer,
    merge_first_pair_reviewer,
    no_merge_reviewer,
)


def pipeline(mstore, estore, embed, reviewer, *, date_fn=None, settings=None) -> EntityResolutionPipeline[Event]:
    return EntityResolutionPipeline(
        mention_store=mstore,
        entity_store=estore,
        embed=embed,
        reviewer=reviewer,
        date_fn=date_fn,
        settings=settings or ExtractionSettings(),
    )


async def store_with(*mentions) -> MemoryMentionStore:
    s = MemoryMentionStore()
    await s.upsert(list(mentions))
    return s


@pytest.mark.anyio
async def test_exact_key_seed_merges_without_llm(estore):
    # two identical payloads → one cluster from seeding alone; reviewer never consulted
    calls = {"n": 0}

    async def counting_reviewer(texts):
        calls["n"] += 1
        return await merge_all_reviewer(texts)

    mstore = await store_with(make_mention("m1", "Shutdown", "s1"), make_mention("m2", "Shutdown", "s2"))
    result = await pipeline(mstore, estore, dict_embed({}), counting_reviewer).resolve()

    assert calls["n"] == 0  # exact-key seed needs no LLM
    assert len(result.entities) == 1
    assert set(result.entities[0].source_ids) == {"s1", "s2"}
    assert result.entities[0].member_mention_ids == ["m1", "m2"]


@pytest.mark.anyio
async def test_semantic_merge_aggregates_provenance(estore):
    mstore = await store_with(make_mention("m1", "a", "s1"), make_mention("m2", "b", "s2"))
    embed = dict_embed({"title: a": [1.0, 0.0, 0.0], "title: b": [1.0, 0.01, 0.0]})
    result = await pipeline(mstore, estore, embed, merge_all_reviewer).resolve()

    assert len(result.entities) == 1
    e = result.entities[0]
    assert set(e.source_ids) == {"s1", "s2"}
    assert sorted(e.member_mention_ids) == ["m1", "m2"]
    assert set(e.aliases) == {"title: a", "title: b"}


@pytest.mark.anyio
async def test_reembed_after_merge_surfaces_second_round_merge(estore):
    # a, b merge in round 1; c is far from both individually but near the MERGED representative.
    embed = dict_embed(
        {
            "title: a": [1.0, 0.0, 0.0],
            "title: b": [1.0, 0.0, 0.0],
            "title: c": [0.0, 1.0, 0.0],
            "title: a | title: b": [0.0, 1.0, 0.0],  # merged rep is now near c
        }
    )
    mentions = (make_mention("m1", "a", "s1"), make_mention("m2", "b", "s2"), make_mention("m3", "c", "s3"))

    # single pass cannot reach c
    one = await pipeline(
        await store_with(*mentions),
        MemoryEntityStore(),
        embed,
        merge_all_reviewer,
        settings=ExtractionSettings(max_iterations=1),
    ).resolve()
    assert len(one.entities) == 2

    # iterating re-embeds the merged representative and pulls c in
    many = await pipeline(
        await store_with(*mentions),
        estore,
        embed,
        merge_all_reviewer,
        settings=ExtractionSettings(max_iterations=3),
    ).resolve()
    assert len(many.entities) == 1
    assert sorted(many.entities[0].member_mention_ids) == ["m1", "m2", "m3"]
    assert many.iterations >= 2


@pytest.mark.anyio
async def test_convergence_stops_when_no_merges(estore):
    mstore = await store_with(make_mention("m1", "a", "s1"), make_mention("m2", "b", "s2"))
    embed = dict_embed({"title: a": [1.0, 0.0, 0.0], "title: b": [1.0, 0.0, 0.0]})
    result = await pipeline(mstore, estore, embed, no_merge_reviewer).resolve()
    assert result.iterations == 1  # one round, no edges → stop
    assert len(result.entities) == 2


@pytest.mark.anyio
async def test_max_iterations_backstop(estore):
    # four mutually-close items, reviewer merges only one pair per round → needs 3 rounds for one entity
    mentions = [make_mention(f"m{i}", c, f"s{i}") for i, c in enumerate("abcd")]
    embed = dict_embed({}, default=[1.0, 0.0, 0.0])  # everything identical

    capped = await pipeline(
        await store_with(*mentions),
        MemoryEntityStore(),
        embed,
        merge_first_pair_reviewer,
        settings=ExtractionSettings(max_iterations=2),
    ).resolve()
    assert len(capped.entities) == 2  # backstop stopped before full convergence

    full = await pipeline(
        await store_with(*mentions),
        estore,
        embed,
        merge_first_pair_reviewer,
        settings=ExtractionSettings(max_iterations=3),
    ).resolve()
    assert len(full.entities) == 1


@pytest.mark.anyio
async def test_pending_tier_not_merged(estore):
    mstore = await store_with(make_mention("m1", "a", "s1"), make_mention("m2", "b", "s2"))
    embed = dict_embed({"title: a": [1.0, 0.0, 0.0], "title: b": [1.0, 0.0, 0.0]})
    result = await pipeline(mstore, estore, embed, low_conf_reviewer).resolve()
    assert len(result.entities) == 2  # below auto_merge_bar → not merged
    assert result.pending  # recorded, not silently dropped


@pytest.mark.anyio
async def test_date_union(estore):
    d1 = FuzzyDate(original_text="1994", edtf="1994", precision="YEAR")
    d2 = FuzzyDate(original_text="1996", edtf="1996", precision="YEAR")
    mstore = await store_with(
        make_mention("m1", "a", "s1", date=d1),
        make_mention("m2", "a", "s2", date=d2),  # same title → exact-key seed merges them
    )
    result = await pipeline(mstore, estore, dict_embed({}), merge_all_reviewer, date_fn=lambda e: e.date).resolve()

    assert len(result.entities) == 1
    merged = result.entities[0].date
    assert merged is not None
    assert merged.start == dt.date(1994, 1, 1)
    assert merged.end == dt.date(1997, 1, 1)  # exclusive end spanning through 1996


@pytest.mark.anyio
async def test_recreatable_and_removed_mention_drops_membership():
    embed = dict_embed({"title: a": [1.0, 0.0, 0.0], "title: b": [0.0, 1.0, 0.0]})  # far apart → never merge
    m_a = make_mention("m1", "a", "s1")
    m_b = make_mention("m2", "b", "s2")

    mstore = await store_with(m_a, m_b)
    estore = MemoryEntityStore()
    first = await pipeline(mstore, estore, embed, no_merge_reviewer).resolve()
    second = await pipeline(mstore, estore, embed, no_merge_reviewer).resolve()
    assert {e.entity_id for e in first.entities} == {e.entity_id for e in second.entities}

    # remove m_b's mention and re-resolve → its entity is gone from the store
    await mstore.delete_by_source("s2")
    third = await pipeline(mstore, estore, embed, no_merge_reviewer).resolve()
    assert {s for e in third.entities for s in e.source_ids} == {"s1"}
    assert set(estore.stored.keys()) == {e.entity_id for e in third.entities}


def test_default_identity_text_renders_payload_with_date():
    e = Event(title="X", date=FuzzyDate(original_text="1994", edtf="1994", precision="YEAR"))
    assert default_identity_text(e) == "title: X\ndate: 1994"


@pytest.mark.anyio
async def test_build_entity_embedder_composes():
    seen = {}

    async def embed(texts):
        seen["texts"] = texts
        return [[1.0, 0.0] for _ in texts]

    embedder = build_entity_embedder(embed, default_identity_text)
    vectors = await embedder([Event(title="X")])
    assert seen["texts"] == ["title: X"]
    assert vectors == [[1.0, 0.0]]


# ---------------------------------------------------------------------------
# make_llm_reviewer degrade semantics (shared LLM layer)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reviewer_refusal_returns_empty_reviewresult():
    """Behavior pin: an LLM refusal degrades to ReviewResult() (no merge groups), never raises."""
    from unittest.mock import AsyncMock, MagicMock

    from ragdoc.extraction.resolution import ReviewResult, make_llm_reviewer

    message = MagicMock()
    message.parsed = None  # refusal
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))

    reviewer = make_llm_reviewer(client, model="test-model")
    result = await reviewer(["Acme Corp", "ACME Corporation"])

    assert isinstance(result, ReviewResult)
    assert result.groups == []
    client.chat.completions.parse.assert_awaited_once()  # refusal not retried


@pytest.mark.anyio
async def test_reviewer_returns_parsed_reviewresult():
    """Happy path: the parsed ReviewResult passes through the shared layer unchanged."""
    from unittest.mock import AsyncMock, MagicMock

    from ragdoc.extraction.resolution import ReviewGroup, ReviewResult, make_llm_reviewer

    parsed = ReviewResult(groups=[ReviewGroup(members=[0, 1], confidence=0.9)])
    message = MagicMock()
    message.parsed = parsed
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=message)]))

    reviewer = make_llm_reviewer(client, model="test-model")
    result = await reviewer(["Acme Corp", "ACME Corporation"])

    assert result is parsed
