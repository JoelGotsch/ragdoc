"""Unit tests for DocumentSummarizerProcessor and helpers (summary_document.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_settings", reason="pydantic_settings not installed")

from ragdoc.processing.summary_document import (
    SUMMARY_SYSTEM_PROMPT,
    DocumentSummarizerSettings,
    build_summary_messages,
    pack_summaries,
)
from ragdoc.utils import GPTTokenizer

_TOKENIZER = GPTTokenizer()


# =============================================================================
# pack_summaries
# =============================================================================


def test_pack_summaries_preserves_order_and_respects_budget():
    # Each word is ~1 token; budget of 3 tokens packs ~3 words per batch.
    summaries = ["alpha", "beta", "gamma", "delta", "epsilon"]
    batches = pack_summaries(summaries, max_tokens=3, tokenizer=_TOKENIZER)

    # Order preserved when flattened.
    assert [s for batch in batches for s in batch] == summaries
    # No batch exceeds the budget.
    for batch in batches:
        assert _TOKENIZER.count(" ".join(batch)) <= 3 or len(batch) == 1


def test_pack_summaries_oversized_string_gets_own_batch():
    big = " ".join(["word"] * 50)  # well over the budget on its own
    batches = pack_summaries(["small", big, "tail"], max_tokens=5, tokenizer=_TOKENIZER)

    # The oversized string is never dropped and occupies a batch alone.
    assert [big] in batches
    assert sum(s == big for batch in batches for s in batch) == 1


def test_pack_summaries_empty_input():
    assert pack_summaries([], max_tokens=10, tokenizer=_TOKENIZER) == []


# =============================================================================
# DocumentSummarizerSettings
# =============================================================================


def test_settings_min_must_be_less_than_max():
    with pytest.raises(ValueError):
        DocumentSummarizerSettings(min_tokens=5000, max_input_tokens=5000)


def test_settings_default_prompt_is_builtin():
    settings = DocumentSummarizerSettings()
    assert settings.system_prompt == SUMMARY_SYSTEM_PROMPT


def test_settings_explicit_prompt_wins(monkeypatch):
    monkeypatch.setenv("DOCUMENT_SUMMARIZER_SYSTEM_PROMPT", "FROM_ENV")
    settings = DocumentSummarizerSettings(system_prompt="EXPLICIT")
    assert settings.system_prompt == "EXPLICIT"


def test_settings_env_prompt_used(monkeypatch):
    monkeypatch.setenv("DOCUMENT_SUMMARIZER_SYSTEM_PROMPT", "FROM_ENV")
    assert DocumentSummarizerSettings().system_prompt == "FROM_ENV"


def test_settings_prompt_file_used_when_prompt_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCUMENT_SUMMARIZER_SYSTEM_PROMPT", raising=False)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("FROM_FILE", encoding="utf-8")
    settings = DocumentSummarizerSettings(system_prompt_file=str(prompt_file))
    assert settings.system_prompt == "FROM_FILE"


def test_settings_inline_prompt_beats_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("FROM_FILE", encoding="utf-8")
    settings = DocumentSummarizerSettings(system_prompt="INLINE", system_prompt_file=str(prompt_file))
    assert settings.system_prompt == "INLINE"


def test_settings_min_tokens_from_env(monkeypatch):
    monkeypatch.setenv("DOCUMENT_SUMMARIZER_MIN_TOKENS", "1234")
    assert DocumentSummarizerSettings().min_tokens == 1234


# =============================================================================
# build_summary_messages
# =============================================================================


def test_build_summary_messages_roles_and_text():
    msgs = build_summary_messages("the body text", system_prompt="SYS")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "SYS"
    assert "the body text" in msgs[1]["content"]


# =============================================================================
# DocumentSummarizerProcessor (mocked LLM)
# =============================================================================

from unittest.mock import AsyncMock, MagicMock

from ragdoc.document import Document, Paragraph
from ragdoc.processing.summary_base import DocumentSummary
from ragdoc.processing.summary_document import DocumentSummarizerProcessor
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt


class _WordTokenizer:
    """Deterministic tokenizer for tests: one token per whitespace-separated word."""

    def __call__(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def count(self, text: str) -> int:
        return len(text.split())


def _renderer() -> Renderer:
    return Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)


def _make_client(summary_fn):
    """Mock client; parse() returns DocumentSummary(summary=summary_fn(input_text)).

    Returns (client, calls) where ``calls`` is the list of ``messages`` lists, in call order.
    """
    calls: list = []

    async def parse(*args, **kwargs):
        messages = kwargs["messages"]
        calls.append(messages)
        user = messages[1]["content"]
        text = user.split("\n\n", 1)[1] if "\n\n" in user else user
        return MagicMock(choices=[MagicMock(message=MagicMock(parsed=DocumentSummary(summary=summary_fn(text))))])

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=parse)
    return client, calls


def _para(text: str) -> Paragraph:
    return Paragraph(html=f"<p>{text}</p>")


def test_unconfigured_client_raises_at_construction():
    """No explicit client and no configured client → LLMNotConfiguredError in __init__, not process()."""
    from ragdoc.config import RagdocConfig, configure
    from ragdoc.llm import LLMNotConfiguredError

    with configure(RagdocConfig()), pytest.raises(LLMNotConfiguredError):
        DocumentSummarizerProcessor()


def test_configured_client_resolved_at_construction():
    from ragdoc.config import RagdocConfig, configure

    configured = MagicMock()
    with configure(RagdocConfig(openai_client=configured)):
        processor = DocumentSummarizerProcessor()
    assert processor._client is configured


@pytest.mark.anyio
async def test_passthrough_below_min_returns_rendered_text_no_call():
    client, _calls = _make_client(lambda t: "UNUSED")
    renderer = _renderer()
    doc = Document(elements=[_para("short body")])
    settings = DocumentSummarizerSettings(min_tokens=1_000_000, max_input_tokens=2_000_000)

    processor = DocumentSummarizerProcessor(client=client, settings=settings, renderer=renderer)
    result = await processor.process(doc)

    assert result.metadata["summary"] == renderer.render(doc)
    client.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_single_call_when_between_min_and_max():
    client, _calls = _make_client(lambda t: "ONE")
    doc = Document(elements=[_para("alpha beta gamma")])
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=1_000_000)

    processor = DocumentSummarizerProcessor(client=client, settings=settings, renderer=_renderer())
    result = await processor.process(doc)

    assert result.metadata["summary"] == "ONE"
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_map_then_single_reduce():
    client, _calls = _make_client(lambda t: "s")  # each summary is 1 word
    pieces = [Document(elements=[_para("a b c")]) for _ in range(3)]
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=5)

    processor = DocumentSummarizerProcessor(
        client=client,
        settings=settings,
        renderer=_renderer(),
        tokenizer=_WordTokenizer(),
        splitter=lambda doc: pieces,
    )
    # Whole doc must exceed max to trigger the split path.
    doc = Document(elements=[_para(" ".join(["word"] * 20))])
    result = await processor.process(doc)

    # 3 map calls + 1 reduce call.
    assert client.chat.completions.parse.call_count == 4
    assert result.metadata["summary"] == "s"


@pytest.mark.anyio
async def test_deep_recursion_terminates():
    client, _calls = _make_client(lambda t: "x")  # always shrinks to 1 word
    # Pieces must be <= max_input_tokens so the map step is single calls; the deep recursion is
    # then forced in the reduce (joining three 1-word summaries exceeds max_input_tokens=2).
    pieces = [Document(elements=[_para("a b")]) for _ in range(3)]
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=2, max_recursion_depth=5)

    processor = DocumentSummarizerProcessor(
        client=client,
        settings=settings,
        renderer=_renderer(),
        tokenizer=_WordTokenizer(),
        splitter=lambda doc: pieces,
    )
    doc = Document(elements=[_para(" ".join(["word"] * 20))])
    result = await processor.process(doc)

    # Reduce had to recurse (more than map + single reduce) but still terminated.
    assert client.chat.completions.parse.call_count > 4
    assert result.metadata["summary"] == "x"


@pytest.mark.anyio
async def test_recursion_cap_raises_when_not_shrinking():
    client, _calls = _make_client(lambda t: " ".join(["big"] * 10))  # never shrinks
    pieces = [Document(elements=[_para("a b c")]) for _ in range(2)]
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=5, max_recursion_depth=2)

    processor = DocumentSummarizerProcessor(
        client=client,
        settings=settings,
        renderer=_renderer(),
        tokenizer=_WordTokenizer(),
        splitter=lambda doc: pieces,
    )
    doc = Document(elements=[_para(" ".join(["word"] * 20))])
    with pytest.raises(RuntimeError, match="recursion"):
        await processor.process(doc)


@pytest.mark.anyio
async def test_map_order_preserved_under_concurrency():
    client, calls = _make_client(lambda t: t)  # echo input back
    pieces = [
        Document(elements=[_para("AAA AAA AAA")]),
        Document(elements=[_para("BBB BBB BBB")]),
        Document(elements=[_para("CCC CCC CCC")]),
    ]
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=100)

    processor = DocumentSummarizerProcessor(
        client=client,
        settings=settings,
        renderer=_renderer(),
        tokenizer=_WordTokenizer(),
        splitter=lambda doc: pieces,
        concurrency=3,
    )
    doc = Document(elements=[_para(" ".join(["word"] * 200))])
    await processor.process(doc)

    # The final (reduce) call's input must contain the pieces in reading order.
    reduce_input = calls[-1][1]["content"]
    assert reduce_input.index("AAA") < reduce_input.index("BBB") < reduce_input.index("CCC")


@pytest.mark.anyio
async def test_idempotent_skips_when_summary_present():
    client, _calls = _make_client(lambda t: "NEW")
    doc = Document(elements=[_para("body")], metadata={"summary": "EXISTING"})

    processor = DocumentSummarizerProcessor(client=client, renderer=_renderer())
    result = await processor.process(doc)

    assert result.metadata["summary"] == "EXISTING"
    client.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_overwrite_replaces_existing_summary():
    client, _calls = _make_client(lambda t: "NEW")
    doc = Document(elements=[_para("body")], metadata={"summary": "EXISTING"})
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=1_000_000)

    processor = DocumentSummarizerProcessor(client=client, settings=settings, renderer=_renderer(), overwrite=True)
    result = await processor.process(doc)

    assert result.metadata["summary"] == "NEW"
    assert client.chat.completions.parse.call_count == 1


@pytest.mark.anyio
async def test_other_metadata_preserved_and_no_split_leak():
    client, _calls = _make_client(lambda t: "ONE")
    doc = Document(elements=[_para("alpha beta")], metadata={"filename": "doc.pdf"})
    settings = DocumentSummarizerSettings(min_tokens=0, max_input_tokens=1_000_000)

    processor = DocumentSummarizerProcessor(client=client, settings=settings, renderer=_renderer())
    result = await processor.process(doc)

    assert result.metadata["filename"] == "doc.pdf"
    assert result.metadata["summary"] == "ONE"
    assert "split_sequence" not in result.metadata
    assert "split_total" not in result.metadata


@pytest.mark.anyio
async def test_empty_document_no_call_no_summary():
    client, _calls = _make_client(lambda t: "ONE")
    doc = Document(elements=[])

    processor = DocumentSummarizerProcessor(client=client, renderer=_renderer())
    result = await processor.process(doc)

    assert "summary" not in result.metadata
    client.chat.completions.parse.assert_not_called()


@pytest.mark.anyio
async def test_prompt_override_reaches_system_message():
    client, calls = _make_client(lambda t: "ONE")
    doc = Document(elements=[_para("alpha beta")])
    settings = DocumentSummarizerSettings(system_prompt="CUSTOM PROMPT", min_tokens=0, max_input_tokens=1_000_000)

    processor = DocumentSummarizerProcessor(client=client, settings=settings, renderer=_renderer())
    await processor.process(doc)

    assert calls[0][0]["content"] == "CUSTOM PROMPT"
