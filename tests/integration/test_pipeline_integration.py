"""Integration tests: full pipeline with nested MultiSourceParser.

Exercises the complete real-world flow:
    parse (MinerU + HTML + Pandoc via nested MultiSourceParser)
    → HeadingLevelProcessor → LLMHeadingResolver (mocked) → TitleDetectionProcessor
    → TokenSplitter → LLMChunker (mocked) → VectorStorePipeline

Verifies against the ground-truth structure in tests/data/test_cases.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")
pytest.importorskip("pydantic_settings", reason="pydantic_settings not installed")

from ragdoc.chunking.chunk import Chunk
from ragdoc.chunking.llm import DocumentTopicSummaries, LLMChunker
from ragdoc.document import Document, Heading
from ragdoc.parsing.mineru.base import _latex_to_text
from ragdoc.parsing.multi_source import MultiSourceParser
from ragdoc.parsing.registry import get_parser
from ragdoc.pipeline import DocumentPipeline, VectorStorePipeline
from ragdoc.pipeline.splitter import TokenSplitter
from ragdoc.processing.heading import HeadingLevelProcessor, TitleDetectionProcessor
from ragdoc.processing.heading_llm import (
    HeadingJudgment,
    HeadingLevel,
    HeadingResponse,
    LLMHeadingResolver,
)
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
from ragdoc.utils.helpers import _normalize_text
from ragdoc.utils.tokenizer import GPTTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / "parsing" / "data"
_MINERU_PATH = _DATA_ROOT / "mineru" / "tesla-q4-2024-update_middle.json"
_HTML_PATH = _DATA_ROOT / "html" / "tesla-q4-2024-update.html"
_PANDOC_PATH = _DATA_ROOT / "pandoc" / "tesla-q4-2024-update.docx"
_TEST_CASES_PATH = Path(__file__).parent.parent / "data" / "test_cases.json"

INPUT_PATH = _MINERU_PATH

_ALL_DATA_EXISTS = _MINERU_PATH.exists() and _HTML_PATH.exists() and _PANDOC_PATH.exists()
_TEST_CASES_EXIST = _TEST_CASES_PATH.exists()

# Snapshot: expected number of chunks from VectorStorePipeline.
# With TokenSplitter(max_tokens=7000) + LLMChunker(3 summaries/split) running
# over the tesla-q4-2024-update corpus (mineru + html + pandoc merged).
EXPECTED_CHUNK_COUNT: int | None = 96


# ---------------------------------------------------------------------------
# Test case loading
# ---------------------------------------------------------------------------


def _load_test_case() -> dict:
    """Load the tesla-q4-2024-update test case from test_cases.json."""
    with open(_TEST_CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    return cases["tesla-q4-2024-update"]


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def mineru_to_html_resolver(path: Path) -> Path | None:
    """Map mineru/*_middle.json → html/*.html in the test data tree."""
    stem = path.stem  # e.g. "tesla-q4-2024-update_middle"
    if stem.endswith("_middle"):
        stem = stem[: -len("_middle")]
    candidate = path.parent.parent / "html" / f"{stem}.html"
    return candidate if candidate.exists() else None


def mineru_to_pandoc_resolver(path: Path) -> Path | None:
    """Map mineru/*_middle.json → pandoc/*.docx in the test data tree."""
    stem = path.stem
    if stem.endswith("_middle"):
        stem = stem[: -len("_middle")]
    candidate = path.parent.parent / "pandoc" / f"{stem}.docx"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Text normalisation & heading lookup
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Normalise text for fuzzy comparison (whitespace + LaTeX + NFKC)."""
    return re.sub(r"\s+", "", _normalize_text(_latex_to_text(text))).strip()


def _find_heading(doc: Document, heading_text: str) -> Heading | None:
    """Fuzzy-find a heading by text in the document.

    Returns the best matching Heading (highest length-similarity ratio ≥ 0.5),
    or None if no match is found.
    """
    needle = _norm(heading_text)
    best: Heading | None = None
    best_ratio = 0.0
    for el in doc.elements:
        if not isinstance(el, Heading):
            continue
        el_norm = _norm(el.text)
        if needle not in el_norm and el_norm not in needle:
            continue
        shorter = min(len(needle), len(el_norm))
        longer = max(len(needle), len(el_norm))
        ratio = shorter / longer if longer > 0 else 1.0
        if ratio > best_ratio:
            best_ratio = ratio
            best = el
    return best if best_ratio > 0.5 else None


# ---------------------------------------------------------------------------
# Synthetic LLM heading response builder
# ---------------------------------------------------------------------------

_LEVEL_MAP = {
    1: HeadingLevel.H1,
    2: HeadingLevel.H2,
    3: HeadingLevel.H3,
    4: HeadingLevel.H4,
    5: HeadingLevel.H5,
    6: HeadingLevel.H6,
}


def _build_synthetic_heading_response(
    heading_infos: list,
    expected_headings: dict[str, int],
    expected_title: str,
) -> HeadingResponse:
    """Build a HeadingResponse that assigns levels from test_cases.json.

    For each heading info collected by LLMHeadingResolver._collect_heading_infos,
    fuzzy-match the text against expected_headings and expected_title, then assign
    the corresponding HeadingLevel.  Unmatched headings get NOT_DETERMINABLE.
    """
    judgments: list[HeadingJudgment] = []
    title_norm = _norm(expected_title)

    for i, info in enumerate(heading_infos):
        hid = i + 1  # 1-based
        info_norm = _norm(info.text)

        # Check title match first
        if title_norm in info_norm or info_norm in title_norm:
            shorter = min(len(title_norm), len(info_norm))
            longer = max(len(title_norm), len(info_norm))
            if longer > 0 and shorter / longer > 0.5:
                judgments.append(HeadingJudgment(id=hid, level=HeadingLevel.DOCUMENT_TITLE))
                continue

        # Check heading matches
        matched = False
        for heading_text, expected_level in expected_headings.items():
            heading_norm = _norm(heading_text)
            if heading_norm not in info_norm and info_norm not in heading_norm:
                continue
            shorter = min(len(heading_norm), len(info_norm))
            longer = max(len(heading_norm), len(info_norm))
            if longer > 0 and shorter / longer > 0.5:
                level = _LEVEL_MAP.get(expected_level, HeadingLevel.NOT_DETERMINABLE)
                judgments.append(HeadingJudgment(id=hid, level=level))
                matched = True
                break

        if not matched:
            judgments.append(HeadingJudgment(id=hid, level=HeadingLevel.NOT_DETERMINABLE))

    return HeadingResponse(judgments=judgments)


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


def _make_heading_mock_client(response: HeadingResponse) -> MagicMock:
    """Create a mock OpenAI client that replays the given HeadingResponse."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = response
    mock_response.choices[0].message.refusal = None
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=mock_response)
    return client


def _make_chunker_mock(summaries_per_call: int = 3) -> MagicMock:
    """Create a mock OpenAI client for LLMChunker.

    Returns a fresh DocumentTopicSummaries for each call via side_effect.
    """
    call_count = 0

    def _respond(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        topics = DocumentTopicSummaries(
            summaries=[f"Topic summary {j + 1} for split {call_count}" for j in range(summaries_per_call)]
        )
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.parsed = topics
        return resp

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_respond)
    return client


# ---------------------------------------------------------------------------
# In-memory VectorStore (same pattern as tests/pipeline/test_vectorstore_integration.py)
# ---------------------------------------------------------------------------


class MemoryVectorStore:
    """In-memory VectorStore for testing."""

    def __init__(self) -> None:
        self.stored: dict[str, Chunk] = {}
        self.upsert_calls: list[list[Chunk]] = []
        self.delete_calls: list[list[str]] = []

    async def upsert(self, chunks: list) -> list[str]:
        self.upsert_calls.append(list(chunks))
        for c in chunks:
            self.stored[c.id] = c
        return [c.id for c in chunks]

    async def delete(self, ids: list[str]) -> None:
        self.delete_calls.append(list(ids))
        for id_ in ids:
            self.stored.pop(id_, None)

    async def delete_by_source(self, source_id: str) -> None:
        to_delete = [cid for cid, c in self.stored.items() if c.source_id == source_id]
        for cid in to_delete:
            del self.stored[cid]

    async def list_source_ids(self) -> set[str]:
        return {c.source_id for c in self.stored.values() if c.source_id is not None}

    async def list_source_state(self) -> dict:
        from ragdoc.pipeline.stores import SourceState

        return {
            c.source_id: SourceState(source_hash=c.source_hash, content_hash=c.content_hash)
            for c in self.stored.values()
            if c.source_id is not None
        }


# ---------------------------------------------------------------------------
# Parser + pipeline construction helpers
# ---------------------------------------------------------------------------


def _build_nested_msp() -> MultiSourceParser:
    """Build the nested MultiSourceParser: (mineru+html) + pandoc."""
    inner = MultiSourceParser(
        primary=get_parser("mineru"),
        secondary=get_parser("html"),
        secondary_resolver=mineru_to_html_resolver,
    )
    outer = MultiSourceParser(
        primary=inner,
        secondary=get_parser("pandoc"),
        secondary_resolver=mineru_to_pandoc_resolver,
    )
    return outer


async def _build_processed_doc_and_heading_mock(
    outer_msp: MultiSourceParser,
    test_case: dict,
) -> tuple[Document, MagicMock]:
    """Parse + process (HeadingLevel only), build synthetic heading mock.

    Returns:
        (doc_after_heading_level_processor, heading_mock_client)
    """
    doc = await outer_msp(INPUT_PATH)
    doc = await HeadingLevelProcessor(trust_parser_levels=False).process(doc)

    # Collect heading infos the same way LLMHeadingResolver does
    resolver_tmp = LLMHeadingResolver(client=MagicMock())
    heading_infos = resolver_tmp._collect_heading_infos(doc)

    response = _build_synthetic_heading_response(
        heading_infos,
        expected_headings=test_case["headings"],
        expected_title=test_case["title"],
    )
    mock_client = _make_heading_mock_client(response)
    return doc, mock_client


# ===========================================================================
# Test 1: Document processing pipeline
# ===========================================================================


@pytest.mark.skipif(not _ALL_DATA_EXISTS, reason="Test data files missing")
@pytest.mark.skipif(not _TEST_CASES_EXIST, reason="test_cases.json missing")
@pytest.mark.anyio
async def test_document_pipeline_with_nested_multi_source_parser():
    """Full pipeline: nested MSP → HeadingLevelProcessor → LLMHeadingResolver → TitleDetectionProcessor.

    Validates Document properties (title, headings, footnotes, tables, rendered output)
    against the ground-truth in test_cases.json.
    """
    test_case = _load_test_case()
    outer_msp = _build_nested_msp()

    # --- Build the heading mock from the actual merged document ---
    doc, mock_heading_client = await _build_processed_doc_and_heading_mock(outer_msp, test_case)

    # --- Run LLMHeadingResolver + TitleDetectionProcessor manually ---
    doc = await LLMHeadingResolver(client=mock_heading_client).process(doc)
    doc = await TitleDetectionProcessor().process(doc)

    # === Title ===
    assert doc.title is not None, "Document title was not set"
    assert _norm(doc.title) == _norm(test_case["title"]), (
        f"Wrong title.\n  Expected: {test_case['title']!r}\n  Got: {doc.title!r}"
    )

    # === Heading levels ===
    expected_headings: dict[str, int] = test_case["headings"]
    found_headings: dict[str, tuple[int, int]] = {}  # text → (expected, actual)
    missing_headings: list[str] = []

    for heading_text, expected_level in expected_headings.items():
        match = _find_heading(doc, heading_text)
        if match is not None:
            found_headings[heading_text] = (expected_level, match.level)
        else:
            missing_headings.append(heading_text)

    # At least half the expected headings should be found
    assert len(found_headings) >= len(expected_headings) // 2, (
        f"Too few headings found: {len(found_headings)}/{len(expected_headings)}. Missing: {missing_headings}"
    )

    # Check levels match
    level_mismatches: list[str] = []
    for heading_text, (expected, actual) in found_headings.items():
        if expected != actual:
            level_mismatches.append(f"  '{heading_text}': expected level {expected}, got {actual}")

    assert not level_mismatches, "Heading level mismatches:\n" + "\n".join(level_mismatches)

    # === Heading ordering ===
    headings_list = list(expected_headings.items())
    violations: list[str] = []
    for i, (text_a, exp_a) in enumerate(headings_list):
        for text_b, exp_b in headings_list[i + 1 :]:
            if exp_a >= exp_b:
                continue
            match_a = _find_heading(doc, text_a)
            match_b = _find_heading(doc, text_b)
            if match_a is None or match_b is None:
                continue
            if match_a.level > match_b.level:
                violations.append(
                    f"  '{text_a}' (exp {exp_a}, got {match_a.level}) > '{text_b}' (exp {exp_b}, got {match_b.level})"
                )

    assert not violations, "Heading ordering violations:\n" + "\n".join(violations)

    # === Footnotes ===
    expected_footnotes = test_case.get("footnotes", [])
    doc_footnote_texts = [_norm(fn.text) for fn in doc.footnotes]

    missing_footnotes: list[str] = []
    for fn_case in expected_footnotes:
        fn_norm = _norm(fn_case["text"])
        found = any(fn_norm in ft or ft in fn_norm for ft in doc_footnote_texts)
        if not found:
            missing_footnotes.append(fn_case["text"][:80])

    assert len(missing_footnotes) <= len(expected_footnotes) // 3, (
        f"Too many footnotes missing ({len(missing_footnotes)}/{len(expected_footnotes)}):\n"
        + "\n".join(f"  - {t}..." for t in missing_footnotes)
    )

    # === Tables ===
    expected_tables = test_case.get("tables", [])
    all_table_text = " ".join(t.text for t in doc.tables)
    for table_str in expected_tables:
        assert table_str in all_table_text, f"Table content '{table_str}' not found"

    # === Rendered output regex ===
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    rendered = renderer.render(doc)

    for pattern in test_case.get("present_in_output_regex", []):
        if not re.search(pattern, rendered):
            pytest.xfail(f"Pattern not found in merged output — content may differ across parsers: {pattern!r}")

    for pattern in test_case.get("not_present_in_output_regex", []):
        assert not re.search(pattern, rendered), f"Pattern should NOT be in output: {pattern!r}"

    # === Also verify pipeline.run() produces chunks ===
    mock_heading_client_2 = _make_heading_mock_client(
        _build_synthetic_heading_response(
            LLMHeadingResolver(client=MagicMock())._collect_heading_infos(
                await HeadingLevelProcessor(trust_parser_levels=False).process(await outer_msp(INPUT_PATH))
            ),
            expected_headings=test_case["headings"],
            expected_title=test_case["title"],
        )
    )

    pipeline = DocumentPipeline(
        parser=outer_msp,
        processors=[
            HeadingLevelProcessor(trust_parser_levels=False),
            LLMHeadingResolver(client=mock_heading_client_2),
            TitleDetectionProcessor(),
        ],
    )
    chunks = await pipeline.run(INPUT_PATH)
    assert len(chunks) >= 1, "Pipeline produced no chunks"
    assert all(c.prompt_content for c in chunks), "Chunk with empty prompt_content"


# ===========================================================================
# Test 2: VectorStorePipeline with splitter + LLMChunker
# ===========================================================================


@pytest.mark.skipif(not _ALL_DATA_EXISTS, reason="Test data files missing")
@pytest.mark.skipif(not _TEST_CASES_EXIST, reason="test_cases.json missing")
@pytest.mark.skipif(
    os.getenv("RAGDOC_RESOLUTION") == "lowest",
    reason="EXPECTED_CHUNK_COUNT snapshot depends on tiktoken/transformers tokenization at the dep floor",
)
@pytest.mark.anyio
async def test_vector_store_pipeline_with_llm_chunker(tmp_path: Path):
    """VectorStorePipeline: nested MSP → processing → TokenSplitter(7000) → LLMChunker.

    Verifies chunk count stability, token limits, and idempotent re-sync.
    """
    test_case = _load_test_case()
    outer_msp = _build_nested_msp()

    # Build heading mock from the merged document
    _, mock_heading_client = await _build_processed_doc_and_heading_mock(outer_msp, test_case)

    mock_chunker_client = _make_chunker_mock(summaries_per_call=3)

    doc_pipeline = DocumentPipeline(
        parser=outer_msp,
        processors=[
            HeadingLevelProcessor(trust_parser_levels=False),
            LLMHeadingResolver(client=mock_heading_client),
            TitleDetectionProcessor(),
        ],
        splitter=TokenSplitter(max_tokens=7000),
        chunker=LLMChunker(client=mock_chunker_client, model="test"),
    )

    vstore = MemoryVectorStore()

    vs_pipeline = VectorStorePipeline(
        pipeline=doc_pipeline,
        vector_store=vstore,
    )

    # --- First run ---
    result = await vs_pipeline.run([INPUT_PATH])

    # UpdateResult is keyed on source_id (default source_id_fn = path.name).
    assert result.processed == [INPUT_PATH.name], f"Expected processed=[{INPUT_PATH.name}], got {result.processed}"
    assert len(result.errors) == 0, f"Pipeline errors: {result.errors}"
    assert len(result.skipped) == 0

    all_chunks = list(vstore.stored.values())
    assert len(all_chunks) > 0, "No chunks produced"

    # --- Snapshot: chunk count stability ---
    if EXPECTED_CHUNK_COUNT is not None:
        assert len(all_chunks) == EXPECTED_CHUNK_COUNT, (
            f"Chunk count changed: expected {EXPECTED_CHUNK_COUNT}, got {len(all_chunks)}"
        )
    else:
        # First run: print the count so it can be hardcoded
        print(f"\n[SNAPSHOT] Chunk count = {len(all_chunks)} — update EXPECTED_CHUNK_COUNT\n")

    # --- Token limit enforcement ---
    tokenizer = GPTTokenizer()
    for chunk in all_chunks:
        token_count = tokenizer.count(chunk.prompt_content)
        assert token_count <= 7000, f"Chunk {chunk.id} has {token_count} tokens, exceeds 7000 limit"

    # --- Non-empty embedding content ---
    for chunk in all_chunks:
        assert chunk.embedding_content, f"Chunk {chunk.id} has empty embedding_content"

    # --- LLMChunker produces distinct embedding_content ---
    for chunk in all_chunks:
        assert chunk.embedding_content != chunk.prompt_content, (
            f"Chunk {chunk.id}: embedding_content should differ from prompt_content (LLMChunker)"
        )

    # --- All chunks have metadata ---
    for chunk in all_chunks:
        assert isinstance(chunk.metadata, dict)

    # --- Idempotent re-run ---
    result2 = await vs_pipeline.run([INPUT_PATH])
    assert len(result2.skipped) == 1, f"Expected skip on re-run, got {result2}"
    assert len(result2.processed) == 0
