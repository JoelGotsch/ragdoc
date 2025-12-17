"""Notebook execution tests — run every notebook in docs/notebooks/ via app.run().

Each test wraps one marimo notebook, executes it end-to-end, and asserts a key
definition so a broken notebook fails loudly in CI.

Skip guards:
- marimo missing            -> all tests skip.
- qdrant_pipeline           -> needs the ``qdrant_client`` extra and LLM credentials
  (``LLM_HEADING_RESOLVER_API_KEY``; the notebook itself loads them via ``.env``).
- pipeline_walkthrough      -> makes REAL LLM calls; opt in with
  ``RAGDOC_RUN_LLM_NOTEBOOKS=1`` (plus LLM credentials).
- mineru_example            -> needs the MinerU parser to be importable.
"""

import os

import pytest

pytest.importorskip("marimo", reason="marimo (docs/notebook tooling) not installed")


def _llm_credentials_available() -> bool:
    """True when LLMHeadingResolver credentials are available (env or repo-root .env)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # same lookup the notebooks themselves perform
    except ImportError:
        pass
    return bool(os.environ.get("LLM_HEADING_RESOLVER_API_KEY"))


def test_chunking_notebook():
    from docs.notebooks.chunking import app

    _outputs, defs = app.run()
    chunks = defs["chunks"]
    assert len(chunks) > 0
    # SimpleChunker semantics demonstrated in the notebook: embedding == prompt
    assert all(c.embedding_content == c.prompt_content for c in chunks)
    assert all(c.source_id and c.source_hash for c in chunks)


def test_custom_elements_notebook():
    from docs.notebooks.custom_elements import app

    _outputs, defs = app.run()
    document = defs["document"]
    assert document.parser == "callout"
    assert len(document.elements) == 2  # two callout lines in the sample file


def test_document_model_notebook():
    from docs.notebooks.document_model import app

    _outputs, defs = app.run()
    assert defs["doc"].title == "Annual Report"
    assert len(defs["demo_doc"].headings) == 2


def test_merging_notebook():
    from docs.notebooks.merging import app

    _outputs, defs = app.run()
    assert len(defs["merged_a"].elements) > 0
    assert len(defs["merged_b"].elements) > 0
    assert len(defs["patch_a"].operations) > 0


def test_mineru_example_notebook():
    try:
        from ragdoc.parsing.mineru import MinerUParser  # noqa: F401
    except ImportError:
        pytest.skip("MinerU parser not importable (pdf-mineru extra not installed)")

    from docs.notebooks.mineru_example import app

    _outputs, defs = app.run()
    assert "MinerUParser" in defs


def test_parsing_notebook():
    from docs.notebooks.parsing import app

    _outputs, defs = app.run()
    assert callable(defs["load"])
    # Placeholder paths (report.docx, page.html, ...) usually don't exist; the
    # notebook must still run and leave the documents as None in that case.
    for name in ("document", "doc_docx", "doc_html", "doc_xlsx", "doc_azure"):
        doc = defs[name]
        assert doc is None or len(doc.elements) >= 0


def test_pipeline_notebook():
    from docs.notebooks.pipeline import app

    _outputs, defs = app.run()
    assert defs["pipeline"] is not None
    assert defs["pipeline_with_split"] is not None
    assert defs["paths"] == []  # placeholder corpus is empty


def test_pipeline_walkthrough_notebook():
    if not os.environ.get("RAGDOC_RUN_LLM_NOTEBOOKS"):
        pytest.skip("pipeline_walkthrough makes real LLM calls; set RAGDOC_RUN_LLM_NOTEBOOKS=1 to run")
    if not _llm_credentials_available():
        pytest.skip("LLM_HEADING_RESOLVER_API_KEY not configured")

    from docs.notebooks.pipeline_walkthrough import app

    _outputs, defs = app.run()
    chunks = defs["chunks"]
    assert len(chunks) > 0
    assert all(c.prompt_content for c in chunks)
    assert all(c.source_id == "bert-paper_middle.json" for c in chunks)


def test_qdrant_pipeline_notebook():
    pytest.importorskip("qdrant_client", reason="qdrant extra not installed")
    if not _llm_credentials_available():
        pytest.skip("LLM_HEADING_RESOLVER_API_KEY not configured (the notebook builds an LLM client at setup)")

    from docs.notebooks.qdrant_pipeline import app

    _outputs, defs = app.run()
    assert defs["doc_pipeline"] is not None
    assert defs["paths"] == []  # placeholder corpus dir does not exist
    assert defs["VECTOR_SIZE"] == 1536


def test_quickstart_notebook():
    from docs.notebooks.quickstart import app

    _outputs, defs = app.run()
    assert defs["pipeline"] is not None
    # FILE_PATH is a placeholder (report.docx); without the file both chunk
    # lists stay empty, but the notebook must run without exceptions.
    assert isinstance(defs["chunks"], list)
    assert isinstance(defs["enriched_chunks"], list)


def test_rendering_notebook():
    from ragdoc.document import Image
    from ragdoc.rendering import render_for_prompt

    # The notebook registers a custom Image renderer on the library-wide
    # singledispatch function; restore the original to avoid leaking into
    # other tests in the same process.
    original_image_renderer = render_for_prompt.dispatch(Image)
    try:
        from docs.notebooks.rendering import app

        _outputs, defs = app.run()
        assert "Introduction" in defs["prompt_text"]
        assert "quarterly revenue" in defs["img"].text_representation
    finally:
        render_for_prompt.register(Image)(original_image_renderer)


def test_splitting_notebook():
    from docs.notebooks.splitting import app

    _outputs, defs = app.run()
    splits = defs["splits"]
    assert all("split_sequence" in d.metadata for d in splits)
    assert [d.metadata["split_sequence"] for d in splits] == list(range(1, len(splits) + 1))
    assert all(d.metadata["split_total"] == len(splits) for d in splits)
