"""Notebook execution tests — run every notebook in docs/notebooks/ via app.run().

Each test wraps one marimo notebook, executes it end-to-end, and asserts on the values
the notebook actually computed (``defs``), so a notebook that runs but produces the wrong
thing fails here — not only one that crashes.

This is layer 4 of the notebook-freshness checks (see CLAUDE.md). Layers 2/3 — the
``test_*`` cells inside the notebooks — run via ``uv run pytest docs/notebooks/``.

Skip guards:
- marimo missing            -> all tests skip.
- qdrant_pipeline           -> needs the ``qdrant`` extra (runs offline against
  ``AsyncQdrantClient(":memory:")``; no LLM credentials needed).
- pipeline_walkthrough      -> two tests: the offline one runs the notebook with the LLM
  key blanked and checks that the LLM-backed cells skip cleanly; the other makes REAL
  LLM calls and is opt-in via ``RAGDOC_RUN_LLM_NOTEBOOKS=1`` (plus LLM credentials).
- mineru_example            -> needs the MinerU parser to be importable.
- extraction                -> needs numpy (``extraction`` extra) for entity resolution.
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
    assert all(c.source_id == "q3_report.docx" for c in chunks)
    assert all(c.source_hash is None for c in chunks)  # hand-built chunks carry no file hash


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
    # concat_documents: first's elements, a bridge <h1> carrying the second title, second's.
    combined = defs["combined"]
    assert [e.text for e in combined.elements] == ["First part.", "Part Two", "Second part."]
    assert combined.metadata["filename"] == "one.html"  # default policy: first wins
    assert combined.source_id is None


def test_extraction_notebook():
    pytest.importorskip("numpy", reason="extraction extra not installed (numpy needed for resolution)")

    from docs.notebooks.extraction import app

    _outputs, defs = app.run()
    # 3 mentions across the 2-file corpus; re-run skipped both sources.
    assert len(defs["mentions"]) == 3
    assert defs["result_second"].processed == []
    assert sorted(defs["result_second"].skipped) == ["letters.html", "notes.html"]
    # Resolution merged the two Ada mentions: 3 mentions -> 2 entities.
    entities = defs["entities"]
    assert [e.payload.name for e in entities] == ["Ada Lovelace", "Charles Babbage"]
    assert len(entities[0].member_mention_ids) == 2


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

    from ragdoc.document import Heading, Paragraph, Table

    _outputs, defs = app.run()
    # The quick-start cell really parses the sample HTML file it writes.
    document = defs["document"]
    assert document.parser == "html"
    assert [type(e) for e in document.elements] == [Heading, Paragraph, Heading, Table]
    assert document.title is None  # no processor ran: title detection is a processing step
    assert document.metadata["filename"] == "page.html"
    assert document.source_path == str(defs["sample_path"])
    # Forcing the parser by name parses the same content.
    assert defs["doc_html"].content_hash() == document.content_hash()
    # The registry listing names every built-in parser.
    for name in ("html", "pandoc", "xlsx", "azure_json", "azure_di", "pdf_basic", "mineru", "ragdoc_json"):
        assert name in defs["registry_text"]


def test_pipeline_notebook():
    from docs.notebooks.pipeline import app

    _outputs, defs = app.run()
    # Scenario A: one chunk without a splitter, several with one; ids are deterministic.
    assert len(defs["chunks"]) == 1
    assert len(defs["split_chunks"]) > 1
    assert [c.id for c in defs["rerun_chunks"]] == [c.id for c in defs["split_chunks"]]
    assert defs["split_chunks"][0].id == defs["expected_first_id"]
    # Scenario D: fan-out collects every file; on_error="skip" reports the missing one.
    assert len(defs["many_result"].chunks) == 3
    assert [p.name for p, _ in defs["skip_result"].errors] == ["does_not_exist.html"]
    # Scenario B (direct sync): plan writes nothing, apply adds, unchanged skips, edit updates.
    all_sources = ["faq.html", "handbook.html", "release_notes.html"]
    assert sorted(sc.source_id for sc in defs["first_plan"].to_add) == all_sources
    assert defs["stored_after_plan"] == 0
    assert sorted(defs["first_apply"].processed) == all_sources
    assert defs["unchanged_run"].processed == []
    assert sorted(defs["unchanged_run"].skipped) == all_sources
    assert [sc.source_id for sc in defs["changed_plan"].to_update] == ["faq.html"]
    assert defs["changed_run"].processed == ["faq.html"]
    # Orphans are kept by default and deleted only on opt-in.
    assert defs["orphan_kept_run"].deleted == []
    assert defs["orphan_deleted_run"].deleted == ["release_notes.html"]
    assert defs["sources_after_opt_in"] == {"faq.html", "handbook.html"}
    # Scenario C (two boundaries): only the edited source moves through each boundary.
    assert defs["b1_changed"].processed == ["faq.html"]
    assert [sc.source_id for sc in defs["b2_changed_plan"].to_update] == ["faq.html"]
    assert defs["b2_changed"].processed == ["faq.html"]


def test_pipeline_walkthrough_notebook_offline(monkeypatch: pytest.MonkeyPatch):
    """Without an LLM key the parsing stage runs and every LLM-backed cell skips cleanly."""
    pytest.importorskip("openai", reason="llm extra not installed")
    try:
        from ragdoc.parsing.mineru import MinerUParser  # noqa: F401
    except ImportError:
        pytest.skip("MinerU parser not importable (pdf-mineru extra not installed)")
    # An empty value wins over any repo-root .env (load_dotenv never overrides the environment).
    monkeypatch.setenv("LLM_HEADING_RESOLVER_API_KEY", "")

    from docs.notebooks.pipeline_walkthrough import app

    _outputs, defs = app.run()
    assert defs["llm_client"] is None
    doc_parsed = defs["doc_parsed"]
    assert doc_parsed.parser == "mineru"
    assert len(doc_parsed.headings) > 0
    assert len(doc_parsed.tables) > 0
    # Stage 2 onwards needs the LLM: those cells were stopped, so their names are undefined.
    for name in ("chunks_quick", "doc_headings", "splits", "chunks"):
        assert name not in defs


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

    from docs.notebooks.qdrant_pipeline import app

    _outputs, defs = app.run()
    all_sources = ["onboarding.html", "security.html", "travel.html"]
    # First run adds everything; the second (run_directory) skips everything.
    assert sorted(defs["first_run"].processed) == all_sources
    assert defs["first_run"].errors == []
    assert defs["second_run"].processed == []
    assert sorted(defs["second_run"].skipped) == all_sources
    # An edited file is the only planned update; apply() writes it.
    assert [sc.source_id for sc in defs["update_plan"].to_update] == ["security.html"]
    assert defs["update_result"].processed == ["security.html"]
    # A removed file stays unless delete_orphans=True.
    assert defs["kept_run"].deleted == []
    assert defs["orphan_run"].deleted == ["travel.html"]
    assert defs["sources_after_orphan"] == {"onboarding.html", "security.html"}
    # Provenance is stored top-level in the Qdrant payload, and retrieval finds the new text.
    assert {"source_id", "source_hash", "content_hash"} <= set(defs["payload_keys"])
    assert "Hardware security keys" in defs["top_hit_payload"]["prompt_content"]
    # Two-boundary sync through QdrantDocumentStore.
    assert sorted(defs["b2_first"].processed) == ["onboarding.html", "security.html"]
    assert defs["b2_second"].processed == []


def test_quickstart_notebook(monkeypatch: pytest.MonkeyPatch):
    from docs.notebooks.quickstart import app

    pytest.importorskip("openai", reason="llm extra not installed (the enrichment cell imports openai)")
    # The enrichment cell is gated on OPENAI_API_KEY; force the offline path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _outputs, defs = app.run()
    # The notebook writes a sample report.html and chunks it for real.
    chunks = defs["chunks"]
    assert len(chunks) == 1
    assert "Quarterly Report" in chunks[0].prompt_content
    assert "subscription tier" in chunks[0].prompt_content
    assert chunks[0].source_id == "report.html"
    assert chunks[0].source_path == str(defs["FILE_PATH"])
    # No OPENAI_API_KEY: the LLM enrichment cell is skipped and produces nothing.
    assert defs["enriched_chunks"] == []


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
