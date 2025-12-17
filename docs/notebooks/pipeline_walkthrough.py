import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Complete Pipeline Walkthrough

    End-to-end walkthrough of the **ragdoc** document processing pipeline using a real
    MinerU-parsed government document as input.

    | Stage | What happens |
    |---|---|
    | **1. Parsing** | Load a MinerU `_middle.json` → `Document` |
    | **2. Processing** | Resolve heading levels (LLM), detect title, resolve footnotes |
    | **3. Splitting** | Break into token-bounded sub-`Document`s |
    | **4. Chunking** | Materialise `Chunk` objects with `prompt_content` + `embedding_content` |

    ---

    **Key design principle:**
    > `Document` is the single source of truth throughout the pipeline.
    > Only at the final chunking step are both content representations materialised into strings.

    ```
    PARSING → PROCESSING ─┐
                           ↓
                       SPLITTING  ← uses rendering for token budget
                           ↓
                       CHUNKING   ← renders prompt_content; embedding_content varies by chunker
                           ↓
                        Chunk
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Full pipeline — ingestion script

    The block below shows the complete pipeline as you would write it in a production ingestion script.
    The rest of the notebook unpacks each step in detail.
    """)
    return


@app.cell
async def _(mo):
    import asyncio
    import hashlib
    from pathlib import Path

    from dotenv import load_dotenv
    from httpx import AsyncClient
    from openai import AsyncOpenAI

    from ragdoc.chunking import Chunk
    from ragdoc.parsing.mineru import MinerUParser
    from ragdoc.parsing.mineru.base import MinerUMiddleDocument
    from ragdoc.processing import (
        FootnoteProcessor,
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
        ProcessingPipeline,
        TitleDetectionProcessor,
    )
    from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt
    from ragdoc.splitting import split_document
    load_dotenv()
    # Resolve the fixture relative to the notebook file so the path works from any CWD
    DATA_FILE = (mo.notebook_dir() / "../../tests/parsing/data/mineru/bert-paper_middle.json").resolve()

    async def ingest(path: Path) -> list[Chunk]:
        doc_parsed = await MinerUParser()(path)
        settings = LLMHeadingResolverSettings()
        http_client = AsyncClient(verify=False)
        llm_client = AsyncOpenAI(max_retries=3, api_key=settings.api_key.get_secret_value(), base_url=settings.base_url, http_client=http_client)
        doc_enriched = await ProcessingPipeline([LLMHeadingResolver(client=llm_client), TitleDetectionProcessor(remove_elements_before_title=True), FootnoteProcessor()]).process(doc_parsed)
        prompt_renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        splits = split_document(doc_enriched, renderer=prompt_renderer, max_tokens=7000, overlap_tokens=200)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: ASYNC240 — doc example; sync read is fine here
        return [Chunk(prompt_content=prompt_renderer.render(_split), embedding_content=prompt_renderer.render(_split), source_id=path.name, source_hash=source_hash, metadata={"filename": path.stem, "title": doc_enriched.title, "heading_path": " > ".join(h.text for h in _split.headings)}) for _split in splits]
    chunks_quick = await ingest(DATA_FILE)
    # In a script: asyncio.run(ingest(DATA_FILE))
    # In a notebook, await works directly:
    print(f"{len(chunks_quick)} chunks produced")
    return (
        AsyncClient,
        AsyncOpenAI,
        Chunk,
        FootnoteProcessor,
        LLMHeadingResolver,
        LLMHeadingResolverSettings,
        MinerUMiddleDocument,
        MinerUParser,
        OutputFormat,
        Path,
        ProcessingPipeline,
        Renderer,
        TitleDetectionProcessor,
        asyncio,
        hashlib,
        load_dotenv,
        render_for_prompt,
        split_document,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step-by-step walkthrough

    The same pipeline, one stage at a time — with explanations and intermediate output at each step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    ### Configuration

    The pipeline uses an LLM for heading resolution and table summarisation.
    Credentials are loaded from a `.env` file in the project root — never hardcoded.

    Create `.env` with the following variables:

    ```dotenv
    LLM_HEADING_RESOLVER_BASE_URL=https://qwen-next-80b-a3b-in-awq.k8s.sg.iaea.org/v1
    LLM_HEADING_RESOLVER_API_KEY=<your-api-key>
    LLM_HEADING_RESOLVER_MODEL_NAME=qwen-next-80b-a3b-in-awq
    ```

    `LLMHeadingResolverSettings` (a `pydantic-settings` model) reads these automatically
    via the `LLM_HEADING_RESOLVER_` prefix. We then build a shared `AsyncOpenAI` client
    from those settings.

    The internal endpoint uses a private CA, so a custom `httpx.AsyncClient` is required.
    Set `verify=False` to skip verification, or `verify=str(path_to_ca_bundle)` to pin
    the corporate certificate bundle.
    """)
    return


@app.cell
def _(AsyncClient, AsyncOpenAI, LLMHeadingResolverSettings, load_dotenv):
    load_dotenv()
    from ragdoc.document import Footnote, Heading, Table
    from ragdoc.utils import GPTTokenizer
    settings = LLMHeadingResolverSettings()
    http_client = AsyncClient(verify=False)
    llm_client = AsyncOpenAI(max_retries=3, api_key=settings.api_key.get_secret_value(), base_url=settings.base_url, http_client=http_client)  # searches upward from CWD; finds the project-root .env
    llm_model = settings.model_name
    print(f"LLM endpoint: {settings.base_url}")
    print(f"LLM model:    {llm_model}")
    # Read LLM_HEADING_RESOLVER_* from environment and build a shared client.
    # The internal endpoint uses a private CA — verify=False skips cert validation.
    # To pin the corporate bundle instead: AsyncClient(verify=str(Path("certs/ca-bundle-all.crt")))
    print("Imports OK")
    return (
        Footnote,
        GPTTokenizer,
        Heading,
        Table,
        llm_client,
        llm_model,
    )


@app.cell
def _(
    Footnote,
    GPTTokenizer,
    Heading,
    OutputFormat,
    Renderer,
    mo,
    render_for_prompt,
):
    # ── Helper utilities ─────────────────────────────────────────────────────────
    _show_tokenizer = GPTTokenizer()

    def show_doc(doc, title=None, max_chars=4000):
        """Render a document with render_for_prompt → Markdown and display it."""
        renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        md = renderer.render(doc)
        token_count = _show_tokenizer.count(md)
        if len(md) > max_chars:
            md = md[:int(max_chars * 2 / 3)] + "\n\n…\n\n" + md[-int(max_chars * 1 / 3):]
        label = f"#### {title}\n\n" if title else ""
        header = label + f"*{token_count} tokens (render_for_prompt)*\n\n---\n\n"
        mo.output.append(mo.md(header + md))

    def doc_structure(doc):
        """Print a compact structural summary of all elements."""
        title_str = f"title={doc.title!r}  " if doc.title else ""
        print(f"Document  {title_str}parser={doc.parser!r}  elements={len(doc.elements)}")
        for _i, el in enumerate(doc.elements):
            name = type(el).__name__
            if isinstance(el, Heading):
                label = f"h{el.level}"
            elif isinstance(el, Footnote):
                label = f"fn#{el.number}"
            else:
                label = name[:3].lower()
            text = el.text[:72].replace("\n", " ")
            print(f"  [{_i:2d}] {name:<14}  ({label:<6}) | {text}")
    print("Helpers defined.")
    return doc_structure, show_doc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stage 1 — Parsing

    The source document is the BERT paper (arXiv:1810.04805) processed by MinerU:

    - **JSON input:** [`bert-paper_middle.json`](../../tests/parsing/data/mineru/bert-paper_middle.json)
    - **Original PDF:** [`bert-paper_origin.pdf`](../../tests/parsing/data/mineru/bert-paper_origin.pdf)

    The `_middle.json` is MinerU's intermediate representation: a list of pages, each containing
    typed blocks (`title`, `text`, `table`, `figure`, …) with bounding boxes and line-level layout data.

    `MinerUParser` runs a chain of extraction stages (`CoreExtractor`) that:

    1. Reads each block type and creates the corresponding `BaseElement` subclass.
    2. Encodes visual properties — font size, text alignment — as **inline CSS** on the element's `html`.
       This CSS-in-HTML convention is what later processors (`LLMHeadingResolver`) depend on.
    3. Returns a flat `Document` with all heading-type blocks tagged `h3` (MinerU's default) and `parser="mineru"` set.

    > **Note on images:** MinerU's `_middle.json` does not expose image blocks as structured elements.
    > Images are detected during PDF rendering but their content is not available in the intermediate format.
    > `ImageSummaryProcessor` is therefore not used in this pipeline.
    """)
    return


@app.cell
async def _(MinerUMiddleDocument, MinerUParser, mo):
    DATA_FILE_1 = (mo.notebook_dir() / "../../tests/parsing/data/mineru/bert-paper_middle.json").resolve()
    source = MinerUMiddleDocument.from_json_path(DATA_FILE_1)
    doc_parsed = await MinerUParser()(DATA_FILE_1)
    print(f"Parsed  parser={doc_parsed.parser!r}  elements={len(doc_parsed.elements)}")
    print(f"Pages in source JSON: {len(source.pdf_info)}")
    print(f"Tables found: {len(doc_parsed.tables)}")
    return DATA_FILE_1, doc_parsed


@app.cell
def _(doc_parsed, doc_structure):
    doc_structure(doc_parsed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observation:** Headings are tagged `h3` — the flat level MinerU assigns to all title-type blocks.
    Each `Heading` carries CSS `font-size` (derived from the average line height in the bounding box)
    and possibly `text-align: center` when the block is horizontally centred on the page.
    This is the raw material `LLMHeadingResolver` uses in Stage 2.
    """)
    return


@app.cell
def _(doc_parsed, show_doc):
    show_doc(doc_parsed, "Stage 1 — raw MinerU output (render_for_prompt → Markdown)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stage 2 — Processing

    Processing enriches the `Document` in place — no rendered strings leave this stage.
    Four processors run in sequence:

    | Processor | Type | What it does |
    |---|---|---|
    | `LLMHeadingResolver` | Async (LLM) | Sends heading metadata to an LLM; maps to `h1`/`h2`/`h3`; sets `document.title` |
    | `TitleDetectionProcessor` | Sync | Removes preamble elements that appear before the detected title |
    | `FootnoteProcessor` | Async | Inserts `<ref>` tags linking footnote citation numbers to their definitions |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2a — LLMHeadingResolver + TitleDetectionProcessor

    `LLMHeadingResolver` sends all heading metadata (font size, centering, page number, text) to an LLM
    using structured output. The LLM assigns a semantic heading level to each heading and identifies
    the document title. With `remove_title_from_elements=True` (default), the title heading is removed
    from `elements` and written to `document.title`.

    We pass the shared `llm_client` built from `LLMHeadingResolverSettings` in the setup cell.
    That client already has the correct base URL, API key, and custom httpx transport baked in.

    `TitleDetectionProcessor` runs afterwards with `remove_elements_before_title=True`.
    This strips any preamble elements that appear before the title on the first pages
    (cover page boilerplate, distribution codes, date headers, etc.).
    """)
    return


@app.cell
async def _(
    LLMHeadingResolver,
    ProcessingPipeline,
    TitleDetectionProcessor,
    doc_parsed,
    llm_client,
):
    doc_headings = await ProcessingPipeline([
        LLMHeadingResolver(client=llm_client),
        TitleDetectionProcessor(remove_elements_before_title=True),
    ]).process(doc_parsed)

    print(f"document.title = {doc_headings.title!r}\n")
    print("Heading levels after LLMHeadingResolver:")
    for h in doc_headings.headings:
        css_hint = "(centred)" if "center" in h.html else ""
        print(f"  h{h.level}  {css_hint:<10}  {h.text[:70]}")
    return (doc_headings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observation:** The LLM has assigned semantic levels to the headings.
    `TitleDetectionProcessor` then confirmed the title and removed any cover-page preamble
    that preceded it, so `elements` now starts with the first substantive content.
    """)
    return


@app.cell
def _(doc_headings, show_doc):
    show_doc(doc_headings, "After LLMHeadingResolver + TitleDetectionProcessor")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2b — FootnoteProcessor

    `FootnoteProcessor` (backed by `SimpleFootnoteResolver`) links inline footnote citation
    numbers in paragraph text to their `Footnote` definition elements.

    For each `Footnote` element in the document it:
    1. Searches paragraph text for occurrences of the footnote number using a heuristic scorer.
    2. Scores candidates by structural signals (end-of-clause punctuation, superscript position, page proximity).
    3. Inserts `<ref id="…" rel="footnote"/>` at the best-match position in the element's HTML.

    The `Renderer` resolves these `<ref>` tags inline when generating prompt content,
    so footnotes appear at the point of citation rather than as a separate appendix.
    """)
    return


@app.cell
async def _(FootnoteProcessor, ProcessingPipeline, doc_headings):
    doc_footnotes = await ProcessingPipeline([
        FootnoteProcessor(),
    ]).process(doc_headings)

    print(f"Footnotes in document: {len(doc_footnotes.footnotes)}\n")
    print("Resolved footnote references:")
    for el in doc_footnotes.elements:
        fn_ids = el.footnote_ids
        if fn_ids:
            for fn_id in fn_ids:
                fn_el = doc_footnotes.get_element(fn_id)
                print(f"  {type(el).__name__:<12} | …{el.text[-50:]}")
                if fn_el is not None:
                    print(f"               → fn {fn_el.number}: {fn_el.text[:70]}")
    return (doc_footnotes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Enriched document

    After processing, the document has semantic heading levels, a detected title, and
    footnote references linked inline. The enriched document is ready for splitting.
    """)
    return


@app.cell
def _(doc_footnotes):
    # Inspect the first table in the enriched document
    if doc_footnotes.tables:
        _tbl = doc_footnotes.tables[0]
        print(f"Table id:      {_tbl.id}")
        print(f"Table page:    {_tbl.page}")
        print()
        print("HTML (first 600 chars):")
        print(_tbl.html[:600])
    else:
        print("No tables found in document.")
    doc_enriched = doc_footnotes  # carry forward as doc_enriched for subsequent cells
    return (doc_enriched,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observation:** Tables are rendered in full by `render_for_prompt` — the LLM context window
    gets every cell. With `SimpleChunker`, `embedding_content` equals `prompt_content`.
    `LLMChunker` can produce distinct `embedding_content` via LLM-generated topic summaries.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stage 3 — Splitting

    `split_document` breaks a large `Document` into token-bounded sub-`Document`s.
    The output is **still a list of `Document` objects**, not strings — processors can still run on them.

    **Three-tier strategy (applied in order):**
    1. Fits within budget → return as-is.
    2. Hierarchical split by heading level → recurse on each part.
    3. Element-level greedy split → accumulate element groups until budget is reached.

    The `renderer` argument is used **only for token counting** during splitting — no text is persisted yet.
    Referenced elements (footnotes, images) always travel with their root element as an atomic group.
    Each split gets the most-recently-seen heading at each level prepended as context, so every chunk is self-contained.
    """)
    return


@app.cell
def _(
    GPTTokenizer,
    OutputFormat,
    Renderer,
    doc_enriched,
    render_for_prompt,
    split_document,
):
    prompt_renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    tokenizer = GPTTokenizer()
    total_tokens = tokenizer.count(prompt_renderer.render(doc_enriched))
    print(f"Document token count (prompt): {total_tokens} tokens")
    MAX_TOKENS = 5000
    OVERLAP_TOKENS = 100
    splits = split_document(doc_enriched, renderer=prompt_renderer, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    print(f"\nSplit into {len(splits)} sub-documents  (max_tokens={MAX_TOKENS}, overlap={OVERLAP_TOKENS})\n")
    print(f"{'#':>3}  {'tokens':>6}  {'elements':>8}  Heading path")
    print("-" * 75)
    for _i, _split in enumerate(splits):
        tokens = tokenizer.count(prompt_renderer.render(_split))
        _heading_path = " > ".join(h.text[:30] for h in _split.headings)
        print(f"{_i + 1:>3}  {tokens:>6}  {len(_split.elements):>8}  {_heading_path[:60]}")
    return prompt_renderer, splits, tokenizer


@app.cell
def _(show_doc, splits):
    # Render a few splits to see their content
    for _i, _split in list(enumerate(splits))[5:8]:
        show_doc(_split, f"Split {_i + 1} / {len(splits)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations:**
    - Each split carries its heading context — every chunk is self-contained for retrieval.
    - Splits are still `Document` objects, so processors can run post-split on focused sub-documents.
    - `parent_ids` on each split links back to the original document for hierarchical retrieval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stage 4 — Chunking

    `Chunk` is the final materialisation point. Each split document is rendered into `prompt_content`
    using `render_for_prompt`. With `SimpleChunker`, `embedding_content` equals `prompt_content`.
    For distinct embedding content, use `LLMChunker` which generates topic summaries via LLM.

    | Field | Source | Format | Used for |
    |---|---|---|---|
    | `prompt_content` | `render_for_prompt` | MARKDOWN | LLM context window, BM25 keyword search |
    | `embedding_content` | `prompt_content` (SimpleChunker) or LLM topics (LLMChunker) | MARKDOWN | Dense vector embedding |

    **Rule:** Keep everything in `Document` form until this point.
    """)
    return


@app.cell
def _(
    Chunk,
    DATA_FILE_1,
    OutputFormat,
    Renderer,
    doc_enriched,
    hashlib,
    prompt_renderer,
    splits,
    tokenizer,
):
    _source_hash = hashlib.sha256(DATA_FILE_1.read_bytes()).hexdigest()
    chunks: list[Chunk] = [Chunk(prompt_content=prompt_renderer.render(_split), embedding_content=prompt_renderer.render(_split), source_id=DATA_FILE_1.name, source_hash=_source_hash, metadata={"filename": DATA_FILE_1.stem, "document_title": doc_enriched.title, "heading_path": " > ".join(h.text for h in _split.headings), "n_elements": len(_split.elements)}) for _split in splits]
    print(f"Created {len(chunks)} chunks.\n")
    print(f"{'ID':10}  {'prompt tok':>10}  {'embed tok':>9}  Heading path")
    print("-" * 80)
    for chunk in chunks:
        pt = tokenizer.count(chunk.prompt_content)
        et = tokenizer.count(chunk.embedding_content)
        print(f"{chunk.id[:8]}…  {pt:>10}  {et:>9}  {chunk.metadata['heading_path']}")
    return (chunks,)


@app.cell
def _(Table, chunks, splits):
    # Find a chunk that contains a table and compare both representations
    table_chunk = next((c for c, s in zip(chunks, splits, strict=True) if any(isinstance(el, Table) for el in s.elements)), chunks[0])
    print("=" * 70)
    print("PROMPT CONTENT  (full-fidelity Markdown — full table preserved)")
    print("=" * 70)
    print(table_chunk.prompt_content[:1000])
    print()
    print("=" * 70)
    print("EMBEDDING CONTENT  (same as prompt_content with SimpleChunker)")
    print("=" * 70)
    print(table_chunk.embedding_content[:1000])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Stage | Class / Function | What it produced |
    |---|---|---|
    | 1. Parsing | `MinerUParser` | `doc_parsed` — flat `Document` with CSS-styled `h3` headings, `parser="mineru"` |
    | 2a. Headings | `LLMHeadingResolver` + `TitleDetectionProcessor` | `doc_headings` — semantic `h1`/`h2`/`h3` levels; `document.title` set; preamble removed |
    | 2b. Footnotes | `FootnoteProcessor` | `doc_enriched` — inline `<ref>` tags linking citations to definitions |
    | 3. Splitting | `split_document` | Token-bounded `Document` list with shared heading context |
    | 4. Chunking | `Chunk` | `prompt_content` + `embedding_content` ready for a vector store |

    **Key takeaways:**
    - `Document` is the single source of truth until the final `Chunk` materialisation.
    - LLM enrichment (heading levels) writes results into structured fields — never as rendered strings.
    - `render_for_prompt` produces full-fidelity Markdown for LLM context windows.
    - With `SimpleChunker`, `embedding_content` equals `prompt_content`. Use `LLMChunker` for distinct topic-based embedding content.
    - `ImageSummaryProcessor` is not used here because MinerU's `_middle.json` does not expose image content as structured elements.

    ---

    ## See Also

    - [Qdrant Sync Pipeline](qdrant_pipeline.py) — production incremental sync with QdrantVectorStore and LLMChunker
    - [Pipeline](pipeline.py) — `DocumentPipeline`, `VectorStorePipeline`, and `TokenSplitter` API reference
    """)
    return


if __name__ == "__main__":
    app.run()
