# Processing

Processors transform [`Document`](document-model.md) objects between parsing and
splitting (and again after splitting, for split-local enrichment). Each one is a small,
universal operation — because elements carry their visual properties as inline CSS, the
same processor works on output from every parser.

## The contract

Every processor subclasses [`DocumentProcessor`](../api/processing.md#documentprocessor)
and implements one method:

```python
class MyProcessor(DocumentProcessor):
    async def process(self, document: Document) -> Document | None:
        ...
```

- **Async-only** — pure-sync processors simply don't `await` anything.
- **Return `None` to drop the document.**
  [`ProcessingPipeline`](../api/processing.md#processingpipeline) short-circuits on
  `None` (later processors never run), `IngestPipeline.run()` returns `None`, and
  `DocumentPipeline.run()` returns `[]` chunks for the filtered file. The sync pipelines
  treat a filtered source as "yields nothing" — its stale stored entries are deleted.
- **Extend metadata, never replace it.** Write individual keys
  (`document.metadata["x"] = ...`); assigning a new dict silently discards `filename`
  and everything earlier processors wrote. Library-written metadata keys are for
  external consumers only — no processor may read or branch on them.

Chain processors with `ProcessingPipeline`, or just pass a list to the pipeline —
`IngestPipeline(processors=[...])` accepts either:

```python
from ragdoc.pipeline import DocumentPipeline, TokenSplitter
from ragdoc.processing import HeadingLevelProcessor, TitleDetectionProcessor

pipeline = DocumentPipeline(
    processors=[HeadingLevelProcessor(), TitleDetectionProcessor()],
    splitter=TokenSplitter(max_tokens=4000),
)
```

## Processor catalog

### Structure (no LLM)

| Processor | What it does |
|---|---|
| [`HeadingLevelProcessor`](../api/processing.md#headinglevelprocessor) | Re-derives heading levels from visual properties (font size, bold, centered, all-caps) via a pluggable `SizeToLevelMapper` |
| [`TitleDetectionProcessor`](../api/processing.md#titledetectionprocessor) | Promotes the most title-like heading to `document.title` |
| [`FootnoteProcessor`](../api/processing.md#footnoteprocessor) | Detects footnote markers and links them to their definitions via a pluggable [`FootnoteResolver`](../api/processing.md#footnoteresolver) ([`SimpleFootnoteResolver`](../api/processing.md#simplefootnoteresolver) is heuristic; [`LLMFootnoteResolver`](../api/processing.md#llmfootnoteresolver) breaks ties with an LLM) |
| [`EmptyDocumentFilter`](../api/processing.md#emptydocumentfilter) | Returns `None` for documents with no meaningful content — the canonical drop-a-document example |
| [`DocumentDumpProcessor`](../api/processing.md#documentdumpprocessor) | Side-effect processor: serializes each passing document to disk (debugging / audit trails), returns it unchanged |

### LLM-based enrichment

| Processor | What it does |
|---|---|
| [`LLMHeadingResolver`](../api/processing.md#llmheadingresolver) | Sends heading candidates to an LLM for semantic level assignment (`document-title` / `h1`–`h6` / `none`) — for documents whose visual styling is inconsistent |
| [`ImageSummaryProcessor`](../api/processing.md#imagesummaryprocessor) | Writes an LLM description into each `Image.text_representation`, which `render_for_prompt` uses as the text fallback; construct with `ImageSummaryProcessor(summarize=openai_image_summarizer(client))` |
| [`DocumentSummarizerProcessor`](../api/processing.md#documentsummarizerprocessor) | Writes a whole-document LLM summary to `document.metadata["summary"]` (recursive fold for documents beyond the token budget); idempotent unless `overwrite=True` |

LLM processors need the `llm` extra (`pip install 'ragdoc[llm]'`).

## Ordering

Order matters — later processors see the output of earlier ones. The natural sequence:

1. **Heading structure first** (`HeadingLevelProcessor`, then `LLMHeadingResolver` if
   needed) — splitting and title detection both depend on correct levels.
2. **`TitleDetectionProcessor`** — after levels are trustworthy.
3. **`FootnoteProcessor`** — once structure is stable.
4. **Filters last** (`EmptyDocumentFilter`) — drop documents only after enrichment had
   its chance; anything after a filter never runs for dropped documents.

## Pre-split vs post-split placement

Processors passed to [`IngestPipeline`](../api/pipeline.md#ingestpipeline) run **once per
document, before splitting** — right for whole-document concerns (headings, title,
footnotes, filtering).

Split-local enrichment (e.g. summarizing each retrieval unit rather than the whole file)
runs **after splitting**: splits are still `Document` objects, so the same processor
classes work. Run the stages yourself when you need this:

```python
from ragdoc.splitting import split_document

doc = await ingest.run(path)                     # parse + process (pre-split)
splits = split_document(doc, renderer, tokenizer)

summarizer = DocumentSummarizerProcessor()       # writes metadata["summary"] per split
# (requires a configured LLM client — raises LLMNotConfiguredError at construction otherwise)
enriched = [await summarizer.process(s) for s in splits]

chunker = SimpleChunker()                        # then chunk each enriched split
chunks = [c for s in enriched if s is not None for c in await chunker.chunk(s)]
```

## Configuring the LLM

All LLM processors resolve their client through one chain (see the
[LLM API](../api/llm.md)): an explicit `client=` argument wins, else
`configure(openai_client=...)`'s global [`RagdocConfig`](../api/config.md#ragdocconfig),
else construction fails loudly with `LLMNotConfiguredError`:

```python
from openai import AsyncOpenAI
from ragdoc import configure

configure(openai_client=AsyncOpenAI(), default_llm_model="gpt-4.1")
```

Every call goes through the shared reliability layer (`ragdoc.llm`): one retry policy
(429/connection/timeout/5xx with backoff, honoring `Retry-After`) and structured parsing
with explicit refusal errors. Degrade semantics are per-processor by design: the heading
resolver returns no changes on failure, the image summarizer skips the image with a
warning, the footnote resolver leaves the footnote unresolved, and the summarizer raises.
`LLMHeadingResolver` additionally supports env-based client configuration via
[`LLMHeadingResolverSettings`](../api/processing.md#llmheadingresolversettings)
(`LLM_HEADING_RESOLVER_API_KEY`, base URL, model).

## Writing your own processor

```python
from ragdoc.document import Document
from ragdoc.processing import DocumentProcessor

class DepartmentTagger(DocumentProcessor):
    """Tags each document with the department that owns its source folder."""

    async def process(self, document: Document) -> Document | None:
        if document.source_path is not None:
            document.metadata["department"] = Path(document.source_path).parent.name
        return document
```

Conventions (see [Contributing](../development/contributing.md)):

- Name it `*Processor`; LLM-based async helpers it delegates to are `*Resolver`s.
- Extract reusable logic into standalone functions so it can be tested without the class.
- Mutating the passed document is fine; returning a new one is too. Return `None` only
  to drop it.

Note that **extraction is not processing**: extractors return typed
[`Mention`](../api/extraction.md#mention) objects instead of writing to documents — see
the [Extraction Guide](extraction.md).

## See Also

- [Processing API reference](../api/processing.md)
- [Pipeline Guide](pipeline.md) — where processors sit in the end-to-end flow
- [Document Model Guide](document-model.md) — the elements processors operate on
