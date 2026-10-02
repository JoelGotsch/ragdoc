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
the `openai_client` of the active [`RagdocConfig`](../api/config.md#ragdocconfig)
(set with `configure(...)`), else construction fails loudly with `LLMNotConfiguredError`.
`RagdocConfig.openai_client` accepts any client with a `chat` and/or an `embeddings`
attribute (`ChatClient | EmbeddingsClient`; an `AsyncOpenAI` has both). Processors need the
chat capability, so an embeddings-only client there also raises `LLMNotConfiguredError`.
`configure()` is a context manager — the config is active inside the block:

```python
from openai import AsyncOpenAI
from ragdoc import RagdocConfig, configure
from ragdoc.processing import DocumentSummarizerProcessor

async with configure(RagdocConfig(openai_client=AsyncOpenAI(), default_llm_model="gpt-4.1")):
    summarizer = DocumentSummarizerProcessor()   # resolves the client from the active config
    document = await summarizer.process(document)
```

Every call goes through the shared reliability layer (`ragdoc.llm`): one retry policy
(429/connection/timeout/5xx with jittered exponential backoff) and structured parsing with
explicit refusal errors. Three things worth knowing:

- **`Retry-After` is capped.** A `Retry-After` header on a 429 replaces the computed delay,
  but never beyond `backoff_max` (30 s by default) — a server asking for an hour does not
  park the pipeline for an hour.
- **One retry layer.** For a real `AsyncOpenAI` / `AsyncAzureOpenAI` client the library calls
  through `client.with_options(max_retries=0)`, so the SDK's own retries do not stack under
  ragdoc's: a call makes exactly `max_retries + 1` HTTP attempts, and the `max_retries` you
  set on the client itself is ignored for ragdoc calls. A custom client object is called as
  is.
- **Timeouts.** Each LLM processor has a per-request timeout knob (`request_timeout`, in
  seconds: `LLMHeadingResolverSettings`, `DocumentSummarizerSettings`,
  `openai_image_summarizer(...)`, `LLMFootnoteResolver(...)`). The default `None` sends no
  per-request timeout, so the client's own timeout applies.

Degrade semantics are per-processor by design: the heading resolver returns no changes on
failure, the image summarizer skips the failing image with a warning (any LLM-call failure —
a refusal, a 400 for one bad image, exhausted retries; programming errors and cancellation
still propagate) and continues with the next image, the footnote resolver leaves the footnote
unresolved, and the summarizer raises.

`LLMHeadingResolver` resolves heading levels even when the document already has a title; in
that case the model's document-title judgment is ignored and the two title-removal options
(`remove_title_from_elements`, `remove_elements_before_title`) do nothing.
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
