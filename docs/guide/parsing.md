# Parsing

> Run interactively: `marimo edit docs/notebooks/parsing.py`

Parsing is the first stage of the pipeline. It converts a raw file into a structured
[`Document`](document-model.md) object. Each parser sets `document.parser` to a provenance
string so downstream processors can adjust behavior.

## Quick start

[`load()`](../api/parsing.md#load) is the primary entry point. It infers the right parser
from the file extension automatically:

```python
from ragdoc.parsing import load

document = await load("report.docx")
```

Pass `parser=` to force a specific parser by name:

```python
document = await load("result.json", parser="azure_di")
```

## Built-in parser registry

ragdoc registers parsers for common file types at import time. Resolution is based on
suffix matching — longer patterns take precedence, and within the same pattern the
highest-priority parser wins:

| Pattern | Parser | Name | Priority | Extra |
|---------|--------|------|----------|-------|
| `.html`, `.htm` | HTML | `html` | — | — |
| `.docx`, `.doc` | Pandoc | `pandoc` | — | — |
| `.xlsx` | Excel | `xlsx` | — | `xlsx` |
| `.azure.json` | Azure DI (JSON) | `azure_json` | — | — |
| `.pdf` | pymupdf (local) | `pdf_basic` | 10 | `pdf` |
| `.pdf` | Azure DI (live) | `azure_di` | 40 | `azure-di` + credentials |
| `_middle.json` | MinerU | `mineru` | 50 | `pdf-mineru` |
| `.ragdoc.json` | RagdocJson | `ragdoc_json` | 60 | — |

Resolution also checks **availability**: a parser whose extra isn't installed (or whose
credentials aren't configured, for `azure_di`) is skipped in favour of the next-best
available one, and if *nothing* usable matches you get an actionable error at resolve
time telling you which extra to install.

### PDF options

| Extra | Parser | Fidelity | Setup |
|---|---|---|---|
| `pdf` | `pdf_basic` | Text + font-size headings (pymupdf) | zero config |
| `azure-di` | `azure_di` | Full layout, tables, figures | Azure credentials |
| `pdf-mineru` | `mineru` | Full layout via MinerU `_middle.json` | run MinerU separately |

With `ragdoc[pdf]` installed, `load("report.pdf")` just works; configure Azure DI (or run
MinerU) when you need higher fidelity — the registry then prefers those automatically.

## Available parsers

### Word documents (`.docx`, `.doc`)

Uses [Pandoc](https://pandoc.org/) to convert Word documents to HTML, then parses the
HTML into a `Document`.

```python
document = await load("report.docx")
# document.parser == "pandoc"
```

### HTML files (`.html`)

Parses HTML files directly into a `Document`, preserving heading structure, paragraphs,
tables, and images.

```python
document = await load("page.html")
# document.parser == "html"
```

### Excel files (`.xlsx`)

Converts Excel workbooks into a `Document`. Each sheet becomes a section; tables are
extracted as `Table` elements.

```python
document = await load("data.xlsx")
# document.parser == "xlsx"
```

### Basic local PDF (`.pdf`)

`pdf_basic` (the `pdf` extra) parses PDFs locally with pymupdf: text extraction plus
font-size/bold heading detection. Zero configuration, no network.

```python
document = await load("report.pdf")
# document.parser == "pdf_basic"   (when no higher-fidelity PDF parser is available)
```

### Azure Document Intelligence (`.azure.json`, `.pdf`)

Parses the JSON output from Azure Document Intelligence into a `Document`. This is the
recommended parser for complex PDFs.

There are two registered parsers: `azure_json` for static `.azure.json` result files, and
`azure_di` for live PDF analysis via the Azure API.

```python
document = await load("result.azure.json")
# document.parser == "azure_di"  (parsed by the "azure_json" registry entry)

document = await load("report.pdf")   # invokes Azure DI (when credentials are configured)
# document.parser == "azure_di"       (parsed by the "azure_di" registry entry)
```

### MinerU (`_middle.json`)

Parses MinerU layout output (the `_middle.json` file produced by
[MinerU PDF extraction](https://github.com/opendatalab/MinerU)):

```python
document = await load("report_middle.json")
# document.parser == "mineru"
```

Requires the `pdf-mineru` extra: `uv add ragdoc[pdf-mineru]`.

#### What the parser extracts

`CoreExtractor` (the default) converts every MinerU block into document
elements:

| MinerU block | Produces | Notes |
|---|---|---|
| `TitleBlock` | `Heading` | CSS `font-size` + optional `text-align: center` |
| `TextBlock` | `Paragraph` | CSS `font-size` + optional `text-align: center` |
| `ListBlock` | `DocumentList` | CSS applied to `<ul>` tag |
| `CodeBlock` | `Paragraph?` + `RawText` | caption before body |
| `TableBlock` | `Paragraph?` + `Table` + `RawText?` | caption, HTML table, footnote |
| `ImageBlock` | `Image?` + `Paragraph?` | Pillow load from disk + caption |

Discarded blocks (headers, footers, page numbers, asides, footnotes) are handled
by the discarded-handler table — see [Default discarded-block handling](#default-discarded-block-handling).

#### Image loading

When the source file path is supplied (the default when using `load()` or `MinerUParser`),
the parser sets an internal `source_dir` so image handlers can resolve relative paths.
Pillow is used to load each image: `width`, `height`, and base64-encoded bytes are stored
directly on the `Image` element — the same representation used by the HTML and Azure DI
parsers.

```python
document = await load("output/report_middle.json")
# Images in output/images/*.png are loaded automatically
```

If Pillow is not installed or an image file is not found, the `Image` element is silently
skipped and a warning is recorded on the document context (not raised as an exception).

#### Metadata from discarded blocks

Headers and other discarded-block text that gets stored by `handle_discarded_as_metadata`
is automatically propagated to `Document.metadata` after parsing:

```python
document = await load("report_middle.json")
print(document.metadata.get("headers"))
# [{"page": 1, "text": "Confidential — Draft v1"}, ...]
```

The accumulated keys depend on which discarded handlers are active (see the default table
below). Internal context keys are never forwarded.

#### Default discarded-block handling

| Block type | Default handler | Effect |
|---|---|---|
| `HEADER` | `handle_discarded_as_metadata` | stored in `document.metadata["headers"]` |
| `FOOTER` | `handle_discarded_as_footnote` | numbered regex → `Footnote`, else warn + drop |
| `PAGE_NUMBER` | `handle_discarded_drop` | dropped silently |
| `ASIDE_TEXT` | `handle_discarded_drop` | dropped silently |
| `PAGE_FOOTNOTE` | `handle_discarded_as_footnote` | numbered regex → `Footnote`, else warn + drop |

#### ExtractionConfig — swapping handlers

Every handler is a plain callable in `ExtractionConfig`. Replace any single entry to
change that block type's behaviour without touching anything else:

```python
from ragdoc.parsing.mineru import MinerUParser, MinerUExtractor, CoreExtractor
from ragdoc.parsing.mineru.handlers import ExtractionConfig, handle_discarded_as_raw_text
from ragdoc.parsing.mineru.base import DiscardedBlockType
from pathlib import Path

# Emit page headers as RawText elements instead of storing them in metadata
config = ExtractionConfig()
config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text

extractor = MinerUExtractor(use_default_stages=False)
extractor.use(CoreExtractor(config=config))

parser = MinerUParser(extractor=extractor)
document = await parser(Path("report_middle.json"))
```

Returning `[]` from a handler is the canonical way to drop a block entirely:

```python
config = ExtractionConfig()
config.handle_image = lambda block, page, ctx: []   # drop all images
config.handle_list  = lambda block, page, ctx: []   # drop all lists
```

#### Available handler functions

Four reusable discarded-block handlers are provided:

| Function | Produces | Use case |
|---|---|---|
| `handle_discarded_as_metadata` | `[]` | accumulate text in `document.metadata` |
| `handle_discarded_as_footnote` | `Footnote` or `[]` | numbered `N text` pattern |
| `handle_discarded_as_raw_text` | `RawText` | preserve text as a document element |
| `handle_discarded_drop` | `[]` | discard silently |

All four are importable from `ragdoc.parsing.mineru`.

#### CSS on output elements

`Heading`, `Paragraph`, and `DocumentList` elements produced by the MinerU parser carry
inline CSS derived from the block's visual geometry:

- **`font-size`** — estimated from the average line height of the block (in points)
- **`text-align: center`** — added when the block is horizontally centred within the page (within a 10 % tolerance)

These CSS properties allow processors like `HeadingLevelProcessor` to refine heading
levels based on font size — the same mechanism used by the HTML and Azure DI parsers.

### ragdoc JSON snapshots (`.ragdoc.json`)

Deserializes `.ragdoc.json` files produced by
[`DocumentDumpProcessor`](../api/processing.md#documentdumpprocessor) back into
`Document` objects. Registered at priority 60 — the highest of any built-in parser —
so `.ragdoc.json` files always resolve here, even if another parser also matches.

```python
document = await load("snapshot.ragdoc.json")
# document.parser == "azure_di"   (original provenance preserved by default)
```

By default the original `filename`, `source_path`, and `parser` values from the JSON
are preserved (`ProvenanceMode.ORIGINAL`). Use `ProvenanceMode.JSON_FILE` to overwrite
them with the snapshot file's own identity instead:

| Mode | `filename` | `source_path` | `parser` |
|------|-----------|--------------|---------|
| `ORIGINAL` (default) | from JSON | from JSON | from JSON |
| `JSON_FILE` | snapshot filename | snapshot path | `"ragdoc_json"` |

```python
from ragdoc.parsing.ragdoc_json import RagdocJsonParser, ProvenanceMode
from pathlib import Path

parser = RagdocJsonParser(provenance_mode=ProvenanceMode.JSON_FILE)
document = await parser(Path("snapshot.ragdoc.json"))
# document.parser == "ragdoc_json"
```

To produce `.ragdoc.json` files, add `DocumentDumpProcessor` to your pipeline:

```python
from ragdoc.processing import DocumentDumpProcessor
from pathlib import Path

proc = DocumentDumpProcessor(output_dir=Path("cache/"))
document = await proc.process(document)  # writes cache/{stem}_{id[:8]}.ragdoc.json
```

---

## Parser registry

### Registering a custom parser

Subclass [`Parser`](../api/parsing.md#parser) and register an instance:

```python
from pathlib import Path
from ragdoc.document import Document
from ragdoc.parsing import load, register_parser
from ragdoc.parsing.parser import Parser


class MyParser(Parser):
    name: str = "my_parser"
    patterns: list[str] = [".myformat"]
    priority: int = 10
    description: str = "My custom format"

    async def __call__(self, path: Path) -> Document:
        # parse a custom format
        return Document(...)


register_parser(MyParser())

document = await load("data.myformat")
```

The `priority` controls which parser wins when multiple patterns match the same file.
Higher values win.

### Resolution rules

When `load()` looks up a parser, it checks three sources in order — the first match wins:

1. **Explicit `parser=` argument** — `load("file.json", parser="azure_json")` bypasses
   the registry and looks up by name.
2. **Config preferences** — `parser_preferences` maps a suffix to a parser name,
   overriding priority without changing the registry (see below).
3. **Registry lookup** — all registrations whose pattern matches the filename are
   collected, then sorted by **longest pattern first**, then **highest priority**.

The longest-pattern rule is how `.azure.json` naturally takes precedence over `.json` —
no priority needed. Priority only breaks ties between parsers registered for the
same pattern (or patterns of equal length).

### Shadowing vs. replacing

Registering a new parser for an existing pattern **does not remove** the old parser.
Both stay in the registry; the higher-priority one wins resolution. The shadowed parser
is still reachable via `parser=` or config preferences.

If you want to fully replace a parser, unregister the old one first:

```python
from ragdoc.parsing import register_parser, unregister_parser

# Remove the built-in MinerU parser for _middle.json
unregister_parser("_middle.json", "mineru")

# Register your replacement
register_parser(my_merged_parser)
```

### Inspecting registered parsers

```python
from ragdoc.parsing import describe_registry, get_registered_parsers

# Human-readable summary
print(describe_registry())

# Programmatic access
for reg in get_registered_parsers():
    print(reg.name, reg.pattern, reg.priority)
```

### Overriding via config

Use `RagdocConfig.parser_preferences` to select a specific parser per extension without
changing the registry globally:

```python
from ragdoc.config import configure, RagdocConfig

with configure(RagdocConfig(parser_preferences={".pdf": "my_custom_pdf_parser"})):
    document = await load("report.pdf")   # uses your registered parser, not Azure DI
```

---

## Multi-source parsing

`MultiSourceParser` runs two parsers and merges the results. This is the pipeline-level
integration of the [merging API](merging.md).

### Two parsers, same file

```python
from ragdoc.parsing import get_parser
from ragdoc.parsing.multi_source import MultiSourceParser

merged_parser = MultiSourceParser(
    primary=get_parser("mineru"),
    secondary=get_parser("azure_di"),
)

document = await merged_parser(Path("report_middle.json"))
# Parsed by both MinerU and Azure DI, then merged
```

### Sibling file resolver

More commonly, the secondary parser reads a different file. Use `sibling_resolver()` to
map from the primary path to the sibling path automatically — it returns `None` if the
sibling does not exist, in which case the primary result is returned unchanged:

```python
from ragdoc.parsing import get_parser, register_parser, load
from ragdoc.parsing.multi_source import MultiSourceParser
from ragdoc.parsing.resolvers import sibling_resolver

merged_parser = MultiSourceParser(
    primary=get_parser("mineru"),
    secondary=get_parser("html"),
    secondary_resolver=sibling_resolver(".html"),
    name="mineru_html",
    patterns=["_middle.json"],
    priority=60,
)

# Register it so load() uses it automatically
register_parser(merged_parser)

document = await load("report_middle.json")
# Merges MinerU output with report_middle.html if it exists
```

See [Document Merging](merging.md#pipeline-level-multi-source-parsing) for customizing
the merge strategy and nesting `MultiSourceParser` for three-way merges.

---

## What parsers produce

All parsers produce a `Document` where:

- Elements store content as `html` (full HTML including the outer tag)
- Visual properties (font size, weight, alignment) are stored as **inline CSS** on the HTML tags
- `document.parser` is set to identify the source parser
- Inline references (`<ref id="..." rel="..."/>`) are embedded in HTML for images and footnotes
- `document.metadata["filename"]` is set to the source filename (e.g. `"report.docx"`)
- `document.source_path` is set to the full path as a string (e.g. `"/data/report.docx"`)

The CSS-in-HTML convention is what allows processors like
[`HeadingLevelProcessor`](../api/processing.md#headinglevelprocessor) to work uniformly
across all parsers without parser-specific logic.

---

## See Also

- [API Reference: Parsing](../api/parsing.md)
- [Document Model Guide](document-model.md)
- [Document Merging Guide](merging.md) — multi-source merging strategies
- [Processing API](../api/processing.md) — refine heading levels, detect titles, summarize
