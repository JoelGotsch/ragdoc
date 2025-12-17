# Document Merging

> Run interactively: `marimo edit docs/notebooks/merging.py`

When the same document is parsed by two different parsers (e.g. MinerU for layout and Azure DI for text accuracy), the outputs typically complement each other. The merging API combines both into a single, higher-quality `Document`.

---

## Two approaches

| | Approach A | Approach B |
|---|---|---|
| Function | `merge_documents` / `compute_patch` | `merge_documents_html` / `compute_html_patch` |
| Mechanism | Aligns structured elements directly | Render → HTML diff → reparse |
| Inspectable intermediate | `DocumentPatch` (typed operations) | `DocumentHtmlPatch` (HTML strings) |
| Standalone Images | ✓ preserved (base64 + URL) | ✓ preserved (base64 + URL) |
| Inline Image refs (`<ref>`) | ✓ intact | ✓ intact |
| Footnote elements | ✓ preserved | ✓ preserved (via `<aside>`) |
| Footnote inline refs | ✓ intact | ✓ reconstructed after reparse |
| doc_a-unique elements | ✓ preserved | ✓ preserved |
| Heading hierarchy detection | ✓ parser-name trust | ✓ distribution heuristic |
| Page / bounding-box metadata | ✓ preserved | ✗ lost (architectural) |

Use **Approach A** when:
- You need to inspect or override individual merge decisions.
- Page numbers or bounding boxes must be retained.
- You need a JSON-serializable audit trail.

Use **Approach B** when:
- You want a simple drop-in merge with no extra configuration.
- You don't need page or bounding-box metadata.
- You want the inspectable `DocumentHtmlPatch` for lightweight overrides.

---

## Quick start

```python
from ragdoc.merging import merge_documents, merge_documents_html

# Approach A (element-alignment)
merged = merge_documents(doc_mineru, doc_azure_di)

# Approach B (render-merge-reparse)
merged = merge_documents_html(doc_mineru, doc_azure_di)
```

Both functions return a `Document` with `parser="merged"` and `metadata` set:

```python
merged.parser          # "merged"
merged.metadata        # {"source_parser_a": "mineru", "source_parser_b": "azure_di"}
```

---

## Pipeline-level multi-source parsing

`MultiSourceParser` integrates the merging step directly into the parsing stage, so the
rest of the pipeline (processing, splitting, chunking) sees a single `Document` with no
awareness of the merge.

### Basic usage

```python
from pathlib import Path
from ragdoc.parsing import get_parser
from ragdoc.parsing.multi_source import MultiSourceParser

merged_parser = MultiSourceParser(
    primary=get_parser("mineru"),
    secondary=get_parser("azure_di"),
)

document = await merged_parser(Path("report_middle.json"))
```

Both parsers receive the same path. The default merge strategy is `merge_documents`
(Approach A).

### Sibling file resolver

When the two parsers read different files, use `sibling_resolver()` to map the primary
path to the companion file. If the companion does not exist, the primary result is
returned unchanged:

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

# Register so load() dispatches automatically (shadows the built-in mineru
# parser because priority 60 > 50 — see the Parsing guide for details)
register_parser(merged_parser)

document = await load("report_middle.json")
# → merges MinerU output with report_middle.html if it exists on disk
```

!!! tip
    The registered parser shadows the built-in MinerU parser because its priority
    (60) is higher than the default (50). See
    [Shadowing vs. replacing](parsing.md#shadowing-vs-replacing) in the Parsing
    guide for details.

### Customising the merge strategy

Pass a custom merge function via `functools.partial`:

```python
from functools import partial
from ragdoc.document import ElementTypeEnum
from ragdoc.merging import merge_documents
from ragdoc.parsing.multi_source import MultiSourceParser

my_merge = partial(
    merge_documents,
    prefer_source={ElementTypeEnum.TABLE: "b"},
)

merged_parser = MultiSourceParser(
    primary=get_parser("mineru"),
    secondary=get_parser("html"),
    merge=my_merge,
)
```

### Nesting for three-way merges

`MultiSourceParser` itself inherits from `Parser`, so it can be nested:

```python
inner = MultiSourceParser(primary=get_parser("mineru"), secondary=get_parser("html"))
outer = MultiSourceParser(primary=inner, secondary=get_parser("azure_di"))

document = await outer(Path("report_middle.json"))
# MinerU + HTML merged first, then merged again with Azure DI
```

---

## Controlling what gets merged

Both approaches accept the same two control parameters.

### `prefer_source`

Force specific element types to always come from one document:

```python
from ragdoc.document import ElementTypeEnum

merged = merge_documents_html(
    doc_a, doc_b,
    prefer_source={
        ElementTypeEnum.HEADING: "b",   # always take headings from doc_b
        ElementTypeEnum.TABLE: "a",     # always take tables from doc_a
    },
)
```

### `allow_insertions_from_b`

Control whether elements that appear only in `doc_b` are included:

```python
# Include everything from doc_b (default)
merged = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=True)

# Suppress all doc_b-only elements
merged = merge_documents_html(doc_a, doc_b, allow_insertions_from_b=False)

# Allow only specific types from doc_b
merged = merge_documents_html(
    doc_a, doc_b,
    allow_insertions_from_b=frozenset({ElementTypeEnum.FOOTNOTE, ElementTypeEnum.IMAGE}),
)
```

---

## Approach A — Element-alignment

### `compute_patch` and `DocumentPatch`

`compute_patch()` returns a `DocumentPatch` whose operations you can inspect and override before applying:

```python
from ragdoc.merging import compute_patch, PatchOperationType

patch = compute_patch(doc_mineru, doc_html)

# Inspect decisions
for op in patch.operations:
    print(op.op, op.reason)
    # PatchOperationType.MERGE  "richer markup from doc_b"
    # PatchOperationType.KEEP_A "identical text"
    # PatchOperationType.DELETE_A "no counterpart in doc_b"
    # ...

# Override a specific decision before applying
from ragdoc.document import Paragraph
patch.operations[2].resolved_elements = [Paragraph(html="<p>Custom</p>")]

merged = patch.apply()
```

`DocumentPatch` is JSON-serializable:

```python
json_str = patch.model_dump_json()
```

Source parser information is available on the patch:

```python
patch.source_parser_a  # "mineru"
patch.source_parser_b  # "html"
```

### Heading level selection (Approach A)

Approach A uses a `trust_parsers` set (default: `{"html", "pandoc"}`) to decide whose heading levels to trust. If only one parser is trusted, its heading levels win:

```python
# mineru (flat h1s) + html (h1/h2/h3) → html's levels win
merged = merge_documents(doc_mineru, doc_html)
```

---

## Approach B — Render-merge-reparse

### `compute_html_patch` and `DocumentHtmlPatch`

`compute_html_patch()` returns a `DocumentHtmlPatch` with HTML-string operations you can inspect and override:

```python
from ragdoc.merging import compute_html_patch

patch = compute_html_patch(doc_mineru, doc_html)

# Inspect
for op in patch.operations:
    print(op.opcode, op.selected)
    # "equal"   ['<h2>Introduction</h2>']
    # "replace" ['<p><em>enriched text</em></p>']
    # "insert"  ['<aside class="footnote" ...>...</aside>']
    # "delete"  ['<p>Only in doc_a</p>']

# Override before applying
patch.operations[0].selected = ["<h1>Custom Title</h1>"]

merged = patch.apply()
```

`DocumentHtmlPatch` is JSON-serializable:

```python
json_str = patch.model_dump_json()
```

`merge_documents_html` is a one-liner equivalent to `compute_html_patch(...).apply()`:

```python
merged = merge_documents_html(doc_a, doc_b)
# same as:
merged = compute_html_patch(doc_a, doc_b).apply()
```

### Heading level selection (Approach B)

Approach B uses a **distribution heuristic** — no parser names required:

- A parser that outputs all headings at the same level did not detect hierarchy.
- A parser with multiple distinct levels did.
- When exactly one document has multi-level hierarchy, its heading levels win.
- The richer content (more inline markup) is preserved regardless of which side's level wins.

```python
# doc_a: all h1 (flat, no hierarchy)
# doc_b: h1/h2/h3 (detected hierarchy)
# → merged uses doc_b's levels
merged = merge_documents_html(doc_a, doc_b)
```

---

## Trade-offs and limitations

### Page and bounding-box metadata

Approach B loses page numbers, bounding boxes, and element-level metadata — `render_raw` emits only visual HTML. If you need this metadata, use Approach A.

### Images

Both approaches handle `Image` elements with base64 data or HTTP URLs:

```python
# base64 round-trips correctly through Approach B
img = Image(image=base64_data, image_type="png", alt="Chart")
```

`Image(image=None)` — where no binary data is available — cannot survive the render-reparse pipeline and is silently absent from the Approach B result. This is expected and documented behaviour; use Approach A to preserve such elements.

### Footnote inline references

Approach B renders inline footnote references as `<a href="#footnote-{uuid}">` during the HTML phase, then reconstructs the `<ref>` tags during reparse. The reconstruction is UUID-based and unambiguous even if footnote numbers repeat across sections.

---

## Validating the merged document

```python
from ragdoc.merging import validate_inline_refs

broken = validate_inline_refs(merged)
if broken:
    print("Broken inline refs:", broken)
else:
    print("All inline refs resolved")
```

`validate_inline_refs` returns a list of target IDs for which no matching element exists in the document.
