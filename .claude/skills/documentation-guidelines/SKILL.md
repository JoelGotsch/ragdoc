---
name: documentation-guidelines
description: Follow these guidelines when writing, reviewing, or updating any documentation for this project — docstrings, guide files, API reference pages, or interactive notebooks. DOCUMENTATION MUST ALWAYS BE CHECKED AFTER ANY CHANGE TO PUBLIC-FACING CODE.
---

# Documentation Best Practices

Use this skill when writing, reviewing, or updating any documentation for this project —
docstrings, guide files, API reference pages, or interactive notebooks. It supersedes
`documentation.md` and consolidates all rules from `DOCUMENTATION-AGENT.md`.

---

## Philosophy

**Docstrings are the single source of truth.** The MkDocs site auto-generates API pages
from docstrings via mkdocstrings. Updating a docstring automatically updates the published
API reference. Never duplicate docstring content in guide files.

**Notebooks are living documentation.** A notebook that crashes or produces wrong output
is a documentation bug. Notebooks must be kept in sync with the code and tested in CI.

---

## Structure: what goes where

| Location | Content |
|---|---|
| Source docstrings | Authoritative API documentation for every public function and class |
| `docs/api/` | API reference pages — contain only `::: module.path` directives, never prose |
| `docs/guide/` | Prose explanations of concepts, design decisions, and usage patterns |
| `docs/getting-started/` | Installation and quickstart guides |
| `docs/notebooks/` | Marimo interactive notebooks — runnable tutorials and walkthroughs |
| `tests/test_notebooks.py` | Programmatic notebook tests using `app.run()` |

**Critical rule:** `::: module.path` directives belong only in `docs/api/`. Guide files
link to API pages; they never embed generated API content.

---

## Docstring format

Use **Google style** throughout. Every public function and class requires a docstring.

```python
def split_document(
    document: Document,
    renderer: Renderer,
    max_tokens: int = 7000,
    overlap_tokens: int = 200,
) -> list[Document]:
    """Split a document into token-bounded sub-documents.

    Splits recursively by heading level, then greedily by element groups when
    a single section still exceeds the token budget. Each output document
    carries heading context so it is self-contained for retrieval.

    Args:
        document: The source document to split.
        renderer: Used only for token counting during splitting. No text is persisted.
        max_tokens: Maximum token budget per output document.
        overlap_tokens: Number of tokens from the end of the previous split to
            prepend to the next one for context continuity.

    Returns:
        A list of `Document` objects. Each carries `ExternalRef` entries linking
        it to the source document and to sibling splits.

    Raises:
        ValueError: If `max_tokens` is less than 1.

    Example:
        ```python
        renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        splits = split_document(doc, renderer=renderer, max_tokens=5000)
        ```

    See Also:
        [`split_by_headings`][ragdoc.splitting.split_by_headings]: Simpler flat split at
            every heading boundary.
        [`Renderer`][ragdoc.rendering.Renderer]: Controls the token measurement format.
    """
```

### Required sections by symbol type

| Symbol | Required sections |
|---|---|
| Public function | Summary, Args, Returns, Raises (if applicable), Example |
| Public class | Summary, extended description of invariants, Example |
| ABC / Protocol | Summary, contract description (what implementors must guarantee) |
| Pydantic model | Summary + `Field(description="...")` on every field |

### Cross-reference syntax

Use mkdocstrings-flavoured cross-references — never bare URLs for internal links:

```python
[`ClassName`][ragdoc.module.ClassName]          # link to class
[`function_name`][ragdoc.module.function_name]  # link to function
[`field`][ragdoc.module.Class.field]            # link to field
```

---

## API documentation files (`docs/api/`)

One file per module. Each entry uses the standard mkdocstrings template:

```markdown
## function_or_class_name

::: ragdoc.module.function_or_class_name
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
```

**Complete coverage rule:** every public symbol (anything not prefixed `_`) must appear
in `docs/api/`. Check after adding new functions or classes.

---

## Guide files (`docs/guide/`)

Guides explain the *why* and *how* of a pipeline stage. Rules:

- Write in prose. Code blocks illustrate patterns — not complete runnable programs.
- Link to API pages: `[`Renderer`](../api/rendering.md#Renderer)` — never `::: ragdoc.rendering.Renderer`.
- For complete runnable examples, link to the companion marimo notebook instead.
- Add the "run interactively" callout at the top of any guide with substantial code:
  ```markdown
  > Run interactively: `marimo edit docs/notebooks/<notebook>.py`
  ```

---

## Interactive notebooks (`docs/notebooks/`)

All runnable tutorials use **marimo**. Jupyter `.ipynb` files are not committed — they
are generated at build time from `.py` sources.

### Notebook structure

- First cell: `import marimo as mo` — always returned so `mo` is available everywhere.
- Prose cells: `mo.md(r"""...""")` with `hide_code=True`.
- Code cells: standalone, importable, return every name used by subsequent cells.
- Private variables (underscore prefix) are not exported and avoid name clashes.
- No `display()` calls from IPython. Use `mo.output.append(mo.md(...))` for rich output.
- Async cells are supported natively — use `async def _():` for cells with `await`.

### Keeping notebooks fresh — four layers

When changing any public API (signatures, imports, field names, class names), update
all affected notebooks. Use these four layers to enforce freshness in CI:

**Layer 1 — static check:**
```bash
uv run marimo check docs/notebooks/*.py   # catches undefined names, reactive-graph errors
```

**Layer 2 — pytest execution:**
Marimo notebooks are pure Python; pytest discovers and runs them natively — no plugin needed:
```bash
uv run pytest docs/notebooks/
```

**Layer 3 — `test_*` cells inside notebooks:**
Name assertion cells `test_*`. These serve dual purpose: pytest runs them in CI, and
readers see exactly what outputs to expect without running the notebook. A good test cell
is also readable documentation:

```python
@app.cell
def test_document_parsed(doc):
    assert len(doc.elements) > 0, "document must have elements after parsing"
    assert doc.filename is not None
    # for notebooks: fixtures defined in one cell can't be used by test cells in another —
    # pass the value as a cell argument instead (as shown above)
```

**Layer 4 — `tests/test_notebooks.py`:**
Every notebook gets a test using `app.run()`, which executes the notebook programmatically
and returns `(outputs, defs)` for assertions:

```python
# tests/test_notebooks.py
from docs.notebooks.pipeline_walkthrough import app
import pytest

@pytest.mark.anyio
async def test_pipeline_walkthrough():
    outputs, defs = app.run()
    assert "chunks" in defs
    assert len(defs["chunks"]) > 0

# Override cell values to test different scenarios:
# outputs, defs = app.run(defs={"max_tokens": 1000})
```

### Build integration

```bash
make notebooks   # export marimo → WASM HTML under docs/notebooks/<name>/
make html        # notebooks + mkdocs build
make serve       # notebooks + mkdocs serve (live reload)
```

The WASM export directories (`docs/notebooks/*/`) are generated artifacts — do not commit them.

### Converting from Jupyter

```bash
marimo convert notebook.ipynb -o notebook.py
marimo check notebook.py
```

---

## Site generation

Built with **MkDocs** (compatible with Zensical as a drop-in). Config lives in `mkdocs.yml`.

| Plugin | Purpose |
|---|---|
| `mkdocstrings[python]` | Auto-generates API pages from Google-style docstrings |
| `mkdocs-autorefs` | Resolves `[name][module.path.name]` cross-references |

```bash
make serve                       # builds notebooks + mkdocs serve
uv run mkdocs build --strict     # fail on warnings
uv run mkdocs gh-deploy          # deploy to GitHub Pages
```

---

## Documentation workflow

### When adding a new public function or class

1. Write comprehensive docstring (Summary, Args, Returns, Raises, Example, See Also).
2. Add `::: module.path` directive to the appropriate `docs/api/*.md` file.
3. Update the guide if the change affects user-facing behavior.
4. Update any companion notebook. Run `marimo check` after editing.
5. Add or update the `app.run()` test in `tests/test_notebooks.py`.
6. If new `.md` files were created, add them to the `nav:` section in `mkdocs.yml`.
7. Verify the build: `uv run zensical build` — must pass with no errors.

### When adding a new documentation file

1. Create the file in the appropriate `docs/` subdirectory.
2. **Add it to `mkdocs.yml` under `nav:`** — a file not in `nav:` is unreachable from the site.
3. Run `uv run zensical build` to confirm the site builds cleanly.

### When changing a function signature

1. Update the docstring (new params, updated example).
2. Update all guide files that reference the old signature.
3. Update all notebooks in `docs/notebooks/`. Run `marimo check`.
4. Update `tests/test_notebooks.py` assertions.
5. Add a migration note if it is a breaking change.
6. Run `uv run zensical build` to confirm no broken cross-references.

---

## What not to do

- Do not add `::: module.path` directives in `docs/guide/` or `docs/getting-started/`.
- Do not write narrative documentation in docstrings — describe the API contract; guides explain design rationale.
- Do not duplicate docstring content in guide pages — link instead.
- Do not commit WASM export directories (`docs/notebooks/*/`) — they are gitignored and rebuilt by `make notebooks`.
- Do not use bare `display()` or `IPython.display` in marimo notebooks.
- Do not use old-style (`:param:` / `:type:`) docstrings — always Google style.
- Do not leave notebooks untested — every notebook must have Layer 3 (`test_*` cells) and Layer 4 (`app.run()` test).

---

## Quality checklist

Before considering documentation complete:

- [ ] All public functions/classes have Google-style docstrings
- [ ] All required sections present (Args, Returns, Example)
- [ ] Cross-references to related functions added
- [ ] API documentation updated in `docs/api/`
- [ ] Guide uses markdown links (no `:::` directives)
- [ ] Examples are realistic and runnable
- [ ] Companion notebook updated and passes `marimo check`
- [ ] `test_*` cells added to notebook for key outputs
- [ ] `tests/test_notebooks.py` updated with `app.run()` assertions
- [ ] No duplicated content — link instead
