# Contributing

## Development setup

```bash
git clone https://github.com/joelgotsch/ragdoc
cd ragdoc
uv sync --all-extras --group docs
uv run pre-commit install
```

**System requirement:** [pandoc](https://pandoc.org/installing.html) must be installed for
rendering and the pandoc parser (`brew install pandoc` on macOS, `apt-get install pandoc` on
Debian/Ubuntu — CI installs it the same way). The bundled `pypandoc-binary` usually covers this,
but a system install is the reliable fallback.

## Running tests

```bash
uv run pytest                         # all tests (no API keys or services needed)
uv run pytest tests/test_document.py  # single file
uv run pytest tests/ -k "test_name"   # single test by name
uv run pytest-watcher tests/          # watch mode
```

LLM-dependent code is tested with mocked clients and pre-recorded fixture replay; Qdrant is fully
stubbed. The suite must stay runnable offline with zero credentials.

## Lint, format, type-check

```bash
uv run ruff check src tests          # lint
uv run ruff check src tests --fix    # lint + autofix
uv run ruff format src tests         # format (black-compatible, line-length 120)
uv run basedpyright                  # type check (standard mode)
```

These mirror the CI jobs; tox wraps them as `tox -e lint` and `tox -e type`.

## Code conventions

- **Models:** Pydantic `BaseModel` with `Field(description="...")` on every field. Use `@property` (not `@computed_field`) for computed values.
- **Functions:** Extract reusable logic into standalone functions (not private methods) so they can be tested independently.
- **Type hints:** Strict — no `Any`, no `**kwargs`. Use `Callable` type aliases for injectable functions.
- **CSS extraction functions:** Always return `None` when a property is absent (not a default value).
- **Async-only:** all public entry points are `async`; do not add sync convenience wrappers. Async tests use `@pytest.mark.anyio`.
- **Naming:** Processors = `*Processor`, LLM-based async resolvers = `*Resolver`, Protocols describe capability, standalone functions = `verb_noun`.
- **xfail discipline:** known bugs are tracked as `xfail(strict=True)` tests with a reason — fixing the bug forces promoting the test.

## Adding a new processor

1. Subclass `DocumentProcessor` and implement `async def process(self, document: Document) -> Document | None`
2. Pure-sync processors simply omit `await` — no sync base class needed; return `None` to drop a document
3. Export from `ragdoc/processing/__init__.py`
4. Add `:::` directive to `docs/api/processing.md`
5. Add usage example to `docs/guide/` if user-facing

## Adding a new parser

1. Subclass the `Parser` ABC (`src/ragdoc/parsing/parser.py`): set `name`, `patterns`
   (filename suffixes), `priority`, `description`, and implement
   `async def __call__(self, path: Path) -> Document`
2. In the parser body, set `document.parser` (provenance string), `document.source_path`,
   and `document.metadata["filename"] = path.name`
3. Register it with `register_parser(...)` (`src/ragdoc/parsing/registry.py`) — resolution
   order is: explicit argument → config `parser_preferences` → longest-suffix match + priority
4. Export from `ragdoc/parsing/__init__.py`
5. Add entry to `docs/api/parsing.md` and update the [Parsing Guide](../guide/parsing.md)

The in-tree parsers (`html/`, `pandoc/`, `xlsx/`, `azure_di/`, `mineru/`, `ragdoc_json/`) are
working references.

## Structural model changes

Before adding or removing fields on `Document`, `BaseElement` subclasses, or other core
Pydantic models, discuss implications with the maintainer. These changes affect all pipeline
stages and all parsers. See `CLAUDE.md` for the collaboration guidelines.

## Building docs locally

```bash
uv run mkdocs serve       # live reload at http://localhost:8000
uv run mkdocs build       # static site in site/
uv run mkdocs build --strict  # fail on warnings
```
