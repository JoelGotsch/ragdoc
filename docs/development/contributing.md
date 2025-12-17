# Contributing

## Development setup

```bash
git clone <repo-url>
cd aa-document-processing
uv sync --all-extras --group docs
```

## Running tests

```bash
uv run pytest                         # all tests
uv run pytest tests/test_document.py  # single file
uv run pytest tests/ -k "test_name"   # single test by name
uv run pytest-watcher tests/          # watch mode
```

## Type checking and formatting

```bash
uv run mypy src/          # type check
uv run black src/ tests/  # format
uv run isort src/ tests/  # sort imports
```

## Code conventions

- **Models:** Pydantic `BaseModel` with `Field(description="...")` on every field. Use `@property` (not `@computed_field`) for computed values.
- **Functions:** Extract reusable logic into standalone functions (not private methods) so they can be tested independently.
- **Type hints:** Strict — no `Any`, no `**kwargs`. Use `Callable` type aliases for injectable functions.
- **CSS extraction functions:** Always return `None` when a property is absent (not a default value).
- **Async tests:** Use `@pytest.mark.anyio`.
- **Naming:** Processors = `*Processor`, LLM-based async resolvers = `*Resolver`, Protocols describe capability, standalone functions = `verb_noun`.
- **Formatter:** black (line-length=100), isort (black profile).

## Adding a new processor

1. Subclass `DocumentProcessor` and implement `async def process(self, document: Document) -> Document`
2. Pure-sync processors simply omit `await` — no sync base class needed
3. Export from `ragdoc/processing/__init__.py`
4. Add `:::` directive to `docs/api/processing.md`
5. Add usage example to `docs/guide/` if user-facing

## Adding a new parser

1. Create a source type (Pydantic model) in a new file under `src/ragdoc/parsing/`
2. Register a `load_file` singledispatch handler
3. Set `document.parser` in the handler
4. Export source type from `ragdoc/parsing/__init__.py`
5. Add entry to `docs/api/parsing.md` and update the [Parsing Guide](../guide/parsing.md)

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
