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
    # Custom Element Types and Parsers

    Note: This is an advanced topic for users who want to extend ragdoc's core functionality.  If you're new to ragdoc, we recommend starting with the [Getting Started](../getting_started.ipynb) guide and referring to the [API Reference](../api_reference.ipynb) for details on existing element types and renderers.

    This notebook shows how to extend ragdoc with:

    1. A custom element type (`CalloutElement`)
    2. A custom parser that produces it
    3. A custom renderer for prompt content
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Define a Custom Element

    `BaseElement.element_type` is typed as `str`, so any subclass can define its own `Literal` value.

    Note: `Document.elements` is a **closed discriminated union** over the built-in element
    types (`ElementType`), so a custom element cannot be stored inside a `Document` directly.
    Custom elements shine as parser-side intermediates: parse into your custom model, then
    map it onto the built-in element vocabulary (step 3 below) before building the `Document`.
    """)
    return


@app.cell
def _():
    from typing import Literal

    from pydantic import Field

    from ragdoc.document import BaseElement


    class CalloutElement(BaseElement):
        """A highlighted callout box (tip, warning, note, …)."""

        element_type: Literal["callout"] = "callout"
        kind: str = Field(description="Callout kind: 'tip' | 'warning' | 'note'")
        title: str = Field(description="Short heading displayed in the callout box")
        body: str = Field(description="Full body text of the callout")

        @property
        def html(self) -> str:  # type: ignore[override]
            return (
                f'<blockquote class="callout callout-{self.kind}">'
                f"<strong>{self.title}</strong>"
                f"<p>{self.body}</p>"
                f"</blockquote>"
            )

    return (CalloutElement,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Register Custom Renderers

    All element renderers use `singledispatch`. Register a handler per renderer per type.

    - **`render_for_prompt`** — full content for LLM context windows.
    """)
    return


@app.cell
def _(CalloutElement):
    from ragdoc.rendering.base import RenderContext
    from ragdoc.rendering.elements import render_for_prompt


    @render_for_prompt.register(CalloutElement)
    def _render_callout_prompt(
        element: CalloutElement, ctx: RenderContext, inline: bool = False
    ) -> str:
        # Full content for LLM context windows.
        if inline:
            return f"<span>[{element.kind.upper()}: {element.title}]</span>"
        return element.html

    return (RenderContext, render_for_prompt)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How Parsers Create Elements

    Parsers construct `BaseElement` subclasses as plain Pydantic models — there is no special factory.
    A parser reads the source format and directly instantiates the appropriate element class.

    Below is a minimal parser for a made-up `.callout` format (one callout per line: `kind|title|body`).
    It parses each line into a `CalloutElement`, then maps it onto the built-in vocabulary
    (`RawText` carrying the callout's HTML) — because `Document.elements` only accepts the
    built-in element types.

    ### 3a. Define a Parser subclass

    The recommended way to register a parser is to subclass [`Parser`][ragdoc.parsing.parser.Parser]
    and register it via [`register_parser()`][ragdoc.parsing.register_parser]. The parser declares
    which file patterns it handles and its priority relative to other parsers.
    """)
    return


@app.cell
def _(CalloutElement):
    from pathlib import Path

    from ragdoc.document import Document, RawText
    from ragdoc.parsing import register_parser
    from ragdoc.parsing.parser import Parser


    class CalloutParser(Parser):
        """Parser for .callout files."""

        name: str = "callout"
        patterns: list[str] = [".callout"]
        priority: int = 10
        description: str = "Callout format (kind|title|body per line)"

        async def __call__(self, path: Path) -> Document:
            elements = []
            for line in path.read_text().splitlines():  # noqa: ASYNC240 — doc example; sync read is fine here
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                kind, title, body = line.split("|", maxsplit=2)
                callout = CalloutElement(kind=kind.strip(), title=title.strip(), body=body.strip())
                # Document.elements is a closed union — store the callout's HTML as RawText
                elements.append(RawText(innerhtml=callout.html))

            return Document(elements=elements, parser="callout")


    register_parser(CalloutParser())

    return CalloutParser, Document, Path, register_parser


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. End-to-End Example
    """)
    return


@app.cell
async def _(
    CalloutElement,
    Document,
    Path,
    RenderContext,
    render_for_prompt,
):
    import tempfile

    from ragdoc.parsing import load
    from ragdoc.rendering import OutputFormat, Renderer
    sample = "# Sample callout file\ntip    | Use type hints      | Always annotate function signatures for better IDE support.\nwarning| Avoid mutable defaults | Never use a mutable default argument like `def f(x=[]):`.\n"
    with tempfile.NamedTemporaryFile(suffix=".callout", mode="w", delete=False) as f:
        f.write(sample)
        tmp_path = Path(f.name)
    # load() resolves to our registered CalloutParser via the .callout extension
    document = await load(tmp_path)
    print(f"Parsed {len(document.elements)} elements, parser='{document.parser}'")
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    print("\n--- PROMPT CONTENT ---")
    print(renderer.render(document))
    # The registered singledispatch renderer also handles CalloutElement instances directly:
    callout = CalloutElement(kind="note", title="Direct rendering", body="Custom elements render outside a Document too.")
    print("\n--- DIRECT CalloutElement RENDER ---")
    print(render_for_prompt(callout, RenderContext(document=Document())))
    return (callout, document, renderer, tmp_path)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Summary

    To add a custom element type and parser:

    1. **Subclass `BaseElement`** with a custom `element_type: Literal[...]` discriminator —
       useful as a parser-side intermediate; `Document.elements` itself only accepts the
       built-in element types, so map custom elements onto that vocabulary before building
       the `Document`.
    2. **Register renderers** via `render_for_prompt.register(YourElement)` for rendering
       custom elements directly.
    3. **Subclass `Parser`** and call `register_parser()` — then `load()` picks it up automatically.
    """)
    return


if __name__ == "__main__":
    app.run()
