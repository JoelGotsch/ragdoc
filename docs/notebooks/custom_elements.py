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
    No changes to `ElementTypeEnum` are needed.
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
    from ragdoc.rendering.elements import render_for_prompt
    from ragdoc.rendering.base import RenderContext


    @render_for_prompt.register(CalloutElement)
    def _render_callout_prompt(
        element: CalloutElement, ctx: RenderContext, inline: bool = False
    ) -> str:
        # Full content for LLM context windows.
        if inline:
            return f"<span>[{element.kind.upper()}: {element.title}]</span>"
        return element.html

    return (render_for_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How Parsers Create Elements

    Parsers construct `BaseElement` subclasses as plain Pydantic models — there is no special factory.
    A parser reads the source format and directly instantiates the appropriate element class.

    Below is a minimal parser for a made-up `.callout` format (one callout per line: `kind|title|body`).

    ### 3a. Define a Parser subclass

    The recommended way to register a parser is to subclass [`Parser`][ragdoc.parsing.parser.Parser]
    and register it via [`register_parser()`][ragdoc.parsing.register_parser]. The parser declares
    which file patterns it handles and its priority relative to other parsers.
    """)
    return


@app.cell
def _(CalloutElement):
    from pathlib import Path
    from ragdoc.document import Document
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
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                kind, title, body = line.split("|", maxsplit=2)
                elements.append(CalloutElement(kind=kind.strip(), title=title.strip(), body=body.strip()))

            return Document(elements=elements, parser="callout")


    register_parser(CalloutParser())

    return CalloutParser, Path, register_parser


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. End-to-End Example
    """)
    return


@app.cell
async def _(
    Path,
    render_for_prompt,
):
    import tempfile
    from ragdoc.parsing import load
    from ragdoc.rendering import Renderer, OutputFormat
    sample = '# Sample callout file\ntip    | Use type hints      | Always annotate function signatures for better IDE support.\nwarning| Avoid mutable defaults | Never use a mutable default argument like `def f(x=[]):`.\n'
    with tempfile.NamedTemporaryFile(suffix='.callout', mode='w', delete=False) as f:
        f.write(sample)
        tmp_path = Path(f.name)
    # load() resolves to our registered CalloutParser via the .callout extension
    document = await load(tmp_path)
    print(f"Parsed {len(document.elements)} elements, parser='{document.parser}'")
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    print('\n--- PROMPT CONTENT ---')
    print(renderer.render(document))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Summary

    To add a custom element type and parser:

    1. **Subclass `BaseElement`** with a custom `element_type: Literal[...]` discriminator.
    2. **Register renderers** via `render_for_prompt.register(YourElement)`.
    3. **Subclass `Parser`** and call `register_parser()` — then `load()` picks it up automatically.
    """)
    return


if __name__ == "__main__":
    app.run()
