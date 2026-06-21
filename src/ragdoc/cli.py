"""Command-line interface for ragdoc."""

import asyncio
from pathlib import Path

import click
from pydantic import TypeAdapter

from ragdoc.document import Document
from ragdoc.parsing import load
from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt


@click.group()
def document_processing():
    pass


@click.command()
@click.argument(
    "files",
    type=click.Path(exists=True, path_type=Path, file_okay=True, dir_okay=False),
    nargs=-1,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default="output",
    help="Output directory.",
)
def parse(files: tuple[Path, ...], output: Path) -> None:
    """Parse files and write each Document as a JSON file."""
    if not output.exists():
        output.mkdir()

    ta = TypeAdapter(Document)
    for file in files:
        click.echo(f"Processing {file}")
        try:
            document = asyncio.run(load(file))
            with open(output / f"{file.stem}.document.json", "wb") as f:
                f.write(ta.dump_json(document))
        except ValueError as e:
            click.echo(f"Error: {e}")
            continue
        except RuntimeError as e:
            click.echo(f"Error: {e}")
            continue


@click.command()
@click.argument(
    "files",
    type=click.Path(exists=True, path_type=Path, file_okay=True, dir_okay=False),
    nargs=-1,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default="output",
    help="Output directory.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["html", "md", "gfm", "rst", "plain"], case_sensitive=False),
    default="md",
    help="Output format.",
)
def chunk(files: tuple[Path, ...], output: Path, fmt: str) -> None:
    """Parse files and render each document as text."""
    if not output.exists():
        output.mkdir()

    renderer = Renderer(format=OutputFormat(fmt), element_renderer=render_for_prompt)
    for file in files:
        click.echo(f"Processing {file}")
        try:
            document = asyncio.run(load(file))
            rendered = renderer.render(document)
            out_file = output / f"{file.stem}.{fmt}"
            out_file.write_text(rendered, encoding="utf-8")
        except ValueError as e:
            click.echo(f"Error: {e}")
            continue
        except RuntimeError as e:
            click.echo(f"Error: {e}")
            continue


document_processing.add_command(parse)
document_processing.add_command(chunk)


if __name__ == "__main__":
    document_processing()
