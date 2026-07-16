"""Scan a folder for *_middle.json files and print a summary of each document.

Repo-only script (not shipped in the wheel). Requires tabulate, which is not a
ragdoc dependency: run via ``uv run --with tabulate python scripts/evaluate_middle_json.py <dir>``.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tabulate import tabulate

from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import MinerUExtractor


class MiddleJsonFileResult(BaseModel):
    path: Path = Field(description="Path to the evaluated file.")
    page_count: int = Field(description="Number of pages in the source PDF.")
    heading_count: int = Field(description="Number of heading elements extracted.")
    paragraph_count: int = Field(description="Number of paragraph elements extracted.")
    table_count: int = Field(description="Number of table elements extracted.")
    image_count: int = Field(description="Number of image elements extracted.")
    list_count: int = Field(description="Number of list elements extracted.")
    footnote_count: int = Field(description="Number of footnote elements extracted.")
    title: str | None = Field(default=None, description="Detected document title.")
    error: str | None = Field(default=None, description="Error message if processing failed.")


async def _process_file(path: Path, extractor: MinerUExtractor) -> MiddleJsonFileResult:
    try:
        source = MinerUMiddleDocument.from_json_path(path)
        document = await extractor.parse(source)
        return MiddleJsonFileResult(
            path=path,
            page_count=source.num_pages,
            heading_count=len(document.headings),
            paragraph_count=len(document.paragraphs),
            table_count=len(document.tables),
            image_count=len(document.images),
            list_count=len(document.lists),
            footnote_count=len(document.footnotes),
            title=document.title or (document.main_heading.text if document.main_heading else None),
        )
    except Exception as e:  # noqa: BLE001 -- eval script collects per-file errors into results
        return MiddleJsonFileResult(
            path=path,
            page_count=0,
            heading_count=0,
            paragraph_count=0,
            table_count=0,
            image_count=0,
            list_count=0,
            footnote_count=0,
            error=str(e),
        )


async def evaluate_middle_jsons(folder: Path | str) -> list[MiddleJsonFileResult]:
    """Parse all *_middle.json files in a folder and return per-file statistics.

    :param folder: Directory to search for ``*_middle.json`` files.
    :return: One :class:`MiddleJsonFileResult` per file found.
    """
    folder = Path(folder)
    paths = sorted(folder.glob("*_middle.json"))
    extractor = MinerUExtractor()
    results = []
    for p in paths:
        results.append(await _process_file(p, extractor))
    return results


def print_summary(results: list[MiddleJsonFileResult]) -> None:
    """Print a formatted summary table of evaluation results."""
    if not results:
        print("No *_middle.json files found.")
        return

    ok = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    # --- per-file table ---
    headers = ["File", "Pages", "Headings", "Paragraphs", "Tables", "Images", "Lists", "Footnotes", "Title"]
    rows = []
    for r in results:
        if r.error:
            rows.append([r.path.name, "ERROR", "", "", "", "", "", "", r.error[:60]])
        else:
            title_preview = (r.title[:40] + "…") if r.title and len(r.title) > 40 else (r.title or "")
            rows.append(
                [
                    r.path.name,
                    r.page_count,
                    r.heading_count,
                    r.paragraph_count,
                    r.table_count,
                    r.image_count,
                    r.list_count,
                    r.footnote_count,
                    title_preview,
                ]
            )

    print()
    print(tabulate(rows, headers=headers, tablefmt="github"))

    # --- aggregate totals ---
    if ok:
        print()
        totals = [
            ["Files processed", len(ok)],
            ["Files with errors", len(errors)],
            ["Total pages", sum(r.page_count for r in ok)],
            ["Total headings", sum(r.heading_count for r in ok)],
            ["Total paragraphs", sum(r.paragraph_count for r in ok)],
            ["Total tables", sum(r.table_count for r in ok)],
            ["Total images", sum(r.image_count for r in ok)],
            ["Total lists", sum(r.list_count for r in ok)],
            ["Total footnotes", sum(r.footnote_count for r in ok)],
        ]
        print(tabulate(totals, headers=["Metric", "Value"], tablefmt="github"))

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for r in errors:
            print(f"  {r.path.name}: {r.error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <folder>")
        sys.exit(1)

    folder_arg = Path(sys.argv[1])
    if not folder_arg.is_dir():
        print(f"Error: '{folder_arg}' is not a directory.")
        sys.exit(1)

    import asyncio

    results = asyncio.run(evaluate_middle_jsons(folder_arg))
    print_summary(results)
