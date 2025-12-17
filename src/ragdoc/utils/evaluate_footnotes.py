from pathlib import Path

from pydantic import BaseModel, Field

from ragdoc.parsing import load
from ragdoc.processing import (
    DocumentProcessor,
    FootnoteProcessor,
    HeadingLevelProcessor,
    ProcessingPipeline,
    TitleDetectionProcessor,
)


class FootnoteFileResult(BaseModel):
    path: Path = Field(description="Path to the evaluated file.")
    total_footnotes: int = Field(description="Total number of footnotes in the document.")
    orphan_count: int = Field(description="Number of footnotes not referenced in the document text.")
    orphan_numbers: list[int] = Field(description="Sorted footnote numbers of orphan footnotes.")
    error: str | None = Field(default=None, description="Error message if the file could not be processed.")


def _default_processors() -> list[DocumentProcessor]:
    return [
        HeadingLevelProcessor(),
        TitleDetectionProcessor(),
        FootnoteProcessor(),
    ]


async def _process_file(
    path: Path,
    pipeline: ProcessingPipeline,
) -> FootnoteFileResult:
    try:
        document = await load(path)
        processed = await pipeline.process(document)
        if processed is None:
            raise ValueError("Processing pipeline dropped the document")
        document = processed
        orphans = document.orphaned_footnotes
        return FootnoteFileResult(
            path=path,
            total_footnotes=len(document.footnotes),
            orphan_count=len(orphans),
            orphan_numbers=sorted(f.number for f in orphans),
        )
    except Exception as e:
        return FootnoteFileResult(
            path=path,
            total_footnotes=0,
            orphan_count=0,
            orphan_numbers=[],
            error=str(e),
        )


async def evaluate_footnotes(
    paths: list[Path | str],
    processors: list[DocumentProcessor] | None = None,
) -> list[FootnoteFileResult]:
    """Parse files and evaluate orphan footnotes (footnotes not linked from any element text).

    An orphan footnote is a :class:`~ragdoc.document.Footnote` element that has
    no corresponding ``<ref rel="footnote"/>`` tag in any other element after
    the footnote processor has run.

    :param paths: File paths to evaluate. Supported extensions: ``.docx``,
        ``.html``, ``.xlsx``, ``.pdf``, ``.azure.json``.
    :param processors: Processors to apply after parsing. Defaults to
        ``[HeadingLevelProcessor(), TitleDetectionProcessor(), FootnoteProcessor()]``.
    :return: One :class:`FootnoteFileResult` per input path.
    """
    resolved_processors = processors if processors is not None else _default_processors()
    pipeline = ProcessingPipeline(resolved_processors)
    resolved_paths = [Path(p) for p in paths]
    results = []
    for path in resolved_paths:
        result = await _process_file(path, pipeline)
        results.append(result)
    return results
