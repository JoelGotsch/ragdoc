"""Basic local PDF parsing via pymupdf: text extraction + font-size heading detection.

The parser aggregates pymupdf's span dictionaries into visual lines, finds the
document's body font size (the modal line size), and treats larger (or bold,
short) lines as heading candidates.  Candidate sizes are mapped to heading
levels 1–6 by reusing :func:`ragdoc.processing.heading.compute_size_to_level_mapping`.

This is deliberately simple — no tables, no images, no columns.  For higher
fidelity use the ``azure-di`` or ``pdf-mineru`` parsers.
"""

from __future__ import annotations

import html as html_stdlib
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ragdoc.document import Document, Heading, Paragraph
from ragdoc.processing.heading import compute_size_to_level_mapping

logger = logging.getLogger(__name__)

_PDF_IMPORT_ERROR = "The pdf_basic parser requires the 'pdf' extra: pip install 'ragdoc[pdf]'"

_BOLD_FLAG = 2**4  # pymupdf span flag bit for bold fonts
_MAX_HEADING_CHARS = 120  # bold same-size lines longer than this are body text, not headings


@dataclass
class _Line:
    """One visual line aggregated from pymupdf spans."""

    text: str
    size: float  # max span size on the line, rounded to 0.5 pt
    bold: bool  # True when every span on the line is bold
    page: int  # 1-based page number


def _round_half_point(size: float) -> float:
    """Round a font size to the nearest 0.5 pt (OCR/subsetting jitter collapses)."""
    return round(size * 2) / 2


def _extract_lines(doc: object) -> list[_Line]:
    """Flatten a pymupdf document into visual lines with (text, size, bold, page)."""
    lines: list[_Line] = []
    for page_index, page in enumerate(doc, 1):  # type: ignore[attr-defined]  # pymupdf.Document iterates pages
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            for raw_line in block.get("lines", []):
                spans = raw_line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                size = _round_half_point(max(span.get("size", 0.0) for span in spans))
                bold = all(span.get("flags", 0) & _BOLD_FLAG for span in spans)
                lines.append(_Line(text=text, size=size, bold=bold, page=page_index))
    return lines


def _body_size(lines: list[_Line]) -> float:
    """The modal line size — the document's body text size."""
    counts = Counter(line.size for line in lines)
    return counts.most_common(1)[0][0]


def _is_heading_candidate(line: _Line, body_size: float) -> bool:
    """Heading candidates are larger than body text, or bold body-size lines that are short."""
    if line.size > body_size:
        return True
    return line.bold and line.size >= body_size and len(line.text) <= _MAX_HEADING_CHARS


def parse_pdf_basic(path: Path) -> Document:
    """Parse *path* into a Document of Headings and Paragraphs (text + font-size headings).

    Provenance (``source_path``, ``metadata["filename"]``) is stamped centrally by
    :func:`ragdoc.parsing.load`; only ``document.parser`` is set here.

    Args:
        path: Path to the PDF file.

    Returns:
        A :class:`~ragdoc.document.Document` with ``parser="pdf_basic"``.

    Raises:
        ImportError: If pymupdf is not installed (the ``pdf`` extra).
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(_PDF_IMPORT_ERROR) from exc

    document = Document()
    with pymupdf.open(path) as pdf:
        lines = _extract_lines(pdf)

    if not lines:
        logger.warning(f"pdf_basic: no text extracted from {path.name} (scanned/image-only PDF?)")
        document.parser = "pdf_basic"
        return document

    body_size = _body_size(lines)
    candidate_sizes = sorted({line.size for line in lines if _is_heading_candidate(line, body_size)})
    size_to_level = compute_size_to_level_mapping(candidate_sizes)

    for line in lines:
        escaped = html_stdlib.escape(line.text)
        # Emit the font size as inline CSS either way so downstream visual
        # processors (HeadingLevelProcessor, TitleDetectionProcessor) keep working.
        if _is_heading_candidate(line, body_size):
            level = size_to_level.get(line.size, 6)
            document.elements.append(
                Heading(html=f'<h{level} style="font-size:{line.size}pt">{escaped}</h{level}>', page=line.page)
            )
        else:
            document.elements.append(
                Paragraph(html=f'<p style="font-size:{line.size}pt">{escaped}</p>', page=line.page)
            )

    document.parser = "pdf_basic"
    return document
