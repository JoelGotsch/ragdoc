from __future__ import annotations

import base64
import logging
from functools import partial
from itertools import islice
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pypandoc import convert_text

from ragdoc.document import Document
from ragdoc.parsing.html.load import HTML, generate_document as html_generate_documents, unwrap_idiotic_tables

logger = logging.getLogger(__name__)


class FigureExtractor:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.doc = pymupdf.open(file_path)

    def _rectangular_hull(self, polygon: list[tuple[float, float]], dpi: int = 72) -> tuple[float, float, float, float]:
        polygon = [(x * dpi, y * dpi) for x, y in polygon]
        x_values, y_values = zip(*polygon)
        return min(y_values), max(y_values), min(x_values), max(x_values)

    def extract_document_image(self, page: int, polygon: list[tuple[float, float]]) -> tuple[tuple[int, int], bytes]:
        pdf_page: pymupdf.Page = self.doc[page - 1]
        top, bottom, left, right = self._rectangular_hull(polygon)
        clip_rectangle = pymupdf.Rect(left, top, right, bottom)
        image: pymupdf.Pixmap = pdf_page.get_pixmap(clip=clip_rectangle)
        return (image.width, image.height), base64.b64encode(image.tobytes())


def _get_figure_information(figure: dict) -> tuple[int | None, list[tuple[float, float]]]:
    if (
        "boundingRegions" not in figure
        or not isinstance(figure["boundingRegions"], list)
        or len(figure["boundingRegions"]) == 0
    ):
        return None, []

    bounding_region = figure["boundingRegions"][0]
    page = bounding_region.get("pageNumber")
    polygon = list(zip(islice(bounding_region["polygon"], 0, None, 2), islice(bounding_region["polygon"], 1, None, 2)))
    return page, polygon


def replace_figure_tags(soup: BeautifulSoup, azure_bundle) -> BeautifulSoup:
    result = azure_bundle.analyze_result
    figures = result.get("figures", [])
    html_figures = soup.find_all("figure")

    if not figures or not azure_bundle.source_path or not azure_bundle.source_path.exists():
        return soup

    if len(figures) != len(html_figures):
        logger.warning(
            f"Number of figures in AzureDI result ({len(figures)}) does not match number of figure tags in HTML ({len(html_figures)})"
        )
        return soup

    try:
        extractor = FigureExtractor(azure_bundle.source_path)
    except Exception:
        return soup

    for azure_figure, html_figure in zip(figures, html_figures):
        page, polygon = _get_figure_information(azure_figure)
        if page is None or len(polygon) == 0:
            continue

        try:
            (width, height), b64_image = extractor.extract_document_image(page, polygon)
        except Exception:
            continue

        img_tag = soup.new_tag(
            "img",
            src=f"data:image/png;base64,{b64_image.decode()}",
            width=str(width),
            height=str(height),
            alt=html_figure.text,
        )
        html_figure.replace_with(img_tag)
    return soup


class AzureDIBundle(BaseModel):
    source_path: Path | None = Field(description="Path to the source file", default=None)
    analyze_result: dict = Field(..., description="Analyze Result output of AzureDI as dict")


def generate_document_azure_di(azure_bundle: AzureDIBundle) -> Document:
    md_content = azure_bundle.analyze_result["content"]
    html_content = convert_text(md_content, to="html", format="markdown")
    document = html_generate_documents(
        HTML(content=html_content),
        soup_transformers=[
            unwrap_idiotic_tables,
            partial(replace_figure_tags, azure_bundle=azure_bundle),
        ],
    )
    document.parser = "azure_di"
    return document
