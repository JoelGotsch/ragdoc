from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# from funcy import rcompose
from pydantic import BaseModel

from ragdoc.config import get_config
from ragdoc.document import Document, DocumentList, Footnote, Heading, Image, Paragraph, Table

logger = logging.getLogger(__name__)


class HTML(BaseModel):
    content: str

    @classmethod
    def from_file(cls, path: str | Path) -> HTML:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        return cls(content=content)


base64_src = re.compile(r"data:image/(?P<image_type>[a-zA-Z\-]+);base64,")


def is_structural(tag: Tag) -> bool:
    if tag_found := tag.find(string=True, recursive=False):
        return not tag_found.strip()
    return True


TSoupTransformer = Callable[[BeautifulSoup], BeautifulSoup]


def unwrap_structural(soup: BeautifulSoup, tags: list[str] | None = None) -> BeautifulSoup:
    """This function unwraps (as default) div and span tags, that have no text in them."""
    if tags is None:
        tags = ["div", "span"]
    for element in soup.find_all(tags):
        if is_structural(element):
            element.unwrap()
    return soup


def unwrap_idiotic_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """This function looks for tables in the soup tree which fulfill the following criteria:

    - contains a THEAD
    - contains a TBODY which is empty
    - THEAD contains exactly one TR
    - TR contains exactly one TH

    Those tables are procuded by some tormented soul deciding its a super cool idea
    to use tables for drawing a broder around a bunch of paragraphs in words.
    Because formatting and styling would be very boring and we do not do this here.

    Therefor those tables now go away and are replaced by the content of the TH cell.
    """
    tables = soup.find_all("table")

    for table in tables:
        thead = table.find("thead")
        tbody = table.find("tbody")

        if thead is not None and tbody is not None and len(tbody.find_all("tr")) == 0:
            thead_rows = thead.find_all("tr")
            if len(thead_rows) == 1:
                th_cells = thead_rows[0].find_all("th")
                if len(th_cells) == 1:
                    th_element = th_cells[0]
                    table.replace_with(th_element)
                    th_element.unwrap()

    return soup


_FOOTNOTE_ID_RE = re.compile(r"^(?:footnote|fn|note)[-_]?(\d+)$", re.I)


def detect_html_footnotes(soup: BeautifulSoup) -> BeautifulSoup:
    """Detect and convert anchor-id footnote paragraphs to ``<aside class="footnote">`` elements.

    Matches ``<p id="footnote-N">``, ``<p id="fn-N">``, ``<p id="note-N">`` etc.
    (case-insensitive, hyphen or underscore separator, number suffix required).
    A leading ``[N]`` prefix in the paragraph text is stripped from the content.

    ``Footnote.html`` renders as ``<aside class="footnote" data-number="N">`` directly,
    so the render → reparse roundtrip requires no transformer — this transformer only
    activates for footnotes arriving from external HTML sources.
    """
    for p in list(soup.find_all("p")):
        p_id = p.get("id", "")
        if not p_id:
            continue
        m = _FOOTNOTE_ID_RE.match(str(p_id))
        if not m:
            continue
        number = int(m.group(1))
        # Strip leading [N] prefix from the first text node, if present
        for child in p.children:
            if isinstance(child, NavigableString):
                child.replace_with(re.sub(r"^\s*\[" + str(number) + r"\]\s*", "", str(child), count=1))
                break
        p.name = "aside"
        p.attrs.clear()
        p["class"] = "footnote"
        p["data-number"] = str(number)
    return soup


def extract_image_attributes(image: Tag) -> dict:
    # This looks funky because bs4 is funky
    kwargs = {}
    for key in ["width", "height", "alt"]:
        if image.has_attr(key):
            kwargs[key] = image[key]
    return kwargs


def generate_image(image: Tag) -> Image | None:
    download_images = get_config().download_images
    if not image.has_attr("src"):
        return None

    image_src = image["src"]
    if not isinstance(image_src, str):
        logger.warning(f"Image src is not a string: {image['src']}")
        return None

    if match := base64_src.match(image_src):
        _, content = image_src.split(",", maxsplit=1)
        image_type = match.group("image_type")
        kwargs = extract_image_attributes(image)
        image = Image(image=content, image_type=image_type, **kwargs)
        return image
    elif download_images and (image_src.startswith("https://") or image_src.startswith("http://")):
        try:
            response = requests.get(image_src)
        except requests.exceptions.ConnectionError:
            logger.warning(f"Failed to download image {image_src}: connection error")
            return None
        if response.status_code == 200:
            content = base64.b64encode(response.content).decode("utf-8")
            kwargs = extract_image_attributes(image)
            return Image(image=content, **kwargs)
        else:
            logger.error(f"Failed to download image: {image_src}")
            pass


def _handle_images(element: Tag) -> list[Image]:
    """Replace <img> tags with <ref id="..." rel="image"/> placeholders and return Image elements."""
    images = []
    for image in element.find_all(["img"]):
        document_image = generate_image(image)
        if document_image is not None:
            images.append(document_image)
            ref_tag = BeautifulSoup(document_image.placeholder_html, "html.parser")
            image.replace_with(ref_tag)
    return images


def handle_tag(element: Tag, document: Document) -> None:
    """
    This function handles the tags that are directly nested under the heading tags.
    It does not handle nested tags, which are handled in the else case where we loop over
    all nested tags and call this function recursively.
    """
    match element.name:
        case "p" | "div":
            images = _handle_images(element)
            document_paragraph = Paragraph(html_content=str(element).strip())
            document.elements.append(document_paragraph)
            document.elements.extend(images)
        case "table":
            images = _handle_images(element)
            document.elements.append(Table(html_content=str(element).strip()))
            document.elements.extend(images)

        case "ul" | "ol" | "dl":
            images = _handle_images(element)
            document.elements.append(DocumentList(html_content=str(element).strip()))
            document.elements.extend(images)
        case "img":
            document_image = generate_image(element)
            if document_image is not None:
                document.elements.append(document_image)
        case "aside":
            if "footnote" in (element.get("class") or []):
                number = int(element.get("data-number") or 0)
                innerhtml = element.decode_contents().strip()
                aside_id = element.get("id", "")
                fn_id = aside_id.removeprefix("footnote-") if aside_id.startswith("footnote-") else None
                document.elements.append(
                    Footnote(number=number, innerhtml=innerhtml, **({"id": fn_id} if fn_id else {}))
                )


def _reconstruct_footnote_refs(document: Document) -> None:
    """Rewrite <a href="#footnote-{uuid}"> citation anchors back to <ref> inline refs.

    Called after all elements are parsed so Footnote UUIDs are known. Safe for
    external HTML: anchors that don't match any Footnote UUID are left untouched.
    """
    id_to_fn = {fn.id: fn for fn in document.footnotes}
    if not id_to_fn:
        return
    for element in document.elements:
        if not hasattr(element, "html_content"):
            continue
        soup = BeautifulSoup(element.html_content, "html.parser")
        changed = False
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if href.startswith("#footnote-"):
                fn_id = href.removeprefix("#footnote-")
                fn = id_to_fn.get(fn_id)
                if fn is not None:
                    a.replace_with(BeautifulSoup(fn.placeholder_html, "html.parser"))
                    changed = True
        if changed:
            element.html_content = str(soup)


def generate_document(html: HTML, soup_transformers: list[TSoupTransformer] | None = None) -> Document:
    document = Document()

    soup = BeautifulSoup(html.content, "html.parser")

    transformers = (
        soup_transformers if soup_transformers is not None else [unwrap_idiotic_tables, detect_html_footnotes]
    )
    for transformer in transformers:
        soup = transformer(soup)

    if title_tag := soup.find("title"):
        document.title = title_tag.get_text(strip=True)

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "h7"}

    headings = soup.find_all(heading_tags)
    # XXX: This is a very very bad hack to deal with <hN> where
    # N > 6 tags that might happen when you convert docx with pandoc to html
    p_headings = soup.find_all(["p"], attrs={"class": "heading"})
    # TODO: Create test case for this kind of heading
    headings.extend(p_headings)

    # Note: headings not necessarily in order!
    # TODO: Maybe include p tags before, but then filter out the ones which doesn't have the heading class attr.

    element_names = {"p", "table", "ol", "ul", "dl", "img", "div", "aside"}
    # Handle preface elements
    root = soup.find("body") or soup

    # TODO: this is probably too simplistic. It might be better to to first find headings again and then for each heading
    # iterate through siblings until another heading is found - and then continue with that heading.
    for element in root.find_all(element_names | heading_tags, recursive=False):
        if element.name in heading_tags or (
            element.name == "p" and "class" in element.attrs and "heading" in element.attrs["class"]
        ):
            heading_innerhtml = element.decode_contents(formatter="html")
            heading_level = int("6" if element.name == "p" else element.name.replace("h", ""))
            document.elements.append(Heading(innerhtml=heading_innerhtml, level=heading_level))
        else:
            handle_tag(element, document)
    document.parser = "html"
    _reconstruct_footnote_refs(document)
    return document
