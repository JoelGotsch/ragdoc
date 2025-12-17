from __future__ import annotations

import logging
import re
import uuid
from enum import Enum
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Annotated, Generic, Literal, TypeVar

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md
from pydantic import BaseModel, Field, SkipValidation, computed_field, model_validator
from pypandoc import convert_text
from typing_extensions import Self

from ragdoc.metadata import MetadataDict, TMetadata, validate_metadata_dict

if TYPE_CHECKING:
    from ragdoc.rendering import Renderer

logger = logging.getLogger(__name__)


class InlineRef(BaseModel):
    """
    Reference to an element that should be rendered inline at the reference location.

    InlineRef represents a `<ref id="..." rel="..."/>` tag embedded in an element's HTML.
    It is NOT stored as a field — it is derived from the element's HTML by the
    `BaseElement.inline_refs` property. The HTML is the single source of truth.

    The renderer replaces `<ref id="..."/>` tags in HTML with the rendered content
    of the referenced element.

    Attributes:
        target_id: The ID of the element being referenced. Must exist in the same Document.
        rel_type: The type of relationship, encoded as the `rel` attribute on the `<ref>` tag:
            - "image": Reference to an Image element
            - "footnote": Reference to a Footnote element
            - "table": Reference to a Table element (for inline table references)
            - "figure": Reference to a figure (image with caption)

    Example (HTML is source of truth):
        ```python
        para = Paragraph(html="<p>See <ref id='img-123' rel='image'/> for the chart.</p>")
        assert para.inline_refs == [InlineRef(target_id="img-123", rel_type="image")]
        ```
    """

    target_id: str = Field(..., description="ID of the referenced element. Must exist in the same Document.")
    rel_type: Literal["image", "footnote", "table", "figure"] = Field(
        ..., description="Type of inline relationship: image, footnote, table, or figure."
    )


class ExternalRef(BaseModel):
    """
    Reference to an element or document outside the current Document.

    ExternalRef is used for cross-document relationships like parent-child
    hierarchies or citations. Unlike InlineRef, external references are NOT
    rendered inline—they are used for navigation and hierarchical rendering
    (e.g., breadcrumbs).

    All rel_type values are prefixed with "external-" to distinguish them from
    inline ref types and to make cross-document semantics explicit.

    Parent/child relationships are bidirectional:
    - If A is parent of B, then B has ExternalRef(target_id=A.id, rel_type="external-parent")
    - If A is parent of B, then A has ExternalRef(target_id=B.id, rel_type="external-child")

    Siblings are NOT stored explicitly—they are computed on-the-fly as other
    children of the same parent.

    The ExternalRefProvider protocol allows renderers to resolve these references
    when needed (e.g., for breadcrumb generation).

    Attributes:
        target_id: The ID of the external element or document being referenced.
        rel_type: The type of relationship (always prefixed with "external-"):
            - "external-parent": This document is a child of the target
            - "external-child": This document is a parent of the target
            - "external-cites": This document cites or references the target
            - "external-related": Generic relationship

    Example:
        ```python
        # Bidirectional parent-child
        parent_doc = Document(
            id="parent-id",
            external_refs=[ExternalRef(target_id="child-id", rel_type="external-child")],
        )
        child_doc = Document(
            id="child-id",
            external_refs=[ExternalRef(target_id="parent-id", rel_type="external-parent")],
        )
        ```
    """

    target_id: str = Field(..., description="ID of the external element or document being referenced.")
    rel_type: Literal["external-parent", "external-child", "external-cites", "external-related"] = Field(
        ...,
        description="Type of external relationship: external-parent, external-child, external-cites, or external-related.",
    )


class ElementTypeEnum(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    DOCUMENT_LIST = "document_list"
    IMAGE = "image"
    RAW_TEXT = "raw_text"
    FOOTNOTE = "footnote"


class BaseElement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the element")
    element_type: str = Field(
        ...,
        description="Type of the element. Built-in types use ElementTypeEnum values (e.g. 'heading', 'paragraph'). Custom subclasses should narrow this to their own Literal string.",
    )
    page: int | None = Field(default=0, description="Page number of the element. Should start with 1, if set.")
    metadata: MetadataDict = Field(
        default_factory=dict, description="Additional metadata for the element. Can be used for custom parsers."
    )
    bounding_box: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box of the element in the format (x0, y0, x1, y1). Coordinates are expected to be normalized between 0 and 1, relative to the page dimensions.",
    )

    @property
    def inline_refs(self) -> list[InlineRef]:
        """Inline references parsed from <ref id="..." rel="..."/> tags in this element's HTML.

        The element's HTML is the single source of truth. To add an inline ref,
        embed a <ref id="<target-id>" rel="<rel-type>"/> tag in the HTML content.
        """
        soup = BeautifulSoup(self.html, "html.parser")
        refs: list[InlineRef] = []
        for ref_tag in soup.find_all("ref"):
            ref_id = ref_tag.get("id")
            rel_attr = ref_tag.get("rel")
            # BeautifulSoup may return list for multi-valued attrs
            rel_type = rel_attr[0] if isinstance(rel_attr, list) else rel_attr
            if ref_id and rel_type:
                try:
                    refs.append(InlineRef(target_id=str(ref_id), rel_type=str(rel_type)))  # type: ignore[arg-type]  # rel_type validated by pydantic
                except Exception:  # noqa: BLE001 -- unknown rel_type must not break parsing; narrowed in Phase 6
                    logger.debug(f"Skipping inline ref with unknown rel_type {rel_type!r} (id={ref_id!r})")
        return refs

    @property
    def sort_key(self) -> tuple[int, float, float]:
        """Return a sort key based on page and bounding_box for ordering elements."""
        page = self.page or 0
        if self.bounding_box:
            return (page, self.bounding_box[1], self.bounding_box[0])  # (page, y, x)
        return (page, 0.0, 0.0)

    @computed_field
    @property
    def html(self) -> str:
        """
        HTML representation of the element content. Should be implemented by subclasses.

        Should include the outer tag (e.g. <p> for paragraphs, <hN> for headings, etc.).
        Footnotes should be represented as <p id='footnote-{id}'>[{number}] {text}</p>.
        Images referenced inline use <ref id='{id}' rel='image'/> tags.
        Lists should be represented as <ul>, <ol>, or <dl> with appropriate child elements.
        """
        raise NotImplementedError

    @html.setter
    def html(self, value: str) -> None:
        """
        HTML representation of the element content. Should be implemented by subclasses.
        This setter should be used for things like updating the text content (e.g. for footnotes parsing)

        Should include the outer tag (e.g. <p> for paragraphs, <hN> for headings, etc.).
        Footnotes should be represented as <p id='footnote-{id}'>[{number}] {text}</p>.
        Images referenced inline use <ref id='{id}' rel='image'/> tags.
        Lists should be represented as <ul> or <ol> with list items as <li>.
        """
        for k, v in type(self)._fields_from_html(value).items():
            setattr(self, k, v)

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:  # pyright: ignore[reportUnusedParameter]  # overridden by subclasses
        """Parse HTML and return a dict of field values for construction.
        Subclasses should override this to enable from_html / from_markdown construction."""
        raise NotImplementedError(f"{cls.__name__} does not support construction from HTML")

    @model_validator(mode="before")
    @classmethod
    def _parse_html_input(cls, data):
        if isinstance(data, dict) and "html" in data:
            try:
                parsed = cls._fields_from_html(data.pop("html"))
                merged = data | parsed
                data = {**merged}
            except NotImplementedError:
                pass
        return data

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        validate_metadata_dict(self.metadata)
        return self

    @classmethod
    def from_html(cls, html: str, **kwargs) -> Self:
        """Create an element from an HTML string."""
        return cls(html=html, **kwargs)  # type: ignore[call-arg]  # 'html' consumed by _parse_html_input validator

    @classmethod
    def from_markdown(cls, markdown_text: str, page: int = 0, **kwargs) -> Self:
        """Create a BaseElement from markdown text.
        Expects img links in markdown format (e.g. ![alt text](document_image/id/{id})).
        Embedded footnotes are expected in markdown format (e.g. [^footnote-<footnote-id>]).
        """
        html = convert_text(markdown_text, "html", format="md").strip()
        return cls(html=html, page=page, **kwargs)  # type: ignore[call-arg]  # 'html' consumed by _parse_html_input validator

    @property
    def footnote_ids(self) -> list[str]:
        """IDs of Footnote elements referenced inline, derived from <ref rel="footnote"/> tags in HTML."""
        return [ref.target_id for ref in self.inline_refs if ref.rel_type == "footnote"]

    @property
    def image_ids(self) -> list[str]:
        """IDs of Image elements referenced inline, derived from <ref rel="image"/> tags in HTML."""
        return [ref.target_id for ref in self.inline_refs if ref.rel_type == "image"]

    @property
    def text(self) -> str:
        """Plain text content of the element, derived from html via BeautifulSoup get_text()."""
        return BeautifulSoup(self.html, "html.parser").get_text().strip("\n -")

    @property
    def html_tag(self) -> Tag:
        """BeautifulSoup Tag representation of the element. Parses the html property."""
        tag = BeautifulSoup(self.html, "html.parser")
        return tag.contents[0] if tag.contents else Tag(name="div")  # type: ignore[return-value]  # first content is a Tag

    @property
    def markdown(self) -> str:
        """Markdown representation of the element. Converts HTML to markdown using markdownify."""
        return md(self.html).rstrip("\n-").strip()


E = TypeVar("E", bound=BaseElement)


class Heading(BaseElement):
    element_type: Literal[ElementTypeEnum.HEADING] = Field(default=ElementTypeEnum.HEADING)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html_content: str = Field(..., description="Full HTML of the heading element including the outer <hN> tag")

    @model_validator(mode="before")
    @classmethod
    def _from_innerhtml_level(cls, data):
        if isinstance(data, dict) and "html_content" not in data and "innerhtml" in data:
            innerhtml = data.pop("innerhtml")
            level = data.pop("level", 1)
            if level == 0:
                raise ValueError("Heading level 0 (h0) is not allowed. Use level 1-6.")
            data["html_content"] = f"<h{level}>{innerhtml}</h{level}>"
        return data

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find(re.compile(r"h[1-6]"))
        if tag:
            inner = tag.decode_contents(formatter="html").strip()
            level = int(tag.name[1])
            attrs_html = "".join(
                f' {k}="{v if not isinstance(v, list) else " ".join(v)}"' for k, v in tag.attrs.items()
            )
            return {"html_content": f"<h{level}{attrs_html}>{inner}</h{level}>"}
        return {"html_content": f"<h1>{soup.get_text().strip()}</h1>"}

    @property
    def html(self) -> str:
        return self.html_content

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter

    @property
    def level(self) -> int:
        tag = BeautifulSoup(self.html_content, "html.parser").find(re.compile(r"h[1-6]"))
        return int(tag.name[1]) if tag else 1

    @level.setter
    def level(self, new_level: int) -> None:
        if new_level == 0:
            raise ValueError("Heading level 0 (h0) is not allowed. Use level 1-6.")
        soup = BeautifulSoup(self.html_content, "html.parser")
        tag = soup.find(re.compile(r"h[1-6]"))
        if tag:
            tag.name = f"h{new_level}"
            self.html_content = str(tag)
        else:
            self.html_content = f"<h{new_level}>{self.html_content}</h{new_level}>"

    @property
    def innerhtml(self) -> str:
        tag = BeautifulSoup(self.html_content, "html.parser").find(re.compile(r"h[1-6]"))
        return tag.decode_contents(formatter="html").strip() if tag else self.html_content


class Paragraph(BaseElement):
    element_type: Literal[ElementTypeEnum.PARAGRAPH] = Field(default=ElementTypeEnum.PARAGRAPH)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html_content: str = ""

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        return {"html_content": html.strip()}

    @property
    def html(self) -> str:
        return self.html_content

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter


class DocumentList(BaseElement):
    element_type: Literal[ElementTypeEnum.DOCUMENT_LIST] = Field(default=ElementTypeEnum.DOCUMENT_LIST)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html_content: str = Field(..., description="HTML content of the list. Should be a <ul>, <ol>, or <dl> element.")

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        return {"html_content": html.strip()}

    @property
    def html(self) -> str:
        """Backward-compatible read alias for html."""
        return self.html_content

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter


class Table(BaseElement):
    element_type: Literal[ElementTypeEnum.TABLE] = Field(default=ElementTypeEnum.TABLE)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html_content: str = Field(..., description="HTML content of the table. Should be a <table> element.")

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        return {"html_content": html.strip()}

    @property
    def html(self) -> str:
        return self.html_content

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter


class Image(BaseElement):
    element_type: Literal[ElementTypeEnum.IMAGE] = Field(default=ElementTypeEnum.IMAGE)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    image: str | None = Field(
        default=None,
        description="Base64 representation of this image. None when no image data is available.",
    )
    image_type: str = Field(default="jpeg", description="Type of image. E.g. jpeg, png, etc.")
    width: int | None = Field(default=None, description="Width of the image in pixels.")
    height: int | None = Field(default=None, description="Height of image in pixels.")
    alt: str | None = Field(default=None, description="Alt text of image content")
    text_representation: str | None = Field(
        default=None,
        description="Text representation of the element, usually generated by a vision-capable-llm. \
                                                E.g. html code if the element content is an image of a table, \
                                                mermaid code if the element content is an image of a diagram, \
                                                formatted text if the element content is some image of a text passage. \
                                                None if image content cannot be represented as text.",
    )

    @property
    def src(self) -> str:
        """Returns a data URI when base64 data is available, or an empty string."""
        if self.image is not None:
            return f"data:image/{self.image_type};base64,{self.image}"
        return ""

    @property
    def placeholder_html(self) -> str:
        """HTML snippet that references this image inline via a <ref> tag.

        Embed this in a parent element (paragraph, table, list) to declare an
        inline image reference.  The renderer resolves the <ref> tag to the
        rendered image content.
        """
        return f'<ref id="{self.id}" rel="image"/>'

    @property
    def html(self) -> str:
        src = self.src
        return (
            f'<img src="{src}"'
            + (f' alt="{self.alt}"' if self.alt else "")
            + (f' width="{self.width}"' if self.width else "")
            + (f' height="{self.height}"' if self.height else "")
            + "/>"
        )

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        img_tag = soup.find("img")
        if not img_tag:
            return {}
        src = img_tag.get("src", "")
        data_pattern = re.compile(r"data:image/(?P<image_type>[^;]+);base64,(?P<image_data>.+)")
        data_match = data_pattern.match(src)  # type: ignore[arg-type]  # src is the str 'src' attribute
        data = {}
        if data_match:
            width = img_tag.get("width")
            height = img_tag.get("height")
            data = {
                "image": data_match.group("image_data"),
                "image_type": data_match.group("image_type"),
                "alt": img_tag.get("alt", None),
                "width": int(width) if width else None,  # type: ignore[arg-type]  # width attr is str
                "height": int(height) if height else None,  # type: ignore[arg-type]  # height attr is str
            }
        return {k: v for k, v in data.items() if v is not None}


class RawText(BaseElement):
    element_type: Literal[ElementTypeEnum.RAW_TEXT] = Field(default=ElementTypeEnum.RAW_TEXT)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    innerhtml: str = Field(default="", description="Inner HTML content of the element (without outer tag)")

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        # Find the first element tag and extract its inner contents
        first_tag = soup.find()
        if first_tag and hasattr(first_tag, "decode_contents"):
            return {"innerhtml": first_tag.decode_contents(formatter="html").strip()}
        return {"innerhtml": soup.decode_contents(formatter="html").strip()}

    @property
    def html(self) -> str:
        return f"<div>{self.innerhtml}</div>"

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter


class Footnote(BaseElement):
    """Represents a footnote definition in the document."""

    element_type: Literal[ElementTypeEnum.FOOTNOTE] = Field(default=ElementTypeEnum.FOOTNOTE)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    number: int = Field(..., description="The footnote number as it appears in the document")
    innerhtml: str = Field(..., description="The footnote text content (without number prefix)")

    @classmethod
    def _fields_from_html(cls, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        aside = soup.find("aside", class_="footnote")
        if aside:
            try:
                number = int(aside.get("data-number", 0))  # type: ignore[union-attr,arg-type]  # find yields a Tag; attr is str
            except (ValueError, TypeError):
                number = 0
            return {"number": number, "innerhtml": aside.decode_contents().strip()}
        # Fallback: [N] text pattern (legacy or external HTML)
        raw = soup.get_text().strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)", raw, re.DOTALL)
        if match:
            return {"number": int(match.group(1)), "innerhtml": match.group(2)}
        return {"innerhtml": raw.strip()}

    @property
    def placeholder_html(self) -> str:
        """HTML snippet that references this footnote inline via a <ref> tag."""
        return f'<ref id="{self.id}" rel="footnote"/>'

    @property
    def html(self) -> str:
        return f'<aside class="footnote" data-number="{self.number}" id="footnote-{self.id}">{self.innerhtml}</aside>'

    @html.setter
    def html(self, value: str) -> None:
        BaseElement.html.fset(self, value)  # type: ignore[misc]  # fset defined via @html.setter

    @property
    def text(self) -> str:
        return BeautifulSoup(self.innerhtml, "html.parser").get_text().strip()


ElementType = Annotated[
    Heading | Paragraph | Table | DocumentList | Image | RawText | Footnote, Field(discriminator="element_type")
]


class Document(BaseModel, Generic[TMetadata]):
    """A structured representation of a document containing various elements.

    Document serves as the primary container for parsed content and supports:
    - Element-based content (headings, paragraphs, tables, images, etc.)
    - Cross-document relationships via external_refs
    - Parser provenance tracking via parser field

    The optional type parameter ``TMetadata`` binds ``metadata`` to a
    user-declared :class:`~ragdoc.metadata.BaseMetadata` subclass.  Parsers
    always produce ``Document[BaseMetadata]`` (they set only ``filename``).
    The cast to ``Document[TMetadata]`` happens inside ``DocumentPipeline``
    *after* all processors have run and populated their fields.

    Bare usage (``Document``, no type argument) is fully valid and treats
    ``metadata`` as :class:`~ragdoc.metadata.BaseMetadata`.

    Attributes:
        id: Unique identifier for the document
        source_path: Full path to the source file. Set by parsers.
        title: Document title, usually from main heading or detected by TitleDetectionProcessor
        elements: List of content elements in document order. E.g. headings, paragraphs, tables, images, footnotes, lists etc.
        metadata: Metadata dict. ``filename`` (set by parsers) and ``split_sequence`` /
            ``split_total`` (set by ``split_document``) are framework-written keys defined in
            :class:`~ragdoc.metadata.BaseMetadata`. Add your own keys in a subclass.
        external_refs: References to external documents (parent, siblings, etc.)
        parser: Name of the parser that created this document (for processor behavior)
        parser_version: Version of the parser (optional)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the document")
    source_path: str = Field(default="", description="Full path to the source file")
    source_id: str | None = Field(
        default=None,
        description=(
            "Sync identity key for this document, derived from its source Path by "
            "DocumentPipeline.source_id_fn. Set before chunking; propagated onto each Chunk. "
            "None when the document was not produced through a sync pipeline."
        ),
    )
    source_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hex digest of the original source file bytes, set by "
            "DocumentPipeline.hash_fn before chunking. True file provenance (not content). "
            "None when not produced through a sync pipeline. For content-based change "
            "detection use the content_hash() method."
        ),
    )
    title: str | None = Field(default=None, description="Title of the document, usually derived from the main heading")
    elements: list[ElementType] = Field(default_factory=list, description="Elements in this document")
    metadata: SkipValidation[TMetadata] = Field(default_factory=dict)  # type: ignore[assignment]
    external_refs: list[ExternalRef] = Field(
        default_factory=list,
        description="References to external documents or elements (parents, siblings, etc.). "
        "Used for hierarchical rendering and navigation, NOT for inline substitution.",
    )
    parser: str | None = Field(
        default=None,
        description="Name of the parser that created this document: 'mineru', 'azure_di', 'html', 'pandoc', etc. "
        "Used by processors to adjust behavior based on parser reliability.",
    )
    parser_version: str | None = Field(default=None, description="Version of the parser that created this document.")

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        validate_metadata_dict(self.metadata)  # type: ignore[arg-type]
        return self

    @property
    def parent_ids(self) -> list[str]:
        """Get IDs of parent documents from external_refs."""
        return [ref.target_id for ref in self.external_refs if ref.rel_type == "external-parent"]

    @property
    def child_ids(self) -> list[str]:
        """Get IDs of child documents from external_refs."""
        return [ref.target_id for ref in self.external_refs if ref.rel_type == "external-child"]

    @property
    def empty(self) -> bool:
        return len(self.elements) == 0

    def _filter_elements_by_type(self, element_type: type[E]) -> list[E]:
        return [e for e in self.elements if isinstance(e, element_type)]

    @property
    def headings(self) -> list[Heading]:
        return self._filter_elements_by_type(Heading)

    @property
    def paragraphs(self) -> list[Paragraph]:
        return self._filter_elements_by_type(Paragraph)

    @property
    def tables(self) -> list[Table]:
        return self._filter_elements_by_type(Table)

    @property
    def lists(self) -> list[DocumentList]:
        return self._filter_elements_by_type(DocumentList)

    @property
    def images(self) -> list[Image]:
        return self._filter_elements_by_type(Image)

    @property
    def raw_texts(self) -> list[RawText]:
        return self._filter_elements_by_type(RawText)

    @property
    def footnotes(self) -> list[Footnote]:
        return self._filter_elements_by_type(Footnote)

    def get_element(self, element_id: str) -> ElementType | None:
        """
        Look up an element by its ID.

        This is used by the renderer to resolve inline references (<ref id="..."/>)
        to their actual elements for inline substitution.

        Args:
            element_id: The unique ID of the element to find

        Returns:
            The element with the given ID, or None if not found
        """
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    @property
    def unnamed(self) -> bool:
        return len(self.headings) == 0 and self.title is None

    @property
    def image_idx(self) -> dict[str, Image]:
        return {image.id: image for image in self.images}

    @property
    def footnote_idx(self) -> dict[str, Footnote]:
        return {footnote.id: footnote for footnote in self.footnotes}

    @property
    def main_heading(self) -> Heading | None:
        if not self.headings:
            return None
        sorted_headings = sorted(self.headings, key=lambda h: (h.level, h.sort_key))
        return sorted_headings[0]

    @property
    def sorted_headings(self) -> list[Heading]:
        """Return headings sorted by level then position."""
        return sorted(self.headings, key=lambda h: (h.level, h.sort_key))

    @property
    def name(self) -> str | None:
        if self.main_heading:
            return self.main_heading.text
        return self.title

    @property
    def level(self) -> int | None:
        return self.main_heading.level if self.main_heading else None

    @property
    def orphaned_footnotes(self) -> list[Footnote]:
        """Footnotes that are not referenced from any element's text."""
        referenced_ids: set[str] = set()
        for e in self.elements:
            referenced_ids.update(e.footnote_ids)
        return [f for f in self.footnotes if f.id not in referenced_ids]

    def content_hash(self, renderer: Renderer | None = None) -> str:
        """Content-stable SHA-256 hash of this document.

        Uses the default prompt renderer (MARKDOWN + render_for_prompt) to produce
        a canonical string, then returns its hex digest.  The same document content
        always yields the same hash, making this safe as a chunk ID for idempotent
        vector-store upserts.

        Subclass Document and override this method to customise the hash strategy —
        for example to incorporate file-path provenance or to switch to a different
        renderer::

            class SourceDocument(Document):
                source_path: str = ""

                def content_hash(self, renderer=None) -> str:
                    base = super().content_hash(renderer)
                    return hashlib.sha256(f"{self.source_path}:{base}".encode()).hexdigest()

        Args:
            renderer: Renderer to use for hashing. Defaults to
                ``Renderer(OutputFormat.MARKDOWN, render_for_prompt)``.

        Returns:
            64-character lowercase hex string (SHA-256 digest).
        """
        import hashlib

        if renderer is None:
            from ragdoc.rendering import OutputFormat, Renderer, render_for_prompt

            renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
        content = renderer.render(self)
        return hashlib.sha256(content.encode()).hexdigest()

    def __or__(self, other: Self) -> Self:
        title = self.title or other.title
        if self.title and other.title:
            new_heading = [Heading(innerhtml=other.title, level=1)]  # type: ignore[call-arg]  # consumed by _from_innerhtml_level validator
        else:
            new_heading = []
        return Document(  # type: ignore[return-value]
            elements=self.elements + new_heading + other.elements,
            title=title,
            source_path=self.source_path or other.source_path,
            metadata=other.metadata | self.metadata,
        )


def join_documents(documents: list[Document]) -> Document:
    if len(documents) == 0:
        return Document()
    return reduce(or_, documents)


DocumentPrimaryElement = Heading | Paragraph | Table | DocumentList | RawText  # elements that get rendered to md
DocumentSecondaryElement = (
    Image | Footnote
)  # elements referenced within primary elements (images via src, footnotes via anchor links)
