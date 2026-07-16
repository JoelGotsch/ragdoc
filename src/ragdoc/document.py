from __future__ import annotations

import hashlib
import html as html_stdlib
import json
import logging
import re
import uuid
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Generic, Literal, TypeVar, cast

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation, field_validator, model_validator
from pypandoc import convert_text
from typing_extensions import Self

from ragdoc.metadata import BaseMetadata, MetadataDict, TMetadata, validate_metadata_dict

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


_INLINE_REL_TYPES = frozenset({"image", "footnote", "table", "figure"})
"""Valid ``rel`` values for inline ``<ref/>`` tags (mirrors :class:`InlineRef`.rel_type)."""

_warned_rel_types: set[str] = set()
"""Unknown inline-ref ``rel`` values already warned about — one warning per value per process."""

_HEADING_TAG_PATTERN = re.compile(r"h[1-6]")
"""Matches heading tag names ``h1``–``h6`` (bs4 name matcher)."""


class BaseElement(BaseModel):
    """Base class for all document elements.

    Every concrete subclass exposes ``html`` — the full HTML of the element including its
    outer tag.  Heading/Paragraph/Table/DocumentList/RawText **store** ``html`` as a plain
    field with a normalizing ``field_validator``; Image and Footnote **derive** ``html`` from
    their structured fields (property + setter), because storing it would duplicate base64
    image data or denormalize the embedded element id.

    ``validate_assignment=True``: assigning ``element.html = value`` (or any field) re-runs
    the field validators, so assignments normalize exactly like construction.
    """

    model_config = ConfigDict(validate_assignment=True)

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

    if TYPE_CHECKING:
        # Typing-only declaration: every concrete subclass provides `html` either as a stored
        # field (Heading/Paragraph/Table/DocumentList/RawText) or as a property with a setter
        # (Image/Footnote). Not a runtime annotation, so Pydantic creates no field here and no
        # "shadows an attribute in parent" warning fires in subclasses. The typing-only default
        # keeps `html` optional in the synthesized constructors of the derived-html subclasses.
        html: str = ""

    _soup: BeautifulSoup | None = PrivateAttr(default=None)
    _soup_source: str | None = PrivateAttr(default=None)

    def _parsed(self) -> BeautifulSoup:
        """Cached BeautifulSoup parse of ``self.html``.

        Cache validity is keyed on the *identity* of the html string object: any assignment
        binds a new ``str``, so ``self._soup_source is not html`` detects staleness without
        assignment hooks (``el.html = el.html`` keeps the cache warm — strings are immutable).
        Never hand the cached soup to callers; read-only derivations only.
        """
        html_str = self.html
        if self._soup is None or self._soup_source is not html_str:
            self._soup = BeautifulSoup(html_str, "html.parser")
            self._soup_source = html_str
        return self._soup

    @property
    def inline_refs(self) -> list[InlineRef]:
        """Inline references parsed from <ref id="..." rel="..."/> tags in this element's HTML.

        The element's HTML is the single source of truth. To add an inline ref,
        embed a <ref id="<target-id>" rel="<rel-type>"/> tag in the HTML content.
        Refs with an unknown ``rel`` value are skipped with a warning (logged once per value).
        """
        refs: list[InlineRef] = []
        for ref_tag in self._parsed().find_all("ref"):
            ref_id = ref_tag.get("id")
            rel_attr = ref_tag.get("rel")
            # BeautifulSoup may return list for multi-valued attrs
            rel_type = rel_attr[0] if isinstance(rel_attr, list) else rel_attr
            if not (ref_id and rel_type):
                continue
            rel_str = str(rel_type)
            if rel_str not in _INLINE_REL_TYPES:
                if rel_str not in _warned_rel_types:
                    _warned_rel_types.add(rel_str)
                    logger.warning("ignoring inline ref with unknown rel type %r", rel_str)
                continue
            refs.append(InlineRef(target_id=str(ref_id), rel_type=rel_str))  # type: ignore[arg-type]  # rel_str checked against _INLINE_REL_TYPES
        return refs

    @property
    def sort_key(self) -> tuple[int, float, float]:
        """Return a sort key based on page and bounding_box for ordering elements."""
        page = self.page or 0
        if self.bounding_box:
            return (page, self.bounding_box[1], self.bounding_box[0])  # (page, y, x)
        return (page, 0.0, 0.0)

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        validate_metadata_dict(self.metadata)
        return self

    @classmethod
    def from_markdown(cls, markdown_text: str, *, page: int | None = 0) -> Self:
        """Create an element from markdown text (converted to HTML via pandoc).

        Expects img links in markdown format (e.g. ![alt text](document_image/id/{id})).
        Embedded footnotes are expected in markdown format (e.g. [^footnote-<footnote-id>]).
        Set further fields (``id``, ``metadata``, ``bounding_box``) by assignment afterwards.
        """
        html_str = convert_text(markdown_text, "html", format="md").strip()
        return cls(html=html_str, page=page)  # pyright: ignore[reportCallIssue]  # concrete subclasses default element_type

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
        return self._parsed().get_text().strip("\n -")

    @property
    def html_tag(self) -> Tag:
        """BeautifulSoup Tag representation of the element.

        Deliberately a **fresh parse per call** (never the cached soup): the returned ``Tag``
        is live and callers may mutate it — handing out the cached soup would let callers
        desync the cache from the stored html.
        """
        tag = BeautifulSoup(self.html, "html.parser")
        return tag.contents[0] if tag.contents else Tag(name="div")  # type: ignore[return-value]  # first content is a Tag

    @property
    def markdown(self) -> str:
        """Markdown representation of the element. Converts HTML to markdown using markdownify."""
        return md(self.html).rstrip("\n-").strip()


E = TypeVar("E", bound=BaseElement)


class Heading(BaseElement):
    element_type: Literal[ElementTypeEnum.HEADING] = Field(default=ElementTypeEnum.HEADING)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html: str = Field(..., description="Full HTML of the heading including the outer <h1>-<h6> tag.")  # pyright: ignore[reportGeneralTypeIssues]  # required override of the typing-only base declaration

    @field_validator("html", mode="after")
    @classmethod
    def _normalize_heading_html(cls, value: str) -> str:
        """Normalize to exactly the first <h1>-<h6> tag, re-serialized by bs4.

        ``tag.decode(formatter="html")`` escapes attribute values, replacing the old manual
        attribute rebuild and its escaping bug. When no hN tag is found the text content is
        escaped and wrapped in ``<h1>``. Out-of-range levels are unrepresentable here (the
        pattern is h[1-6]); the explicit 1-6 range guard lives in the ``level`` setter.
        """
        soup = BeautifulSoup(value, "html.parser")
        tag = soup.find(_HEADING_TAG_PATTERN)
        if isinstance(tag, Tag):
            return tag.decode(formatter="html").strip()
        return f"<h1>{html_stdlib.escape(soup.get_text().strip())}</h1>"

    @property
    def level(self) -> int:
        """Heading level (1-6) parsed from the stored html; 1 when no hN tag is present."""
        tag = self._parsed().find(_HEADING_TAG_PATTERN)
        return int(tag.name[1]) if isinstance(tag, Tag) else 1

    @level.setter
    def level(self, new_level: int) -> None:
        """Rebuild html with the heading tag renamed to ``h{new_level}``.

        Raises ValueError for any level outside 1-6: a tag like ``<h7>`` would be missed by
        the normalizing validator's h[1-6] pattern and silently destroy inline content (the
        text-escape fallback strips e.g. ``<ref/>`` tags). Assigning ``self.html``
        re-validates and invalidates the soup cache.
        """
        if not 1 <= new_level <= 6:
            raise ValueError(f"Heading level {new_level} is not allowed. Use level 1-6.")
        soup = BeautifulSoup(self.html, "html.parser")  # fresh parse — the tree is mutated below
        tag = soup.find(_HEADING_TAG_PATTERN)
        if isinstance(tag, Tag):
            tag.name = f"h{new_level}"
            self.html = str(tag)
        else:
            self.html = f"<h{new_level}>{self.html}</h{new_level}>"

    @property
    def innerhtml(self) -> str:
        """Inner HTML of the heading tag (without the outer <hN>). Read-only convenience."""
        tag = self._parsed().find(_HEADING_TAG_PATTERN)
        return tag.decode_contents(formatter="html").strip() if isinstance(tag, Tag) else self.html


class Paragraph(BaseElement):
    element_type: Literal[ElementTypeEnum.PARAGRAPH] = Field(default=ElementTypeEnum.PARAGRAPH)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html: str = Field(default="", description="Full HTML of the paragraph including the outer tag (typically <p>).")

    @field_validator("html", mode="after")
    @classmethod
    def _normalize_html(cls, value: str) -> str:
        return value.strip()


class DocumentList(BaseElement):
    element_type: Literal[ElementTypeEnum.DOCUMENT_LIST] = Field(default=ElementTypeEnum.DOCUMENT_LIST)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html: str = Field(..., description="Full HTML of the list. Expected outer tag: <ul>, <ol>, or <dl>.")  # pyright: ignore[reportGeneralTypeIssues]  # required override of the typing-only base declaration

    @field_validator("html", mode="after")
    @classmethod
    def _normalize_html(cls, value: str) -> str:
        return value.strip()


class Table(BaseElement):
    element_type: Literal[ElementTypeEnum.TABLE] = Field(default=ElementTypeEnum.TABLE)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html: str = Field(..., description="Full HTML of the table. Expected outer tag: <table>.")  # pyright: ignore[reportGeneralTypeIssues]  # required override of the typing-only base declaration

    @field_validator("html", mode="after")
    @classmethod
    def _normalize_html(cls, value: str) -> str:
        return value.strip()


_IMG_DATA_URI_PATTERN = re.compile(r"data:image/(?P<image_type>[^;]+);base64,(?P<image_data>.+)")
"""Matches an ``<img src>`` data URI, capturing the image type and base64 payload."""


def image_fields_from_html(html_str: str) -> dict[str, str | int | None]:
    """Parse an ``<img>`` tag into :class:`Image` field values.

    Only ``data:`` URIs yield image data; keys with ``None`` values are dropped so parsed
    values can serve as defaults that explicit kwargs override.  Returns ``{}`` when no
    ``<img>`` tag (or no data URI) is present.

    Args:
        html_str: HTML containing an ``<img>`` tag (e.g. ``<img src="data:image/png;base64,..."/>``).

    Returns:
        Field values for ``image``, ``image_type``, ``alt``, ``width``, ``height`` (subset).
    """
    soup = BeautifulSoup(html_str, "html.parser")
    img_tag = soup.find("img")
    if not isinstance(img_tag, Tag):
        return {}
    src = img_tag.get("src", "")
    data_match = _IMG_DATA_URI_PATTERN.match(src)  # type: ignore[arg-type]  # src is the str 'src' attribute
    data: dict[str, str | int | None] = {}
    if data_match:
        width = img_tag.get("width")
        height = img_tag.get("height")
        alt = img_tag.get("alt", None)
        data = {
            "image": data_match.group("image_data"),
            "image_type": data_match.group("image_type"),
            "alt": str(alt) if alt is not None else None,
            "width": int(str(width)) if width else None,
            "height": int(str(height)) if height else None,
        }
    return {k: v for k, v in data.items() if v is not None}


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

    @model_validator(mode="before")
    @classmethod
    def _from_html_kwarg(cls, data: object) -> object:
        """Allow ``Image(html='<img .../>')``: parsed fields are defaults, explicit kwargs win."""
        if isinstance(data, dict) and "html" in data:
            html_value = data.pop("html")
            if isinstance(html_value, str):
                data = image_fields_from_html(html_value) | data
        return data

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
        """Derived ``<img>`` HTML — a projection of the structured image fields (never stored)."""
        return (
            f'<img src="{self.src}"'
            + (f' alt="{html_stdlib.escape(self.alt, quote=True)}"' if self.alt else "")
            + (f' width="{self.width}"' if self.width else "")
            + (f' height="{self.height}"' if self.height else "")
            + "/>"
        )

    @html.setter
    def html(self, value: str) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]  # property implements the typing-only base declaration
        """Re-derive **all** derivable fields from an ``<img>`` tag with a ``data:`` URI src.

        Attributes absent from the new tag reset to ``None`` — assignment never leaves stale
        values from the previous tag. A missing ``<img>`` tag or a non-``data:`` src raises
        ``ValueError``: :class:`Image` stores base64 content, so such an assignment cannot be
        represented and silently ignoring it would violate ``validate_assignment`` semantics.
        (Construction via ``Image(html=...)`` keeps its lenient parsed-fields-as-defaults
        behavior — see :func:`image_fields_from_html`.)
        """
        soup = BeautifulSoup(value, "html.parser")
        img_tag = soup.find("img")
        if not isinstance(img_tag, Tag):
            raise ValueError("Image.html assignment requires an <img> tag; none found in the assigned HTML.")
        src = str(img_tag.get("src") or "")
        data_match = _IMG_DATA_URI_PATTERN.match(src)
        if data_match is None:
            raise ValueError(
                "Image.html assignment requires a data: URI src (Image stores base64 content); "
                f"got src={src[:100]!r}. Set structured fields directly for non-embedded images."
            )
        width = img_tag.get("width")
        height = img_tag.get("height")
        alt = img_tag.get("alt")
        self.image = data_match.group("image_data")
        self.image_type = data_match.group("image_type")
        self.alt = str(alt) if alt is not None else None
        self.width = int(str(width)) if width else None
        self.height = int(str(height)) if height else None


class RawText(BaseElement):
    element_type: Literal[ElementTypeEnum.RAW_TEXT] = Field(default=ElementTypeEnum.RAW_TEXT)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    html: str = Field(default="<div></div>", description="Full HTML: a single outer <div> wrapping the raw content.")

    @field_validator("html", mode="after")
    @classmethod
    def _normalize_rawtext_html(cls, value: str) -> str:
        """Normalize to a single outer ``<div>``.

        A single root ``<div>`` passes through unchanged; any other single root tag is
        unwrapped and its contents re-wrapped in ``<div>``; tag-less (or mixed) content is
        wrapped in ``<div>`` verbatim.
        """
        stripped = value.strip()
        soup = BeautifulSoup(stripped, "html.parser")
        roots = [node for node in soup.contents if not (isinstance(node, str) and not node.strip())]
        if len(roots) == 1 and isinstance(roots[0], Tag):
            tag = roots[0]
            if tag.name == "div":
                return stripped
            return f"<div>{tag.decode_contents(formatter='html').strip()}</div>"
        return f"<div>{stripped}</div>"

    @property
    def innerhtml(self) -> str:
        """Inner HTML of the outer <div> (the raw content). Read-only convenience."""
        tag = self._parsed().find("div")
        return tag.decode_contents(formatter="html") if isinstance(tag, Tag) else self.html


def footnote_fields_from_html(html_str: str) -> dict[str, int | str]:
    """Parse footnote HTML into :class:`Footnote` field values.

    Accepts the canonical ``<aside class="footnote" data-number="N">text</aside>`` form, the
    legacy ``[N] text`` pattern, or bare text (``number`` omitted — the Footnote constructor
    will then require it explicitly).

    Args:
        html_str: Footnote HTML in one of the accepted forms.

    Returns:
        Field values for ``number`` and/or ``innerhtml``.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    aside = soup.find("aside", class_="footnote")
    if isinstance(aside, Tag):
        try:
            number = int(str(aside.get("data-number", "0")))
        except (ValueError, TypeError):
            number = 0
        return {"number": number, "innerhtml": aside.decode_contents().strip()}
    # Fallback: [N] text pattern (legacy or external HTML)
    raw = soup.get_text().strip()
    match = re.match(r"^\[(\d+)\]\s*(.*)", raw, re.DOTALL)
    if match:
        return {"number": int(match.group(1)), "innerhtml": match.group(2)}
    return {"innerhtml": raw.strip()}


class Footnote(BaseElement):
    """Represents a footnote definition in the document."""

    element_type: Literal[ElementTypeEnum.FOOTNOTE] = Field(default=ElementTypeEnum.FOOTNOTE)  # pyright: ignore[reportIncompatibleVariableOverride]  # pydantic discriminator narrowing
    number: int = Field(..., description="The footnote number as it appears in the document")
    innerhtml: str = Field(..., description="The footnote text content (without number prefix)")

    @model_validator(mode="before")
    @classmethod
    def _from_html_kwarg(cls, data: object) -> object:
        """Allow ``Footnote(html='<aside class="footnote" .../>')``: parsed fields are defaults."""
        if isinstance(data, dict) and "html" in data:
            html_value = data.pop("html")
            if isinstance(html_value, str):
                data = footnote_fields_from_html(html_value) | data
        return data

    @property
    def placeholder_html(self) -> str:
        """HTML snippet that references this footnote inline via a <ref> tag."""
        return f'<ref id="{self.id}" rel="footnote"/>'

    @property
    def html(self) -> str:
        """Derived footnote HTML — embeds the live element id, so it is never stored."""
        return f'<aside class="footnote" data-number="{self.number}" id="footnote-{self.id}">{self.innerhtml}</aside>'

    @html.setter
    def html(self, value: str) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]  # property implements the typing-only base declaration
        for k, v in footnote_fields_from_html(value).items():
            setattr(self, k, v)

    @property
    def text(self) -> str:
        """Plain text of the footnote content (fresh parse of ``innerhtml`` — not hot)."""
        return BeautifulSoup(self.innerhtml, "html.parser").get_text().strip()


ElementType = Annotated[
    Heading | Paragraph | Table | DocumentList | Image | RawText | Footnote, Field(discriminator="element_type")
]


# ---------------------------------------------------------------------------
# Canonical content hashing (Document.content_hash)
# ---------------------------------------------------------------------------

_CONTENT_HASH_DOMAIN = b"ragdoc.content_hash.v1\x00"
"""Domain-separation prefix for :meth:`Document.content_hash`.

Any change to the canonical payload (field participation, normalization, JSON encoding) MUST
bump this version tag. Each bump changes every stored ``content_hash`` once — i.e. one full
corpus re-chunk/re-embed on the next Boundary-2 sync — and requires a CHANGELOG migration note.
"""

_REF_TAG_PATTERN = re.compile(r"<ref\b[^>]*/?>")
_REF_ID_ATTR_PATTERN = re.compile(r"""(\bid\s*=\s*)(["'])(.*?)\2""")


def normalize_ref_ids(html: str, ref_ordinals: dict[str, int]) -> str:
    """Rewrite the ``id`` attribute of every ``<ref .../>`` tag to a parse-stable form.

    Element ids are uuid4s minted per parse, and they are embedded inside stored HTML as
    ``<ref id="{uuid}" rel="..."/>`` tags. Hashing the raw HTML would therefore make
    :meth:`Document.content_hash` unstable across re-parses of an unchanged file. This function
    replaces each ref's ``id`` value with the target element's document-order ordinal
    (``"#3"``), or the literal ``"unresolved"`` for dangling refs (targets not in
    *ref_ordinals*). The ``rel`` attribute is left untouched (it participates in the hash).
    Only ``<ref>`` tags are rewritten — ``id`` attributes on any other tag are content and
    pass through unchanged.

    Args:
        html: Element HTML possibly containing ``<ref id="..." rel="..."/>`` tags.
        ref_ordinals: Map of element id -> document-order ordinal (position in
            ``document.elements``), as built by :func:`document_content_payload`.

    Returns:
        The HTML with every ref-tag ``id`` normalized; all other content byte-identical.
    """

    def _rewrite_tag(tag_match: re.Match[str]) -> str:
        def _rewrite_id(id_match: re.Match[str]) -> str:
            ordinal = ref_ordinals.get(id_match.group(3))
            normalized = f"#{ordinal}" if ordinal is not None else "unresolved"
            quote = id_match.group(2)
            return f"{id_match.group(1)}{quote}{normalized}{quote}"

        return _REF_ID_ATTR_PATTERN.sub(_rewrite_id, tag_match.group(0))

    return _REF_TAG_PATTERN.sub(_rewrite_tag, html)


def element_content_payload(element: BaseElement, ref_ordinals: dict[str, int]) -> dict[str, str | int | None]:
    """Canonical hash payload for one element (content fields only).

    Exhaustive over the built-in element types; raises ``TypeError`` on any unknown
    :class:`BaseElement` subclass — never silently skips a variant (an unhashed element would
    silently break change detection).

    Per-type payloads:

    * ``Heading`` / ``Paragraph`` / ``Table`` / ``DocumentList`` / ``RawText`` —
      ``{"t": <element_type>, "html": normalize_ref_ids(element.html, ...)}``. The element
      type participates explicitly, so e.g. a ``RawText`` and a ``Paragraph`` with identical
      inner text hash differently.
    * ``Image`` — the six structured content fields (``image``, ``image_type``, ``alt``,
      ``text_representation``, ``width``, ``height``).
    * ``Footnote`` — ``number`` plus the normalized ``innerhtml``. Deliberately **not**
      ``element.html``, which embeds the volatile ``id="footnote-{uuid}"`` attribute.

    Excluded on every element: ``id``, ``metadata``, ``page``, ``bounding_box`` — identity,
    layout, and per-element metadata do not participate; content fields do.

    Args:
        element: The element to build a payload for.
        ref_ordinals: Element id -> document-order ordinal map for ref normalization.

    Returns:
        A JSON-serializable dict of the element's content fields.

    Raises:
        TypeError: If *element* is not one of the seven built-in element types.
    """
    if isinstance(element, Image):
        return {
            "t": element.element_type.value,
            "image": element.image,
            "image_type": element.image_type,
            "alt": element.alt,
            "text_representation": element.text_representation,
            "width": element.width,
            "height": element.height,
        }
    if isinstance(element, Footnote):
        return {
            "t": element.element_type.value,
            "number": element.number,
            "html": normalize_ref_ids(element.innerhtml, ref_ordinals),
        }
    if isinstance(element, (Heading, Paragraph, Table, DocumentList, RawText)):
        return {
            "t": element.element_type.value,
            "html": normalize_ref_ids(element.html, ref_ordinals),
        }
    raise TypeError(f"unhandled element type {type(element).__name__} in content hash")


def document_content_payload(document: Document) -> dict[str, object]:
    """Canonical hash payload for a document: ``{"title": ..., "elements": [...]}``.

    Only ``title`` and the ordered ``elements`` participate (see
    :meth:`Document.content_hash` for the full field-participation rationale). The ref-ordinal
    map is built once here from element document order.

    Args:
        document: The document to build a payload for.

    Returns:
        A JSON-serializable dict fed into the canonical-JSON hash.
    """
    ref_ordinals = {element.id: i for i, element in enumerate(document.elements)}
    return {
        "title": document.title,
        "elements": [element_content_payload(element, ref_ordinals) for element in document.elements],
    }


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

    def content_hash(self) -> str:
        """Content-stable SHA-256 hex digest of ``(title, elements)`` — pure Python, no rendering.

        SHA-256 of the version-tagged (``ragdoc.content_hash.v1``) canonical JSON of the
        document's content: ``title`` plus the ordered element payloads built by
        :func:`document_content_payload` / :func:`element_content_payload`. No pandoc, no
        renderer — the hash is stable across pandoc releases and cheap enough to recompute
        freely (no caching; a stale cache would be a silent change-detection bug).

        Field participation:

        * ``title`` — **yes** (a title edit must re-chunk: ``prompt_content`` renders it).
        * ``elements`` — **yes**, order-sensitive, per-type content payloads. Inline
          ``<ref id=...>`` uuids are normalized to document-order ordinals
          (:func:`normalize_ref_ids`), so a re-parse of an unchanged file — which mints fresh
          element ids — yields the same hash.
        * ``id``, ``source_path``, ``source_id``, ``source_hash``, ``parser``,
          ``parser_version``, ``external_refs`` — **no** (provenance, not content;
          ``source_hash`` is the *other* change token of the two-hash design).
        * ``metadata`` — **no**, on the document and on every element. Metadata exists for
          external consumers and would otherwise poison the hash with library-written keys
          (``split_sequence``/``split_total``). **Consequence:** editing only ``metadata`` in a
          stored Document does *not* trigger a re-chunk, so chunk metadata in a vector store can
          go stale until content changes. Override in a subclass for metadata-sensitive
          detection.
        * element ``page`` / ``bounding_box`` — **no** (layout moving an unchanged element to
          another page must not re-embed).

        The same content always yields the same hash, making this safe as a chunk ID for
        idempotent vector-store upserts. Subclass Document and override this method to customise
        the strategy — compose over ``super().content_hash()``::

            class SourceDocument(Document):
                def content_hash(self) -> str:
                    base = super().content_hash()
                    return hashlib.sha256(f"{self.source_path}:{base}".encode()).hexdigest()

        Returns:
            64-character lowercase hex string (SHA-256 digest).
        """
        payload = document_content_payload(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(_CONTENT_HASH_DOMAIN + canonical.encode("utf-8")).hexdigest()


MetadataMergePolicy = Literal["first", "second", "strict"]
"""Conflict policy for metadata keys present on both inputs of :func:`concat_documents`."""


def concat_documents(first: Document, second: Document, *, metadata_policy: MetadataMergePolicy = "first") -> Document:
    """Concatenate two documents into a new one. Inputs are never mutated.

    Not to be confused with :func:`ragdoc.merging.merge_documents`, which *aligns* two parses
    of the same source — this function appends ``second``'s elements after ``first``'s.

    Field policy:

    ========================= =====================================================================
    Field                     Result
    ========================= =====================================================================
    ``id``                    new uuid — the merged document is a new identity
    ``title``                 ``first.title or second.title``
    ``elements``              ``[*first.elements, *bridge, *second.elements]`` where *bridge* is a
                              single ``<h1>`` heading carrying ``second.title`` (HTML-escaped) iff
                              **both** titles are set, else empty. Always a fresh list — never
                              aliases either input's list.
    ``source_path``           ``first.source_path or second.source_path``
    ``source_id`` / ``source_hash``  ``None`` — merged content is new provenance; re-stamped by
                              the pipeline if synced
    ``parser`` / ``parser_version``  kept iff identical on both inputs, else ``None``
    ``external_refs``         ``[]`` — refs point at pre-merge identities
    ``metadata``              ``"first"`` (default): ``{**second.metadata, **first.metadata}``
                              (first wins on conflicts); ``"second"``: second wins; ``"strict"``:
                              raise ``ValueError`` naming the keys whose values differ. Always a
                              fresh dict.
    ========================= =====================================================================

    Args:
        first: Left document; wins title/source_path/metadata conflicts under the default policy.
        second: Right document.
        metadata_policy: How to resolve metadata keys present on both documents.

    Returns:
        A new :class:`Document` containing both documents' elements.

    Raises:
        ValueError: Under ``metadata_policy="strict"`` when a shared metadata key differs.
    """
    first_meta: MetadataDict = dict(first.metadata)
    second_meta: MetadataDict = dict(second.metadata)
    if metadata_policy == "strict":
        conflicts = sorted(key for key in first_meta.keys() & second_meta.keys() if first_meta[key] != second_meta[key])
        if conflicts:
            raise ValueError(f"concat_documents(metadata_policy='strict'): conflicting metadata keys {conflicts}")
        metadata = {**first_meta, **second_meta}
    elif metadata_policy == "second":
        metadata = {**first_meta, **second_meta}
    else:
        metadata = {**second_meta, **first_meta}

    bridge: list[ElementType] = (
        [Heading(html=f"<h1>{html_stdlib.escape(second.title)}</h1>")] if first.title and second.title else []
    )
    return Document(
        title=first.title or second.title,
        elements=[*first.elements, *bridge, *second.elements],
        source_path=first.source_path or second.source_path,
        parser=first.parser if first.parser == second.parser else None,
        parser_version=first.parser_version if first.parser_version == second.parser_version else None,
        metadata=cast("BaseMetadata", metadata),
    )


def join_documents(documents: list[Document], *, metadata_policy: MetadataMergePolicy = "first") -> Document:
    """Concatenate documents left-to-right via :func:`concat_documents`; an empty list yields ``Document()``."""
    if len(documents) == 0:
        return Document()
    return reduce(lambda a, b: concat_documents(a, b, metadata_policy=metadata_policy), documents)
