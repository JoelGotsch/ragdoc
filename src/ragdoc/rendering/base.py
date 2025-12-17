"""
Core rendering infrastructure.

This module provides the base classes and protocols for document rendering:
- OutputFormat: Enum of supported output formats
- RenderContext: Context passed to element renderers
- ExternalRefProvider: Protocol for cross-document reference resolution
- Renderer: Main renderer class with format conversion

Architecture:
-------------
Element renderers (singledispatch functions) ALWAYS output HTML. The Renderer
class handles:
1. Inline reference resolution (<ref id="..."/> → rendered element HTML)
2. Skipping elements that are only referenced inline
3. Format conversion (HTML → Markdown, plain text, etc.)

This separation ensures element renderers are simple and focused on content,
while the Renderer handles structural concerns.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from bs4 import BeautifulSoup
from pypandoc import convert_text

from ragdoc.metadata import MetadataValue

if TYPE_CHECKING:
    from ragdoc.document import BaseElement, Document


class OutputFormat(str, Enum):
    """
    Supported output formats for rendering.

    The renderer produces HTML internally and converts to the target format
    using pypandoc. HTML is the native format (no conversion needed).
    """

    HTML = "html"
    MARKDOWN = "md"
    GFM = "gfm"  # GitHub Flavored Markdown
    RST = "rst"  # reStructuredText
    PLAIN = "plain"  # Plain text


@runtime_checkable
class ExternalRefProvider(Protocol):
    """
    Protocol for resolving references to external documents or elements.

    This is used by the hierarchical renderer to look up parent documents
    and other external references. The protocol is dict-compatible,
    so a simple dict[str, Document] works as a provider.

    Design Rationale:
    -----------------
    ExternalRefProvider enables renderers to access content outside the
    current document for features like:
    - Breadcrumb generation (parent document titles)
    - Cross-document link resolution

    Siblings are computed on-the-fly: given a parent, its children (excluding
    the current document) are the siblings.

    The dict-compatible interface makes it easy to use in simple cases
    while allowing more sophisticated implementations (e.g., lazy loading
    from a database).

    Example:
        ```python
        # Simple dict usage
        external_docs = {
            parent_doc.id: parent_doc,
        }
        renderer = Renderer(external_refs=external_docs)

        # Or custom provider
        class DBProvider:
            def get(self, id: str) -> Document | None:
                return db.load_document(id)

            def __getitem__(self, id: str) -> Document:
                doc = self.get(id)
                if doc is None:
                    raise KeyError(id)
                return doc
        ```
    """

    def __getitem__(self, element_id: str) -> Document | BaseElement:
        """Get external element/document by ID. Raises KeyError if not found."""
        ...

    def get(self, element_id: str, default: Any = None) -> Document | BaseElement | None:
        """Get external element/document by ID, or default if not found."""
        ...


class RenderContext:
    """
    Context passed to element renderers during document rendering.

    RenderContext provides access to:
    - The document being rendered (for element lookup)
    - External reference provider (for cross-document access)

    Note: Inline reference resolution (<ref id="..."/>) is handled by the
    Renderer class, NOT by RenderContext. Element renderers just output HTML
    with <ref> tags, and the Renderer substitutes them.

    Attributes:
        document: The document being rendered
        external_refs: Provider for cross-document references
    """

    def __init__(
        self,
        document: Document,
        external_refs: ExternalRefProvider | dict | None = None,
    ):
        """
        Initialize render context.

        Args:
            document: The document being rendered
            external_refs: Optional provider for external references.
                Can be a dict[str, Document] or any ExternalRefProvider.
        """
        self.document = document
        self.external_refs: ExternalRefProvider | dict = external_refs or {}
        self._inline_ref_ids: set[str] | None = None

    @property
    def inline_ref_ids(self) -> set[str]:
        """IDs of elements that are referenced inline by at least one other element."""
        if self._inline_ref_ids is None:
            ids: set[str] = set()
            for element in self.document.elements:
                for ref in element.inline_refs:
                    ids.add(ref.target_id)
            self._inline_ref_ids = ids
        return self._inline_ref_ids

    def is_inline_referenced(self, element_id: str) -> bool:
        """Return True if *element_id* is the target of at least one inline ref."""
        return element_id in self.inline_ref_ids

    def get_external(self, element_id: str) -> Document | BaseElement | None:
        """
        Get an external element or document by ID.

        This is used for hierarchical rendering features like breadcrumbs,
        NOT for inline substitution.

        Args:
            element_id: ID of the external element/document

        Returns:
            The external element/document, or None if not found
        """
        if isinstance(self.external_refs, dict):
            return self.external_refs.get(element_id)
        return self.external_refs.get(element_id)

    def get_siblings(self, document: Document) -> list[Document]:
        """
        Get sibling documents (other children of the same parent).

        Siblings are computed on-the-fly from the parent's children,
        excluding the current document.

        Args:
            document: The document to find siblings for

        Returns:
            List of sibling documents (may be empty)
        """
        siblings: list[Document] = []
        for parent_id in document.parent_ids:
            parent = self.get_external(parent_id)
            if parent is None:
                continue
            # Get parent's children, excluding this document
            for child_id in getattr(parent, "child_ids", []):
                if child_id != document.id:
                    child = self.get_external(child_id)
                    if child is not None:
                        siblings.append(child)
        return siblings


class DocumentMetadata(dict[str, MetadataValue]):
    """dict subclass for document-level metadata passed to element renderers.

    The sole reason for subclassing dict (rather than using dict directly)
    is singledispatch: dispatch resolves on the *runtime type*, so
    registering DocumentMetadata only catches instances we explicitly
    construct here — not every stray dict in the codebase.
    Usage is identical to a plain dict: indexing, iteration, .get(), etc.

    Not a BaseElement subclass — this is a rendering concern only.

    Values must be JSON-serializable (see MetadataValue).  Pydantic BaseModel
    instances are serialized via model_dump_json().  dict/list values are
    serialized via json.dumps() — complex nested Pydantic models inside them
    will cause a TypeError at render time.
    """


# Union of types accepted by element renderer functions
Renderable = "BaseElement | DocumentMetadata"

# Type for element render functions
# Element renderers ALWAYS return HTML - format conversion is done by Renderer
# The `inline` parameter indicates whether the element is being rendered for
# inline substitution (True) or standalone rendering (False).
ElementRenderer = Callable[["BaseElement | DocumentMetadata", RenderContext, bool], str]


class Renderer:
    """
    Document renderer with pluggable element rendering and format conversion.

    Renderer is the main entry point for converting Document to string output.
    It uses a configurable element renderer (singledispatch function) and
    converts the final HTML to the target format.

    Design Rationale:
    -----------------
    The Renderer is SYNCHRONOUS - there's no I/O involved in rendering.

    Instead of separate renderer classes for different purposes (prompt,
    embedding, raw), we use a single Renderer class with pluggable element
    renderers. This provides:

    - Flexibility: Custom element renderers for any purpose
    - Consistency: Same rendering pipeline for all use cases
    - Simplicity: One class to understand and configure

    Rendering flow:
    1. Element renderers produce HTML (with <ref id="..."/> for inline refs)
    2. Renderer resolves inline refs by substituting rendered element HTML
    3. Renderer converts final HTML to target format (markdown, plain, etc.)

    Element renderers ALWAYS output HTML. The Renderer handles:
    - Inline reference resolution
    - Skipping elements only used inline
    - Format conversion

    Example:
        ```python
        # For LLM prompts
        prompt_renderer = Renderer(
            format=OutputFormat.MARKDOWN,
            element_renderer=render_for_prompt,
            metadata_keys=["source"],
        )
        content = prompt_renderer.render(document)

        # For embeddings
        embedding_renderer = Renderer(
            format=OutputFormat.PLAIN,
            element_renderer=render_for_embedding,
        )
        embedding_text = embedding_renderer.render(document)
        ```

    Attributes:
        format: Output format (markdown, html, etc.)
        element_renderer: Function to render individual elements (outputs HTML)
        external_refs: Provider for cross-document references
        metadata_keys: Which metadata keys to include in output
    """

    def __init__(
        self,
        format: OutputFormat = OutputFormat.MARKDOWN,
        element_renderer: ElementRenderer | None = None,
        external_refs: ExternalRefProvider | dict | None = None,
        metadata_keys: list[str] | None = None,
        include_title: bool = True,
    ):
        """
        Initialize the renderer.

        Args:
            format: Target output format (default: markdown)
            element_renderer: Element rendering function. If None, uses
                the default render_element from rendering.elements.
                MUST return HTML - format conversion is handled by Renderer.
            external_refs: Provider for external references (for hierarchical
                rendering). Can be a dict[str, Document].
            metadata_keys: List of metadata keys to include in output.
                Empty list or None means no metadata.
            include_title: If True and document.title is set, prepend an <h1>
                with the document title before the element content. Useful when
                the title heading was removed from elements by TitleDetectionProcessor
                or when the title comes from the HTML <title> tag. Default: True.
        """
        self.format = format
        self._element_renderer = element_renderer
        self.external_refs = external_refs
        self.metadata_keys = metadata_keys or []
        self.include_title = include_title

    @property
    def element_renderer(self) -> ElementRenderer:
        """Get the element renderer, lazily importing default if needed."""
        if self._element_renderer is None:
            from ragdoc.rendering.elements import render_for_prompt

            return render_for_prompt
        return self._element_renderer

    def render(self, document: Document) -> str:
        """
        Render a document to string.

        The rendering process:
        1. Create RenderContext with document and external refs
        2. Optionally render metadata header
        3. Render each element (skipping inline-only elements)
        4. Resolve inline references (<ref id="..."/> substitution)
        5. Convert final HTML to target format

        Args:
            document: The document to render

        Returns:
            Rendered content in the target format
        """
        ctx = RenderContext(
            document=document,
            external_refs=self.external_refs,
        )

        parts: list[str] = []

        # Metadata header (title + selected metadata keys, routed through singledispatch)
        meta_data: dict[str, MetadataValue] = {}
        if self.include_title and document.title:
            meta_data["title"] = document.title
        for key in self.metadata_keys:
            val = document.metadata.get(key)
            if val is not None:
                meta_data[key] = val
        if meta_data:
            meta_html = self.element_renderer(DocumentMetadata(meta_data), ctx, False)
            if meta_html:
                parts.append(meta_html)

        # Collect IDs of elements that are only referenced inline
        inline_ref_ids = self._collect_inline_ref_ids(document)

        # First pass: render all elements for standalone display, can still contain <ref> tags for inline refs
        rendered_elements: dict[str, str] = {}
        for element in document.elements:
            rendered_elements[element.id] = self.element_renderer(element, ctx, False)

        # Also render inline versions of referenced elements
        inline_rendered: dict[str, str] = {}
        for element_id in inline_ref_ids:
            element = document.get_element(element_id)
            if element:
                inline_rendered[element_id] = self.element_renderer(element, ctx, True)

        # Second pass: collect elements that are not rendered inline elsewhere.
        # Footnotes (and other inline-referenced elements) are omitted from the
        # main output because they appear at the <ref> insertion point instead.
        parts.extend(rendered_elements[element.id] for element in document.elements)
        parts = list(filter(lambda x: x is not None and x.strip() != "", parts))
        # Resolve inline references
        html = "\n".join(parts)
        html = self._resolve_inline_refs(html, inline_rendered, rendered_elements)

        return self._convert(html)

    def _resolve_inline_refs(
        self,
        html: str,
        inline_rendered: dict[str, str],
        fallback_rendered: dict[str, str],
    ) -> str:
        """
        Replace <ref id="..."/> tags with rendered element content.

        Uses inline-rendered versions when available (compact format for
        footnotes, placeholders for images). Falls back to standard rendering
        if inline version not found.

        Args:
            html: HTML string containing <ref id="..."/> tags
            inline_rendered: Dict mapping element IDs to inline-rendered HTML
            fallback_rendered: Dict mapping element IDs to standard-rendered HTML

        Returns:
            HTML with refs replaced by rendered content
        """
        soup = BeautifulSoup(html, "html.parser")

        for ref_tag in soup.find_all("ref"):
            ref_id = ref_tag.get("id")
            if not ref_id:
                ref_tag.replace_with("[invalid ref: no id]")
                continue

            # Prefer inline rendering, fall back to standard
            if ref_id in inline_rendered:
                rendered = inline_rendered[ref_id]
            else:
                ref_tag.replace_with(f"[missing ref: {ref_id}]")
                continue

            ref_tag.replace_with(BeautifulSoup(rendered, "html.parser"))

        return str(soup)

    def _collect_inline_ref_ids(self, document: Document) -> set[str]:
        """Collect IDs of elements that are referenced inline."""
        ids: set[str] = set()
        for element in document.elements:
            for ref in element.inline_refs:
                ids.add(ref.target_id)
        return ids

    def _convert(self, html: str) -> str:
        """Convert HTML to target format.

        When a <header class="document-metadata"> block is present and the
        target format is not HTML, the metadata is extracted and forwarded to
        pandoc as a proper HTML <head> (title + <meta> tags).  Calling pandoc
        with --standalone causes it to emit a YAML frontmatter block at the
        top of the markdown/RST output — no manual YAML construction needed.
        """
        if self.format == OutputFormat.HTML:
            return html
        if self.format == OutputFormat.PLAIN:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()

        soup = BeautifulSoup(html, "html.parser")
        header = soup.find("header", class_="document-metadata")

        if header:
            head_parts: list[str] = []
            h1 = header.find("h1")
            if h1:
                head_parts.append(f"<title>{h1.get_text()}</title>")
            dl = header.find("dl")
            if dl:
                for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
                    head_parts.append(f'<meta name="{dt.get_text()}" content="{dd.get_text()}"/>')
            header.extract()
            full_html = f"<html><head>{''.join(head_parts)}</head><body>{soup}</body></html>"
            result = convert_text(full_html, to=self.format.value, format="html", extra_args=["--standalone"])
        else:
            result = convert_text(html, to=self.format.value, format="html")

        return _reduce_blank_lines(result)


def _reduce_blank_lines(text: str) -> str:
    """Halve consecutive blank lines produced by pandoc (ceil division).

    Normalises mixed line endings first, then maps any run of N newlines
    (N >= 2) to ceil(N/2), so:
        \\n\\n       (1 blank line)  → \\n       (no blank line)
        \\n\\n\\n     (2 blank lines) → \\n\\n     (1 blank line)
        \\n\\n\\n\\n   (3 blank lines) → \\n\\n     (1 blank line)
    """
    text = text.replace("\r", "\n")
    return re.sub(r"\n{2,}", lambda m: "\n" * math.ceil(len(m.group()) / 2), text)
