from __future__ import annotations

from pathlib import Path

from ragdoc.document import Document

from .base import (
    AsyncCallNext,
    BaseMiddleware,
    ChartBlock,
    CodeBlock,
    ImageBlock,
    InterlineEquationBlock,
    ListBlock,
    MinerUMiddleDocument,
    ParseContext,
    ParsedElement,
    ParserMiddleware,
    RefTextBlock,
    TableBlock,
    TextBlock,
    TitleBlock,
)
from .handlers import ExtractionConfig


class CoreExtractionMiddleware(BaseMiddleware):
    """
    Core middleware that converts MinerU blocks into :class:`ParsedElement` objects.

    Dispatches each block to the appropriate handler in *config*.  Replace any
    handler to change extraction behaviour for that block type without
    subclassing::

        config = ExtractionConfig()
        config.handle_image = lambda block, page, ctx: []  # drop all images
        middleware = CoreExtractionMiddleware(config=config)

    See :class:`~ragdoc.parsing.mineru.handlers.ExtractionConfig` for the full
    set of configurable handlers.
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config: ExtractionConfig = config if config is not None else ExtractionConfig()

    async def process(self, context: ParseContext) -> ParseContext:
        """Iterate all pages and dispatch every block to its configured handler."""
        context.metadata.setdefault("_default_heading_level", self.config.default_heading_level)

        for page in context.source.pdf_info:
            for block in page.para_blocks:
                results: list[ParsedElement]
                if isinstance(block, TitleBlock):
                    results = self.config.handle_title(block, page, context)
                elif isinstance(block, TextBlock):
                    results = self.config.handle_text(block, page, context)
                elif isinstance(block, ListBlock):
                    results = self.config.handle_list(block, page, context)
                elif isinstance(block, CodeBlock):
                    results = self.config.handle_code(block, page, context)
                elif isinstance(block, ImageBlock):
                    results = self.config.handle_image(block, page, context)
                elif isinstance(block, TableBlock):
                    results = self.config.handle_table(block, page, context)
                elif isinstance(block, ChartBlock):
                    results = self.config.handle_chart(block, page, context)
                elif isinstance(block, (RefTextBlock, InterlineEquationBlock)):
                    # Top-level ref_text (bibliography entries) and
                    # interline_equation blocks have the same shape as a
                    # TextBlock — a flat list of lines — so reuse the text
                    # handler. They become Paragraph elements in the Document.
                    results = self.config.handle_text(
                        block,  # type: ignore[arg-type]  # RefTextBlock/InterlineEquationBlock are structurally TextBlock (flat .lines)
                        page,
                        context,
                    )
                else:
                    results = []
                context.elements.extend(results)

            for block in page.discarded_blocks:
                handler = self.config.discarded_handlers.get(block.type)
                if handler is not None:
                    context.elements.extend(handler(block, page, context))

        return context


# =============================================================================
# Parser Class
# =============================================================================


class MinerUExtractor:
    """Internal extraction pipeline for MinerU ``_middle.json`` documents.

    Accepts a :class:`~ragdoc.parsing.mineru.base.MinerUMiddleDocument` (not a
    ``Path``) and converts it to a :class:`~ragdoc.document.Document` via a
    configurable middleware chain.  This is **not** a :class:`~ragdoc.parsing.parser.Parser`
    subclass and is not the public entry point — use :class:`MinerUParser` or
    :func:`parse_mineru_file` instead.

    Customise extraction by composing middleware::

        from ragdoc.parsing.mineru.handlers import (
            ExtractionConfig, handle_discarded_as_raw_text,
        )
        from ragdoc.parsing.mineru.base import DiscardedBlockType

        config = ExtractionConfig()
        config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text
        extractor = MinerUExtractor(use_default_middlewares=False)
        extractor.use(CoreExtractionMiddleware(config=config))

        doc = await extractor.parse(mineru_document, source_path=Path("report/_middle.json"))
    """

    def __init__(
        self,
        middlewares: list[ParserMiddleware] | None = None,
        use_default_middlewares: bool = True,
    ) -> None:
        self.middlewares: list[ParserMiddleware] = []

        if use_default_middlewares:
            self.middlewares.append(CoreExtractionMiddleware())

        if middlewares:
            for mw in middlewares:
                self.use(mw)

    def use(self, middleware: ParserMiddleware) -> MinerUExtractor:
        """Append a middleware to the pipeline."""
        self.middlewares.append(middleware)
        return self

    def use_before(
        self,
        middleware: ParserMiddleware,
        before: type[ParserMiddleware],
    ) -> MinerUExtractor:
        """Insert a middleware immediately before the first instance of *before*."""
        for i, mw in enumerate(self.middlewares):
            if isinstance(mw, before):
                self.middlewares.insert(i, middleware)
                return self
        self.middlewares.insert(0, middleware)
        return self

    def use_after(
        self,
        middleware: ParserMiddleware,
        after: type[ParserMiddleware],
    ) -> MinerUExtractor:
        """Insert a middleware immediately after the first instance of *after*."""
        for i, mw in enumerate(self.middlewares):
            if isinstance(mw, after):
                self.middlewares.insert(i + 1, middleware)
                return self
        self.middlewares.append(middleware)
        return self

    async def parse(
        self,
        source: MinerUMiddleDocument,
        source_path: Path | None = None,
    ) -> Document:
        """
        Parse a MinerU document into a :class:`~ragdoc.document.Document`.

        Args:
            source: The MinerU middle document to parse.
            source_path: Path to the ``_middle.json`` file.  When provided the
                parent directory is stored in ``context.metadata["source_dir"]``
                so that image handlers can resolve relative image paths.

        After all middlewares run, ``document.metadata`` contains only
        ``"filename"`` (when *source_path* is provided).  Parsers must not
        inject any other keys into document metadata; if discarded content
        (headers, footers, …) should be preserved it must be emitted as
        :class:`~ragdoc.document.BaseElement` objects via a custom handler.
        """
        context = ParseContext(source=source)
        if source_path is not None:
            context.metadata["source_dir"] = source_path.parent

        context = await self._run_middlewares(context)
        doc = Document(
            elements=[parsed.element for parsed in context.elements],
            title=context.document_title,
            parser="mineru",
        )
        if source_path is not None:
            doc.metadata["filename"] = source_path.name
            doc.source_path = str(source_path)
        return doc

    async def _run_middlewares(self, context: ParseContext) -> ParseContext:
        def create_chain(index: int) -> AsyncCallNext:
            async def call_next(ctx: ParseContext) -> ParseContext:
                if index >= len(self.middlewares):
                    return ctx
                return await self.middlewares[index](ctx, create_chain(index + 1))

            return call_next

        return await create_chain(0)(context)
