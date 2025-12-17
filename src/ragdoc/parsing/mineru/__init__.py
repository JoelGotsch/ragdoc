"""MinerU ``_middle.json`` parsing.

The module is importable without the ``pdf-mineru`` extra; LaTeX equation
conversion (``latex2mathml``) is imported lazily at conversion time and the
parser reports itself unavailable via :meth:`MinerUParser.is_available` until
the extra is installed.
"""

import importlib.util
from pathlib import Path

from pydantic import ConfigDict

from ragdoc.document import Document
from ragdoc.parsing.mineru.base import parse_middle_json_file
from ragdoc.parsing.mineru.handlers import (
    ExtractionConfig,
    build_heading_html,
    build_text_html,
    check_mineru_images,
    handle_chart_block,
    handle_code_block,
    handle_discarded_as_footnote,
    handle_discarded_as_metadata,
    handle_discarded_as_raw_text,
    handle_discarded_drop,
    handle_image_block,
    handle_list_block,
    handle_table_block,
    handle_text_block,
    handle_title_block,
)
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor
from ragdoc.parsing.parser import Parser


async def parse_mineru_file(path: Path) -> Document:
    """Parse a MinerU ``_middle.json`` file into a :class:`~ragdoc.document.Document`.

    This is the standalone entry point for one-off parsing without going through
    the registry. Equivalent to ``MinerUParser()(path)``.

    Args:
        path: Path to the ``_middle.json`` file produced by MinerU.

    Returns:
        Parsed :class:`~ragdoc.document.Document` with ``parser="mineru"``.
    """
    source = parse_middle_json_file(path)
    extractor = MinerUExtractor()
    return await extractor.parse(source, source_path=path)


class MinerUParser(Parser):
    """Public parser entry point for MinerU ``_middle.json`` files.

    Registered in the ragdoc parser registry under the name ``"mineru"`` at
    priority 50.  Accepts an optional *extractor* argument to inject a custom
    :class:`MinerUExtractor` (e.g., one with a modified stage chain) without
    exposing :class:`~ragdoc.parsing.mineru.base.MinerUMiddleDocument` to callers.

    The only key written to ``document.metadata`` is ``"filename"``.  If
    discarded content (running headers, footers, …) should be preserved it
    must be emitted as document elements by configuring the extractor — see
    :class:`~ragdoc.parsing.mineru.handlers.ExtractionConfig` for the full
    handler reference.

    Example — preserve running page headers as :class:`~ragdoc.document.RawText`::

        from ragdoc.parsing.mineru import (
            MinerUParser, MinerUExtractor, CoreExtractor, ExtractionConfig,
            handle_discarded_as_raw_text,
        )
        from ragdoc.parsing.mineru.base import DiscardedBlockType

        config = ExtractionConfig()
        config.discarded_handlers[DiscardedBlockType.HEADER] = handle_discarded_as_raw_text
        extractor = MinerUExtractor(use_default_stages=False)
        extractor.use(CoreExtractor(config=config))

        parser = MinerUParser(extractor=extractor)
        document = await parser(Path("report_middle.json"))
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "mineru"
    patterns: list[str] = ["_middle.json"]
    priority: int = 50
    description: str = "MinerU middle JSON files"

    extractor: MinerUExtractor | None = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("latex2mathml") is not None

    def unavailable_reason(self) -> str:
        return "mineru: install 'ragdoc[pdf-mineru]' for MinerU middle-JSON parsing"

    async def __call__(self, path: Path) -> Document:
        if self.extractor is not None:
            source = parse_middle_json_file(path)
            return await self.extractor.parse(source, source_path=path)
        return await parse_mineru_file(path)


__all__ = [
    "CoreExtractor",
    "ExtractionConfig",
    "MinerUExtractor",
    "MinerUParser",
    "build_heading_html",
    "build_text_html",
    "check_mineru_images",
    "handle_code_block",
    "handle_discarded_as_footnote",
    "handle_discarded_as_metadata",
    "handle_discarded_as_raw_text",
    "handle_discarded_drop",
    "handle_image_block",
    "handle_list_block",
    "handle_table_block",
    "handle_text_block",
    "handle_title_block",
]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(MinerUParser())
