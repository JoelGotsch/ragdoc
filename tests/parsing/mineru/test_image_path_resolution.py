"""
Tests for MineRU image path resolution.

MineRU stores ``image_path`` as a bare filename (e.g. ``4ab1fbb6….jpg``) in
``_middle.json``, but always writes the actual file into an ``images/``
subdirectory alongside the JSON.  These tests verify that ``handle_image_block``
resolves correctly via ``<source_dir>/images/<image_path>`` and that the
``check_mineru_images`` pre-flight helper surfaces missing images.
"""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")

from ragdoc.document import Image, Paragraph
from ragdoc.parsing.mineru.base import (
    ImageBlock,
    ImageBodyBlock,
    ImageCaptionBlock,
    ImageSpan,
    Line,
    MinerUMiddleDocument,
    PageInfo,
    ParseContext,
    TextSpan,
)
from ragdoc.parsing.mineru.handlers import handle_image_block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BBOX = [0.0, 0.0, 100.0, 80.0]


def _page(page_idx: int = 0) -> PageInfo:
    return PageInfo(para_blocks=[], discarded_blocks=[], page_idx=page_idx)


def _context(**metadata) -> ParseContext:
    ctx = ParseContext(source=MinerUMiddleDocument(pdf_info=[]))
    ctx.metadata.update(metadata)
    return ctx


def _image_block(image_path: str) -> ImageBlock:
    span = ImageSpan(bbox=_BBOX, image_path=image_path)
    line = Line(bbox=_BBOX, spans=[span])
    body = ImageBodyBlock(bbox=_BBOX, index=0, lines=[line])
    return ImageBlock(bbox=_BBOX, index=0, blocks=[body])


def _make_png(path: Path) -> None:
    """Write a minimal 1×1 PNG using Pillow."""
    pil = pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image as PILImage

    PILImage.new("RGB", (4, 6), color=(0, 128, 255)).save(str(path))


# ---------------------------------------------------------------------------
# Unit tests — path resolution logic
# ---------------------------------------------------------------------------


def test_handle_image_block_resolves_via_images_subdir(tmp_path):
    """Image file under images/ subdir is found when image_path is bare filename."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_png(images_dir / "fig1.png")

    ctx = _context(source_dir=tmp_path)
    result = handle_image_block(_image_block("fig1.png"), _page(), ctx)

    images = [r for r in result if isinstance(r.element, Image)]
    assert len(images) == 1, "Expected one Image element"
    img = images[0].element
    assert img.width == 4
    assert img.height == 6
    assert img.image  # non-empty base64


def test_handle_image_block_images_subdir_missing_file_logs_warning(tmp_path):
    """No images/ dir → no Image element and warning in context.warnings."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    # tmp_path has no images/ subdirectory
    ctx = _context(source_dir=tmp_path)
    result = handle_image_block(_image_block("nonexistent.png"), _page(), ctx)
    assert not any(isinstance(r.element, Image) for r in result)
    assert any("not found" in w for w in ctx.warnings)


def test_handle_image_block_image_file_absent_from_images_subdir(tmp_path):
    """images/ dir exists but the specific file is absent → no Image element, warning in context."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    (tmp_path / "images").mkdir()
    # specific file NOT written
    ctx = _context(source_dir=tmp_path)
    result = handle_image_block(_image_block("missing.png"), _page(), ctx)
    assert not any(isinstance(r.element, Image) for r in result)
    assert any("not found" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# Integration test — real test-data fixture
# ---------------------------------------------------------------------------

_REAL_JSON = Path(__file__).parent.parent / "data" / "mineru" / "nist-sp-800-63b" / "hybrid_auto" / "nist-sp-800-63b_middle.json"


@pytest.mark.skipif(not _REAL_JSON.exists(), reason="real test data not present")
@pytest.mark.anyio
async def test_parse_real_mineru_document_loads_image(caplog):
    """Parsing the real NIST middle JSON produces at least one Image element."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    import logging

    from ragdoc.parsing.mineru.base import parse_middle_json_file
    from ragdoc.parsing.mineru.parser import MinerUExtractor

    source = parse_middle_json_file(_REAL_JSON)
    extractor = MinerUExtractor()

    with caplog.at_level(logging.WARNING, logger="ragdoc.parsing.mineru.handlers"):
        doc = await extractor.parse(source, source_path=_REAL_JSON)

    image_elements = [e for e in doc.elements if isinstance(e, Image)]
    assert len(image_elements) >= 1, "Expected at least one Image element"
    img = image_elements[0]
    assert img.image, "image base64 data must be non-empty"
    assert img.width > 0
    assert img.height > 0
    assert not any("Could not load image" in r.message for r in caplog.records), (
        "Unexpected 'Could not load image' warning — path resolution may still be broken"
    )


# ---------------------------------------------------------------------------
# Pre-flight check tests
# ---------------------------------------------------------------------------


def _make_minimal_middle_json(tmp_path: Path, image_filenames: list[str]) -> Path:
    """Write a minimal _middle.json referencing the given image filenames."""
    page: dict = {
        "page_idx": 0,
        "page_size": [595, 842],
        "para_blocks": [],
        "discarded_blocks": [],
        "need_drop": False,
        "layout_bboxes": [],
        "page_no": 0,
        "images": [],
        "tables": [],
        "interline_equations": [],
        "preproc_blocks": [],
        "pymudf_origin_page": 0,
    }
    for fname in image_filenames:
        block = {
            "type": "image",
            "bbox": [0.0, 0.0, 100.0, 80.0],
            "index": 0,
            "blocks": [
                {
                    "type": "image_body",
                    "bbox": [0.0, 0.0, 100.0, 80.0],
                    "index": 0,
                    "lines": [
                        {
                            "bbox": [0.0, 0.0, 100.0, 80.0],
                            "spans": [
                                {
                                    "type": "image",
                                    "bbox": [0.0, 0.0, 100.0, 80.0],
                                    "image_path": fname,
                                    "image_caption_list": [],
                                    "image_footnote_list": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        page["para_blocks"].append(block)

    doc = {"pdf_info": [page], "_version_name": "test"}
    json_path = tmp_path / "test_middle.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")
    return json_path


def test_check_mineru_images_all_present_returns_empty_list(tmp_path):
    from ragdoc.parsing.mineru import check_mineru_images

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.png").write_bytes(b"fake")
    (images_dir / "img2.png").write_bytes(b"fake")

    json_path = _make_minimal_middle_json(tmp_path, ["img1.png", "img2.png"])
    assert check_mineru_images(json_path) == []


def test_check_mineru_images_missing_file_returned(tmp_path):
    from ragdoc.parsing.mineru import check_mineru_images

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.png").write_bytes(b"fake")
    # img2.png intentionally absent

    json_path = _make_minimal_middle_json(tmp_path, ["img1.png", "img2.png"])
    missing = check_mineru_images(json_path)
    assert missing == [Path("images") / "img2.png"]


def test_check_mineru_images_no_image_blocks_returns_empty_list(tmp_path):
    from ragdoc.parsing.mineru import check_mineru_images

    json_path = _make_minimal_middle_json(tmp_path, [])  # no image blocks
    assert check_mineru_images(json_path) == []
