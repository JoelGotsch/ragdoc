"""Optional-dependency hygiene (Phase 8 extras split).

Two layers:

1. ``import ragdoc`` must not pull any heavyweight optional dependency — verified in a
   subprocess (this process's ``sys.modules`` is already polluted by other tests).
2. Every lazy-import guard raises an actionable ``ImportError`` naming the extra.
   Real absence coverage lives in the CI base-install smoke leg; here the guard
   *messages* are pinned by blocking the module in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# import ragdoc pulls no optional heavyweight deps
# ---------------------------------------------------------------------------

_BANNED_ON_IMPORT = ("torch", "transformers", "pandas", "openai", "PIL", "pymupdf", "numpy")


def test_import_ragdoc_pulls_no_optional_deps():
    """`import ragdoc` must not import torch/transformers/pandas/openai/PIL/pymupdf/numpy."""
    script = textwrap.dedent(
        f"""
        import sys
        import ragdoc
        leaked = [m for m in {_BANNED_ON_IMPORT!r} if m in sys.modules]
        assert not leaked, f"import ragdoc pulled optional deps: {{leaked}}"
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


# ---------------------------------------------------------------------------
# guard messages
# ---------------------------------------------------------------------------


def _block_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import <name>`` raise ImportError inside this test."""
    monkeypatch.setitem(sys.modules, name, None)


def test_reranker_tokenizer_guard_message(monkeypatch: pytest.MonkeyPatch):
    from ragdoc.utils import RerankerTokenizer

    _block_module(monkeypatch, "transformers")
    with pytest.raises(ImportError, match=r"'tokenizers' extra.*ragdoc\[tokenizers\]"):
        RerankerTokenizer()


def test_xlsx_load_excel_guard_message(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from ragdoc.parsing.xlsx import load_excel

    _block_module(monkeypatch, "pandas")
    with pytest.raises(ImportError, match=r"'xlsx' extra.*ragdoc\[xlsx\]"):
        load_excel(tmp_path / "x.xlsx")


def test_pdf_basic_guard_message(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from ragdoc.parsing.pdf_basic.load import parse_pdf_basic

    _block_module(monkeypatch, "pymupdf")
    with pytest.raises(ImportError, match=r"'pdf' extra.*ragdoc\[pdf\]"):
        parse_pdf_basic(tmp_path / "x.pdf")


def test_figure_extractor_guard_message(monkeypatch: pytest.MonkeyPatch):
    from ragdoc.parsing.azure_di.load import FigureExtractor

    _block_module(monkeypatch, "pymupdf")
    with pytest.raises(ImportError, match=r"pymupdf.*ragdoc\[azure-di\].*ragdoc\[pdf\]"):
        FigureExtractor("whatever.pdf")


def test_openai_errors_guard_message(monkeypatch: pytest.MonkeyPatch):
    from ragdoc.llm import _openai_errors

    _openai_errors.cache_clear()
    _block_module(monkeypatch, "openai")
    try:
        with pytest.raises(ImportError, match=r"'llm' extra.*ragdoc\[llm\]"):
            _openai_errors()
    finally:
        _openai_errors.cache_clear()  # don't poison the cache for other tests


def test_numpy_guard_message(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("edtf")  # extraction extra
    from ragdoc.extraction.resolution import EntityResolutionPipeline

    _block_module(monkeypatch, "numpy")

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    import asyncio

    pipeline = EntityResolutionPipeline.__new__(EntityResolutionPipeline)
    with pytest.raises(ImportError, match=r"numpy.*'extraction' extra"):
        # _propose_and_review imports numpy before touching any instance state beyond clusters
        asyncio.run(pipeline._propose_and_review([object(), object()]))  # type: ignore[list-item]


def test_processing_getattr_hint_message(monkeypatch: pytest.MonkeyPatch):
    """The processing package's module __getattr__ raises an actionable hint for absent names."""
    import ragdoc.processing as processing

    # Simulate a base install where the summary_image import failed.
    monkeypatch.delattr(processing, "ImageSummaryProcessor", raising=False)
    with pytest.raises(ImportError, match=r"ImageSummaryProcessor requires the 'llm' extra"):
        _ = processing.ImageSummaryProcessor

    with pytest.raises(AttributeError):
        _ = processing.definitely_not_a_real_name


def test_mineru_latex_guard_message(monkeypatch: pytest.MonkeyPatch):
    from ragdoc.parsing.mineru.base import latex_to_mathml

    _block_module(monkeypatch, "latex2mathml")
    _block_module(monkeypatch, "latex2mathml.converter")
    with pytest.raises(ImportError, match=r"'pdf-mineru' extra.*ragdoc\[pdf-mineru\]"):
        latex_to_mathml("x^2")
