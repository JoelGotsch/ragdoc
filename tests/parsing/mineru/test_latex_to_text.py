"""Tests for latex_to_text conversion used by the MinerU pipeline."""

import pytest

pytest.importorskip("pylatexenc", reason="pdf_mineru extra not installed")
from ragdoc.parsing.mineru.base import _latex_to_text


@pytest.mark.parametrize(
    "latex, expected",
    [
        (r"\mathrm{ Fe_235 }", "Fe235"),
        (r"\mathrm { S F } _ { 6 }", "S F 6"),
        (r"4 . 5 \%", "4 . 5 %"),
        (r"\mathrm{SF_6}", "SF6"),
        (r"\text{Hello}", "Hello"),
        (r"\alpha", "\u03b1"),
        (r"x^{2}", "x2"),
        (r"H_2O", "H2O"),
        ("plain text", "plain text"),
        ("", ""),
    ],
    ids=[
        "mathrm_Fe235_spaced",
        "mathrm_SF6_spaced",
        "percent",
        "mathrm_SF6_compact",
        "text_command",
        "greek_alpha",
        "superscript",
        "subscript_H2O",
        "plain_text",
        "empty",
    ],
)
def test_latex_to_text(latex: str, expected: str) -> None:
    assert _latex_to_text(latex) == expected
