"""Test-only LaTeX → plain-text helper.

Moved out of ``ragdoc.parsing.mineru.base`` (Phase 8 dead-code sweep): the
library itself never calls it — it exists purely to normalize text for
content-based comparisons in tests.  Requires ``pylatexenc`` (installed with
the ``pdf-mineru`` extra); importing this module without it skips the test
module.
"""

import re

import pytest

pylatexenc = pytest.importorskip("pylatexenc")

from pylatexenc.latex2text import LatexNodes2Text

_LATEX2TEXT = LatexNodes2Text(math_mode="text")


def latex_to_text(latex: str) -> str:
    """Convert a LaTeX string to plain text using pylatexenc.

    Handles commands like ``\\mathrm{H_2O}`` → ``H2O``,
    ``\\mathrm { C O } _ { 2 }`` → ``CO2``, etc.

    Sub/superscript markers (``_`` / ``^``) are stripped so that
    chemical formulae and isotope names read naturally as plain text.
    """
    result = _LATEX2TEXT.latex_to_text(latex).strip()
    # Strip sub/superscript markers left by pylatexenc
    result = re.sub(r"[_^]", "", result)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result
