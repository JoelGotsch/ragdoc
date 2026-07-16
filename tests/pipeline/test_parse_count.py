"""Phase-6 perf exit criterion: at most one BS4 parse per element per pipeline pass.

The counter is scoped to ``ragdoc.document.BeautifulSoup`` — the element model's own
parses (normalizing validators + the identity-keyed soup cache).  The renderer's and
splitter's parses live in other modules and are deliberately out of scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ragdoc.document as document_module
from ragdoc.document import Document, Heading, Paragraph


@pytest.fixture
def parse_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counter = {"n": 0}
    real_bs = document_module.BeautifulSoup

    def counting_bs(*args: object, **kwargs: object):
        counter["n"] += 1
        return real_bs(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(document_module, "BeautifulSoup", counting_bs)
    return counter


@pytest.mark.anyio
async def test_one_parse_per_element_per_pass(parse_counter: dict[str, int]):
    """A full DocumentPipeline pass performs O(elements) parses, not O(accesses)."""
    from ragdoc.pipeline import DocumentPipeline, TokenSplitter

    n_elements = 12

    async def parser(path: Path) -> Document:
        elements = []
        for i in range(n_elements // 2):
            elements.append(Heading(html=f"<h2>Section {i}</h2>"))
            elements.append(Paragraph(html=f"<p>{'word ' * 30}(section {i})</p>"))
        return Document(elements=elements, source_path=str(path))

    pipeline = DocumentPipeline(parser=parser, splitter=TokenSplitter(max_tokens=120))
    parse_counter["n"] = 0
    chunks = await pipeline.run(Path("report.html"))
    assert len(chunks) > 1  # the splitter actually split

    # Budget: construction normalization (1/element) + first cached derived access
    # (1/element) + small fixed slack for split-created RawText wrappers.  The old
    # fresh-parse-per-access model measured in the hundreds here.
    budget = n_elements * 3 + 10
    assert parse_counter["n"] <= budget, f"{parse_counter['n']} document-module parses > budget {budget}"


def test_repeated_derived_access_is_one_parse(parse_counter: dict[str, int]):
    """The headline cache property: N derived reads on one element cost one parse."""
    p = Paragraph(html="<p>See <ref id='x' rel='image'/> details</p>")
    parse_counter["n"] = 0
    for _ in range(10):
        _ = p.text
        _ = p.inline_refs
        _ = p.footnote_ids
        _ = p.image_ids
    assert parse_counter["n"] == 1
