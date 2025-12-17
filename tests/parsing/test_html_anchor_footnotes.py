"""Tests for HTML parsing and rendering of anchor-linked footnotes.

Source HTML pattern
-------------------
HTML documents sometimes express footnotes using ``<a href="#footnote-N">``
anchors in the body and matching ``<p id="footnote-N">`` paragraphs at the
bottom::

    <p>Some text <a href="#footnote-1">[1]</a>. More text.</p>
    <p id="footnote-1">[1] Footnote content.</p>

The tests are split into two sections:

Part 1 — Parsing
    Checks whether ``generate_document`` converts these anchor pairs into
    proper ``Footnote`` elements with ``InlineRef`` links.

Part 2 — Rendering
    Checks the output of all three built-in element renderers
    (``render_for_prompt``, ``render_raw``) when
    the document *does* contain correct ``Footnote`` / ``InlineRef`` structure
    (supplied by the ``anchor_footnote_document`` fixture).

Tests marked ``xfail(strict=True)`` document desired-but-not-yet-implemented
behaviour.  They will turn into failures as soon as the feature is added,
reminding you to flip them to ordinary passing tests.

Implementation suggestion
-------------------------
**Parsing — anchor-footnote detection**

Add a ``DocumentProcessor`` (e.g. ``AnchorFootnoteExtractor``) that runs
after ``generate_document``::

    class AnchorFootnoteExtractor(DocumentProcessor):
        # Flexible id pattern — matches footnote-N, fn-N, fn_N, note-N, …
        _ID_RE = re.compile(r"^(?:footnote|fn|note)[-_]?(\\d+)$", re.I)
        # Text guard — paragraph must start with [N] or N. to avoid false positives
        _TEXT_RE = re.compile(r"^\\[?\\d+[\\].]")

        async def process(self, document: Document) -> Document:
            # Pass 1: find footnote paragraphs → build {html-id: Footnote}
            # Pass 2: rewrite body paragraphs — replace <a href="#…"> with <ref id="…"/>
            #         and attach InlineRef entries
            # Replace Paragraph elements with Footnote elements in document.elements
            ...

Keeping detection in a ``DocumentProcessor`` (rather than inside the
parser) means:
* ``generate_document`` stays format-agnostic.
* The extractor is testable in isolation.
* Other parsers that produce similar HTML structures benefit for free.

**Rendering — pandoc footnote syntax**

Pandoc converts its own HTML footnote convention to ``[^label]:`` markdown.
The ``Renderer`` can emit that convention by:

1. Numbering inline footnotes during ``_resolve_inline_refs`` and emitting::

       <a href="#fn-{fn.id}" class="footnote-ref" id="fnref-{fn.id}"
          role="doc-noteref"><sup>N</sup></a>

2. Appending a ``<section class="footnotes">`` block at the end of the HTML
   before calling ``convert_text``::

       <section id="footnotes" class="footnotes footnotes-end-of-document"
                role="doc-endnotes">
         <ol>
           <li id="fn-{fn.id}">
             <p>{fn.innerhtml}
               <a href="#fnref-{fn.id}" class="footnote-back"
                  role="doc-backlink">↩︎</a>
             </p>
           </li>
         </ol>
       </section>

Pandoc then produces::

    [^fn-id]: Footnote text.

Using the element UUID as the label keeps references stable across re-renders
and avoids the need for a separate numbering registry.

Proof that pandoc handles this correctly — see
``test_pandoc_html_footnote_convention`` at the bottom of this file.
"""

from __future__ import annotations

import re

import pytest

from ragdoc.document import Document, Footnote


def replace_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


from ragdoc.parsing.html.load import HTML, generate_document
from ragdoc.rendering.base import OutputFormat, Renderer
from ragdoc.rendering.elements import render_for_prompt, render_raw

# =============================================================================
# Part 1 — Parsing
# =============================================================================


def test_anchor_links_survive_in_raw_paragraph_html(anchor_footnote_html: str) -> None:
    """``<a href="#footnote-N">`` anchors are preserved inside the body Paragraph's HTML."""
    doc = generate_document(HTML(content=anchor_footnote_html))
    body_paragraphs = [p for p in doc.paragraphs if "footnote link" in p.text]
    assert len(body_paragraphs) == 1
    assert 'href="#footnote-1"' in body_paragraphs[0].html
    assert 'href="#footnote-2"' in body_paragraphs[0].html


def test_anchor_footnotes_detected_as_footnote_elements(anchor_footnote_html: str) -> None:
    """``<p id="footnote-N">`` paragraphs become ``Footnote`` elements via detect_html_footnotes."""
    doc = generate_document(HTML(content=anchor_footnote_html))

    assert len(doc.footnotes) == 2

    fn1 = next((f for f in doc.footnotes if f.number == 1), None)
    fn2 = next((f for f in doc.footnotes if f.number == 2), None)

    assert fn1 is not None and "first footnote" in fn1.text
    assert fn2 is not None and "longer footnote" in fn2.text


@pytest.mark.xfail(
    reason="AnchorFootnoteExtractor not yet implemented — no InlineRefs are created.",
    strict=True,
)
def test_anchor_links_become_inline_refs(anchor_footnote_html: str) -> None:
    """Desired: ``<a href="#footnote-N">`` links become ``InlineRef(rel_type='footnote')`` entries."""
    doc = generate_document(HTML(content=anchor_footnote_html))

    body_paragraphs = [p for p in doc.paragraphs if "footnote link" in p.text]
    assert len(body_paragraphs) == 1
    body = body_paragraphs[0]

    footnote_refs = [r for r in body.inline_refs if r.rel_type == "footnote"]
    assert len(footnote_refs) == 2

    footnote_ids = {fn.id for fn in doc.footnotes}
    assert all(r.target_id in footnote_ids for r in footnote_refs)


# =============================================================================
# Part 2 — Rendering
# All tests in this section use ``anchor_footnote_document``, which supplies
# a correctly-structured Document (Footnote elements + InlineRefs) regardless
# of whether the parser can produce it yet.
# =============================================================================


# ---------------------------------------------------------------------------
# render_for_prompt
# ---------------------------------------------------------------------------


def test_prompt_renderer_resolves_footnotes_inline(anchor_footnote_document: Document) -> None:
    """Current behaviour: inline footnotes appear as ``[Footnote: text]`` spans."""
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    normalized = replace_whitespace(renderer.render(anchor_footnote_document))

    assert "[Footnote:" in normalized
    # No pandoc [^label] references yet.
    assert "[^" not in normalized


def test_prompt_renderer_does_not_drop_long_footnote(anchor_footnote_document: Document) -> None:
    """Long footnote content must not be truncated regardless of line wrapping."""
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    output = renderer.render(anchor_footnote_document)

    # Normalise whitespace — pandoc may wrap long lines mid-phrase.
    assert "multi-sentence footnote content" in replace_whitespace(output)


@pytest.mark.xfail(
    reason="Renderer does not yet emit pandoc [^label] syntax for footnotes.",
    strict=True,
)
def test_prompt_renderer_uses_pandoc_footnote_syntax(anchor_footnote_document: Document) -> None:
    """Desired: ``render_for_prompt`` emits ``[^{fn.id}]`` references and ``[^{fn.id}]:`` blocks."""
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt)
    output = renderer.render(anchor_footnote_document)

    for fn in anchor_footnote_document.footnotes:
        assert f"[^{fn.id}]" in output, f"Missing inline reference [^{fn.id}]"
        assert f"[^{fn.id}]:" in output, f"Missing definition block [^{fn.id}]:"

    normalized = replace_whitespace(output)
    assert "Here is my first footnote" in normalized
    assert "multi-sentence footnote content" in normalized


# ---------------------------------------------------------------------------
# render_raw
# ---------------------------------------------------------------------------


def test_raw_renderer_resolves_footnotes_inline(anchor_footnote_document: Document) -> None:
    """Current behaviour: raw renderer resolves footnotes as ``[N: text]`` spans."""
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_raw)
    normalized = replace_whitespace(renderer.render(anchor_footnote_document))

    # Footnote number prefix appears in the inline span.
    assert "[\\[1\\]]" in normalized


def test_raw_renderer_does_not_drop_long_footnote(anchor_footnote_document: Document) -> None:
    """Long footnote text must survive raw rendering without truncation."""
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_raw)
    output = renderer.render(anchor_footnote_document)

    assert "multi-sentence footnote content" in replace_whitespace(output)


@pytest.mark.xfail(
    reason="Renderer does not yet emit pandoc [^label] syntax for footnotes.",
    strict=True,
)
def test_raw_renderer_uses_pandoc_footnote_syntax(anchor_footnote_document: Document) -> None:
    """Desired: ``render_raw`` emits ``[^{fn.id}]`` inline references and ``[^{fn.id}]:`` definitions.

    Expected markdown structure (element IDs abbreviated)::

        This is the main body … footnote link[^a1b2c3…]. … footnotes as well[^d4e5f6…].

        [^a1b2c3…]: Here is my first footnote.

        [^d4e5f6…]: This is a considerably longer footnote that spans multiple
            sentences. … multi-sentence footnote content …
    """
    renderer = Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_raw)
    output = renderer.render(anchor_footnote_document)

    for fn in anchor_footnote_document.footnotes:
        assert f"[^{fn.id}]" in output, f"Missing inline reference [^{fn.id}]"
        assert f"[^{fn.id}]:" in output, f"Missing definition block [^{fn.id}]:"

    normalized = replace_whitespace(output)
    assert "Here is my first footnote" in normalized
    assert "multi-sentence footnote content" in normalized


# ---------------------------------------------------------------------------
# Standalone (un-referenced) footnote blocks
# ---------------------------------------------------------------------------


def test_unreferenced_footnote_standalone_rendering_per_renderer() -> None:
    """Unreferenced footnotes should be rendered by all renderers."""
    fn = Footnote(number=1, innerhtml="Standalone footnote block.")
    doc = Document()
    doc.elements = [fn]

    # render_for_prompt: orphaned footnote
    prompt_out = replace_whitespace(
        Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_for_prompt).render(doc)
    )
    assert "Standalone footnote block" in prompt_out

    # render_raw: orphaned footnote appears as a plain paragraph.
    raw_out = replace_whitespace(Renderer(format=OutputFormat.MARKDOWN, element_renderer=render_raw).render(doc))
    assert "Standalone footnote block" in raw_out


# =============================================================================
# Proof-of-concept: pandoc understands its own HTML footnote convention
# =============================================================================


def test_pandoc_html_footnote_convention() -> None:
    """Pandoc converts its own HTML footnote markup to ``[^N]:`` markdown.

    This test demonstrates both what pandoc *can* do and an important
    limitation for the desired ``[^{fn.id}]`` output:

    **Pandoc normalises footnote labels to sequential integers.**

    Even though the source HTML uses ``id="fn-abc123"``, the output markdown
    uses ``[^1]``, not ``[^fn-abc123]``.  This means the HTML->markdown
    route via pandoc cannot preserve UUID-based labels.

    **Implication for the desired implementation:** To emit stable
    ``[^{fn.id}]`` labels, the ``Renderer`` must write the footnote reference
    lines directly as markdown strings rather than relying on pandoc's
    HTML-to-markdown conversion.  See the module docstring for details.
    """
    from pypandoc import convert_text

    fn_id = "abc123"
    html = (
        f"<p>Main text"
        f'<a href="#fn-{fn_id}" class="footnote-ref" id="fnref-{fn_id}" role="doc-noteref">'
        f"<sup>1</sup></a>."
        f"</p>"
        f'<section id="footnotes" class="footnotes footnotes-end-of-document"'
        f' role="doc-endnotes">'
        f"<ol>"
        f'<li id="fn-{fn_id}">'
        f"<p>Here is the footnote."
        f'<a href="#fnref-{fn_id}" class="footnote-back" role="doc-backlink">&#8617;&#65038;</a>'
        f"</p>"
        f"</li>"
        f"</ol>"
        f"</section>"
    )

    md = convert_text(html, to="markdown", format="html")
    normalized = replace_whitespace(md)

    # Pandoc produces [^1] / [^1]: — it normalises to sequential integers.
    assert "[^1]" in normalized
    assert "[^1]:" in normalized
    assert "Here is the footnote." in normalized

    # The original UUID label is NOT preserved by pandoc.
    assert f"[^fn-{fn_id}]" not in normalized
