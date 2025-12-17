"""Shared fixtures for merging tests."""

import pytest

from ragdoc.document import Document, Footnote, Image, Paragraph, RawText


@pytest.fixture
def real_world_docs():
    """Two-parser scenario: mineru flat RawTexts vs html-structured Document.

    doc_a (mineru): flat RawTexts — some have inline markup, no element structure.
    doc_b (html):   properly structured with Paragraph+ref, Image, Footnote, and
                    one extra paragraph that only exists in doc_b.

    The merged output (Approach A) should:
    - Use doc_b's Paragraph for the footnote sentence (has proper <ref>)
      and inject doc_a's <strong> into it.
    - Preserve doc_a's "Some additional text" (unique to doc_a, no doc_b match).
    - Use doc_b's para2 / Image / Footnote for the middle section.
    - Include or exclude para_extra based on allow_insertions_from_b.
    """
    fn = Footnote(number=1, innerhtml="This is the footnote text")
    img = Image(image=None, image_type="png", alt="Figure 1: example diagram")
    para1 = Paragraph(html=f'<p>This is just some text with a footnote<ref id="{fn.id}" rel="footnote"/>.</p>')
    para2 = Paragraph(html="<p>Some text in the middle</p>")
    para_extra = Paragraph(html="<p>This paragraph appears only in doc_b.</p>")

    doc_a = Document(
        elements=[
            RawText(html="This is just some <strong>text</strong> with a footnote1."),
            RawText(html="Some additional text"),
            RawText(html="Some text in the middle"),
            RawText(html="Figure 1: example diagram"),
            RawText(html="1: This is the footnote text"),
        ],
        parser="mineru",
    )
    # para_extra is AFTER all aligned elements → becomes a pure INSERT_B
    doc_b = Document(elements=[para1, para2, img, fn, para_extra], parser="html")

    return doc_a, doc_b, fn, img, para1, para2, para_extra
