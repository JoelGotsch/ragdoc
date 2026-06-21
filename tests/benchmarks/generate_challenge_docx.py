#!/usr/bin/env python3
"""
Generate tests/benchmarks/fixtures/slice_d/challenge_01.docx

A single DOCX file packed with deliberate parsing challenges that exercise:
  1. Mixed real/fake footnotes
  2. Fake headings (bold paragraph, all-caps paragraph)
  3. Fake lists (manually typed numbers, no list style)
  4. Layout tables vs. real data tables
  5. Nested table (table inside table cell)
  6. Fake code block (monospace + tabs, no Code style)
  7. Image with unlinked caption paragraph
  8. Manual line breaks (w:br) masquerading as paragraphs
  9. Empty spacing paragraphs
  10. Unicode math notation in body text
  11. Footer-like body paragraph (not a real Word footer)

Run with:
    uv run python tests/benchmarks/generate_challenge_docx.py
"""

import io
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.packuri import PackURI
from docx.opc.part import XmlPart
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree
from PIL import Image as PILImage, ImageDraw

OUTPUT_PATH = Path(__file__).parent / "fixtures" / "slice_d" / "challenge_01.docx"

# ── Namespace / relationship constants ──────────────────────────────────────
FN_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
FN_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"
FN_URI = "/word/footnotes.xml"

_INITIAL_FN_XML = etree.fromstring(
    b"<w:footnotes"
    b' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    b' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    b'<w:footnote w:type="separator" w:id="-1">'
    b"<w:p><w:r><w:separator/></w:r></w:p>"
    b"</w:footnote>"
    b'<w:footnote w:type="continuationSeparator" w:id="0">'
    b"<w:p><w:r><w:continuationSeparator/></w:r></w:p>"
    b"</w:footnote>"
    b"</w:footnotes>"
)


# ── Footnote manager ─────────────────────────────────────────────────────────


class FootnoteManager:
    """Manages real OOXML footnotes in a python-docx Document."""

    def __init__(self, doc: Document) -> None:
        self._fn_root = _INITIAL_FN_XML
        self._next_id = 1
        fn_part = XmlPart(PackURI(FN_URI), FN_CT, self._fn_root, doc.part.package)
        doc.part.relate_to(fn_part, FN_REL)

    def add(self, paragraph, text: str) -> int:
        """Append a footnote reference after the last run in *paragraph*."""
        fn_id = self._next_id
        self._next_id += 1

        # ── Build the <w:footnote> element ───────────────────────────────────
        fn = OxmlElement("w:footnote")
        fn.set(qn("w:id"), str(fn_id))

        p = OxmlElement("w:p")
        pPr = OxmlElement("w:pPr")
        pStyle = OxmlElement("w:pStyle")
        pStyle.set(qn("w:val"), "FootnoteText")
        pPr.append(pStyle)
        p.append(pPr)

        # Footnote reference mark
        r_ref = OxmlElement("w:r")
        r_ref_pr = OxmlElement("w:rPr")
        r_ref_style = OxmlElement("w:rStyle")
        r_ref_style.set(qn("w:val"), "FootnoteReference")
        r_ref_pr.append(r_ref_style)
        r_ref.append(r_ref_pr)
        r_ref.append(OxmlElement("w:footnoteRef"))
        p.append(r_ref)

        # Footnote text
        r_text = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = f" {text}"
        r_text.append(t)
        p.append(r_text)

        fn.append(p)
        self._fn_root.append(fn)

        # ── Insert reference run into the body paragraph ─────────────────────
        b_ref = OxmlElement("w:r")
        b_ref_pr = OxmlElement("w:rPr")
        b_ref_style = OxmlElement("w:rStyle")
        b_ref_style.set(qn("w:val"), "FootnoteReference")
        b_ref_pr.append(b_ref_style)
        b_ref.append(b_ref_pr)
        b_ref_elem = OxmlElement("w:footnoteReference")
        b_ref_elem.set(qn("w:id"), str(fn_id))
        b_ref.append(b_ref_elem)
        paragraph._p.append(b_ref)

        return fn_id


# ── Small helpers ────────────────────────────────────────────────────────────


def superscript_run(para, text: str):
    """Add a plain superscript run (NOT a footnote reference)."""
    run = para.add_run(text)
    rPr = run._r.get_or_add_rPr()
    v = OxmlElement("w:vertAlign")
    v.set(qn("w:val"), "superscript")
    rPr.append(v)
    return run


def linebreak(para):
    """Add a manual line break (w:br) inside a paragraph."""
    run = para.add_run()
    br = OxmlElement("w:br")
    run._r.append(br)


def set_table_borders(table, val: str = "single", sz: str = "4", color: str = "000000"):
    tblPr = table._tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        table._tbl.insert(0, tblPr)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), val)
        b.set(qn("w:sz"), sz)
        b.set(qn("w:color"), color)
        borders.append(b)
    tblPr.append(borders)


def remove_table_borders(table):
    set_table_borders(table, val="none", sz="0", color="auto")


def make_placeholder_image() -> io.BytesIO:
    img = PILImage.new("RGB", (280, 100), color=(230, 240, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([3, 3, 276, 96], outline=(120, 160, 200), width=2)
    draw.text((12, 38), "[ pipeline diagram ]", fill=(80, 110, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Document builder ─────────────────────────────────────────────────────────


def build_document() -> Document:
    doc = Document()
    fn = FootnoteManager(doc)

    # ════════════════════════════════════════════════════════════════════════
    # §1  The Footnote Gauntlet
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("The Footnote Gauntlet", level=1)

    # Three proper OOXML footnotes embedded inline
    p1 = doc.add_paragraph()
    p1.add_run("The International Verification Commission published its annual report in March 2024.")
    fn.add(p1, "Published as IVC/2024/Report-7.")
    p1.add_run(" The report cited findings from three independent laboratories")
    fn.add(p1, "Laboratories operated in Vienna, Geneva, and Tokyo respectively.")
    p1.add_run(", confirming earlier estimates of the compound\u2019s stability under pressure.")
    fn.add(p1, "See also Appendix C for the full dataset.")
    p1.add_run(" All measurements were conducted under standard conditions.")

    doc.add_paragraph()  # visual spacer — this empty paragraph is a challenge too

    # "Fourth footnote" — just body text with a Unicode superscript, NOT an OOXML footnote
    p_fake_ref = doc.add_paragraph()
    p_fake_ref.add_run("Methodological note\u00a0")
    superscript_run(p_fake_ref, "4")
    p_fake_ref.add_run(
        ": The sampling interval was adjusted to account for seasonal variation. "
        "Readers should consult the supplementary materials for details."
    )

    # The body-text "footnote definition" — no OOXML link whatsoever
    p_fake_fn = doc.add_paragraph()
    superscript_run(p_fake_fn, "4")
    p_fake_fn.add_run(
        "\u00a0Note on methodology: The seasonal correction factor was derived from a "
        "10-year moving average of baseline measurements taken at monthly intervals. "
        "This annotation is plain body text formatted to resemble a footnote — "
        "it has no OOXML footnote reference and should NOT be parsed as a Footnote element."
    )

    # ════════════════════════════════════════════════════════════════════════
    # §2  Headings and Their Impostors
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("Headings and Their Impostors", level=1)
    doc.add_heading("The Heading Hierarchy", level=2)

    doc.add_paragraph(
        "This section tests whether the parser can distinguish Word heading styles "
        "from visual imitations created with font formatting alone."
    )

    # FAKE HEADING — bold 14 pt, Normal paragraph style (not a heading style)
    p_fake_h1 = doc.add_paragraph()
    r_fake_h1 = p_fake_h1.add_run("The Subcommittee on Subheadings")
    r_fake_h1.bold = True
    r_fake_h1.font.size = Pt(14)
    # paragraph style remains 'Normal' — no w:pStyle heading applied

    doc.add_paragraph(
        "The bold 14 pt paragraph above masquerades as a heading. "
        "A style-aware parser will recognise it has no heading paragraph style; "
        "a naive parser keying only on font weight and size may promote it."
    )

    # FAKE HEADING — all-caps body text, visually prominent
    p_allcaps = doc.add_paragraph()
    r_allcaps = p_allcaps.add_run("THIS ENTIRELY UPPERCASE PARAGRAPH COULD FOOL A NAIVE CLASSIFIER")
    r_allcaps.font.size = Pt(11)
    # Still Normal style — only text transform makes it look like a section title

    doc.add_paragraph(
        "The all-caps paragraph above is also styled as body text. "
        "Its visual prominence is an artefact of the author typing in uppercase, "
        "not of applying a heading style."
    )

    doc.add_heading("Genuine Subsection", level=3)
    doc.add_paragraph(
        "Only this paragraph carries a proper Word heading style (Heading 3). "
        "A compliant parser must preserve the three-level gap between the H1 above "
        "and this H3 — no implicit H2 should be inferred."
    )

    # ════════════════════════════════════════════════════════════════════════
    # §3  The List Labyrinth
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("The List Labyrinth", level=1)

    # ── Real bullet list, three nesting levels ───────────────────────────────
    doc.add_paragraph("Alpha-level item", style="List Bullet")
    doc.add_paragraph("Beta-level item (nested once)", style="List Bullet 2")
    doc.add_paragraph("Gamma-level item (nested twice)", style="List Bullet 3")
    doc.add_paragraph("Another gamma-level item — parser should see list continuation", style="List Bullet 3")
    doc.add_paragraph("Beta-level item two", style="List Bullet 2")
    doc.add_paragraph("Alpha-level item two", style="List Bullet")
    doc.add_paragraph("Alpha-level item three", style="List Bullet")

    doc.add_paragraph()  # spacer

    # ── Fake numbered list — plain paragraphs, no list style ────────────────
    doc.add_paragraph("1. The first pseudo-item stands alone as a plain paragraph.")
    doc.add_paragraph("2. The second pseudo-item looks numbered but lacks list formatting.")
    doc.add_paragraph("3. The third pseudo-item completes the impostor sequence.")
    doc.add_paragraph("   a. This sub-item is also fake — just indented plain text.")

    doc.add_paragraph()  # spacer

    # ── Real numbered list ───────────────────────────────────────────────────
    doc.add_paragraph("Genuine first numbered item", style="List Number")
    doc.add_paragraph("Genuine nested numbered item", style="List Number 2")
    doc.add_paragraph("Genuinely deep numbered item", style="List Number 3")
    doc.add_paragraph("Genuine second numbered item", style="List Number")

    # ════════════════════════════════════════════════════════════════════════
    # §4  Tables: Truth Versus Decoration
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("Tables: Truth Versus Decoration", level=1)

    # Caption paragraph — italic body text, NOT semantically linked to the table
    p_cap = doc.add_paragraph()
    r_cap = p_cap.add_run("Table 1: Regional Performance Data (2020\u20132023)")
    r_cap.italic = True

    # ── Real data table: merged header + data rows ───────────────────────────
    real_tbl = doc.add_table(rows=5, cols=3)
    real_tbl.style = "Table Grid"
    # Row 0: merged header spanning all three columns
    real_tbl.cell(0, 0).merge(real_tbl.cell(0, 2))
    hdr = real_tbl.cell(0, 0).paragraphs[0]
    run_hdr = hdr.add_run("Annual Performance Summary")
    run_hdr.bold = True
    # Column headers
    real_tbl.cell(1, 0).text = "Region"
    real_tbl.cell(1, 1).text = "YoY Growth"
    real_tbl.cell(1, 2).text = "Absolute Change"
    # Data rows
    real_tbl.cell(2, 0).text = "North"
    real_tbl.cell(2, 1).text = "8.5%"
    real_tbl.cell(2, 2).text = "+14 units"
    real_tbl.cell(3, 0).text = "South"
    real_tbl.cell(3, 1).text = "5.1%"
    real_tbl.cell(3, 2).text = "+5 units"
    real_tbl.cell(4, 0).text = "Total"
    real_tbl.cell(4, 1).text = "7.1%"
    real_tbl.cell(4, 2).text = "+19 units"

    doc.add_paragraph()  # spacer

    # ── 1×1 table used as a visual separator (no data content) ──────────────
    sep_tbl = doc.add_table(rows=1, cols=1)
    sep_tbl.cell(0, 0).text = ""
    set_table_borders(sep_tbl, val="single", sz="12", color="4472C4")  # thick blue border

    doc.add_paragraph()  # spacer

    # ── Layout table: 2-column side-by-side content, no visible borders ─────
    layout_tbl = doc.add_table(rows=1, cols=2)
    layout_tbl.cell(0, 0).paragraphs[0].add_run(
        "Key Findings\n"
        "The northern region showed consistent growth of 8.5% year-over-year, "
        "driven primarily by expansion into adjacent markets and new product lines."
    )
    layout_tbl.cell(0, 1).paragraphs[0].add_run(
        "Recommendations\n"
        "Expand northern operations while monitoring southern performance metrics "
        "more closely. A review cycle of six months is recommended."
    )
    remove_table_borders(layout_tbl)

    # ════════════════════════════════════════════════════════════════════════
    # §5  Nested Structures
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("Nested Structures", level=1)

    # ── Nested table: outer 2×2, inner 2×2 inside cell (1,1) ────────────────
    outer_tbl = doc.add_table(rows=2, cols=2)
    outer_tbl.style = "Table Grid"
    outer_tbl.cell(0, 0).text = "Outer A1: project name"
    outer_tbl.cell(0, 1).text = "Outer A2: lead analyst"
    outer_tbl.cell(1, 0).text = "Outer B1: start date"
    # Cell (1,1) hosts the inner table
    inner_cell = outer_tbl.cell(1, 1)
    inner_tbl = inner_cell.add_table(rows=2, cols=2)
    inner_tbl.style = "Table Grid"
    inner_tbl.cell(0, 0).text = "Inner metric"
    inner_tbl.cell(0, 1).text = "Inner value"
    inner_tbl.cell(1, 0).text = "Throughput"
    inner_tbl.cell(1, 1).text = "1 240 req/s"

    doc.add_paragraph()  # spacer

    # ── Fake code block: Courier New + tab indentation + manual line breaks ──
    # A single paragraph — NOT using a Code or Preformatted style.
    p_code = doc.add_paragraph()
    code_lines = [
        "\tdef analyse_report(data: list[float]) -> dict:",
        "\t\tmean = sum(data) / len(data)",
        "\t\tvariance = sum((x - mean) ** 2 for x in data) / len(data)",
        '\t\treturn {"mean": round(mean, 4), "variance": round(variance, 4)}',
    ]
    first = True
    for line in code_lines:
        if not first:
            linebreak(p_code)
        run = p_code.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(10)
        first = False

    # ════════════════════════════════════════════════════════════════════════
    # §6  The Caption Confusion
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("The Caption Confusion", level=1)

    doc.add_paragraph(
        "The diagram below illustrates the document processing architecture described in the previous section:"
    )

    # Inline image (actual PNG bytes, no alt text via high-level API)
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_img = p_img.add_run()
    run_img.add_picture(make_placeholder_image(), width=Inches(3.0))

    # Caption paragraph — italic plain text, NO semantic link to the image above
    p_img_cap = doc.add_paragraph()
    r_img_cap = p_img_cap.add_run(
        "Figure 1: The document parsing and chunking pipeline as deployed in v2.3. "
        "Arrows indicate data flow; shaded boxes represent async processing stages."
    )
    r_img_cap.italic = True

    doc.add_paragraph(
        "Note that the pipeline processes documents asynchronously via a "
        "queue-based architecture with configurable concurrency limits."
    )

    # ════════════════════════════════════════════════════════════════════════
    # §7  The Chaos Zone
    # ════════════════════════════════════════════════════════════════════════
    doc.add_heading("The Chaos Zone", level=1)

    # Manual line breaks inside ONE paragraph — looks like four paragraphs visually
    p_chaos = doc.add_paragraph()
    p_chaos.add_run("Chaos line one: The plenary session commenced at 09:00 sharp.")
    linebreak(p_chaos)
    p_chaos.add_run("Chaos line two: Attendance was recorded by the duty secretary.")
    linebreak(p_chaos)
    p_chaos.add_run("Chaos line three: The minutes of the previous meeting were adopted without amendment.")
    linebreak(p_chaos)
    p_chaos.add_run("Chaos line four: The session was adjourned sine die at 10:45.")

    # Three empty spacing paragraphs — layout hack that should be dropped
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()

    # Unicode math — the parser should preserve these codepoints verbatim
    doc.add_paragraph(
        "Statistical analysis revealed that \u03c3\u00b2 = \u03a3(x\u1d62 \u2212 \u03bc)\u00b2\u00a0/\u00a0n = 14.3, "
        "where n\u00a0=\u00a0847 observations. "
        "The significance threshold \u03b1 was set to 0.05 throughout; "
        "all p-values below this threshold are marked with an asterisk."
    )

    # Two more empty paragraphs
    doc.add_paragraph()
    doc.add_paragraph()

    # Footer-like body paragraph — grey, small, centred, looks like a page footer
    # but is plain body text with no Word footer section.
    p_footer = doc.add_paragraph()
    r_footer = p_footer.add_run(
        "CONFIDENTIAL\u2003|\u2003Internal Use Only\u2003|\u2003Document Ref: TMP-2024-001\u2003|\u2003Page 12 of 15"
    )
    r_footer.font.size = Pt(9)
    r_footer.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    p_footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

    return doc


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.save(str(OUTPUT_PATH))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
