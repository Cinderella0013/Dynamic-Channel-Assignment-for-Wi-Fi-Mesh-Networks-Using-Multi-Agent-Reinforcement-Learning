# -*- coding: utf-8 -*-
"""English paper draft: python docs/paper/build/build_paper_en.py OUT.docx
Single-column A4 draft from paper_content_en.py; tables numbered above, figures below. PDF: render.ps1."""
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
import paper_content_en as C  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
FONT = "Times New Roman"


def para(d, text="", size=10, bold=False, italic=False, align=None, space_after=4):
    p = d.add_paragraph()
    r = p.add_run(text)
    r.font.size, r.bold, r.italic = Pt(size), bold, italic
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def table(d, rows, widths, hl=()):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            r = c.paragraphs[0].add_run(val)
            r.font.size, r.bold = Pt(8.5), i == 0 or i in hl
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
    d.add_paragraph().paragraph_format.space_after = Pt(2)


def build(out):
    d = docx.Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = FONT, Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for s in d.sections:
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.2)
    center = WD_ALIGN_PARAGRAPH.CENTER
    para(d, C.TITLE, 15, bold=True, align=center, space_after=8)
    para(d, C.AUTHORS, 11, align=center, space_after=2)
    para(d, C.AFFIL, 10, italic=True, align=center, space_after=2)
    para(d, C.EMAILS, 10, align=center, space_after=10)
    para(d, "Abstract", 10, bold=True, space_after=2)
    for kind, text in C.ABSTRACT:
        p = para(d, text, 9.5, italic=kind == "p", bold=False)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    n_tab = n_fig = 0
    for item in C.BODY:
        kind = item[0]
        if kind == "h1":
            para(d, item[1], 12, bold=True, space_after=4).paragraph_format.space_before = Pt(8)
        elif kind == "h2":
            para(d, item[1], 10.5, bold=True, italic=True, space_after=3).paragraph_format.space_before = Pt(4)
        elif kind == "p":
            para(d, item[1]).alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "note":
            para(d, item[1], 8.5, italic=True)
        elif kind == "ct":
            n_tab += 1
            para(d, f"Table {n_tab}. {item[1]}", 9, bold=True, align=center, space_after=2)
        elif kind == "tbl":
            table(d, item[1], item[2], (item[3] if len(item) > 3 else {}).get("hl", ()))
        elif kind == "img":
            d.add_picture(str(FIG / item[1]), width=Inches(item[2]))
            d.paragraphs[-1].alignment = center
        elif kind == "cf":
            n_fig += 1
            para(d, f"Fig. {n_fig}. {item[1]}", 9, align=center, space_after=8)
        else:
            raise ValueError(kind)
    para(d, "References", 12, bold=True, space_after=4).paragraph_format.space_before = Pt(8)
    for ref in C.REFS:
        para(d, ref, 8.5, space_after=2)
    d.save(out)
    print("saved", out, "tables", n_tab, "figures", n_fig)


if __name__ == "__main__":
    build(sys.argv[1])
