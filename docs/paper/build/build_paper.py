# -*- coding: utf-8 -*-
"""Rebuild the Thai conference paper around EGAT-MAPPO: python docs/paper/build/build_paper.py SRC.docx OUT.docx
Keeps the title block and the paper's own formatting; the body runs in two columns, and each "wide" block
(full-width tables) gets its own single-column section, as in the source paper."""
import copy
import sys
from pathlib import Path

from docx.oxml.ns import qn
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
import paper_content as C  # noqa: E402
from build_thesis import W, Builder, text_of  # noqa: E402

CELL_RPR = ('<w:rPr><w:rFonts w:ascii="TH Sarabun New" w:hAnsi="TH Sarabun New" w:eastAsia="TH Sarabun New" '
            'w:cs="TH Sarabun New"/><w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')


class PaperBuilder(Builder):
    def __init__(self, src):  # noqa: D107 - templates differ from the thesis
        import docx
        self.d = docx.Document(src)
        self.body = self.d.element.body
        self.n_tab = self.n_fig = 0
        kids = list(self.body.iterchildren())
        para = lambda pred: next(el for el in kids if el.tag == qn("w:p") and pred(text_of(el).strip()))
        self.t_h1 = para(lambda t: t == "1 บทนำ")
        self.t_h2 = para(lambda t: t.startswith("2.1 "))
        self.t_p = para(lambda t: t.startswith("การจัดสรรช่องต้องพิจารณา"))
        self.t_ct = para(lambda t: t.startswith("ตาราง 1 "))
        self.t_cf = para(lambda t: t.startswith("ภาพ 1 "))
        self.t_img = next(el for el in kids if el.tag == qn("w:p") and el.xpath(".//w:drawing"))
        self.t_ref = para(lambda t: t.startswith("[1] "))
        sects = [el.find(qn("w:pPr") + "/" + qn("w:sectPr")) for el in kids
                 if el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None]
        two = [s for s in sects if 'w:num="2"' in etree.tostring(s).decode() and s.find(qn("w:type")) is not None]
        one = [s for s in sects if 'w:num="2"' not in etree.tostring(s).decode() and s.find(qn("w:type")) is not None]
        self.sect2, self.sect1 = two[0], one[0]  # continuous two-column / one-column section ends
        self.cell_rpr = CELL_RPR

    def h1(self, s):
        return [self._para(self.t_h1, [(s, False, False)])]

    def p(self, s):
        return [self._para(self.t_p, [(s, False, False)])]

    def ct(self, s):
        self.n_tab += 1
        return [self._para(self.t_ct, [(f"ตาราง {self.n_tab} ", True, False), (s, False, False)])]

    def cf(self, s):
        self.n_fig += 1
        return [self._para(self.t_cf, [(f"ภาพ {self.n_fig} ", True, False), (s, False, False)])]

    def _sect_end(self, sect):
        p = etree.Element(qn("w:p"))
        ppr = etree.SubElement(p, qn("w:pPr"))
        etree.SubElement(ppr, qn("w:spacing")).set(qn("w:after"), "0")
        ppr.append(copy.deepcopy(sect))
        return p

    def wide(self, blocks):
        """End the two-column section, render blocks full width, then end that one-column section."""
        return [self._sect_end(self.sect2)] + self.render(blocks) + [self._sect_end(self.sect1)]

    def tbl(self, rows, widths, opts=None):
        out = super().tbl(rows, widths, opts, check=False)
        for r in out[0].iter(qn("w:r")):
            rpr = etree.fromstring(f'<w:x xmlns:w="{W}">{self.cell_rpr}</w:x>')[0]
            if r.find(qn("w:rPr")) is not None:
                rpr.insert(0, etree.Element(qn("w:b")))
                r.remove(r.find(qn("w:rPr")))
            r.insert(0, rpr)
        return out

    def run(self, out):
        eq = lambda s: (lambda t: t == s)
        body = self.render(C.BODY) + [self._para(self.t_ref, [(r, False, False)]) for r in C.REFS]
        start = self.index(eq("1 บทนำ"))
        kids = self.children()
        final = kids[-1]  # body-level sectPr (last section: one column)
        for el in kids[start:-1]:
            self.body.remove(el)
        for el in body:
            final.addprevious(el)
        # the body ends in the two-column layout: close it before the final (one-column) section
        final.addprevious(self._sect_end(self.sect2))
        a_en = self.index(eq("Abstract"))
        sect_end = next(i for i, el in enumerate(self.children()) if i > a_en and el.tag == qn("w:p")
                        and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None)
        self.replace(a_en + 1, sect_end, self.render(C.ABSTRACT_EN))
        a_th = self.index(eq("บทคัดย่อ"))
        self.replace(a_th + 1, self.index(eq("Abstract")), self.render(C.ABSTRACT_TH))
        self.d.save(out)
        print("saved", out, "tables", self.n_tab, "figures", self.n_fig)


if __name__ == "__main__":
    PaperBuilder(sys.argv[1]).run(sys.argv[2])
