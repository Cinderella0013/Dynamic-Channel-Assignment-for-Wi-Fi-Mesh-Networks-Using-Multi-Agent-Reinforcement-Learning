# -*- coding: utf-8 -*-
"""Rebuild the Thai conference paper around EGAT-MAPPO: python docs/paper/build/build_paper.py SRC.docx OUT.docx
Keeps the title block and the paper's own formatting. As in the source paper (Research_Paper_NW14), sections up to
the method run in two columns (each "wide" block gets its own single-column section) and everything from the
experiment design onward, references included, runs in one column."""
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
        self.flat = False  # True once the body has switched to one column

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
        if self.flat:
            return self.render(blocks)
        return [self._sect_end(self.sect2)] + self.render(blocks) + [self._sect_end(self.sect1)]

    def img(self, name, width_cm):
        return super().img(name, max(width_cm, 10.5) if self.flat else width_cm)

    def tbl(self, rows, widths, opts=None):
        if self.flat and sum(widths) < 9026:  # column-width table in the one-column part: stretch to text width
            widths = [round(w * 9026 / sum(widths)) for w in widths]
            widths[-1] += 9026 - sum(widths)
        out = super().tbl(rows, widths, opts, check=False)
        # source-paper table look: pale grey rules, white body cells, tight margins
        for el in out[0].iter(qn("w:top"), qn("w:left"), qn("w:bottom"), qn("w:right"), qn("w:insideH"), qn("w:insideV")):
            if el.get(qn("w:color")) == "BFBFBF":
                el.set(qn("w:color"), "D9D9D9")
        for shd in out[0].iter(qn("w:shd")):
            if shd.get(qn("w:fill")) == "F7F9FC":
                shd.set(qn("w:fill"), "FFFFFF")
        for kn in out[0].findall(qn("w:tr"))[-1].iter(qn("w:keepNext")):  # don't chain the next block to the table
            kn.getparent().remove(kn)
        for mar in out[0].iter(qn("w:tcMar")):
            for side in mar:
                side.set(qn("w:w"), "25" if side.tag in (qn("w:top"), qn("w:bottom")) else "55")
        for r in out[0].iter(qn("w:r")):
            rpr = etree.fromstring(f'<w:x xmlns:w="{W}">{self.cell_rpr}</w:x>')[0]
            if r.find(qn("w:rPr")) is not None:
                rpr.insert(0, etree.Element(qn("w:b")))
                r.remove(r.find(qn("w:rPr")))
            r.insert(0, rpr)
        return out

    def run(self, out):
        eq = lambda s: (lambda t: t == s)
        cut = next(i for i, b in enumerate(C.BODY) if b[0] == "h1" and b[1].startswith("5 "))  # experiment design
        body = self.render(C.BODY[:cut]) + [self._sect_end(self.sect2)]
        self.flat = True
        body += self.render(C.BODY[cut:]) + [self._para(self.t_ref, [(r, False, False)]) for r in C.REFS]
        start = self.index(eq("1 บทนำ"))
        kids = self.children()
        final = kids[-1]  # body-level sectPr (last section: one column)
        for el in kids[start:-1]:
            self.body.remove(el)
        for el in body:
            final.addprevious(el)
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
