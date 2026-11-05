# -*- coding: utf-8 -*-
"""Rebuild the thesis around EGAT-MAPPO: python docs/paper/build/build_thesis.py SRC.docx OUT.docx
Replaces whole body ranges between anchor paragraphs, reusing the source document's own paragraph,
caption and table formatting. Paragraphs carrying a section break (w:sectPr) are never removed."""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
import thesis_content as C  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _script_runs(s):
    """Split text into (segment, is_thai) runs; spaces, digits and punctuation join the run before them."""
    out = []
    for ch in s:
        thai = "฀" <= ch <= "๿"
        latin = ch.isascii() and ch.isalpha() or ch in "αβγλκηεθσ"
        cls = True if thai else (False if latin else None)
        if out and (cls is None or out[-1][1] == cls):
            out[-1][0] += ch
        elif out and out[-1][1] is None:
            out[-1] = [out[-1][0] + ch, cls]
        else:
            out.append([ch, cls])
    return [(seg, bool(c)) for seg, c in out] or [("", False)]


def text_of(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


class Builder:
    def __init__(self, src):
        self.d = docx.Document(src)
        self.body = self.d.element.body
        self.n_tab, self.n_fig = 2, 0  # tables 1-2 (chapter 2-3) are kept as they are
        ps = self.d.paragraphs
        find = lambda pred: next(p._p for p in ps if pred(p))
        self.t_h2 = find(lambda p: p.style.name == "Heading 2" and p.text.strip().startswith("4.4"))
        self.t_h3 = find(lambda p: p.style.name == "Heading 3" and p.text.strip().startswith("4.4.1"))
        self.t_p = find(lambda p: p.text.strip().startswith("เครือข่าย Wi-Fi Mesh [1]"))
        self.t_ct = find(lambda p: p.text.strip().startswith("ตารางที่ 13"))
        self.t_cf = find(lambda p: p.text.strip().startswith("ภาพที่ 1"))
        self.t_img = next(p._p for p in ps[5:] if p._p.xpath(".//w:drawing"))
        self.t_ref = find(lambda p: p.text.strip().startswith("26.  ns-3"))
        self.old_tables = [copy.deepcopy(t._tbl) for t in self.d.tables]

    # ---------- element factories ----------
    def _para(self, tmpl, runs, keep_ppr=True):
        p = etree.SubElement(etree.Element(qn("w:body")), qn("w:p"))
        ppr = tmpl.find(qn("w:pPr"))
        if keep_ppr and ppr is not None:
            ppr = copy.deepcopy(ppr)
            for tag in ("w:pageBreakBefore", "w:rPr", "w:sectPr", "w:numPr"):
                for x in ppr.findall(qn(tag)):
                    ppr.remove(x)
            p.append(ppr)
        base_r = tmpl.find(qn("w:r"))
        base_rpr = base_r.find(qn("w:rPr")) if base_r is not None else None
        for txt, bold, tab in runs:
            for k, (seg, thai) in enumerate(_script_runs(txt)):
                r = etree.SubElement(p, qn("w:r"))
                rpr = copy.deepcopy(base_rpr) if base_rpr is not None else etree.Element(qn("w:rPr"))
                for x in rpr.findall(qn("w:b")) + rpr.findall(qn("w:cs")) + rpr.findall(qn("w:lastRenderedPageBreak")):
                    rpr.remove(x)
                if bold:
                    rpr.insert(0, etree.Element(qn("w:b")))
                if thai:  # Word applies Thai word breaking only to complex-script runs
                    lang = rpr.find(qn("w:lang"))
                    cs = etree.Element(qn("w:cs"))
                    lang.addprevious(cs) if lang is not None else rpr.append(cs)
                if len(rpr):
                    r.append(rpr)
                if tab and k == 0:
                    etree.SubElement(r, qn("w:tab"))
                t = etree.SubElement(r, qn("w:t"))
                t.text = seg
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p

    def h2(self, s):
        return [self._para(self.t_h2, [(s, False, False)])]

    def h3(self, s):
        return [self._para(self.t_h3, [(s, False, False)])]

    def p(self, s):
        el = self._para(self.t_p, [(s, False, False)])
        el.find(qn("w:pPr")).find(qn("w:jc")).addprevious(etree.fromstring(f'<w:ind xmlns:w="{W}" w:firstLine="567"/>'))
        return [el]

    def ct(self, s):
        self.n_tab += 1
        p = self._para(self.t_ct, [(f"ตารางที่ {self.n_tab}", True, False), (s, False, True)])
        if self.n_tab >= 10:  # two-digit label is wider than the 850-twip tab stop
            ppr = p.find(qn("w:pPr"))
            ppr.find(qn("w:tabs")).find(qn("w:tab")).set(qn("w:pos"), "1050")
            ppr.find(qn("w:ind")).set(qn("w:left"), "1050")
            ppr.find(qn("w:ind")).set(qn("w:hanging"), "1050")
        return [p]

    def cf(self, s):
        self.n_fig += 1
        return [self._para(self.t_cf, [(f"ภาพที่ {self.n_fig}", True, False), (s, False, True)])]

    def img(self, name, width_cm):
        para = self.d.add_paragraph()  # oxml-typed element; moved into place by replace()
        para.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
        p = para._p
        p.insert(0, copy.deepcopy(self.t_img.find(qn("w:pPr"))))
        p.getparent().remove(p)
        return [p]

    def tbl(self, rows, widths, check=True):
        assert not check or sum(widths) == 9026, widths
        x = [f'<w:tbl xmlns:w="{W}"><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>']
        x += [f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
              for b in ("top", "left", "bottom", "right", "insideH", "insideV")]
        x.append('</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" '
                 'w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>')
        x += [f'<w:gridCol w:w="{w}"/>' for w in widths]
        x.append("</w:tblGrid>")
        for i, row in enumerate(rows):
            head = i == 0
            x.append("<w:tr><w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if head else "") + '<w:jc w:val="center"/></w:trPr>')
            for j, (cell, w) in enumerate(zip(row, widths)):
                fill = "E8EDF4" if head else "F7F9FC"
                numeric = cell[:1] in "+−-0123456789—"
                jc = "left" if not head and (j == 0 or (len(cell) > 18 and not numeric)) else "center"
                esc = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                x.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
                         '<w:tcMar><w:top w:w="55" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="55" w:type="dxa"/>'
                         '<w:right w:w="80" w:type="dxa"/></w:tcMar><w:vAlign w:val="center"/></w:tcPr>'
                         f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="40"/><w:ind w:firstLine="0"/><w:jc w:val="{jc}"/></w:pPr>'
                         f'<w:r>{"<w:rPr><w:b/></w:rPr>" if head else ""}<w:t xml:space="preserve">{esc}</w:t></w:r></w:p></w:tc>')
            x.append("</w:tr>")
        x.append("</w:tbl>")
        return [etree.fromstring("".join(x))]

    def copy(self, idx, caption):
        return self.ct(caption) + [copy.deepcopy(self.old_tables[idx])]

    def render(self, blocks):
        out = []
        for b in blocks:
            out += getattr(self, b[0])(*b[1:])
        return out

    # ---------- body surgery ----------
    def children(self):
        return list(self.body.iterchildren())

    def index(self, pred, start=0):
        for i, el in enumerate(self.children()):
            if i >= start and el.tag == qn("w:p") and pred(text_of(el).strip()):
                return i
        raise KeyError("anchor not found")

    def replace(self, start, end, new):
        """Remove children [start, end) (keeping section-break paragraphs) and insert new before end."""
        kids = self.children()
        anchor = kids[end]
        for el in kids[start:end]:
            if el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None:
                for r in el.findall(qn("w:r")):
                    el.remove(r)
                if anchor is kids[end]:
                    anchor = el  # new content stays inside the section that ends here
                continue
            self.body.remove(el)
        for el in new:
            anchor.addprevious(el)

    def insert_before(self, i, new):
        anchor = self.children()[i]
        for el in new:
            anchor.addprevious(el)

    def run(self, out):
        eq = lambda s: (lambda t: t == s)
        st = lambda s: (lambda t: t.startswith(s))
        # chapter 5 and 4 first: later edits do not shift earlier anchors
        refs = self.index(st("26.  ns-3"))
        for line in reversed(C.NEW_REFS):
            self.children()[refs].addnext(self._para(self.t_ref, [(line, False, False)]))

        # numbering must follow document order, so render in order and place afterwards
        self.n_tab = 2
        ch3 = self.render(C.CH3_INTRO_AND_31_32)
        ch34 = self.render(C.CH3_34)
        ch4 = self.render(C.CH4)
        ch5 = self.render(C.CH5)

        c5 = self.index(eq("บทสรุป"), self.index(st("บทที่ 5")))
        self.replace(c5 + 1, self.index(eq("เอกสารอ้างอิง"), c5), ch5)
        c4 = self.index(eq("การวิเคราะห์ระบบและผลการทดลอง"), self.index(st("บทที่ 4")))
        self.replace(c4 + 1, self.index(st("บทที่ 5"), c4), ch4)
        s34 = self.index(st("3.4  วิธีประเมิน"))
        self.replace(s34, self.index(st("บทที่ 4"), s34), ch34)
        c3 = self.index(eq("วิธีการดำเนินงาน"), self.index(st("บทที่ 3")))
        self.replace(c3 + 1, self.index(st("3.3  แผน"), c3), ch3)

        # chapter 2: theory 2.1.7-2.1.8, related work 2.2.10-2.2.12, rows of table 1
        self.insert_before(self.index(st("ช่องว่างของงานที่เกี่ยวข้อง")), self.render(C.CH2_RELATED))
        self.insert_before(self.index(st("2.2  งานวิจัยที่เกี่ยวข้อง")), self.render(C.CH2_THEORY))
        t1 = self.d.tables[7]._tbl  # last part of table 1
        rows = t1.findall(qn("w:tr"))
        proto = rows[1]
        last = rows[-1]
        if "การจัดสรรช่อง" in text_of(last):  # stray running-title row: drop it
            t1.remove(last)
            last = rows[-2]
        for vals in reversed(C.TABLE1_ROWS):
            r = copy.deepcopy(proto)
            for tc, v in zip(r.findall(qn("w:tc")), vals):
                ps_ = tc.findall(qn("w:p"))
                for extra in ps_[1:]:
                    tc.remove(extra)
                runs = ps_[0].findall(qn("w:r"))
                for extra in runs[1:]:
                    ps_[0].remove(extra)
                ts = runs[0].findall(qn("w:t"))
                ts[0].text = v
                for extra in ts[1:]:
                    runs[0].remove(extra)
            last.addnext(r)

        # chapter 1 (from the old scope paragraph to the end of the chapter)
        s1 = self.index(st("โครงงานพัฒนาต้นแบบ GAT ร่วมกับ MARL"))
        self.replace(s1, self.index(st("บทที่ 2"), s1), self.render(C.CH1))

        # abstracts
        a_en = self.index(eq("ABSTRACT"))
        self.replace(a_en + 1, self.index(eq("กิตติกรรมประกาศ"), a_en), self.render(C.ABSTRACT_EN))
        a_th = self.index(eq("บทคัดย่อ"))
        self.replace(a_th + 1, self.index(st("Mr.Wissarut"), a_th), self.render(C.ABSTRACT_TH))
        self.toc()
        self.d.save(out)
        print("saved", out, "tables", self.n_tab, "figures", self.n_fig)

    # ---------- table of contents / lists (page numbers filled by fill_pages) ----------
    def toc(self):
        """Rebuild the TOC and the lists of tables and figures from the new headings and captions."""
        ps = [el for el in self.body.iterchildren(qn("w:p"))]

        def style(el):
            st_ = el.find(qn("w:pPr") + "/" + qn("w:pStyle"))
            return st_.get(qn("w:val")) if st_ is not None else ""

        front, chapters, tabs, figs = ["บทคัดย่อ", "ABSTRACT", "กิตติกรรมประกาศ", "สารบัญตาราง", "สารบัญภาพ"], [], {}, {}
        for i, p in enumerate(ps):
            t, s = text_of(p).strip(), style(p)
            if s == "Heading1" and re.match(r"บทที่ \d$", t):
                chapters.append((f"{t}  {text_of(ps[i + 1]).strip()}", True))
            elif s == "Heading2" and re.match(r"\d\.\d\s", t) and chapters:
                chapters.append((re.sub(r"\s+", "  ", t, count=1), False))
            elif s == "NW14Caption" and "(ต่อ)" not in t:
                m = re.match(r"(ตารางที่|ภาพที่) (\d+)\s*(.*)", t)
                if m:
                    (tabs if m[1] == "ตารางที่" else figs).setdefault(int(m[2]), f"{m[1]} {m[2]}  {m[3]}")
        toc_entries = [(x, "front") for x in front] + [(x, "chap" if b else "sec") for x, b in chapters] \
            + [("เอกสารอ้างอิง", "chap"), ("ประวัติผู้เขียน", "chap")]
        tl = self.d.tables
        split = next(k for k, (x, _) in enumerate(toc_entries) if x.startswith("5.1"))
        self._fill_list(tl[0]._tbl, toc_entries[:split + 1])
        self._fill_list(tl[1]._tbl, toc_entries[split + 1:])
        self._fill_list(tl[2]._tbl, [(tabs[k], "sec") for k in sorted(tabs)])
        self._fill_list(tl[4]._tbl, [(figs[k], "sec") for k in sorted(figs)])
        # the old continuation of the list of tables is no longer needed
        t4 = tl[3]._tbl
        head = t4.getprevious()
        if head is not None and "สารบัญตาราง (ต่อ)" in text_of(head):
            sect = head.find(qn("w:pPr") + "/" + qn("w:sectPr")) if head.find(qn("w:pPr")) is not None else None
            if sect is None:
                head.getparent().remove(head)
        t4.getparent().remove(t4)

    def _fill_list(self, tbl, entries):
        rows = tbl.findall(qn("w:tr"))
        proto = {"front": rows[1], "chap": next((r for r in rows if text_of(r).startswith("บทที่")), rows[1]),
                 "sec": rows[min(2, len(rows) - 1)]}
        sec = next((r for r in rows if re.match(r"\d\.\d|ตารางที่|ภาพที่", text_of(r).strip())), None)
        if sec is not None:
            proto["sec"] = sec
        proto = {k: copy.deepcopy(v) for k, v in proto.items()}
        for r in rows[1:]:
            tbl.remove(r)
        for txt, kind in entries:
            r = copy.deepcopy(proto[kind])
            cells = r.findall(qn("w:tc"))
            for tc, v in zip(cells, (txt, "")):
                for p in tc.findall(qn("w:p"))[1:]:
                    tc.remove(p)
                p0 = tc.find(qn("w:p"))
                runs = p0.findall(qn("w:r"))
                for x in runs[1:]:
                    p0.remove(x)
                if runs:
                    for x in runs[0]:
                        if x.tag != qn("w:rPr"):
                            runs[0].remove(x)
                    t = etree.SubElement(runs[0], qn("w:t"))
                else:
                    t = etree.SubElement(etree.SubElement(p0, qn("w:r")), qn("w:t"))
                t.text = v
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            tbl.append(r)


def _norm(s):
    return re.sub(r"\s+", "", s.replace("ํา", "ำ"))


def fill_pages(docx_path, pdf_path):
    """Second pass: write printed page numbers into the TOC/list tables from a Word-rendered PDF."""
    import pymupdf
    pages = [_norm(p.get_text()) for p in pymupdf.open(pdf_path)]
    firsts = [p.get_text().strip().split("\n")[0].strip() for p in pymupdf.open(pdf_path)]
    d = docx.Document(docx_path)
    lists_end = next(i for i, p in enumerate(pages) if "บทที่1" in p[:12])  # first body page
    off = lists_end  # printed page 1 is pdf page lists_end+1

    def page_of(txt, start):
        key = _norm(txt)[:22]
        for i in range(start, len(pages)):
            if key and key in pages[i]:
                return i
        return None

    for t in d.tables[:4]:
        for r in t._tbl.findall(qn("w:tr"))[1:]:
            cells = r.findall(qn("w:tc"))
            label = text_of(cells[0]).strip()
            front = label in ("บทคัดย่อ", "ABSTRACT", "กิตติกรรมประกาศ", "สารบัญตาราง", "สารบัญภาพ")
            if front:
                i = next((k for k in range(1, lists_end)
                          if _norm(label) in pages[k] and "สารบัญหน้า" not in pages[k][:30]), None)
                num = firsts[i] if i is not None else ""
            else:
                key = label.split("  ")[0] if label.startswith("บทที่") else label
                if label.startswith(("ตารางที่", "ภาพที่")):
                    key = label.replace("  ", "")
                i = page_of(key, lists_end)
                num = str(i - off + 1) if i is not None else ""
            ts = list(cells[1].iter(qn("w:t")))
            ts[0].text = num
    d.save(docx_path)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--pages":
        fill_pages(sys.argv[2], sys.argv[3])
    else:
        Builder(sys.argv[1]).run(sys.argv[2])
