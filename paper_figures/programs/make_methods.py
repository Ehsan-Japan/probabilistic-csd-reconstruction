# -*- coding: utf-8 -*-
"""
make_methods.py -- section 3, the method, as one Word document.

    method_docs/3_methods/3_Methods.docx

The four subsections 3.1 ... 3.4 are written as four part documents so
they can be read and commented on one at a time.  This puts them together
as the section they are: one "3. Method" document, 3.1 ... 3.4 under it,
and the figures and tables of all four at the end.

It is built the way make_manuscript builds the paper -- from the same
source as the parts, not by merging their .docx files -- and the figure
and table numbers are the paper's (the first method figure is the one
after the introduction's), so a number here is the number in
Manuscript.docx.

    python paper_figures/programs/make_methods.py
    python paper_figures/programs/make_pdfs.py          # and the PDF

Same hand-edit guard as every other document: a 3_Methods.docx that Word
has touched is never overwritten; the rebuild goes beside it as
3_Methods_rebuilt.docx and the replaced copy goes to backups/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _figures                                                # noqa: E402
import make_method_docs as M                                   # noqa: E402
import make_manuscript as MS                                   # noqa: E402

OUT = os.path.join(M.OUT_DIR, _figures.METHODS, "3_Methods.docx")


def _is_method(label):
    return label.startswith("3.")


def offsets():
    """Figures and tables the paper has used before section 3."""
    fig_at = tbl_at = 0
    for spec in M.parts():
        number, sections, figures = spec[0], spec[4], spec[5]
        extra = spec[6] if len(spec) > 6 else {}
        if number in ("0", "1") or extra.get("letter") \
                or extra.get("title_page"):
            continue
        if _is_method(M.shown(number)):
            break
        fig_at += len(figures)
        tbl_at += sum(1 for _h, ps in sections for p in ps
                      if isinstance(p, tuple))
    return fig_at, tbl_at


def main():
    base = offsets()
    _title, _abstract, sections, figures, _refs = MS.assemble(
        keep=_is_method, base=base)
    # the document's own title is the section heading
    if sections and sections[0][0] == MS.METHOD_HEADING:
        sections = sections[1:]
    path, edited = M.build(None, "methods", MS.METHOD_HEADING, None,
                           sections, figures, fig_offset=base[0],
                           tbl_offset=base[1], out=OUT)
    n_tables = sum(1 for _h, ps in sections for p in ps
                   if isinstance(p, tuple))
    print("  %-40s %d headings, Fig. %d-%d, %d tables"
          % (os.path.basename(path), len(sections), base[0] + 1,
             base[0] + len(figures), n_tables))
    if edited:
        print()
        print("3_Methods.docx was edited by hand, so it was NOT touched.")
        print("The rebuild is beside it as 3_Methods_rebuilt.docx.")
    print()
    print("-> %s" % path)
    return path


if __name__ == "__main__":
    main()
