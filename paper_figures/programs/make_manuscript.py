# -*- coding: utf-8 -*-
"""
make_manuscript.py -- the parts, assembled into one JJAP manuscript.

The parts in method_docs/ are written to be read one at a time, so each
one starts its figures at 1.  This script builds the paper they are parts
OF: one document, from the same JJAP template, with

    * the author block and the abstract once, at the front
    * every section under the number its folder already carries --
      2. Introduction, 3. Method, 3.1 ... 3.4, 4, 5
    * ONE run of figure numbers (Fig. 1 ... Fig. 22) and one run of table
      numbers (Table I ...), continuous across the parts
    * the references, then the tables, then the figures, at the end --
      the order JJAP asks for, which a per-part document cannot give
      because its figures have to sit under its own text

It is not a merge of the .docx files: it is a build from the same source
the parts are built from, so no number, caption or reference can be a
stale copy of one in a part document.

    python paper_figures/programs/make_manuscript.py     # Manuscript.docx
    python paper_figures/programs/make_pdfs.py           # and the PDF

The cover letter is not part of the paper and is left out.  The hand-edit
guard is the same as the parts': a Manuscript.docx that Word has touched
is never overwritten, the rebuild goes beside it as
Manuscript_rebuilt.docx, and the copy replaced goes to backups/.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_method_docs as M                                   # noqa: E402

OUT = os.path.join(M.OUT_DIR, "Manuscript.docx")

# The four 3.x parts are subsections of one section that no part carries a
# title for, because it is not a part: the method itself.
METHOD_HEADING = "3. Method"


def _subheading(label, index, heading):
    """`3.2` + `1. The ray fan`  ->  `3.2.1. The ray fan`.

    A part numbers its own sections from 1; in the paper those numbers
    hang off the part's number."""
    return "%s.%d. %s" % (label, index, re.sub(r"^\s*\d+\.\s*", "", heading))


def assemble(keep=None, base=(0, 0)):
    """(title, abstract, sections, figures, references) for the whole paper.

    Walks parts() in order, keeping a running count of the figures and
    tables used so far and rewriting each part's text through
    M.renumber with it.

    `keep(label)` limits the output to some parts (make_methods takes the
    3.x ones); the others are still counted, so the numbers stay the
    paper's.  `base` is subtracted from that count, for a document whose
    build() adds it back as fig_offset / tbl_offset."""
    sections, figures, references = [], [], ()
    abstract = None
    fig_at = tbl_at = 0
    method_open = False

    for spec in M.parts():
        number, slug, title, part_abstract, part_sections, part_figures = \
            spec[:6]
        extra = spec[6] if len(spec) > 6 else {}
        if number == "0":
            continue                 # the cover letter is not the paper
        if number == "1":
            # the title page: its abstract is the paper's abstract, and the
            # author block build() writes for us is the rest of it
            abstract = part_abstract
            continue
        # The per-part abstracts are a reading aid for a part on its own.
        # In the paper they would be five more abstracts, so they are gone.
        references = extra.get("references", references)

        label = M.shown(number)                       # "2", "3.1", ...
        if keep is not None and not keep(label):
            fig_at += len(part_figures)
            tbl_at += sum(1 for _h, ps in part_sections for p in ps
                          if isinstance(p, tuple))
            continue
        fa, ta = fig_at - base[0], tbl_at - base[1]
        if label.startswith("3.") and not method_open:
            sections.append((METHOD_HEADING, []))
            method_open = True
        sections.append(("%s. %s" % (label, title), []))
        # Only a HEADED subsection takes a number: an unheaded opening
        # paragraph (3.4, 2026-09-26) must not push 3.4.1 to 3.4.2.
        n_headed = 0
        for heading, paragraphs in part_sections:
            if heading:
                n_headed += 1
            body = []
            for text in paragraphs:
                if isinstance(text, tuple):
                    body.append(M.renumber_table(text, fa, ta))
                elif isinstance(text, list):
                    # a list of items; each one is a string in its own right
                    body.append([M.renumber(i, fa, ta)
                                 for i in text])
                else:
                    body.append(M.renumber(text, fa, ta))
            # a part whose text is one continuous section (the
            # introduction) has no subheading to hang off the number
            sections.append((_subheading(label, n_headed, heading)
                             if heading else None, body))

        for fname, caption in part_figures:
            figures.append((fname, M.renumber(caption, fa, ta)))
        fig_at += len(part_figures)
        tbl_at += sum(1 for _h, ps in part_sections for p in ps
                      if isinstance(p, tuple))

    if abstract is None:
        raise SystemExit("no title-page part: nothing to take the abstract "
                         "from")
    return M.PAPER_TITLE, abstract, sections, figures, references


def main():
    title, abstract, sections, figures, references = assemble()
    # The text is already renumbered part by part, so build() must not
    # shift it again: it is handed offsets of zero and prints the figure
    # captions straight off its own running index.
    path, edited = M.build(None, "manuscript", title, abstract, sections,
                           figures, references=references, front_matter=True,
                           out=OUT)
    n_tables = sum(1 for _h, ps in sections for p in ps
                   if isinstance(p, tuple))
    print("  %-40s %d headings, %d figures, %d tables"
          % (os.path.basename(path), len(sections), len(figures), n_tables))
    if edited:
        print()
        print("Manuscript.docx was edited by hand, so it was NOT touched.")
        print("The rebuild is beside it as Manuscript_rebuilt.docx.")
    print()
    print("-> %s" % path)
    return path


if __name__ == "__main__":
    main()
