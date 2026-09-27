# -*- coding: utf-8 -*-
"""
make_method_notes.py -- one comment sheet per part, as a plain .txt.

    method_docs/0_Cover_letter/0_Cover_letter.txt
    method_docs/3_methods/3_Methods.txt       (3.1 ... 3.4 together)
    ...

The sheet is named after the folder it sits in, opens in Notepad, and is
written in two halves: YOU WRITE, where the requests go, and THE REPLY,
where what was actually changed is written back.  One file is the whole
conversation about that part, beside the document it is about.

NEVER OVERWRITES AN EXISTING SHEET.  A comment is the one thing here that
cannot be regenerated, so a sheet already on disk is left exactly as it is
and reported as kept.  --refresh-inventory rewrites only the bracketed
block at the top, which lists the part's sections and figures, and leaves
everything below it untouched.

    python paper_figures/programs/make_method_notes.py
    python paper_figures/programs/make_method_notes.py --refresh-inventory
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # programs/
sys.path.insert(0, HERE)

import _figures                                                # noqa: E402
import make_method_docs as M                                   # noqa: E402

BEGIN = "[INVENTORY -- regenerated, do not write in this block]"
END = "[END INVENTORY]"

RULE = "=" * 70
THIN = "-" * 70

TEMPLATE = """{rule}
 {number}.  {utitle}
 Document:  {slug}.docx  /  {slug}.pdf
{rule}

{begin}

 SECTIONS
{sections}

 FIGURES
{figures}

{end}


{rule}
 YOU WRITE  --  anything below this line is read
{rule}

 Point at something:   S2 P3   section 2, paragraph 3
                       FIG 4   figure 4 of this part
                       TABLE I / ABSTRACT / TITLE
                       FORMAT  font, spacing, margins, captions
 Or just say it in prose -- "the third paragraph is too long" is enough.

 One request per line, starting with a dash.  Nothing here is ever
 deleted or overwritten by a rebuild.

-
-
-


{rule}
 THE REPLY  --  what was actually changed, and where
{rule}

 (nothing yet)


{rule}
 HOW THIS PART IS BUILT
{rule}

 The document is generated, so a request is applied by editing the
 generator and rebuilding -- that way it survives the next rebuild:

   wording, a section, the abstract   programs/make_method_docs.py
   which figures this part carries    the parts() list in the same file
   a figure itself                    programs/make_figures.py
                                      (or make_tau_grids.py,
                                       make_threshold_figure.py)
   fonts, spacing, margins            the format block in
                                      make_method_docs.py
   the cover letter's look            templates/cover_letter-RP.docx

   python paper_figures/programs/make_method_docs.py    the .docx
   python paper_figures/programs/make_pdfs.py           the .pdf

 IF YOU EDIT THE WORD FILE YOURSELF -- that is safe.  Every document is
 fingerprinted when it is written, so a rebuild can tell you changed it:

   * a document you edited is NEVER overwritten;
   * the new machine-made version is written beside it as
     {slug}_rebuilt.docx, and the run says so;
   * the copy any replaced document had before goes to backups/ with a
     timestamp, and the last ten are kept;
   * make_pdfs.py exports whatever .docx is on disk, so the PDF matches
     the Word file, hand-edited or not.

 What that does NOT do is put your edit back into the generator.  So if
 an edit should become the way the document is built, say so above in
 one line -- "keep my S1 P2 rewrite" -- and it will be moved into
 make_method_docs.py.
"""


def inventory(sections, figures):
    sec = "\n".join(
        "  %d. %s  -- %d paragraph%s"
        % (i, head or "(continuous text, no heading)", len(paras),
           "" if len(paras) == 1 else "s")
        for i, (head, paras) in enumerate(sections, 1))
    fig = "\n".join(
        "  %d. %s\n     %s" % (i, name, _wrap(cap))
        for i, (name, cap) in enumerate(figures, 1))
    return sec or "  (none)", fig or "  (none)"


def _wrap(text, width=64, indent=" " * 5):
    """Captions are long; a comment sheet is read in Notepad."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    lines.append(line)
    return ("\n" + indent).join(lines)


def write(number, slug, title, sections, figures, refresh):
    path = os.path.join(M.part_dir(number, slug),
                        M.stem(number, slug) + ".txt")
    sec, fig = inventory(sections, figures)
    block = TEMPLATE.format(number=M.shown(number), utitle=title.upper(),
                            slug=M.stem(number, slug),
                            sections=sec, figures=fig,
                            begin=BEGIN, end=END, rule=RULE)

    if not os.path.exists(path):
        _save(path, block)
        return "created"

    if not refresh:
        return "kept (has your comments)"

    # rewrite ONLY between the markers, so everything you wrote survives
    with open(path, encoding="utf-8") as fh:
        old = fh.read()
    if BEGIN not in old or END not in old:
        return "kept (markers missing -- not touched)"
    head, rest = old.split(BEGIN, 1)
    _stale, tail = rest.split(END, 1)
    new_mid = block.split(BEGIN, 1)[1].split(END, 1)[0]
    _save(path, head + BEGIN + new_mid + END + tail)
    return "inventory refreshed"


def _save(path, text):
    """CRLF, so the sheet is readable in Notepad as well as in an editor."""
    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(text.replace("\r\n", "\n"))


def main():
    refresh = "--refresh-inventory" in sys.argv
    os.makedirs(M.OUT_DIR, exist_ok=True)
    for spec in M.parts():
        # a part may carry an extras dict (front matter, references)
        number, slug, title, _abstract, sections, figures = spec[:6]
        if M.stem(number, slug) in _figures.METHOD_PARTS:
            # 3.1 ... 3.4 share one sheet, 3_methods/3_Methods.txt: the
            # four sheets they had, one after another.  Left alone here.
            continue
        what = write(number, slug, title, sections, figures, refresh)
        print("  %-40s %s" % (M.stem(number, slug) + ".txt", what))
    print()
    print("comment sheets ->", M.OUT_DIR)


if __name__ == "__main__":
    main()
