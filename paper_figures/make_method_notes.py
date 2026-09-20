# -*- coding: utf-8 -*-
"""
make_method_notes.py -- one feedback file per method document.

    method_docs/01_device_simulation_and_dataset.md
    method_docs/02_ray_based_measurement.md
    ...

Each note lists that part's sections and figures as they currently stand, so
feedback can point at something precisely -- "S2 P3", "Fig. 4" -- instead of
describing it.  Write under the headings; anything under "Notes" is read too.

NEVER OVERWRITES AN EXISTING NOTE.  Feedback is the one thing here that
cannot be regenerated, so a note that is already on disk is left exactly as
it is and reported as kept.  Use --refresh-inventory to update only the
bracketed inventory block at the top of each note, which is rewritten
between the INVENTORY markers and leaves everything below untouched.

    python paper_figures/make_method_notes.py
    python paper_figures/make_method_notes.py --refresh-inventory
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_method_docs as M                                   # noqa: E402

BEGIN = "<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->"
END = "<!-- INVENTORY:END  write your feedback below -->"

TEMPLATE = """# Part {number}. {title}

{begin}

**Document**  `{slug}.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
{sections}

### Figures
{figures}

{end}

## Notes

<!--
How to point at things:
  S2 P3        section 2, paragraph 3
  Fig. 4       figure 4 of this part
  ABSTRACT     the abstract
  FORMAT       font, spacing, margins, captions
Anything else in prose is fine too.

What can be changed, and where it lives:
  wording, a section, the abstract   paper_figures/make_method_docs.py
  a figure itself                    paper_figures/make_figures.py
                                     (or make_tau_grids.py /
                                      make_threshold_figure.py)
  which figures a part carries       the parts() list in make_method_docs.py
  fonts, spacing, margins            the format block in make_method_docs.py
-->

-
"""


def inventory(sections, figures):
    sec = "\n".join(
        "%d. **%s** — %d paragraph%s"
        % (i, head, len(paras), "" if len(paras) == 1 else "s")
        for i, (head, paras) in enumerate(sections, 1))
    fig = "\n".join(
        "%d. `%s`\n   > %s" % (i, name, cap)
        for i, (name, cap) in enumerate(figures, 1))
    return sec, fig


def write(number, slug, title, sections, figures, refresh):
    path = os.path.join(M.OUT_DIR, "%02d_%s.md" % (number, slug))
    sec, fig = inventory(sections, figures)
    block = TEMPLATE.format(number=number, title=title,
                            slug="%02d_%s" % (number, slug),
                            sections=sec, figures=fig,
                            begin=BEGIN, end=END)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(block)
        return "created"

    if not refresh:
        return "kept (has your feedback)"

    # rewrite ONLY between the markers, so the notes below survive
    old = open(path, encoding="utf-8").read()
    if BEGIN not in old or END not in old:
        return "kept (markers missing -- not touched)"
    head, rest = old.split(BEGIN, 1)
    _stale, tail = rest.split(END, 1)
    new_mid = block.split(BEGIN, 1)[1].split(END, 1)[0]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + BEGIN + new_mid + END + tail)
    return "inventory refreshed"


def main():
    refresh = "--refresh-inventory" in sys.argv
    os.makedirs(M.OUT_DIR, exist_ok=True)
    for number, slug, title, _abstract, sections, figures in M.parts():
        what = write(number, slug, title, sections, figures, refresh)
        print("  %-40s %s" % ("%02d_%s.md" % (number, slug), what))
    print()
    print("notes ->", M.OUT_DIR)


if __name__ == "__main__":
    main()
