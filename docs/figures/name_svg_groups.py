# -*- coding: utf-8 -*-
"""
name_svg_groups.py — give every label in a dvisvgm SVG a name.

    python docs/figures/name_svg_groups.py

dvisvgm already wraps each piece of text in its own <g transform=...>, so a
label is one group by the time Illustrator opens the SVG -- which is why the
SVG, not the PDF, is the file to edit: a PDF's text arrives as one object per
font change, and "c_{s_1g_1}" is three of those.

What this adds is the NAME.  It writes id='c_s1g1' on each group, so the
Layers panel says what you are grabbing instead of listing twenty-one
<Group>s.  The names are assigned in drawing order, which is the order the
labels appear in the .tex; the glyph count of each group is checked against
the expected label before anything is written, so a change to the figure that
moves a label makes this fail rather than mislabel it.

Re-run after changing a figure:
    pdflatex dqd_model_full.tex
    latex    dqd_model_full.tex
    dvisvgm  --no-fonts --exact --output=dqd_model_full_paths.svg \
             dqd_model_full.dvi
    python   name_svg_groups.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, glyphs) in the order the .tex draws them.  The glyph count is the
# check: QD_1 is three, c_{d_1g_1} is five, "(a)" is three.
FIGURES = {
    "dqd_model_full": [
        ("QD1", 3), ("QD2", 3), ("QDs", 3),
        ("Vg1", 3), ("Vg2", 3), ("Vg3", 3),
        ("c_d1g1", 5), ("c_d2g2", 5), ("c_d2g1", 5), ("c_d1g2", 5),
        ("c_d1d2", 5), ("c_s1d1", 5), ("c_s1d2", 5), ("c_s1g3", 5),
        ("c_d1g3", 5), ("c_d2g3", 5), ("c_s1g1", 5), ("c_s1g2", 5),
        ("panel_a", 3), ("matrices", None), ("panel_b", 3),
    ],
    "dqd_model_sketch": [
        ("QDs", 3), ("QD1", 3), ("QD2", 3),
        ("Vg3", 3), ("Vg1", 3), ("Vg2", 3),
        ("c_s1g1", 5), ("c_s1g2", 5), ("c_s1d1", 5), ("c_s1d2", 5),
        ("c_d1g1", 5), ("c_d2g2", 5), ("c_d2g1", 5), ("c_d1g2", 5),
        ("c_d1d2", 5), ("c_s1g3", 5), ("c_d1g3", 5), ("c_d2g3", 5),
        ("panel_a", 3), ("matrices", None), ("panel_b", 3),
    ],
}


LEGEND_ROWS = [("key_primary", 41), ("key_cross", 43), ("key_interdot", 24),
               ("key_sensor", 28), ("key_weak", 41)]

# the coloured twins are the same figure plus the five key rows
for _stem in ("dqd_model_full", "dqd_model_sketch"):
    FIGURES[_stem + "_color"] = FIGURES[_stem] + LEGEND_ROWS

# an already-named file must re-name cleanly, so the id is optional here
GROUP = re.compile(r"<g (?:id='[^']*' )?transform='[^']*'>(.*?)</g>", re.S)


def name_groups(stem):
    path = os.path.join(HERE, stem + "_paths.svg")
    if not os.path.isfile(path):
        return "%s: no %s_paths.svg -- run dvisvgm first" % (stem, stem)
    svg = io.open(path, encoding="utf-8").read()
    head, body = svg.split("</defs>", 1)

    expected = FIGURES[stem]
    blocks = list(GROUP.finditer(body))
    if len(blocks) != len(expected):
        return ("%s: %d groups, expected %d -- the figure changed, so the "
                "list in this script has to change with it"
                % (stem, len(blocks), len(expected)))

    for block, (name, glyphs) in zip(blocks, expected):
        found = block.group(1).count("<use")
        if glyphs is not None and found != glyphs:
            return ("%s: group for %s has %d glyphs, expected %d -- refusing "
                    "to name it" % (stem, name, found, glyphs))

    out, last = [], 0
    for block, (name, _) in zip(blocks, expected):
        out.append(body[last:block.start()])
        tag = re.sub(r"<g (?:id='[^']*' )?transform=",
                     "<g id='%s' transform=" % name, block.group(0), 1)
        out.append(tag)
        last = block.end()
    out.append(body[last:])

    io.open(path, "w", encoding="utf-8").write(head + "</defs>" + "".join(out))
    # the .svg beside the figure is the one anything else reads
    io.open(os.path.join(HERE, stem + ".svg"), "w", encoding="utf-8").write(
        io.open(path, encoding="utf-8").read())
    return "%s: %d groups named" % (stem, len(expected))


if __name__ == "__main__":
    bad = False
    for stem in sorted(FIGURES):
        msg = name_groups(stem)
        print(" ", msg)
        bad = bad or "named" not in msg
    sys.exit(1 if bad else 0)
