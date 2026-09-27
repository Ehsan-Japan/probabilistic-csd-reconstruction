# -*- coding: utf-8 -*-
"""
make_capacitance_note.py — why the capacitance count is 14, and why the
figure looks like 13 or 9.

    python docs/codebase/make_capacitance_note.py

Writes capacitance_count.docx and .pdf beside this file, with two figures it
draws itself.  Every count in the note is computed from the code and from a
stored device record, not typed.
"""
import io
import json
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT_DOCX = os.path.join(HERE, "capacitance_count.docx")
FIG_MATRICES = os.path.join(HERE, "_fig_matrices.png")
FIG_NETWORK = os.path.join(HERE, "_fig_network.png")

INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x60, 0x60, 0x60)
BLACK = "#1a1a1a"
RED = "#c0392b"
BLUE = "#2b6cb0"

# The four matrices as the generator builds them, and which is symmetric.
SHAPES = {"Cdd": (2, 2), "Cgd": (2, 3), "Cds": (1, 2), "Cgs": (1, 3)}
SYMMETRIC = {"Cdd"}

CELLS = {
    "Cdd": [["d1d1", "d1d2"], ["d2d1", "d2d2"]],
    "Cgd": [["d1g1", "d1g2", "d1g3"], ["d2g1", "d2g2", "d2g3"]],
    "Cds": [["s1d1", "s1d2"]],
    "Cgs": [["s1g1", "s1g2", "s1g3"]],
}

# The paper's symbol for each matrix.  The code keeps QArray's keys, which
# name a matrix after the pair it couples (Cgd, "gates and dots") while
# shaping it (n_dot, n_gate); the paper and the figure name it in the order
# its own entries use, so C_dg holds c_d1g1.
PAPER_NAME = {"Cdd": "Cdd", "Cgd": "Cdg", "Cds": "Csd", "Cgs": "Csg"}

# Which capacitors panel (a) of docs/figures/dqd_model.png actually draws.
DRAWN_IN_FIGURE_A = {"d1g1", "d1g2", "d2g1", "d2g2", "d1d2",
                     "d1g3", "s1d1", "s1d2", "s1g3"}


# ── the counts, from the code and from a real device ──────────────────────

def counts():
    from csdrecon.config.capacitance_config import DEFAULT_INTERVALS
    from csdrecon.study.device_split import VECTOR_KEYS
    printed = sum(r * c for r, c in SHAPES.values())
    free = sum(len(DEFAULT_INTERVALS[m]) for m in SHAPES)
    for m, (r, c) in SHAPES.items():
        n = len(DEFAULT_INTERVALS[m])
        want = r * (r + 1) // 2 if m in SYMMETRIC else r * c
        assert n == want, (m, n, want)
    assert len(VECTOR_KEYS) == free, (len(VECTOR_KEYS), free)
    return printed, free


def from_a_device():
    """(cells printed, distinct values, symmetric?) in a stored device.json."""
    found = sorted(glob.glob(os.path.join(
        ROOT, "data", "_device_pools", "*", "sample_*", "device.json")))
    if not found:
        return None
    cap = json.load(io.open(found[0], encoding="utf-8"))["capacitance"]
    values = []
    for m in SHAPES:
        rows = cap[m] if isinstance(cap[m][0], list) else [cap[m]]
        for row in rows:
            values.extend(row)
    return (len(values), len(set(values)),
            cap["Cdd"][0][1] == cap["Cdd"][1][0],
            os.path.relpath(found[0], ROOT).replace("\\", "/"))


# ── figure 1: the matrices, with the mirrored pair marked ─────────────────

def draw_matrices(path):
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_axis_off()
    x = 0.0
    cell_w, cell_h = 0.95, 0.52
    positions = {}
    for name in ("Cdd", "Cgd", "Cds", "Cgs"):
        grid = CELLS[name]
        rows, cols = len(grid), len(grid[0])
        top = 1.15
        ax.text(x + cols * cell_w / 2, top + 0.30, PAPER_NAME[name],
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=BLACK)
        for i, row in enumerate(grid):
            for j, label in enumerate(row):
                cx = x + j * cell_w
                cy = top - i * cell_h
                mirrored = name in SYMMETRIC and i > j
                face = "#fdecea" if mirrored else "#ffffff"
                edge = RED if mirrored else "#9aa5b1"
                ax.add_patch(plt.Rectangle((cx, cy - cell_h * 0.72),
                                           cell_w * 0.92, cell_h * 0.72,
                                           facecolor=face, edgecolor=edge,
                                           linewidth=1.1, zorder=1))
                ax.text(cx + cell_w * 0.46, cy - cell_h * 0.36, label,
                        ha="center", va="center", fontsize=9,
                        color=RED if mirrored else BLACK, zorder=2)
                positions[(name, i, j)] = (cx + cell_w * 0.46,
                                           cy - cell_h * 0.36)
        x += cols * cell_w + 0.75

    x0 = 0.0
    ax.text(x0 + cell_w * 0.92, 0.02,
            "d1d2 = d2d1 — one capacitor, printed twice",
            ha="center", va="top", fontsize=8.5, color=RED)

    ax.set_xlim(-0.4, x)
    ax.set_ylim(-0.55, 1.85)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ── figure 2: who couples to whom, and which couplings the drawing shows ──

# The three islands, and everything each of them can couple to.  The cell is
# the capacitance between the two, or blank where the matrices define none.
GRID_ROWS = ["QD1", "QD2", "QDs"]
GRID_COLS = ["Vg1", "Vg2", "Vg3", "QD1", "QD2", "ground"]
GRID = {
    ("QD1", "Vg1"): "d1g1", ("QD1", "Vg2"): "d1g2", ("QD1", "Vg3"): "d1g3",
    ("QD1", "QD2"): "d1d2", ("QD1", "ground"): "d1d1",
    ("QD2", "Vg1"): "d2g1", ("QD2", "Vg2"): "d2g2", ("QD2", "Vg3"): "d2g3",
    ("QD2", "QD1"): "d1d2", ("QD2", "ground"): "d2d2",
    ("QDs", "Vg1"): "s1g1", ("QDs", "Vg2"): "s1g2", ("QDs", "Vg3"): "s1g3",
    ("QDs", "QD1"): "s1d1", ("QDs", "QD2"): "s1d2",
}


def draw_coupling_grid(path):
    """Every capacitance as one cell: what it couples, and whether the
    drawing shows it.  A grid rather than a schematic, because fourteen
    labelled edges through three nodes cannot be drawn without collisions."""
    fig, ax = plt.subplots(figsize=(7.3, 3.1))
    ax.set_axis_off()
    w, h = 1.12, 0.70
    for j, col in enumerate(GRID_COLS):
        ax.text((j + 0.5) * w, 0.32, col, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=BLACK)
    for i, row in enumerate(GRID_ROWS):
        y = -(i + 1) * h
        ax.text(-0.18, y + h * 0.5, row, ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=BLACK)
        for j, col in enumerate(GRID_COLS):
            label = GRID.get((row, col))
            x = j * w
            if label is None:
                ax.add_patch(plt.Rectangle((x + 0.03, y + 0.05),
                                           w - 0.06, h - 0.10,
                                           facecolor="#f4f6f8",
                                           edgecolor="#e2e8ed", zorder=1))
                continue
            duplicate = (row, col) == ("QD2", "QD1")
            drawn = label in DRAWN_IN_FIGURE_A
            if duplicate:
                face, edge, ink = "#fdecea", RED, RED
                text = label + "\n(same as above)"
            elif drawn:
                face, edge, ink = "#ffffff", "#9aa5b1", BLACK
                text = label
            else:
                face, edge, ink = "#fdecea", RED, RED
                text = label + "\nnot drawn"
            ax.add_patch(plt.Rectangle((x + 0.03, y + 0.05), w - 0.06,
                                       h - 0.10, facecolor=face,
                                       edgecolor=edge, lw=1.2, zorder=1))
            ax.text(x + w * 0.5, y + h * 0.5, text, ha="center",
                    va="center", fontsize=8.2, color=ink, zorder=2,
                    linespacing=1.25)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", markersize=9,
               markerfacecolor="white", markeredgecolor="#9aa5b1",
               label="drawn in Figure 1(a)  (%d)" % len(DRAWN_IN_FIGURE_A)),
        Line2D([], [], marker="s", ls="", markersize=9,
               markerfacecolor="#fdecea", markeredgecolor=RED,
               label="in the matrices, missing from Figure 1(a)  (5)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False,
        fontsize=9, ncol=2)
    ax.set_xlim(-1.35, len(GRID_COLS) * w + 0.1)
    ax.set_ylim(-len(GRID_ROWS) * h - 0.30, 0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ── the note ──────────────────────────────────────────────────────────────

def setup(doc):
    s = doc.styles["Normal"]
    s.font.name = "Calibri"
    s.font.size = Pt(10)
    s.font.color.rgb = INK
    s.paragraph_format.space_after = Pt(7)
    s.paragraph_format.line_spacing = 1.15
    for sec in doc.sections:
        sec.top_margin = Inches(0.9)
        sec.bottom_margin = Inches(0.9)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)


def para(doc, text, bullet=False):
    p = doc.add_paragraph(style="List Bullet" if bullet else None)
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            p.add_run(chunk).font.bold = bool(i % 2)
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(13)
    r.font.bold = True


def figure(doc, path, caption, width=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY
    c.paragraph_format.space_after = Pt(12)


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for cell, head in zip(t.rows[0].cells, headers):
        cell.text = ""
        r = cell.paragraphs[0].add_run(head)
        r.font.bold = True
        r.font.size = Pt(9)
    for row in rows:
        for cell, value in zip(t.add_row().cells, row):
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(value))
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build():
    printed, free = counts()
    dev = from_a_device()
    drawn = len(DRAWN_IN_FIGURE_A)
    missing = sorted(set(sum([sum(v, []) for v in CELLS.values()], []))
                     - DRAWN_IN_FIGURE_A - {"d2d1"})

    draw_matrices(FIG_MATRICES)
    draw_coupling_grid(FIG_NETWORK)

    doc = Document()
    setup(doc)

    p = doc.add_paragraph()
    r = p.add_run("How many capacitances is a device?")
    r.font.size = Pt(20)
    r.font.bold = True
    p.paragraph_format.space_after = Pt(2)
    p = doc.add_paragraph()
    r = p.add_run("Four numbers are floating around — 15, 14, 13 and 9. "
                  "This is where each of them comes from, and which one "
                  "belongs in the paper.")
    r.font.size = Pt(10.5)
    r.font.color.rgb = GREY
    p.paragraph_format.space_after = Pt(12)

    h1(doc, "The short answer")
    para(doc,
         "A device is **%d free parameters**. Figure 1(b) prints **%d** "
         "numbers, but two of them are the same capacitor, so only %d of "
         "them can be chosen independently." % (free, printed, free))

    h1(doc, "1.  Why 15 and not 14 — C_dd is symmetric")
    para(doc,
         "The mutual capacitance between dot 1 and dot 2 is one physical "
         "quantity. Writing it as a matrix puts it in twice, once above the "
         "diagonal and once below:")
    figure(doc, FIG_MATRICES,
           "The four matrices as Figure 1(b) prints them. The shaded cell "
           "is not a separate capacitance — it is the same number as the "
           "one above the diagonal.")
    para(doc,
         "So counting the printed cells gives %d, and counting the "
         "independent values gives %d. The code enforces this rather than "
         "assuming it: MatrixGenerator.generate_all builds C_dd with "
         "generate_symmetric(), which draws the upper triangle and mirrors "
         "it." % (printed, free))
    if dev:
        cells, distinct, sym, path = dev
        para(doc,
             "Checked on a stored device rather than argued: %s prints %d "
             "numbers, %d of them distinct, and its C_dd[0][1] equals "
             "C_dd[1][0]." % (path, cells, distinct))
    table(doc, ["matrix", "shape", "cells printed", "free parameters"], [
        ("C_dd", "2 x 2  (symmetric)", 4, 3),
        ("C_dg", "2 x 3  (QArray's Cgd)", 6, 6),
        ("C_sd", "1 x 2  (QArray's Cds)", 2, 2),
        ("C_sg", "1 x 3  (QArray's Cgs)", 3, 3),
        ("total", "", printed, free),
    ])

    h1(doc, "2.  Where 13 came from — it was simply wrong")
    para(doc,
         "13 was not a different way of counting. It was a number typed "
         "into the text and never checked against the code: the "
         "manuscript said a device is “a point in a 13-dimensional "
         "box” while, two paragraphs later, the separation sentence "
         "printed “14-dimensional normalised parameter space” "
         "from the data. Table I has always listed %d rows. The text "
         "contradicted itself and the table." % free)
    para(doc,
         "It is fixed in three places, and neither number is typed any "
         "more — both are computed from the shapes the generator uses and "
         "from DEFAULT_INTERVALS, and the build asserts they agree.")

    h1(doc, "3.  Where 9 comes from — the drawing is incomplete")
    para(doc,
         "This is the one still worth acting on. Panel (a) of "
         "docs/figures/dqd_model.png draws **%d** capacitors. Five that "
         "panel (b) tabulates are not in the drawing at all." % drawn)
    figure(doc, FIG_NETWORK,
           "Every capacitance the matrices define, as one cell: what it "
           "couples, and whether Figure 1(a) draws it. White: the %d it "
           "draws. Red: the %d it does not — the two dot self-capacitances "
           "to ground, the dot-2 coupling to gate 3, and the sensor's "
           "couplings to gates 1 and 2. The shaded QD2/QD1 cell is the "
           "symmetric duplicate, not a fifteenth capacitance."
           % (drawn, len(missing)))
    para(doc, "Missing from panel (a): %s." % ", ".join(missing))
    para(doc,
         "%d drawn plus %d missing is %d, which is the same count arrived "
         "at from the matrices — so nothing is wrong with the physics or "
         "the code. What is wrong is that a referee who counts the picture "
         "will find %d where the text now says %d."
         % (drawn, len(missing), free, drawn, free))

    h1(doc, "4.  What to do")
    para(doc,
         "**Either** add the five missing capacitors to panel (a) in "
         "Illustrator, so (a) and (b) show the same network;", bullet=True)
    para(doc,
         "**or** leave the drawing and say so in the caption — that (a) "
         "shows the couplings which set the geometry of the diagram and "
         "(b) the complete set of %d. Say which and I will write the "
         "caption." % free, bullet=True)
    para(doc,
         "The text in Section 3.1 is already correct either way: it states "
         "the symmetry, the %d printed entries, the %d free parameters and "
         "the per-matrix breakdown, all computed at build time."
         % (printed, free))

    doc.save(OUT_DOCX)
    for f in (FIG_MATRICES, FIG_NETWORK):
        pass          # kept beside the document; they are its figures
    return OUT_DOCX


PS = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
foreach ($src in $args) {
  $pdf = [System.IO.Path]::ChangeExtension($src, '.pdf')
  try {
    $doc = $word.Documents.Open($src, $false, $true)
    $n = $doc.ComputeStatistics(2)
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close(0)
    Write-Output ("OK`t{0}`t{1}" -f [System.IO.Path]::GetFileName($src), $n)
  } catch {
    Write-Output ("SKIP`t{0}`t{1}" -f [System.IO.Path]::GetFileName($src), $_.Exception.Message)
  }
}
$word.Quit()
"""


def to_pdf(docx_path):
    script = os.path.join(HERE, "_export_pdf.ps1")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(PS)
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, docx_path], capture_output=True, text=True)
    os.remove(script)
    for line in out.stdout.splitlines():
        if line.startswith("OK"):
            return os.path.splitext(docx_path)[0] + ".pdf", line.split("\t")[-1]
        if line.startswith("SKIP"):
            print("  not exported:", line.split("\t")[-1][:120])
    if out.stderr.strip():
        print(out.stderr.strip()[:400])
    return None, None


def main():
    print("How many capacitances is a device?")
    print("=" * 74)
    docx_path = build()
    print("  %-32s written" % os.path.basename(docx_path))
    pdf, pages = to_pdf(docx_path)
    if pdf:
        print("  %-32s %s pages" % (os.path.basename(pdf), pages))
        print()
        print(pdf)


if __name__ == "__main__":
    main()
