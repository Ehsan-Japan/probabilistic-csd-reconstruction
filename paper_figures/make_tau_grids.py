# -*- coding: utf-8 -*-
"""
make_tau_grids.py -- two text-free pictures that DEFINE the tolerance.

    tau_example.png        why tolerance exists
    tau_neighbourhood.png  what tau admits, cell by cell

Panel 3 superimposes the two lines rather than drawing their (empty)
intersection: the point is that they are one cell apart everywhere, and
an empty box cannot show that.

THE EXAMPLE is the one that matters: the model draws the line perfectly --
same shape, same length, same slope -- but one pixel to the left.  Not one
pixel coincides, so strict pixel F1 = 0.00, for a result any physicist would
call correct.  At tau = 1 every predicted pixel has a true pixel one step
away, and precision = recall = F1 = 1.00.  Both numbers are computed here
with the pipeline's own grid_metrics.tolerant_f1, not asserted.

THE NEIGHBOURHOOD strip shows what "within tau" is, around a single pixel
the model drew.  The distance is Euclidean (one distance transform per map,
grid_metrics.py), so

    tau = 0   the same pixel only
    tau = 1   the four side neighbours; a diagonal is 1.41 away, so no
    tau = 2   diagonals, and everything out to 2 pixels
    tau = 3   everything out to 3 pixels

NO TEXT IS DRAWN IN EITHER FIGURE.  Every label belongs on the slide or in
the caption, so it can be reworded or deleted; the panel centres are written
to tau_layout.json so a deck can place them exactly.

    python paper_figures/make_tau_grids.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

# Write a vector .pdf beside every .png.  Off: the PNGs are 600 dpi
# and the PDFs doubled the folder for nothing.
WITH_PDF = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _run import ROOT                                          # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1               # noqa: E402

TRUTH_C, PRED_C, BAND_C = "#000000", "#C00000", "#D9D9D9"
EDGE, FRAME = "#9a9a9a", "#444444"

N = 9                       # the example grid
# The line runs edge to edge, top to bottom: a transition line crosses the
# whole diagram, and a short floating segment reads as a stray blob instead.
ROWS = range(0, N)
COLS = [2, 2, 3, 3, 4, 4, 5, 5, 6]   # a staircase, as a transition line runs
M = 7                       # the neighbourhood grid
TAUS = (0, 1, 2, 3)
LEFTS = (0.015, 0.260, 0.505, 0.750)


def _grid(ax, n):
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n + 1))
    ax.set_yticks(np.arange(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color=EDGE, linewidth=0.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color(FRAME)
        sp.set_linewidth(0.9)


def _cells(ax, mask, colour, inset=0.0, zorder=2):
    for r, c in zip(*np.nonzero(mask)):
        ax.add_patch(plt.Rectangle((c + inset, r + inset),
                                   1 - 2 * inset, 1 - 2 * inset,
                                   facecolor=colour, edgecolor="none",
                                   zorder=zorder))


def _save(fig, stem):
    # PNG only.  The vector twin was written for a journal that wants
    # one; set WITH_PDF = True to get it back.
    for ext, kw in ((".png", {"dpi": 600}),) + (
            ((".pdf", {}),) if WITH_PDF else ()):
        p = os.path.join(HERE, stem + ext)
        fig.savefig(p, facecolor="white", **kw)
        print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)


def example():
    truth = np.zeros((N, N), bool)
    pred = np.zeros((N, N), bool)
    for r, c in zip(ROWS, COLS):
        truth[r, c] = True
        pred[r, c - 1] = True
    dist = distance_transform_edt(~truth)

    scores = {t: tolerant_f1(pred, truth, float(t)) for t in (0, 1)}
    overlap = truth & pred                      # empty, and that is the point

    fig = plt.figure(figsize=(9.0, 2.35))
    # truth | prediction | the two superimposed | tau = 1
    for k, left in enumerate(LEFTS):
        ax = fig.add_axes([left, 0.03, 0.235, 0.94])
        _grid(ax, N)
        if k == 0:
            _cells(ax, truth, TRUTH_C)
        elif k == 1:
            _cells(ax, pred, PRED_C)
        elif k == 2:
            # BOTH lines, on the same grid.  Drawing `truth & pred` here
            # is literally correct -- the strict overlap is empty -- but
            # an empty box states the result without showing it.  Side by
            # side the two staircases are visibly adjacent and visibly
            # never in the same cell, which is the whole argument for a
            # tolerance.  No band: tau has not been introduced yet.
            _cells(ax, truth, TRUTH_C, zorder=2)
            _cells(ax, pred, PRED_C, zorder=2)
        else:
            _cells(ax, dist <= 1, BAND_C, zorder=1)
            _cells(ax, truth, TRUTH_C, zorder=2)
            _cells(ax, pred, PRED_C, inset=0.17, zorder=3)
    _save(fig, "tau_example")
    print("     overlap = %d pixels, strict F1 = %.2f, tau = 1 F1 = %.2f"
          % (int(overlap.sum()), scores[0]["f1"], scores[1]["f1"]))
    return [left + 0.1175 for left in LEFTS], scores


def neighbourhood():
    one = np.zeros((M, M), bool)
    one[M // 2, M // 2] = True
    dist = distance_transform_edt(~one)

    # The centre cell is the MODEL'S OUTPUT, in red, not a truth pixel: the
    # point is that the output never moves, only how strictly it is read.
    # The grey patch is what that one drawn pixel may stand for at each tau.
    fig = plt.figure(figsize=(9.0, 2.30))
    for left, tau in zip(LEFTS, TAUS):
        ax = fig.add_axes([left, 0.03, 0.235, 0.94])
        _grid(ax, M)
        _cells(ax, (dist <= tau) & ~one, BAND_C, zorder=1)
        _cells(ax, one, PRED_C, zorder=2)
    _save(fig, "tau_neighbourhood")
    return [left + 0.1175 for left in LEFTS]


def main():
    centres_a, scores = example()
    centres_b = neighbourhood()
    layout = {"example": {"centres": centres_a,
                          "f1_strict": round(scores[0]["f1"], 2),
                          "f1_tau1": round(scores[1]["f1"], 2)},
              "neighbourhood": {"centres": centres_b, "taus": list(TAUS)}}
    path = os.path.join(HERE, "tau_layout.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)
    print("  ->", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
