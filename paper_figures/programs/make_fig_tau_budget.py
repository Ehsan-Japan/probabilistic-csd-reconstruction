"""
make_fig_tau_budget.py -- one figure from two (user, 2026-10-26): (a) the
tolerance curve (precision, recall and F1 against tau, 8 x 60) and (b) F1@1
against the fraction of the plane measured for every budget.

    python paper_figures/programs/make_fig_tau_budget.py

Both panels are the PNGs that make_figures_ver3.py writes into
method_docs_ver2 (p2l_7_tolerance_curve.png, results_budget_ladder.png),
placed side by side at their native 600 dpi, so that their fonts keep the
same size, bottom-aligned, with the panel letters added. Writes
method_docs_ver2/4_Results_and_Discussion/figures/fig_tau_budget.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "method_docs_ver2")
A = os.path.join(DOCS, "3_methods", "figures", "p2l_7_tolerance_curve.png")
B = os.path.join(DOCS, "4_Results_and_Discussion", "figures",
                 "results_budget_ladder.png")
OUT = os.path.join(DOCS, "4_Results_and_Discussion", "figures",
                   "fig_tau_budget.png")
DPI = 600
GAP, TOP = 120, 110          # pixels: between the panels; above, for letters


def rgb(path):
    im = plt.imread(path)
    if im.dtype != np.uint8:
        im = (im * 255).round().astype(np.uint8)
    return im[..., :3]


def main():
    a, b = rgb(A), rgb(B)
    h = max(a.shape[0], b.shape[0]) + TOP
    w = a.shape[1] + GAP + b.shape[1]
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[h - a.shape[0]:, :a.shape[1]] = a           # bottom-aligned
    canvas[h - b.shape[0]:, a.shape[1] + GAP:] = b
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.figimage(canvas, 0, 0)
    for x, lab in ((0, "(a)"), (a.shape[1] + GAP, "(b)")):
        fig.text((x + 20) / w, 1 - (TOP // 2 + 20) / h, lab, ha="left",
                 va="center",
                 fontsize=10, fontweight="bold", family="Arial")
    fig.savefig(OUT, dpi=DPI)
    plt.close(fig)
    print("wrote", OUT, (w, h))


if __name__ == "__main__":
    main()
