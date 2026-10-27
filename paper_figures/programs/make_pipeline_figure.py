"""
make_pipeline_figure.py -- the whole method on one test device, step by
step (Fig. 12, Sec. 3.6): the input, the U-Net, its binarized output, the
+-BAND margin, the local sweeps and the result.

    python scripts/run_19_two_step.py                     # table.csv
    python paper_figures/programs/make_pipeline_figure.py

(a), (b) the U-Net input: the raw charge-sensor signal at the ray pixels
    (grey elsewhere) and the binary ray mask;
(c) the U-Net probability map;
(d) the binarized output (threshold chosen on validation), over the true
    lines;
(e) the +-BAND margin around it, swept in step 2 (with the rays);
(f) the step value of every tested pixel of the local sweeps;
(g) the pixels kept as line pixels, over the true lines.
Arrows between the panels name each operation.  Same device and threshold
as make_step_decision_figure.py.  Writes, into method_docs_ver2/3_methods/
figures/: fig_pipeline.png and pipeline.json.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.collections import PatchCollection      # noqa: E402
from matplotlib.colors import LogNorm, to_rgb            # noqa: E402
from matplotlib.patches import (Circle, ConnectionPatch,  # noqa: E402
                                FancyBboxPatch)

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from csdrecon.ml.ray_peaks import CH_SIGNAL, CH_VISITED  # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 2
DEVICE = "sample_453"              # the one test device with exactly four
#                                    charge states, (0,0), (0,1), (1,0), (1,1)
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
# the colours of the other step-2 figures
LINE_C, CHECK_C, KEPT_C, TRUE_C, RAY_C = (
    "#4f8fc8", "#a9c9e8", "#0b2f6b", "#d00000", "#555555")
# the two steps, as in the step-detection figure (user, 2026-10-26)
PRED_C, PRED_BG = "#3f51b5", "#eef0fa"
VER_C, VER_BG = "#b26a00", "#fbf3e6"
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def main(conceptual=False):
    # conceptual=True (user, 2026-10-26): the labels of the conceptual
    # version -- no numbers, the margin named by what it does; written to
    # method_docs_ver2/conceptual/figures/fig_pipeline_conceptual.png
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        q = float(next(r for r in csv.DictReader(f)
                       if r["setting"] == f"band ±{BAND} px")["q"])
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr_b = float(ck["threshold"])
    test = R.load(net, thr_b, "test")
    i = test["names"].index(DEVICE)
    Y, B, Z, X = test["Y"][i], test["B"][i], test["Z"][i], test["X"][i]
    P = grid_train.predict(net, X[None])[0]          # probability map
    assert np.array_equal(P > thr_b, B)
    rays = X[CH_VISITED] > 0.5
    strip = R.dilate(B, BAND)
    measured = strip | rays
    kept, Rmap, tested = R.sweep_lines(Z, strip, q, measured)
    assert np.array_equal(kept, R.two_step(test, i, q, BAND)[0])
    f1_u, f1_k = tolerant_f1(B, Y, 1.0)["f1"], tolerant_f1(kept, Y, 1.0)["f1"]

    # layout (user, 2026-10-26): 2 x 3. The two input channels (a), (b)
    # are drawn as stacked cards in the first column, (b) behind and offset
    # up and to the right; (c), (d) complete the first row, (e)-(g) the
    # second. Arrows in the gaps name each operation.
    # (user, 2026-10-26) each row is one step, on its own tinted band
    # with the step named down its left edge: prediction above,
    # verification below
    W, gx, gy, top, bot = 15.49 / 2.54, 0.50, 0.62, 0.10, 0.30
    lab = 0.30                          # the column of the step names
    hs = (W - lab - 0.10 - 2 * gx) / 3 * 0.80
    off = 0.20 * hs                     # offset of the card behind
    cs = hs - off                       # size of each card
    x0 = lab + (W - lab - 3 * hs - 2 * gx) / 2
    H = top + 2 * hs + gy + bot

    def xcol(k):
        return x0 + k * (hs + gx)
    fig = plt.figure(figsize=(W, H))
    yt = bot + hs + gy
    ax_b = fig.add_axes(((xcol(0) + off) / W, (yt + off) / H, cs / W,
                         cs / H))
    ax_a = fig.add_axes((xcol(0) / W, yt / H, cs / W, cs / H))
    ax = [ax_a, ax_b]
    ax += [fig.add_axes((xcol(k) / W, yt / H, hs / W, hs / H))
           for k in (1, 2)]
    ax += [fig.add_axes((xcol(k) / W, bot / H, hs / W, hs / H))
           for k in range(3)]

    def rgb(*layers):
        im = np.ones(Y.shape + (3,))
        for m, colr in layers:
            im[m] = to_rgb(colr)
        return im

    def dots(a):
        yy, xx = np.nonzero(Y)
        a.add_collection(PatchCollection(
            [Circle((x, y), 0.45) for y, x in zip(yy, xx)], fc=TRUE_C,
            ec="none", zorder=5))

    # (a), (b): the two input channels of the U-Net -- the raw
    # charge-sensor signal at the ray pixels (grey elsewhere), and the
    # binary mask of the ray pixels
    raw = np.ma.masked_where(~rays, Z)
    ax[0].imshow(np.full(Y.shape + (3,), 0.88), origin="lower")
    ax[0].imshow(raw, origin="lower", cmap="hot", interpolation="nearest",
                 vmin=float(Z[rays].min()), vmax=float(Z[rays].max()))
    ax[1].imshow(rgb((rays, "#111111")), origin="lower",
                 interpolation="nearest")
    # (c) the U-Net probability map, (d) its binarized output
    ax[2].imshow(P, origin="lower", cmap="Blues", vmin=0, vmax=1,
                 interpolation="nearest")
    ax[3].imshow(rgb((B, LINE_C)), origin="lower", interpolation="nearest")
    # (e) the margin, swept with the rays
    ax[4].imshow(rgb((strip, CHECK_C), (B, LINE_C), (rays, RAY_C)),
                 origin="lower", interpolation="nearest")
    # (f) the step value of the tested pixels (log scale), untested white
    rv = np.ma.masked_where(~tested, np.maximum(Rmap, 1e-3))
    ax[5].imshow(np.ones(Y.shape + (3,)), origin="lower")
    ax[5].imshow(rv, origin="lower", cmap="Blues",
                 norm=LogNorm(vmin=0.02, vmax=float(Rmap[tested].max())),
                 interpolation="nearest")
    # (g) the result
    ax[6].imshow(rgb((kept, KEPT_C)), origin="lower", interpolation="nearest")
    for a in (ax[3], ax[6]):
        dots(a)

    titles = ("signal along the rays", "ray mask", "U-Net probability",
              "predicted lines",
              "swept points" if conceptual else f"±{BAND} px margin",
              "step value", "verified lines")
    notes = (f"U-Net input, {100 * rays.mean():.1f}% of the plane measured",
             "", "",
             f"F1@1 = {f1_u:.3f}",
             f"{100 * measured.mean():.1f}% of the plane measured", "",
             f"F1@1 = {f1_k:.3f}")
    if conceptual:
        notes = ("U-Net input", "", "", "", "", "", "")
    for k, a in enumerate(ax):
        a.set_xticks([])
        a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_linewidth(0.6)
        a.text(0.03, 0.97, f"({'abcdefg'[k]}) {titles[k]}",
               transform=a.transAxes, ha="left", va="top", fontsize=5.8,
               zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="#999999", lw=0.5))
        if notes[k]:
            a.text(0.5, -0.025, notes[k], transform=a.transAxes, ha="center",
                   va="top", fontsize=6, color="#222222")

    def arrow(a0, a1, label, row):
        # from the right edge of a0 (for the card stack: of the card
        # behind) at the height of a1's centre
        y1 = (a1.get_position().y0 + a1.get_position().y1) / 2
        fig.add_artist(ConnectionPatch(
            xyA=(a0.get_position().x1, y1), coordsA="figure fraction",
            xyB=(0.0, 0.5), coordsB=a1.transAxes, arrowstyle="-|>",
            mutation_scale=8, lw=0.9, color="#333333", shrinkA=2, shrinkB=2,
            zorder=9))
        x = (a0.get_position().x1 + a1.get_position().x0) / 2
        y = (bot + (hs + gy if row == 0 else 0) + hs / 2 + 0.04) / H
        fig.text(x, y, label, ha="center", va="bottom", fontsize=5.6,
                 color="#333333")
    arrow(ax[1], ax[2], "U-Net", 0)
    arrow(ax[2], ax[3], "threshold" if conceptual
          else f"threshold\n{thr_b:g}", 0)
    arrow(ax[4], ax[5], "local\nsweeps", 1)
    arrow(ax[5], ax[6], "step\ndetector", 1)
    # from (d) down to (e), across the row gap
    fig.add_artist(ConnectionPatch(
        xyA=(0.5, -0.12), coordsA=ax[3].transAxes, xyB=(0.5, 1.0),
        coordsB=ax[4].transAxes, arrowstyle="-|>", mutation_scale=8, lw=0.9,
        color="#333333", shrinkA=2, shrinkB=2,
        connectionstyle="angle,angleA=180,angleB=90,rad=0", zorder=9))
    fig.text(0.5, (bot + hs + gy * 0.38) / H,
             "sweep across the predicted lines" if conceptual
             else f"±{BAND} px margin",
             ha="center", va="bottom", fontsize=5.8, color="#333333",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for y0, name, fc, ec in ((yt, "Step 1: Prediction", PRED_BG, PRED_C),
                             (bot, "Step 2: Verification", VER_BG, VER_C)):
        fig.add_artist(FancyBboxPatch(
            (0.04 / W, (y0 - 0.25) / H), (W - 0.08) / W, (hs + 0.29) / H,
            boxstyle="round,pad=0,rounding_size=0.01", fc=fc, ec=ec,
            lw=0.8, zorder=-1, transform=fig.transFigure))
        fig.text((0.04 + lab / 2) / W, (y0 + hs / 2 - 0.105) / H, name,
                 rotation=90, ha="center", va="center", fontsize=7.5,
                 fontweight="bold", color=ec)
    if conceptual:
        out = os.path.join(ROOT, "paper_figures", "method_docs_ver2",
                           "conceptual", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "fig_pipeline_conceptual.png"),
                    dpi=400)
        plt.close(fig)
        return
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=400)
    plt.close(fig)

    vals = {"device": DEVICE, "band": BAND, "threshold": thr_b, "q": q,
            "rays_measured": 100 * float(rays.mean()),
            "measured": 100 * float(measured.mean()),
            "f1_unet": f1_u, "f1_final": f1_k}
    with open(os.path.join(OUT, "pipeline.json"), "w") as f:
        json.dump(vals, f, indent=1)
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main(conceptual="--conceptual" in sys.argv)
