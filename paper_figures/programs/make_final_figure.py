"""
make_final_figure.py -- what the local sweeps add: the binarized U-Net
output and the final estimate, each over the true transition lines, on the
whole device of Fig. 12 (Fig. 13, Sec. 3.6).

    python scripts/run_19_two_step.py                     # table.csv
    python paper_figures/programs/make_final_figure.py

(a) the binarized U-Net output (step 1);
(b) the estimated transition lines after the local sweeps and the step
    detector (step 2, margin BAND);
(c), (d) the dashed square of (a), (b) magnified: the ZOOM x ZOOM window
    where the sweeps remove the most errors, in the colours of (a), (b);
    below each column, the fraction of the plane measured and F1@1.

Recomputed with run_19's own functions.  Writes, into
method_docs_ver2/3_methods/figures/: fig_unet_vs_final.png and
unet_vs_final.json (the numbers the caption quotes).
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
from matplotlib.colors import to_rgb                     # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.collections import PatchCollection      # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle  # noqa
from scipy.ndimage import distance_transform_edt         # noqa: E402

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 2
DEVICE = "sample_223"              # the device of Fig. 12
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
LINE_C, FOUND_C, TRUE_C = "#4f8fc8", "#0b2f6b", "#d00000"   # as in Fig. 12
ZOOM = 22                          # side of the magnified square, px
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444"})


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        q = float(next(r for r in csv.DictReader(f)
                       if r["setting"] == f"band ±{BAND} px")["q"])
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    test = R.load(net, float(ck["threshold"]), "test")
    i = test["names"].index(DEVICE)
    Y, B, X = test["Y"][i], test["B"][i], test["X"][i]
    found, m_f = R.two_step(test, i, q, BAND)
    m_u = float(X[1].mean())                      # the U-Net alone: the rays
    f_u, f_f = tolerant_f1(B, Y, 1.0), tolerant_f1(found, Y, 1.0)
    H, W = Y.shape
    # the errors F1@1 counts: a predicted pixel with no true pixel within
    # 1 px, and a true pixel with no predicted pixel within 1 px
    dt = distance_transform_edt(~Y)
    false_u, false_f = B & (dt > 1), found & (dt > 1)
    miss_u = Y & (distance_transform_edt(~B) > 1)
    miss_f = Y & (distance_transform_edt(~found) > 1)

    # the magnified region: the ZOOM x ZOOM window where the local sweeps
    # remove the most errors (false plus missed pixels)
    best = None
    for r0 in range(0, H - ZOOM + 1, 2):
        for c0 in range(0, W - ZOOM + 1, 2):
            sl = (slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM))
            gain = (int(false_u[sl].sum() + miss_u[sl].sum())
                    - int(false_f[sl].sum() + miss_f[sl].sum()))
            if best is None or gain > best[0]:
                best = (gain, r0, c0)
    _, r0, c0 = best
    r, c = slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM)

    def rgb(*layers):
        img = np.ones(Y.shape + (3,))
        for msk, colr in layers:
            img[msk] = to_rgb(colr)
        return img

    def grid(a, rows, cols, lw):
        for k in range(cols.start, cols.stop + 1):
            a.axvline(k - 0.5, color="#d8d8d8", lw=lw, zorder=3)
        for k in range(rows.start, rows.stop + 1):
            a.axhline(k - 0.5, color="#d8d8d8", lw=lw, zorder=3)

    def true_dots(a, ys, xs):
        """The true line pixels as red circles that touch the sides of
        their pixel."""
        a.add_collection(PatchCollection(
            [Circle((x, y), 0.5) for y, x in zip(ys, xs)], fc=TRUE_C,
            ec="none", zorder=5))

    def letter(a, lab):
        a.text(0.97, 0.97, f"({lab})", transform=a.transAxes, ha="right",
               va="top", fontsize=7.5, fontweight="bold", zorder=10,
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999",
                         lw=0.6))

    def frame(a):
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)

    fig, ax = plt.subplots(2, 2, figsize=(15.49 / 2.54, 6.55),
                           gridspec_kw=dict(wspace=0.06, hspace=0.12,
                                            left=0.005, right=0.995,
                                            top=0.997, bottom=0.09))
    full = (slice(0, H), slice(0, W))
    yy, xx = np.nonzero(Y)
    cols = ((B, LINE_C, false_u), (found, FOUND_C, false_f))
    for k, (m, colr, fal) in enumerate(cols):
        top, bot = ax[0, k], ax[1, k]
        top.imshow(rgb((m, colr)), origin="lower", interpolation="nearest")
        grid(top, *full, lw=0.12)
        true_dots(top, yy, xx)
        top.add_patch(Rectangle((c.start - 0.5, r.start - 0.5), ZOOM, ZOOM,
                                fill=False, ec="#444444", lw=0.8, ls="--",
                                zorder=6))
        ext = (c.start - 0.5, c.stop - 0.5, r.start - 0.5, r.stop - 0.5)
        bot.imshow(rgb((m, colr))[r, c], origin="lower",
                   interpolation="nearest", extent=ext)
        grid(bot, r, c, lw=0.3)
        ty, tx = np.nonzero(Y[r, c])
        true_dots(bot, ty + r.start, tx + c.start)
        # the square of the top panel enlarged below, joined as in Fig. 3
        for x_box, x_ax in ((c.start - 0.5, 0.0), (c.stop - 0.5, 1.0)):
            fig.add_artist(ConnectionPatch(
                xyA=(x_box, r.start - 0.5), coordsA=top.transData,
                xyB=(x_ax, 1.0), coordsB=bot.transAxes,
                color="#888888", lw=0.6, ls="--", zorder=8))
        for a, lab in ((top, "ab"[k]), (bot, "cd"[k])):
            frame(a)
            letter(a, lab)
    # what each column costs and achieves, below the panels
    for k, (name, meas, f1) in enumerate(
            (("U-Net output", m_u, f_u), ("U-Net + local sweeps", m_f, f_f))):
        ax[1, k].text(0.5, -0.035, f"{name}: {100 * meas:.1f}% of the plane "
                      f"measured, F1@1 = {f1['f1']:.3f}",
                      transform=ax[1, k].transAxes, ha="center", va="top",
                      fontsize=7)
    fig.legend(handles=[Patch(color=LINE_C, label="binarized U-Net output"),
                        Patch(color=FOUND_C,
                              label="estimated transition line"),
                        Line2D([], [], ls="none", marker="o", ms=2.6,
                               color=TRUE_C, label="true transition line")],
               loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               handlelength=1.0, columnspacing=1.4, handletextpad=0.4)
    fig.savefig(os.path.join(OUT, "fig_unet_vs_final.png"), dpi=400)
    plt.close(fig)

    def errs(fal, mis, sl):
        return {"false_px": int(fal[sl].sum()), "missed_px": int(mis[sl].sum())}
    vals = {"device": DEVICE, "band": BAND,
            "zoom_rows": [r.start, r.stop], "zoom_cols": [c.start, c.stop],
            "unet": {**{k: f_u[k] for k in ("precision", "recall", "f1")},
                     "measured": 100 * m_u, **errs(false_u, miss_u, full),
                     "zoom": errs(false_u, miss_u, (r, c))},
            "final": {**{k: f_f[k] for k in ("precision", "recall", "f1")},
                      "measured": 100 * m_f, **errs(false_f, miss_f, full),
                      "zoom": errs(false_f, miss_f, (r, c))}}
    with open(os.path.join(OUT, "unet_vs_final.json"), "w") as f:
        json.dump(vals, f, indent=1)
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
