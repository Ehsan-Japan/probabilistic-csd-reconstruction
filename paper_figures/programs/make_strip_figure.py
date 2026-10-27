"""
make_strip_figure.py -- the local sweeps of step 2: which pixels are
measured and tested, and what the step detector uses
(Fig. 12, Sec. 3.6).

    python scripts/run_19_two_step.py                     # table.csv
    python paper_figures/programs/make_strip_figure.py

One test device (the device of the two-step figure), margin BAND = 2:
(a) step 1, the rays, and the binarized U-Net output;
(b) step 2, the pixels measured by the local sweeps (within BAND grid steps
    of the binarized output), every one of which is tested;
(c) magnified: the binarized U-Net output alone;
(d) magnified: the same with the swept pixels of the local sweeps;
(e) magnified: the step value of the swept pixels;
(f) magnified: the estimated transition lines, over the true lines.

The square of (b) is joined to (c) by dashed lines, as in Fig. 3.
Everything is recomputed with run_19's own functions and checked to equal
run_19.two_step.  Writes, into method_docs_ver2/3_methods/figures/:
fig_strip_check.png and strip_check.json (the numbers the text quotes).
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
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa
from matplotlib.lines import Line2D                      # noqa: E402
from matplotlib.collections import PatchCollection      # noqa: E402
from matplotlib.patches import (Circle, ConnectionPatch, Patch,  # noqa
                                Rectangle)

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from csdrecon.study import charge_states as C           # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 2
DEVICE = "sample_223"              # the device of the two-step figure
ZOOM = (slice(26, 44), slice(19, 37))   # rows, cols; square, so that
#                                         (c)-(f) are as tall as (a), (b)
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
RAY_C, LINE_C = "#555555", "#4f8fc8"             # rays, U-Net output
MEAS_C = "#a9c9e8"                       # swept: measured and tested
FOUND_C, TRUE_C = "#0b2f6b", "#d00000"   # dark blue, red
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444"})


def outline(a, m, r, c, color, lw=1.1):
    """The border of mask m on the pixel grid, within rows r and cols c."""
    for y in range(r.start, r.stop):
        for x in range(c.start, c.stop):
            if not m[y, x]:
                continue
            for dy, dx, seg in ((1, 0, ((x - .5, x + .5), (y + .5, y + .5))),
                                (-1, 0, ((x - .5, x + .5), (y - .5, y - .5))),
                                (0, 1, ((x + .5, x + .5), (y - .5, y + .5))),
                                (0, -1, ((x - .5, x - .5), (y - .5, y + .5)))):
                yy, xx = y + dy, x + dx
                inside = 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]
                if not (inside and m[yy, xx]):
                    a.plot(*seg, color=color, lw=lw, solid_capstyle="butt",
                           zorder=6)


def true_dots(a, ys, xs):
    """The true line pixels as red circles that touch the sides of their
    pixel."""
    a.add_collection(PatchCollection(
        [Circle((x, y), 0.5) for y, x in zip(ys, xs)], fc=TRUE_C, ec="none",
        zorder=7))


def grid(a, r, c, lw=0.2):
    for k in range(c.start, c.stop + 1):
        a.axvline(k - 0.5, color="#c8c8c8", lw=lw, zorder=3)
    for k in range(r.start, r.stop + 1):
        a.axhline(k - 0.5, color="#c8c8c8", lw=lw, zorder=3)


def stencil(rr, cc, H, W):
    """The pixels the step value of pixel (rr, cc) uses: k-2 ... k+3 along
    each axis (run_19.inside3)."""
    out = []
    for s in range(-1, 2):
        for p in ((rr + s, cc), (rr, cc + s)):
            if 0 <= p[0] < H and 0 <= p[1] < W and p not in out:
                out.append(p)
    return out


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

    def parts(D, k):
        B, rays = D["B"][k], D["X"][k, 1] > 0.5
        strip = R.dilate(B, BAND)
        measured = strip | rays
        kept, step, tested = R.sweep_lines(D["Z"][k], strip, q, measured)
        return B, rays, strip, measured, kept, step, tested

    # over the 50 test devices: how many swept pixels have a direction with
    # both neighbours measured, and so are tested
    frac = []
    for k in range(len(test["Y"])):
        _, _, st, _, _, _, te = parts(test, k)
        frac.append(te.sum() / st.sum())
    i = test["names"].index(DEVICE)
    Y, Z = test["Y"][i], test["Z"][i]
    B, rays, strip, measured, found, step, tested = parts(test, i)
    f2, m = R.two_step(test, i, q, BAND)
    assert np.isclose(m, measured.mean()) and np.array_equal(f2, found)
    thr = float(np.quantile(step[tested], q))
    f1 = tolerant_f1(found, Y, 1.0)
    H, W = B.shape

    def rgb(*layers):
        img = np.ones(B.shape + (3,))
        for msk, colr in layers:
            img[msk] = to_rgb(colr)
        return img

    # one swept pixel, one step from the U-Net output, with all the pixels
    # its step value uses measured and inside the magnified region, for (d)
    r, c = ZOOM
    dist = np.zeros(B.shape, int)
    for k in range(1, BAND):
        dist[R.dilate(B, k) & ~R.dilate(B, k - 1)] = k
    cand = [tuple(p) for p in np.argwhere(R.inside3(measured)
                                          & strip & (dist == BAND - 1))
            if r.start + 1 <= p[0] < r.stop - 1
            and c.start + 1 <= p[1] < c.stop - 1]
    pc = cand[len(cand) // 2]
    sc = stencil(*pc, H, W)
    assert all(measured[p] for p in sc)

    ext = (c.start - 0.5, c.stop - 0.5, r.start - 0.5, r.stop - 0.5)
    ty, tx = np.nonzero(Y[r, c])
    fig, ax = plt.subplots(2, 3, figsize=(15.49 / 2.54, 4.3),
                           gridspec_kw=dict(wspace=0.05, hspace=0.06,
                                            left=0.005, right=0.995,
                                            top=0.995, bottom=0.065))
    ax = ax.ravel()
    yall, xall = np.nonzero(Y)
    # (b)-(d): the swept pixels and the binarized U-Net output on top
    base = rgb((strip, MEAS_C), (B, LINE_C))
    ax[0].imshow(rgb((rays, RAY_C), (B, LINE_C)), origin="lower",
                 interpolation="nearest")
    ax[1].imshow(base, origin="lower", interpolation="nearest")
    for a in ax[:2]:
        grid(a, slice(0, H), slice(0, W), lw=0.12)
        true_dots(a, yall, xall)
    ax[1].add_patch(Rectangle((c.start - 0.5, r.start - 0.5),
                              c.stop - c.start, r.stop - r.start,
                              fill=False, ec="#444444", lw=0.8, ls="--",
                              zorder=6))
    # the square of (b) enlarged in (c), joined as in Fig. 3
    for y_box, y_ax in ((r.stop - 0.5, 1.0), (r.start - 0.5, 0.0)):
        fig.add_artist(ConnectionPatch(
            xyA=(c.stop - 0.5, y_box), coordsA=ax[1].transData,
            xyB=(0.0, y_ax), coordsB=ax[2].transAxes,
            color="#888888", lw=0.6, ls="--", zorder=8))
    # (c): the raw U-Net output alone; (d): with the local measurement
    ax[2].imshow(rgb((B, LINE_C))[r, c], origin="lower",
                 interpolation="nearest", extent=ext)
    ax[3].imshow(base[r, c], origin="lower", interpolation="nearest",
                 extent=ext)
    # the step value in the colour of the swept pixels, darker if larger
    blues = LinearSegmentedColormap.from_list("swept", [MEAS_C, FOUND_C])
    ax[4].imshow(np.where(strip, step, np.nan)[r, c], origin="lower",
                 cmap=blues, vmin=0, vmax=np.quantile(step[tested], 0.99),
                 interpolation="nearest", extent=ext)
    # (e), (f): the binarized U-Net output as an outline, so that it does
    # not hide the step values or the estimated lines
    # (e): the U-Net output outlined over the step value; (f) shows only
    # the final estimate against the true lines
    outline(ax[4], B, r, c, LINE_C)
    ax[5].imshow(rgb((found, FOUND_C))[r, c], origin="lower",
                 interpolation="nearest", extent=ext)
    heads = ["rays and U-Net output",
             f"local sweeps ({100 * measured.mean():.1f}% measured)",
             "local sweeps (magnified)",
             "pixels used by one step value",
             "step value of the swept pixels",
             "estimated transition lines"]
    for k, (a, head) in enumerate(zip(ax, heads)):
        if k >= 2:
            grid(a, r, c, lw=0.3)
            true_dots(a, ty + r.start, tx + c.start)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        # the panel letter inside the panel, as in Fig. 7; no title
        a.text(0.97, 0.97, f"({'abcdef'[k]})", transform=a.transAxes,
               ha="right", va="top", fontsize=7.5, fontweight="bold",
               zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="#999999", lw=0.6))
    dot = Line2D([], [], ls="none", marker="o", ms=2.4, color=TRUE_C,
                 label="true transition line")
    fig.legend(handles=[Patch(color=RAY_C, label="ray"),
                        Patch(color=LINE_C, label="binarized U-Net output"),
                        Patch(color=MEAS_C,
                              label="swept: measured and tested"),
                        Line2D([], [], color=LINE_C, lw=1.2,
                               label="U-Net output (outline)"),
                        Patch(color=FOUND_C,
                              label="estimated transition line"), dot],
               loc="lower center", ncol=6, frameon=False, fontsize=5.8,
               handlelength=1.0, columnspacing=0.9, handletextpad=0.35)
    fig.savefig(os.path.join(OUT, "fig_strip_check.png"), dpi=400)
    plt.close(fig)

    vals = {"device": DEVICE, "band": BAND, "q": q, "threshold": thr,
            "device_measured": 100 * float(measured.mean()),
            "device_f1": f1["f1"],
            "device_tested_of_swept": float(tested.sum() / strip.sum()),
            "test_tested_of_swept_mean": float(np.mean(frac)),
            "test_tested_of_swept_min": float(np.min(frac)),
            "stencil_px": len(sc)}
    with open(os.path.join(OUT, "strip_check.json"), "w") as f:
        json.dump(vals, f, indent=1)
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
