"""
make_step_decision_figure.py -- how step 2 decides where a line is, on one
test device (a figure for Sec. 3.6).

    python scripts/run_19_two_step.py                     # table.csv
    python paper_figures/programs/make_step_decision_figure.py

(a) a magnified region on the pixel grid: the swept pixels, the true
    lines and one cut along V2;
(b), (c) the measured signal along the vertical and horizontal sweeps;
(d), (e) the step value along them and the threshold.
Each sweep is numbered from its first pixel (1), marked by an arrow in (a)
along the sweep direction; the true-line pixel where they cross is ringed.

Everything is recomputed with the study's own functions
(run_19_two_step.two_step, charge_states.step_residual_map) for the +-5 px
strip and the threshold quantile run_19 chose on the validation devices;
the kept pixels are checked to equal the study's.  Writes, into
method_docs_ver2/3_methods/figures/: fig_step_decision.png and
step_decision.json (the numbers the text quotes).
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
from matplotlib.patches import (Circle, Patch, Polygon,  # noqa: E402
                                Rectangle)
from scipy.ndimage import median_filter                  # noqa: E402

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.study import charge_states as C           # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 2
DEVICE = "sample_451"
CUT_COL = 30
CUT_ROWS = (9, 21)                 # inclusive start, exclusive end
ZOOM = (slice(0, 28), slice(9, 37))    # the line, the rejected stripe at
#                                        the bottom, and room left of every
#                                        sweep for the arrow into it
REJ_ROW, REJ_COL = 1, 25           # a third sweep, along row 1, through a
#                                    U-Net prediction with no line under it
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
# the colours of Figs. 13, 14 and 18: swept pixels light blue, kept
# (estimated) line pixels dark blue, true line pixels red dots
CHECK_C, KEPT_C, TRUE_C, CUT_C = "#bfd3f8", "#0b2f6b", "#d00000", "#b45309"
DOT_R = 0.3                        # red dot radius, px (as in Fig. 18)
LINE_C = "#2563eb"                 # binarized U-Net output
RAY_C = "#555555"                  # the rays (step 1)
# the two steps, as in the pipeline figure (user, 2026-10-26)
PRED_C, PRED_BG = "#1e40af", "#ffffff"         # as in Fig. 1
VER_C, VER_BG = "#b45309", "#ffffff"
INK = "#111827"
CUT3_C = "#7b3fa0"                 # the third sweep
CUT2_C = "#1b9e9e"                 # the horizontal sweep
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444"})


def sweep_counts(test, band, i_dev):
    """How many sweeps along V1 (rows) and V2 (columns) the strip of
    margin band is made of, and how long they are."""
    def runs(m):
        out = []
        for line in m:
            d = np.diff(np.r_[0, line.astype(int), 0])
            out += list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])
        return np.array(out)
    n1, n2, l1 = [], [], []
    for k in range(len(test["Y"])):
        strip = R.dilate(test["B"][k], band)
        h, v = runs(strip), runs(strip.T)
        n1.append(len(h))
        n2.append(len(v))
        l1.append(h.mean())
        if k == i_dev:
            dev = {"dev_sweeps_v1": int(len(h)), "dev_sweeps_v2": int(len(v)),
                   "dev_len_v1": float(h.mean())}
    return {**dev, "sweeps_v1_min": int(min(n1)), "sweeps_v1_max": int(max(n1)),
            "sweeps_v1_mean": float(np.mean(n1)),
            "sweeps_v2_mean": float(np.mean(n2)),
            "len_v1_mean": float(np.mean(l1))}


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
    Y, B, Z, X = test["Y"][i], test["B"][i], test["Z"][i], test["X"][i]

    strip = R.dilate(B, BAND)
    measured = strip | (X[1] > 0.5)
    kept, Rmap, tested = R.sweep_lines(Z, strip, q, measured)
    assert np.array_equal(kept, R.two_step(test, i, q, BAND)[0])
    thr = float(np.quantile(Rmap[tested], q))
    # along the cut (V2), a pixel is tested in this direction if its
    # neighbours above and below were measured
    v2 = (measured & R._neighbour_measured(measured, 0, -1)
          & R._neighbour_measured(measured, 0, 1))

    # the two sweeps through one true-line pixel, from the same formula as
    # step3_map: the change to the next pixel against the change just
    # before it; along V2 (rows) and along V1 (columns)
    ref = np.asarray(Z, float)[measured]
    Zs = (np.asarray(Z, float) - ref.min()) / np.ptp(ref)

    def sweep_values(ax):
        d = np.diff(Zs, axis=ax)
        b = np.empty_like(d)
        if ax == 0:
            b[1:], b[0] = d[:-1], d[0]
            both = measured[1:] & measured[:-1]
        else:
            b[:, 1:], b[:, 0] = d[:, :-1], d[:, 0]
            both = measured[:, 1:] & measured[:, :-1]
        scale = float(np.median(np.abs(d[both])))
        step = np.pad(np.abs(d - b) / (np.abs(b) + scale),
                      [(0, 1), (0, 0)] if ax == 0 else [(0, 0), (0, 1)])
        ok = (measured & R._neighbour_measured(measured, ax, -1)
              & R._neighbour_measured(measured, ax, 1))
        assert np.all(step[ok] <= Rmap[ok] + 1e-12)
        return d, b, step, ok

    col = CUT_COL
    t0 = next(k for k in range(*CUT_ROWS) if Y[k, col])

    def run(line, k0):
        """The measured run of pixels of a 1-D boolean line through k0."""
        lo = hi = k0
        while lo - 1 >= 0 and line[lo - 1]:
            lo -= 1
        while hi + 1 < len(line) and line[hi + 1]:
            hi += 1
        return np.arange(lo, hi + 1)

    d, b, step, v2 = sweep_values(0)
    d1, b1, step1, v1 = sweep_values(1)
    rr = run(measured[:, col], t0)                  # vertical sweep: rows
    cc = run(measured[t0, :], col)                  # horizontal sweep: cols
    c3 = run(measured[REJ_ROW, :], REJ_COL)          # the third sweep: cols
    # it crosses one true line, and U-Net pixels far from any true line
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(~Y)
    false3 = B[REJ_ROW, c3] & (dt[REJ_ROW, c3] >= 3)
    assert Y[REJ_ROW, c3].any() and false3.sum() >= 3
    assert not (kept[REJ_ROW, c3] & false3).any()   # all of them rejected
    # each sweep is numbered from its first pixel (1, arrowed in (a)) in
    # the direction of increasing gate voltage
    sweeps = [("V$_2$", rr - rr[0] + 1, Zs[rr, col], step[rr, col],
               v2[rr, col], rr[Y[rr, col]] - rr[0] + 1, CUT_C),
              ("V$_1$", cc - cc[0] + 1, Zs[t0, cc], step1[t0, cc],
               v1[t0, cc], cc[Y[t0, cc]] - cc[0] + 1, CUT2_C)]

    # layout: row 1 -- (a) step 1, (b) the swept strip, (c) the region of
    # its dashed square with the three sweeps; row 2 -- (d)-(f) the signals
    # and (g)-(i) the step values of the three sweeps; row 3 -- (j) all the
    # measured pixels, (k) the result
    # row 1: (a) step 1, (b) the band, (c) its dashed square magnified with
    # the three sweeps; row 2: (d)-(f) signals, (g)-(i) step values; row 3:
    # (j) what was measured, (k) the result, (l) its dashed square magnified.
    # One column grid for all three rows, so that the panels line up.
    W, top_m = 15.49 / 2.54, 0.34       # top: the step names
    X0, GX = 0.48, 0.30                    # left margin (axis labels); gap
    ht = (W - X0 - 0.08 - 2 * GX) / 3      # every panel is ht wide
    hq = ht                                # row 3 panels are ht x ht

    def xcol(k):
        return X0 + k * (ht + GX)
    hp, gp = 0.88, 0.10                    # a sweep plot; gap in a column
    bot3 = 0.32
    bot2 = bot3 + hq + 0.90                # the step values (g)-(i); room
    #                                        for the two stage boxes between
    bot1 = bot2 + 2 * hp + gp + 0.42       # row 1
    H = bot1 + ht + top_m
    fig = plt.figure(figsize=(W, H))

    def box(x, y, w, hh):
        return fig.add_axes([x / W, y / H, w / W, hh / H])
    top3 = [box(xcol(k), bot1, ht, ht) for k in range(2)]
    a = box(xcol(2), bot1, ht, ht)          # (c): magnified
    sig_ax = [box(xcol(k), bot2 + hp + gp, ht, hp) for k in range(3)]
    stp_ax = [box(xcol(k), bot2, ht, hp) for k in range(3)]
    # row 3: what was measured -> (arrow: local step detection) -> the
    # result -> its dashed square magnified
    GA = GX
    ax_m = box(xcol(0), bot3, hq, hq)
    ax_j = box(xcol(1), bot3, hq, hq)
    ax_k = box(xcol(2), bot3, hq, hq)

    def letter(ax_, lab):
        ax_.text(0.97, 0.95, f"({lab})", transform=ax_.transAxes, ha="right",
                 va="top", fontsize=7.5, fontweight="bold", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="#999999", lw=0.6))

    def frame(ax_):
        ax_.set_xticks([])
        ax_.set_yticks([])
        for s_ in ax_.spines.values():
            s_.set_visible(True)
            s_.set_linewidth(0.6)

    # ---- (c): the magnified region, with the three sweeps: the band, the
    # binarized U-Net output and the true lines
    r, c = ZOOM
    img = np.ones(Y.shape + (3,))
    for m, colr in ((strip, CHECK_C), (B, LINE_C)):
        img[m] = to_rgb(colr)
    a.imshow(img[r, c], origin="lower", interpolation="nearest",
             extent=(c.start - 0.5, c.stop - 0.5, r.start - 0.5,
                     r.stop - 0.5))
    for k in range(c.start, c.stop + 1):
        a.axvline(k - 0.5, color="#c8c8c8", lw=0.2, zorder=3)
    for k in range(r.start, r.stop + 1):
        a.axhline(k - 0.5, color="#c8c8c8", lw=0.2, zorder=3)
    ty, tx = np.nonzero(Y[r, c])                 # the true line: red dots
    a.add_collection(PatchCollection(
        [Circle((x + c.start, y + r.start), DOT_R) for y, x in zip(ty, tx)],
        fc=TRUE_C, ec="none", zorder=5))
    a.add_patch(Rectangle((col - 0.5, rr[0] - 0.5), 1, len(rr), fill=False,
                          ec=CUT_C, lw=1.1, zorder=4))
    a.add_patch(Rectangle((cc[0] - 0.5, t0 - 0.5), len(cc), 1, fill=False,
                          ec=CUT2_C, lw=1.1, zorder=4))
    a.add_patch(Rectangle((c3[0] - 0.5, REJ_ROW - 0.5), len(c3), 1,
                          fill=False, ec=CUT3_C, lw=1.1, zorder=4))
    # the pixel where the sweeps cross, and an arrow into the first pixel
    # of each sweep, along its direction
    a.add_patch(Circle((col, t0), 0.5, fill=False, ec="#111111", lw=0.9,
                       zorder=6))
    for (x1, y1), (x0_, y0_), colr in (
            ((col, rr[0] - 0.55), (col, rr[0] - 2.0), CUT_C),
            ((cc[0] - 0.55, t0), (cc[0] - 2.0, t0), CUT2_C),
            ((c3[0] - 0.55, REJ_ROW), (c3[0] - 2.0, REJ_ROW), CUT3_C)):
        a.annotate("", xy=(x1, y1), xytext=(x0_, y0_), zorder=7,
                   annotation_clip=False,
                   arrowprops=dict(arrowstyle="-|>", color=colr, lw=0.9,
                                   mutation_scale=6, shrinkA=0, shrinkB=0))
    a.set_xlim(c.start - 0.5, c.stop - 0.5)
    a.set_ylim(r.start - 0.5, r.stop - 0.5)

    # ---- (d)-(i): the three sweeps -------------------------------------
    idx3 = c3 - c3[0] + 1
    sweeps.append(("V$_1$", idx3, Zs[REJ_ROW, c3], step1[REJ_ROW, c3],
                   v1[REJ_ROW, c3], c3[Y[REJ_ROW, c3]] - c3[0] + 1, CUT3_C))
    for k, (name, idx, sig, sv, on, true_px, colr) in enumerate(sweeps):
        top, bt = sig_ax[k], stp_ax[k]
        bt.sharex(top)
        for ax_ in (top, bt):
            ax_.tick_params(labelsize=6)
            for s in ax_.spines.values():
                s.set_linewidth(0.6)
            if k == 2:      # the U-Net prediction with no true line under it
                for k_ in idx3[false3]:
                    ax_.axvspan(k_ - 0.5, k_ + 0.5, color=LINE_C,
                                alpha=0.22, lw=0, zorder=0)
        top.plot(idx, 1e3 * sig, "o-", ms=2.0, lw=0.8, color=colr)
        top.plot(true_px, 1e3 * sig[true_px - 1], "o", ms=3.6,
                 color=TRUE_C, zorder=5)
        top.tick_params(labelbottom=False)
        bt.bar(idx[on], sv[on], width=0.75,
               color=np.where(sv[on] > thr, KEPT_C, "#9a9a9a"))
        bt.axhline(thr, color="#333333", ls="--", lw=0.7)
        bt.plot(true_px, sv[true_px - 1], "o", ms=3.6, color=TRUE_C,
                zorder=5)
        # (review, 2026-10-27) the decision made visible: "kept" beside
        # the bar of the true line, as "predicted line, rejected" in (i)
        jk = true_px[int(np.argmax(sv[true_px - 1]))]
        assert sv[jk - 1] > thr
        bt.text(jk + 0.55, sv[jk - 1], "kept", ha="left", va="center",
                fontsize=5.8, color=KEPT_C)
        # the strip is swept along V1; the V2 trace is read from pixels
        # already measured, so all three are "traces", not sweeps
        bt.set_xlabel(f"pixel along {name}", fontsize=6.5)
        bt.set_xlim(0.4, idx[-1] + 0.6)
        step_t = 4 if idx[-1] > 16 else (2 if idx[-1] > 8 else 1)
        bt.set_xticks(range(1, idx[-1] + 1, step_t))
        if k == 0:
            top.set_ylabel("signal (×10$^{-3}$)", fontsize=6.5)
            bt.set_ylabel("step value", fontsize=6.5)
        letter(top, "def"[k])
        letter(bt, "ghi"[k])
    stp_ax[2].text(idx3[false3].mean(), 0.5 * stp_ax[2].get_ylim()[1],
                   # (user, 2026-10-27) "U-Net": the network proposes,
                   # the step detector decides
                   "U-Net\npredicted\nline,\nrejected", ha="center",
                   va="center",
                   fontsize=5.8, color="#2b5d8f")

    # ---- (a), (b), (j), (k): the whole device --------------------------
    H_, W_ = Y.shape
    yy, xx = np.nonzero(Y)
    rays = X[1] > 0.5
    panels = ((top3[0], ((rays, RAY_C), (B, LINE_C)), "a"),
              (top3[1], ((strip, CHECK_C), (B, LINE_C)), "b"),
              (ax_j, ((kept, KEPT_C),), "k"))
    for ax_, layers, lab in panels:
        im = np.ones(Y.shape + (3,))
        for m, colr in layers:
            im[m] = to_rgb(colr)
        ax_.imshow(im, origin="lower", interpolation="nearest")
        for g_ in range(W_ + 1):
            ax_.axvline(g_ - 0.5, color="#d8d8d8", lw=0.1, zorder=3)
        for g_ in range(H_ + 1):
            ax_.axhline(g_ - 0.5, color="#d8d8d8", lw=0.1, zorder=3)
        ax_.add_collection(PatchCollection(
            [Circle((x, y), 0.45) for y, x in zip(yy, xx)], fc=TRUE_C,
            ec="none", zorder=5))
        letter(ax_, lab)
    # (k): the dashed square of (j), magnified -- the kept pixels against
    # the true line, pixel by pixel
    img_k = np.ones(Y.shape + (3,))
    img_k[kept] = to_rgb(KEPT_C)
    ax_k.imshow(img_k[r, c], origin="lower", interpolation="nearest",
                extent=(c.start - 0.5, c.stop - 0.5, r.start - 0.5,
                        r.stop - 0.5))
    for k_ in range(c.start, c.stop + 1):
        ax_k.axvline(k_ - 0.5, color="#c8c8c8", lw=0.2, zorder=3)
    for k_ in range(r.start, r.stop + 1):
        ax_k.axhline(k_ - 0.5, color="#c8c8c8", lw=0.2, zorder=3)
    ax_k.add_collection(PatchCollection(
        [Circle((x + c.start, y + r.start), DOT_R) for y, x in zip(ty, tx)],
        fc=TRUE_C, ec="none", zorder=5))
    ax_k.set_xlim(c.start - 0.5, c.stop - 0.5)
    ax_k.set_ylim(r.start - 0.5, r.stop - 0.5)
    # what was measured: rays, U-Net output and the band around it, named
    # together in one text box
    im = np.ones(Y.shape + (3,))
    for m, colr in ((strip, CHECK_C), (B, LINE_C), (rays, RAY_C)):
        im[m] = to_rgb(colr)
    ax_m.imshow(im, origin="lower", interpolation="nearest")
    for g_ in range(W_ + 1):
        ax_m.axvline(g_ - 0.5, color="#d8d8d8", lw=0.1, zorder=3)
    for g_ in range(H_ + 1):
        ax_m.axhline(g_ - 0.5, color="#d8d8d8", lw=0.1, zorder=3)
    # (review, 2026-10-27) no coverage under (j), no F1 under (k): the
    # metric is defined only in Sec. 3.5
    # the arrow from what was measured to the result (j)
    from matplotlib.patches import ConnectionPatch
    fig.add_artist(ConnectionPatch(
        xyA=(1.0, 0.5), coordsA=ax_m.transAxes, xyB=(0.0, 0.5),
        coordsB=ax_j.transAxes, arrowstyle="-|>", mutation_scale=9, lw=1.2,
        color=INK, shrinkA=3, shrinkB=3, zorder=9))
    # the operation the arrow stands for is named in the caption
    for ax_ in top3 + [a, ax_m, ax_j, ax_k]:
        frame(ax_)
    letter(a, "c")
    letter(ax_m, "j")
    letter(ax_k, "l")
    # the dashed squares, each joined to its magnified panel by two dashed
    # lines from its near corners, as in Fig. 3
    from matplotlib.patches import ConnectionPatch
    for src, dst in ((top3[1], a), (ax_j, ax_k)):
        src.add_patch(Rectangle((c.start - 0.5, r.start - 0.5),
                                c.stop - c.start, r.stop - r.start,
                                fill=False, ec="#222222", lw=0.8, ls="--",
                                zorder=6))
        for y_box, y_ax in ((r.stop - 0.5, 1.0), (r.start - 0.5, 0.0)):
            fig.add_artist(ConnectionPatch(
                xyA=(c.stop - 0.5, y_box), coordsA=src.transData,
                xyB=(0.0, y_ax), coordsB=dst.transAxes, color="#888888",
                lw=0.6, ls="--", zorder=8))
    # no legend: the caption names the colours
    # (user, 2026-10-26) four boxes, every panel in exactly one: (a) step 1,
    # line prediction; step 2 in three stages -- (b), (c) local
    # acquisition, (d)-(i) step detection, (j)-(l) line selection
    from matplotlib.patches import FancyBboxPatch

    def frame_box(x0_, x1_, y0_, y1_, name, c, bg):
        fig.add_artist(FancyBboxPatch(
            (x0_ / W, y0_ / H), (x1_ - x0_) / W, (y1_ - y0_) / H,
            boxstyle="round,pad=0,rounding_size=%g" % (0.08 / W),
            mutation_aspect=W / H, fc=bg, ec=c, lw=1.1, zorder=-1,
            transform=fig.transFigure))
        fig.text((x0_ + 0.14) / W, y1_ / H, name, ha="left", va="center",
                 fontsize=7.5, fontweight="bold", color=c, zorder=10,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white",
                           ec="none"))
    xr = W - 0.04
    # (a)'s box from the same left edge as the full-width boxes to the
    # middle of the gap before (b)
    xm = xcol(0) + ht + GX / 2
    frame_box(0.04, xm - 0.04, bot1 - 0.06,
              bot1 + ht + 0.24, "Step 1: Line prediction", PRED_C, PRED_BG)
    frame_box(xm + 0.04, xr, bot1 - 0.06, bot1 + ht + 0.24,
              "Step 2a: Local acquisition", VER_C, VER_BG)
    frame_box(0.04, xr, bot2 - 0.40, bot1 - 0.14,
              "Step 2b: Step detection", VER_C, VER_BG)
    frame_box(0.04, xr, bot3 - 0.26, bot2 - 0.48,
              "Step 2c: Line selection", VER_C, VER_BG)
    fig.savefig(os.path.join(OUT, "fig_step_decision.png"), dpi=400)
    plt.close(fig)

    # the numbers the text quotes (the vertical sweep, as before, and the
    # horizontal one)
    sv, on = step[rr, col], v2[rr, col]
    true_rows = rr[Y[rr, col]]
    j = int(np.argmax(sv))
    k = int(rr[j])
    sv1, on1 = step1[t0, cc], v1[t0, cc]
    sv3, on3 = step1[REJ_ROW, c3], v1[REJ_ROW, c3]
    vals = {"threshold": thr, "q": q,
            "jump_step": float(step[k, col]),
            "jump_change": float(1e3 * d[k, col]),
            "usual_change": float(1e3 * b[k, col]),
            "background_max_step": float(np.max(sv[on & ~np.isin(
                rr, np.r_[true_rows, true_rows - 1, true_rows + 1])])),
            "n_true_on_cut": int(len(true_rows)),
            "min_true_step": float(np.min(step[true_rows, col])),
            "kept_on_cut": [int(x) for x in rr[on & (sv > thr)]],
            "true_on_cut": [int(x) for x in true_rows],
            "h_jump_step": float(np.max(sv1[on1])),
            "h_kept_on_cut": [int(x) for x in cc[on1 & (sv1 > thr)]],
            "h_true_on_cut": [int(x) for x in cc[Y[t0, cc]]],
            # the third sweep: a true line, then a rejected U-Net prediction
            "rej_row": REJ_ROW, "rej_cols": [int(c3[0]), int(c3[-1])],
            "rej_true_step": float(np.max(sv3[on3 & Y[REJ_ROW, c3]])),
            "rej_false_px": int(false3.sum()),
            "rej_false_dist": [float(dt[REJ_ROW, c3][false3].min()),
                               float(dt[REJ_ROW, c3][false3].max())],
            "rej_false_step": [float(sv3[on3 & false3].min()),
                               float(sv3[on3 & false3].max())],
            # the whole device, (a)-(c)
            "measured": 100 * float(measured.mean()),
            "rays_measured": 100 * float((X[1] > 0.5).mean()),
            "f1_unet": tolerant_f1(B, Y, 1.0)["f1"],
            "f1_final": tolerant_f1(kept, Y, 1.0)["f1"],
            "zoom_rows": [r.start, r.stop], "zoom_cols": [c.start, c.stop],
            # the strip as one-dimensional sweeps: every continuous run of
            # strip pixels along V1 (a row) is one sweep; this device and
            # the 50 test devices
            **sweep_counts(test, BAND, i)}
    with open(os.path.join(OUT, "step_decision.json"), "w") as f:
        json.dump(vals, f, indent=1)
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
