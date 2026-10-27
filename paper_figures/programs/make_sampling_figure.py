"""
make_sampling_figure.py -- what the U-Net adds: the same local sweeps
(+-BAND) and step detector around equally spaced seed points
(run_23_equal_seeds.py), against U-Net + local sweeps (+-BAND), on the
device of Fig. 12 (Fig. 18, Sec. 4.3).  Derived from make_final_figure.py.

    python scripts/run_19_two_step.py                     # table.csv
    python scripts/run_23_equal_seeds.py                  # equal_seeds.csv
    python paper_figures/programs/make_sampling_figure.py

columns: equally spaced seeds (left), U-Net + local sweeps (right);
(a), (b) a ZOOM x ZOOM window inside one charge state (Coulomb blockade,
    no true line within 1 px) where U-Net + sweeps measure the least;
(c), (d) the whole grid: measured pixels and estimated lines, the two
    magnified squares dashed and joined to their panels;
(e), (f) the ZOOM x ZOOM window, with at least MIN_TRUE true-line pixels,
    where the U-Net removes the most errors.
In each magnified panel the pixels measured in it; the fraction of the
plane measured and F1@1 of each column are in the caption (json).

The budgets are matched on this device: the seed spacing is the largest
whose strip measures at least as much of the plane as U-Net + local sweeps
do here, and its threshold is chosen on the 75 validation devices by F1@1,
as in run_23 (whose 50-device means, matched to the mean cost, are also
stored).
Writes,
into method_docs_ver2/3_methods/figures/: fig_sampling_vs_final.png and
sampling_vs_final.json (the numbers the text quotes, with the 50-device
means of both).
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
import run_23_equal_seeds as E                           # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 2
DEVICE = "sample_223"              # the device of Fig. 12
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
LINE_C, FOUND_C, TRUE_C = "#4f8fc8", "#0b2f6b", "#d00000"   # as in Fig. 12
ZOOM = 22                          # side of the magnified square, px
MIN_TRUE = 40                      # true-line pixels the square must hold
MEAS_C = "#bfd3f8"                 # measured pixels, as in Fig. 8
SLATE, BLUE = "#4a5568", "#1e40af"   # the two columns, as in Fig. 1
RAY_C = "#8a8a8a"                  # the rays, measured before the U-Net
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
    # the equally spaced seeds matched by run_23 to this band
    with open(os.path.join(R.OUT, "equal_seeds.csv"), newline="",
              encoding="utf-8") as f:
        eq = next(r for r in csv.DictReader(f)
                  if r["matched"] == f"band ±{BAND} px")
    # this device: the equally spaced seeds get at least its budget
    per = max(p for p in range(1, Y.shape[0])
              if E.strip_of(Y.shape, p, BAND).mean() >= m_f)
    val = R.load(net, float(ck["threshold"]), "validation")
    q_e = R.choose(val, E.equal_seeds, per, BAND)
    B, m_u = E.equal_seeds(test, i, q_e, per, BAND)
    assert m_u >= m_f
    meas_u = E.strip_of(Y.shape, per, BAND)
    meas_f = R.dilate(test["B"][i], BAND) | (X[1] > 0.5)
    assert np.isclose(meas_f.mean(), m_f) and np.isclose(meas_u.mean(), m_u)
    f_u, f_f = tolerant_f1(B, Y, 1.0), tolerant_f1(found, Y, 1.0)
    H, W = Y.shape
    # the errors F1@1 counts: a predicted pixel with no true pixel within
    # 1 px, and a true pixel with no predicted pixel within 1 px
    dt = distance_transform_edt(~Y)
    false_u, false_f = B & (dt > 1), found & (dt > 1)
    miss_u = Y & (distance_transform_edt(~B) > 1)
    miss_f = Y & (distance_transform_edt(~found) > 1)

    # magnified region 1: the ZOOM x ZOOM window, crossing true lines,
    # where the U-Net removes the most errors (false plus missed pixels)
    best = None
    for r0 in range(0, H - ZOOM + 1, 2):
        for c0 in range(0, W - ZOOM + 1, 2):
            sl = (slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM))
            gain = (int(false_u[sl].sum() + miss_u[sl].sum())
                    - int(false_f[sl].sum() + miss_f[sl].sum()))
            # a region that crosses true lines, not an empty corner
            if Y[sl].sum() < MIN_TRUE:
                continue
            if best is None or gain > best[0]:
                best = (gain, r0, c0)
    _, r0, c0 = best
    r, c = slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM)
    # magnified region 2: a ZOOM x ZOOM window inside one charge state
    # (Coulomb blockade, no true line within 1 px) where U-Net + sweeps
    # measure the fewest pixels; ties go to the window with the fewest
    # estimated line pixels of either method, then to the highest one
    # (nearest the panel above it, so the joining lines stay short)
    near = dt <= 1
    best = None
    for r0 in range(0, H - ZOOM + 1):
        for c0 in range(0, W - ZOOM + 1):
            sl = (slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM))
            if near[sl].any():
                continue
            key = (int(meas_f[sl].sum()), int((B | found)[sl].sum()), -r0)
            if best is None or key < best[0]:
                best = (key, r0, c0)
    assert best is not None, "no line-free window of this size"
    _, r0, c0 = best
    r2, c2 = slice(r0, r0 + ZOOM), slice(c0, c0 + ZOOM)

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

    def true_dots(a, ys, xs, est):
        """The true line pixels as red dots of one size, small enough that
        an estimate under the dot shows around it (est is kept for the
        callers)."""
        a.add_collection(PatchCollection(
            [Circle((x, y), 0.3) for y, x in zip(ys, xs)], fc=TRUE_C,
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

    # rows: the region inside one charge state (a, b), the whole grid
    # (c, d), the region crossed by lines (e, f); each magnified region is
    # joined to its dashed square by two dashed lines from the near corners
    # the magnified panels (a, b, e, f) at the left and right edges, the
    # whole grids (c, d) larger and close together in the centre
    FW, Z, M = 7.2, 2.45, 2.95                 # figure width, panel sides, in
    GAP, LEG = 0.30, 0.26                       # between rows, bottom margin
    FH = LEG + 2 * Z + M + 2 * GAP + 0.40        # counts above (a), (b); tabs
    EDGE = 0.12                                 # panels inside the boxes
    fig = plt.figure(figsize=(FW, FH))
    ax = np.empty((3, 2), dtype=object)
    for row, (side, y0) in enumerate(((Z, LEG + Z + M + 2 * GAP),
                                       (M, LEG + Z + GAP), (Z, LEG))):
        if side == Z:
            xs = (EDGE, FW - EDGE - Z)
        else:
            # (user, 2026-10-27) symmetric margins: the inner gap to each
            # box edge equals the outer one (EDGE - 0.03)
            xs = (FW / 2 - 0.035 - (EDGE - 0.03) - M,
                  FW / 2 + 0.035 + (EDGE - 0.03))
        for k, x0 in enumerate(xs):
            ax[row, k] = fig.add_axes((x0 / FW, y0 / FH, side / FW,
                                       side / FH))
    full = (slice(0, H), slice(0, W))
    yy, xx = np.nonzero(Y)
    rays = X[1] > 0.5
    no_rays = np.zeros_like(rays)
    cols = ((B, meas_u, no_rays), (found, meas_f, rays))
    for k, (m, ms, ry) in enumerate(cols):
        mid = ax[1, k]
        img = rgb((ms, MEAS_C), (ry, RAY_C), (m, FOUND_C))
        mid.imshow(img, origin="lower", interpolation="nearest")
        grid(mid, *full, lw=0.12)
        true_dots(mid, yy, xx, m)
        for row, (rr, cc) in ((0, (r2, c2)), (2, (r, c))):
            zax, lab = ax[row, k], "abef"[row + k]
            # the dashed square in the whole grid
            mid.add_patch(Rectangle((cc.start - 0.5, rr.start - 0.5), ZOOM,
                                    ZOOM, fill=False, ec="#444444", lw=0.8,
                                    ls="--", zorder=6))
            ext = (cc.start - 0.5, cc.stop - 0.5, rr.start - 0.5,
                   rr.stop - 0.5)
            zax.imshow(img[rr, cc], origin="lower", interpolation="nearest",
                       extent=ext)
            grid(zax, rr, cc, lw=0.3)
            ty, tx = np.nonzero(Y[rr, cc])
            true_dots(zax, ty + rr.start, tx + cc.start, m)
            frame(zax)
            letter(zax, lab)
            zax.text(0.5, 1.02 if row == 0 else -0.02,
                     f"{int(ms[rr, cc].sum())} of {ZOOM * ZOOM} pixels "
                     "measured", transform=zax.transAxes, ha="center",
                     va="bottom" if row == 0 else "top", fontsize=7,
                     color="#222222")
            # upper region: from the top corners of the square to the
            # bottom corners of the panel above; lower: bottom to top
            y_box = rr.stop - 0.5 if row == 0 else rr.start - 0.5
            y_ax = 0.0 if row == 0 else 1.0
            for x_box, x_ax in ((cc.start - 0.5, 0.0), (cc.stop - 0.5, 1.0)):
                fig.add_artist(ConnectionPatch(
                    xyA=(x_box, y_box), coordsA=mid.transData,
                    xyB=(x_ax, y_ax), coordsB=zax.transAxes,
                    color="#888888", lw=0.6, ls="--", zorder=8))
        frame(mid)
        letter(mid, "cd"[k])
    # the two columns, each in a box as in Fig. 1
    from matplotlib.patches import FancyBboxPatch
    for x0_, x1_, name, colr in (
            # (user, 2026-10-27) the terms of Fig. 11 and Table II
            (0.03, FW / 2 - 0.035, "Uniform-grid local sweeps", SLATE),
            (FW / 2 + 0.035, FW - 0.03,
             "U-Net-guided local sweeps", BLUE)):
        y0_, y1_ = 0.03, FH - 0.09
        fig.add_artist(FancyBboxPatch(
            (x0_ / FW, y0_ / FH), (x1_ - x0_) / FW, (y1_ - y0_) / FH,
            boxstyle="round,pad=0,rounding_size=%g" % (0.08 / FW),
            mutation_aspect=FW / FH, fc="white", ec=colr, lw=1.1,
            zorder=-1, transform=fig.transFigure))
        fig.text((x0_ + 0.14) / FW, y1_ / FH, name, ha="left", va="center",
                 fontsize=8, fontweight="bold", color=colr, zorder=10,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white",
                           ec="none"))
    # no legend: the caption names the colours
    fig.savefig(os.path.join(OUT, "fig_sampling_vs_final.png"), dpi=400)
    plt.close(fig)

    def errs(fal, mis, sl):
        return {"false_px": int(fal[sl].sum()), "missed_px": int(mis[sl].sum())}
    vals = {"device": DEVICE, "band": BAND, "period": per, "q": q_e,
            "mean_period": int(eq["period"]),
            "mean_test": {"sampling": {k: float(eq[k]) for k in
                                       ("measured", "f1")},
                          "final": {"measured": float(eq["matched_measured"]),
                                    "f1": float(eq["matched_f1"])}},
            "zoom_rows": [r.start, r.stop], "zoom_cols": [c.start, c.stop],
            "blockade_rows": [r2.start, r2.stop],
            "blockade_cols": [c2.start, c2.stop],
            "blockade_measured_px": {"sampling": int(meas_u[r2, c2].sum()),
                                     "final": int(meas_f[r2, c2].sum()),
                                     "of": ZOOM * ZOOM},
            "blockade_final_split": {
                "rays": int(rays[r2, c2].sum()),
                "strip_only": int((meas_f & ~rays)[r2, c2].sum()),
                "unet_px": int(test["B"][i][r2, c2].sum())},
            "blockade_line_px": {"sampling": int(B[r2, c2].sum()),
                                 "final": int(found[r2, c2].sum())},
            "sampling": {**{k: f_u[k] for k in ("precision", "recall", "f1")},
                     "measured": 100 * m_u, **errs(false_u, miss_u, full),
                     # where the false pixels lie (origin lower: high rows
                     # are the top of the plot)
                     "false_upper_right": int(false_u[H // 2:, W // 2:].sum()),
                     "zoom": errs(false_u, miss_u, (r, c))},
            "final": {**{k: f_f[k] for k in ("precision", "recall", "f1")},
                      "measured": 100 * m_f, **errs(false_f, miss_f, full),
                      "zoom": errs(false_f, miss_f, (r, c))}}
    with open(os.path.join(OUT, "sampling_vs_final.json"), "w") as f:
        json.dump(vals, f, indent=1)
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
