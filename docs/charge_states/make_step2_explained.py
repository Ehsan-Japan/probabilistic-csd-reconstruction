# -*- coding: utf-8 -*-
"""
make_step2_explained.py -- a short PDF explaining step 2 (the local check)
on one test device, with a picture of every stage.

    python scripts/run_19_two_step.py            # the evidence (table.csv)
    python docs/charge_states/make_step2_explained.py

Every intermediate quantity is recomputed here with the same functions the
study uses (run_19_two_step.two_step, charge_states.step_residual_map), for
the +-5 px strip and the threshold quantile run_19 chose on the validation
devices.  Writes Step2_explained.pdf into paper_figures/method_docs_ver2/.
"""
import csv
import os
import subprocess
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
from matplotlib.patches import Patch, Rectangle          # noqa: E402
from scipy.ndimage import (distance_transform_cdt,       # noqa: E402
                           distance_transform_edt, median_filter)
from skimage.morphology import skeletonize               # noqa: E402

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from csdrecon.study import charge_states as C           # noqa: E402
from run_18_equal_cost import coarse_image               # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

BAND = 5
DEVICE = "sample_223"
CUT_COL = 22                       # the 1-D cut: this column, along V2
CUT_ROWS = slice(4, 46)
ZOOM = (slice(4, 24), slice(34, 60))   # rows, cols of the centre-line zoom
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2")
BUILD = os.path.join(OUT, "step2_explained_build")
STRIP_C, CHECK_C, CENTRE_C, KEPT_C, TRUE_C, DROP_C = (
    "#e3eef8", "#a9c9e8", "#1f4e8c", "#c00000", "#9a9a9a", "#f2a65a")
FALSE_C, MISS_C, LINE_C = "#111111", "#21a0a0", "#7fa8d6"
# found vs true at the 1-px tolerance of F1@1: a found pixel is correct if a
# true line pixel lies within 1 px, a true pixel is missed if no found pixel
# lies within 1 px
SCORE_KEY = [(KEPT_C, "found, within 1 px of a true line"),
             (FALSE_C, "found, no true line within 1 px"),
             (MISS_C, "true line missed (nothing found within 1 px)"),
             (TRUE_C, "true line, found")]
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def grid(a, H, W, lw=0.12):
    for k in range(W + 1):
        a.axvline(k - 0.5, color="#d0d0d0", lw=lw, zorder=3)
    for k in range(H + 1):
        a.axhline(k - 0.5, color="#d0d0d0", lw=lw, zorder=3)


def scored(found, Y):
    """The four classes of SCORE_KEY, as (mask, colour) layers, and the
    device's precision / recall / F1@1 from the study's own metric."""
    # Euclidean, as in tolerant_f1: at 1 px only the four neighbours count,
    # the diagonals (1.41 px) do not
    d_true = distance_transform_edt(~Y)
    d_found = distance_transform_edt(~found)
    layers = [(Y & (d_found <= 1), TRUE_C), (Y & (d_found > 1), MISS_C),
              (found & (d_true > 1), FALSE_C), (found & (d_true <= 1), KEPT_C)]
    return layers, tolerant_f1(found, Y, 1.0)


def main():
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["setting"] == f"band ±{BAND} px")
    q = float(row["q"])
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    test = R.load(net, float(ck["threshold"]), "test")
    i = test["names"].index(DEVICE)
    Y, B, Z, X = test["Y"][i], test["B"][i], test["Z"][i], test["X"][i]
    H, W = Y.shape

    # exactly the study's step 2
    centre = skeletonize(B)
    measured = R.dilate(centre, BAND) | (X[1] > 0.5)
    checked = R.dilate(centre, BAND - 2) & R.inside(measured)
    kept, cov = R.two_step(test, i, q, BAND)
    Rmap = C.step_residual_map(Z, 5, True, measured)
    thr = float(np.quantile(Rmap[checked], q))
    assert np.array_equal(kept, checked & (Rmap > thr))
    # the strip is a dilation with the 4-neighbour cross, i.e. every pixel
    # within 5 grid steps (up/down/left/right) of the centre line
    steps_away = distance_transform_cdt(~centre, metric="taxicab")
    assert np.array_equal(R.dilate(centre, BAND), steps_away <= BAND)

    # the raster scan run_19 matched to this band: k x k points, upsampled
    # linearly to the full grid, the same step value over the whole plane
    ras = next(r for r in rows if r["method"] == "coarse scan"
               and r["matched"] == row["setting"])
    k_r = int(ras["setting"].split(" x ")[0])
    assert "upsampled" in ras["setting"]
    q_r = float(ras["q"])
    Zr, _ = coarse_image(Z, k_r * k_r, 1)
    R_r = C.step_residual_map(Zr, 5, True)
    thr_r = float(np.quantile(R_r, q_r))
    found_r = R_r > thr_r
    assert np.array_equal(found_r, R.coarse_scan(test, i, q_r, k_r,
                                                 "upsampled")[0])
    pts = np.zeros((H, W), bool)
    pts[np.ix_(np.rint(np.linspace(0, H - 1, k_r)).astype(int),
               np.rint(np.linspace(0, W - 1, k_r)).astype(int))] = True
    lay_t, s_t = scored(kept, Y)
    lay_r, s_r = scored(found_r, Y)

    # the 1-D cut along V2 (rows) at one column, recomputed by hand from the
    # same formula: d = Z[r+1] - Z[r], b = running median of d over 5,
    # step value = |d - b| / (|b| + typical |d| of the measured pixels)
    Z64 = np.asarray(Z, dtype=float)          # as the study computes it
    ref = Z64[measured]
    Zs = (Z64 - ref.min()) / np.ptp(ref)
    d = np.diff(Zs, axis=0)
    b = median_filter(d, size=(5, 1), mode="nearest")
    both = measured[1:] & measured[:-1]
    scale = np.median(np.abs(d[both]))
    r_cut = (np.abs(d - b) / (np.abs(b) + scale))[:, CUT_COL]
    # the other direction (V1, along columns), and the check that the hand
    # computation is the study's step value: the larger of the two
    d1 = np.diff(Zs, axis=1)
    b1 = median_filter(d1, size=(1, 5), mode="nearest")
    scale1 = np.median(np.abs(d1[measured[:, 1:] & measured[:, :-1]]))
    r1 = np.pad(np.abs(d1 - b1) / (np.abs(b1) + scale1), [(0, 0), (0, 1)])
    r0 = np.pad(np.abs(d - b) / (np.abs(b) + scale), [(0, 1), (0, 0)])
    assert np.allclose(np.maximum(r0, r1), Rmap, rtol=1e-9, atol=1e-12)

    # worked example on the cut: A the strongest step, B the pixel after it
    # (its window still holds the jump), C a background pixel (the checked
    # pixel of median step value among those not kept)
    on_cut = np.arange(CUT_ROWS.start, CUT_ROWS.stop)
    ok = on_cut[checked[on_cut, CUT_COL]]
    k_t = int(ok[np.argmax(r_cut[ok])])
    calm = ok[r_cut[ok] <= thr]
    k_b = int(calm[np.argsort(r_cut[calm])[len(calm) // 2]])

    def example(kk):
        return {"k": kk, "Zk": Zs[kk, CUT_COL], "Zk1": Zs[kk + 1, CUT_COL],
                "d": d[kk, CUT_COL], "win": d[kk - 2:kk + 3, CUT_COL],
                "b": b[kk, CUT_COL], "r0": r0[kk, CUT_COL],
                "r1": r1[kk, CUT_COL], "R": Rmap[kk, CUT_COL],
                "kept": bool(kept[kk, CUT_COL]),
                "true": [int(t) for t in (kk, kk + 1) if Y[t, CUT_COL]]}
    k_n = k_t + 1
    assert checked[k_n, CUT_COL]
    ex_t, ex_n, ex_b = example(k_t), example(k_n), example(k_b)
    # what the text says about them
    assert ex_t["kept"] and not ex_n["kept"] and not ex_b["kept"]
    assert d[k_t, CUT_COL] in ex_n["win"]           # B's window holds A's jump
    w_c = ex_b["win"]
    assert ex_b["r0"] == 0 and (np.all(np.diff(w_c) > 0)
                                or np.all(np.diff(w_c) < 0))
    # the same check with a mean instead of the median, for the text
    for e in (ex_t, ex_n):
        e["bm"] = float(np.mean(e["win"]))
        e["rm"] = abs(e["d"] - e["bm"]) / (abs(e["bm"]) + scale)
    assert ex_n["rm"] > thr and ex_t["true"] == [k_t] and ex_t["r1"] == 0
    # kept pixels on the cut that are not on a true line
    extra = [(int(kk), float(Rmap[kk, CUT_COL])) for kk in on_cut
             if kept[kk, CUT_COL] and not Y[kk, CUT_COL]]
    print("kept off-line on the cut", extra)
    assert np.isclose(np.median(ex_t["win"]), ex_t["b"])

    # -- figure 1: the stages on the whole device ---------------------------
    def rgb(*layers):
        img = np.ones((H, W, 3))
        for m, col in layers:
            img[m] = to_rgb(col)
        return img
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.6),
                           gridspec_kw=dict(wspace=0.08, hspace=0.16,
                                            left=0.005, right=0.995,
                                            top=0.96, bottom=0.135))
    ax = ax.ravel()
    ax[0].imshow(rgb((B, LINE_C), (centre, CENTRE_C)), origin="lower",
                 interpolation="nearest")
    r0, c0 = ZOOM
    ax[0].add_patch(Rectangle((c0.start - 0.5, r0.start - 0.5),
                              c0.stop - c0.start, r0.stop - r0.start,
                              fill=False, ec=KEPT_C, lw=0.9, zorder=4))
    ax[0].set_title("(a) U-Net lines and their centre line")
    ax[1].imshow(rgb((measured, STRIP_C), (checked, CHECK_C),
                     (centre, CENTRE_C)), origin="lower",
                 interpolation="nearest")
    ax[1].set_title(f"(b) measured ±{BAND} px; checked ±{BAND - 2} px")
    show = np.where(measured, Z, np.nan)
    ax[2].imshow(show, origin="lower", cmap="viridis",
                 interpolation="nearest")
    ax[2].plot([CUT_COL, CUT_COL], [CUT_ROWS.start, CUT_ROWS.stop - 1],
               color="#ff3b3b", lw=1.2)
    ax[2].set_title("(c) the signal measured in step 2")
    rv = np.where(checked, Rmap, np.nan)
    im = ax[3].imshow(rv, origin="lower", cmap="magma",
                      vmax=np.nanquantile(rv, 0.99), interpolation="nearest")
    ax[3].set_title("(d) step value in the checked pixels")
    fig.colorbar(im, ax=ax[3], fraction=0.045, pad=0.01).ax.tick_params(
        labelsize=6)
    dropped = centre & ~kept & ~R.dilate(kept, 1)
    ax[4].imshow(rgb((checked, "#f4f4f4"), (Y, TRUE_C), (dropped, DROP_C),
                     (kept, KEPT_C)), origin="lower",
                 interpolation="nearest")
    ax[4].set_title("(e) kept pixels vs true lines")
    ax[5].imshow(rgb(*lay_t), origin="lower", interpolation="nearest")
    ax[5].set_title(f"(f) final lines, scored: F1@1 = {s_t['f1']:.3f}")
    for a in ax:
        grid(a, H, W)
        a.set_xticks([])
        a.set_yticks([])
    fig.legend(handles=[
        Patch(color=LINE_C, label="U-Net line (a)"),
        Patch(color=CENTRE_C, label="U-Net centre line (a, b)"),
        Patch(color=STRIP_C, label=f"measured strip, ±{BAND} px (b)"),
        Patch(color=CHECK_C, label=f"checked region, ±{BAND - 2} px (b)"),
        Patch(color=KEPT_C, label="kept: a step was measured (e)"),
        Patch(color=DROP_C, label="U-Net line dropped: no step (e)"),
        Patch(color=TRUE_C, label="true line (e)")],
        loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2,
        frameon=False, fontsize=7, handlelength=1.4, columnspacing=1.2)
    fig.legend(handles=[Patch(color=c, label=f"{l} (f)")
                        for c, l in SCORE_KEY],
               loc="lower right", bbox_to_anchor=(0.995, 0.0), ncol=1,
               frameon=False, fontsize=7, handlelength=1.4)
    fig.savefig(os.path.join(BUILD, "stages.png"), dpi=300)
    plt.close(fig)

    # -- the centre line, magnified ------------------------------------------
    r0, c0 = ZOOM
    ext = (c0.start - 0.5, c0.stop - 0.5, r0.start - 0.5, r0.stop - 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0),
                           gridspec_kw=dict(wspace=0.06, left=0.01,
                                            right=0.99, top=0.9,
                                            bottom=0.02))
    ax[0].imshow(rgb((B, LINE_C), (centre, CENTRE_C))[r0, c0],
                 origin="lower", interpolation="nearest", extent=ext)
    ax[0].set_title("(a) U-Net line pixels (light) and centre line (dark)")
    ax[1].imshow(rgb((measured & ~(X[1] > 0.5), STRIP_C), (checked, CHECK_C),
                     (centre, CENTRE_C))[r0, c0], origin="lower",
                 interpolation="nearest", extent=ext)
    for rr_ in range(r0.start, r0.stop):
        for cc_ in range(c0.start, c0.stop):
            s = steps_away[rr_, cc_]
            if 0 < s <= BAND:
                ax[1].text(cc_, rr_, str(s), ha="center", va="center",
                           fontsize=4.5, color="#333333", zorder=5)
    ax[1].set_title("(b) grid steps from the centre line")
    for a in ax:
        for kk in range(c0.start, c0.stop + 1):
            a.axvline(kk - 0.5, color="#c8c8c8", lw=0.3, zorder=3)
        for kk in range(r0.start, r0.stop + 1):
            a.axhline(kk - 0.5, color="#c8c8c8", lw=0.3, zorder=3)
        a.set_xticks([])
        a.set_yticks([])
    fig.savefig(os.path.join(BUILD, "centre_zoom.png"), dpi=300)
    plt.close(fig)

    # -- the raster scan at the same cost, in the same panels ----------------
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.4),
                           gridspec_kw=dict(wspace=0.08, hspace=0.16,
                                            left=0.005, right=0.965,
                                            top=0.96, bottom=0.09))
    ax = ax.ravel()
    ax[0].imshow(rgb((pts, "#111111")), origin="lower",
                 interpolation="nearest")
    ax[0].set_title(f"(a) raster {k_r} × {k_r}: {100 * pts.mean():.1f} % "
                    "measured")
    ax[1].imshow(Zr, origin="lower", cmap="viridis", interpolation="nearest")
    ax[1].set_title("(b) the raster, interpolated to the grid")
    im = ax[2].imshow(R_r, origin="lower", cmap="magma",
                      vmax=np.quantile(R_r, 0.99), interpolation="nearest")
    ax[2].set_title("(c) step value in every pixel")
    fig.colorbar(im, ax=ax[2], fraction=0.045, pad=0.01).ax.tick_params(
        labelsize=6)
    ax[3].imshow(rgb(*lay_r), origin="lower", interpolation="nearest")
    ax[3].set_title(f"(d) raster lines: F1@1 = {s_r['f1']:.3f}")
    ax[4].imshow(rgb(*lay_t), origin="lower", interpolation="nearest")
    ax[4].set_title(f"(e) two steps: F1@1 = {s_t['f1']:.3f}")
    ax[5].imshow(rgb((measured, "#111111")), origin="lower",
                 interpolation="nearest")
    ax[5].set_title(f"(f) two steps: {100 * measured.mean():.1f} % "
                    "measured")
    for a in ax:
        grid(a, H, W)
        a.set_xticks([])
        a.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=l) for c, l in SCORE_KEY],
               loc="lower center", ncol=2, frameon=False, fontsize=7,
               handlelength=1.4)
    fig.savefig(os.path.join(BUILD, "raster.png"), dpi=300)
    plt.close(fig)

    # -- figure 2: the 1-D cut ---------------------------------------------
    rr = np.arange(CUT_ROWS.start, CUT_ROWS.stop)
    meas = measured[rr, CUT_COL]
    chk = checked[rr, CUT_COL]
    true_rows = rr[Y[rr, CUT_COL]]
    fig, ax = plt.subplots(3, 1, figsize=(6.0, 5.0), sharex=True,
                           gridspec_kw=dict(hspace=0.12, left=0.12,
                                            right=0.98, top=0.96,
                                            bottom=0.09))
    for a in ax:
        for t in true_rows:
            a.axvspan(t - 0.5, t + 0.5, color="#dddddd", zorder=0)
    ax[0].plot(rr[meas], Zs[rr[meas], CUT_COL], "o-", ms=3, color="#333333",
               lw=0.8)
    ax[0].set_ylabel("signal $Z_k$\n(scaled)")
    ax[0].text(0.01, 0.92, "grey bands: true transition lines",
               transform=ax[0].transAxes, fontsize=7, color="#555555",
               va="top")
    # d_k is the jump from pixel k to k+1: drawn between the two pixels
    pair = meas[:-1] & meas[1:]
    xd = rr[:-1][pair] + 0.5
    ax[1].plot(xd, 1e3 * d[rr[:-1][pair], CUT_COL], "o", ms=3,
               color="#333333", label="$d_k$, jump to the next pixel")
    ax[1].plot(xd, 1e3 * b[rr[:-1][pair], CUT_COL], "-", color="#2a6ab0",
               lw=1.2, label="$b_k$, median of 5 jumps")
    for sg in (1, -1):
        ax[1].axhline(sg * 1e3 * scale, color="#2a6ab0", ls=":", lw=0.9)
    x_end = rr[meas][-1]
    ax[1].text(x_end, 1e3 * scale, "$+s$", color="#2a6ab0",
               fontsize=7, va="bottom", ha="right")
    ax[1].text(x_end, -1e3 * scale, "$-s$", color="#2a6ab0",
               fontsize=7, va="top", ha="right")
    ax[1].set_ylabel("jump ($10^{-3}$)")
    ax[1].legend(fontsize=7, frameon=False, loc="upper right",
                 bbox_to_anchor=(1.0, 0.92))
    rc = r_cut[rr]
    k = chk & meas
    ax[2].bar(rr[k], rc[k], width=0.8, color=np.where(rc[k] > thr, KEPT_C,
                                                         "#8c8c8c"))
    ax[2].axhline(thr, color=KEPT_C, ls="--", lw=0.9)
    ax[2].text((k_t + k_t + 9) / 2, thr + 0.04, "threshold", color=KEPT_C,
               fontsize=7, va="bottom", ha="center")
    for kk, lab in ((k_t, "A"), (k_n, "B"), (k_b, "C")):
        ax[2].text(kk, r_cut[kk] + 0.08, lab, ha="center", va="bottom",
                   fontsize=7, fontweight="bold")
    ax[2].set_ylabel("step value $r_k$\n(V$_2$ direction)")
    ax[2].set_xlabel("pixel row $k$ along the cut (V$_2$ increases →)")
    fig.savefig(os.path.join(BUILD, "cut.png"), dpi=300)
    plt.close(fig)

    v = {"q": q, "keep_pct": 100 * (1 - q), "thr": thr,
         "measured": float(row["measured"]), "p": float(row["precision"]),
         "r": float(row["recall"]), "f1": float(row["f1"]),
         "u": next(r for r in rows if r["method"] == "U-Net only"),
         "full": next(r for r in rows if r["method"] == "full scan"),
         "ras": ras, "k": k_r, "q_r": q_r, "s_t": s_t, "s_r": s_r,
         "dev_meas": 100 * measured.mean(), "dev_meas_r": 100 * pts.mean(),
         "s0": scale, "s1": scale1}
    print("examples", ex_t, ex_n, ex_b, sep="\n")

    def m3(x):                       # in units of 1e-3 of the signal range
        return f"{1e3 * x:.3f}"

    def ex_col(e):
        win = [m3(x) for x in e["win"]]
        win[2] = r"\textbf{" + win[2] + "}"          # d_k itself
        return [str(e["k"]), m3(e["Zk"]), m3(e["Zk1"]), m3(e["d"]),
                *win, m3(e["b"]),
                m3(e["d"] - e["b"]), m3(abs(e["b"]) + scale),
                f"{e['r0']:.3f}", f"{e['r1']:.3f}", f"{e['R']:.3f}",
                "yes" if e["kept"] else "no"]
    names = [r"row $k$", r"$Z_k$", r"$Z_{k+1}$", r"$d_k = Z_{k+1}-Z_k$",
             r"\quad $d_{k-2}$", r"\quad $d_{k-1}$", r"\quad $d_k$",
             r"\quad $d_{k+1}$", r"\quad $d_{k+2}$",
             r"$b_k$ (median of the five)",
             r"$d_k - b_k$", r"$|b_k| + s$",
             r"$r_k$ along $V_2$", r"$r_k$ along $V_1$",
             r"step value (larger)",
             r"kept (threshold " + f"{thr:.3f}" + ")"]
    ex_rows = "\n".join(f"{n} & {a} & {c} & {e} \\\\" for n, a, c, e in
                        zip(names, ex_col(ex_t), ex_col(ex_n),
                            ex_col(ex_b)))
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}\usepackage{graphicx}\usepackage{parskip}
\usepackage{enumitem}\setlist{itemsep=3pt}\usepackage{float}
\usepackage{caption}\captionsetup{font=small,labelfont=bf}
\usepackage{booktabs}\usepackage{amsmath}
\begin{document}
\begin{center}{\Large Step 2, the local check: how it works}\\[3pt]
\small one test device (sample\_223), 8 rays $\times$ 60 points, strip of
$\pm$""" + str(BAND) + r""" pixels\end{center}

\section*{The idea in one sentence}
The U-Net says where the lines probably are; we measure only near those
places, and keep a line only where the measured signal really jumps.

\section*{Definition: the centre line}
The \emph{centre line} of a predicted line is its skeleton: the set of
pixels left after the outer pixels of the line are peeled off, one layer
at a time and from both sides alike, until the line is one pixel wide.
A pixel is never removed if removing it would cut the line into two pieces
or shorten it at an end, so the centre line keeps the shape, the length and
the junctions of the predicted line and runs along its middle (Fig.~\ref{fig:zoom}a).
The thinning is the Zhang--Suen algorithm (scikit-image
\texttt{skeletonize}). Distances to the centre line are counted in
\emph{grid steps}, up, down, left or right, one pixel per step (Fig.~\ref{fig:zoom}b).

\begin{figure}[H]\centering
\includegraphics[width=0.85\textwidth]{centre_zoom.png}
\caption{The red box of Fig.~\ref{fig:stages}a, magnified. (a) The pixels the U-Net
marks as line (light blue) and their centre line (dark blue). (b) The
number of grid steps from the centre line; the strip measured in step~2
is every pixel up to """ + str(BAND) + r""" steps away (light blue), the
checked region up to """ + str(BAND - 2) + r""" steps (darker blue).
Ray pixels, measured in step~1, are not shaded here.}
\label{fig:zoom}\end{figure}

\section*{The steps}
\begin{enumerate}
\item \textbf{Thin the U-Net lines to their centre line.} The U-Net's
  predicted lines are a few pixels thick (Fig.~\ref{fig:stages}a, light blue). They are
  thinned to a centre line one pixel wide (Fig.~\ref{fig:stages}a and Fig.~\ref{fig:zoom}a, dark blue).
\item \textbf{Measure a strip around them.} Every pixel up to
  """ + str(BAND) + r""" grid steps from the centre line is measured
  (Fig.~\ref{fig:stages}b, light blue; Fig.~\ref{fig:zoom}b gives the number of steps). This is the
  extra measurement of
  step~2: here """ + f"{v['measured']:.1f}" + r"""\,\% of the plane,
  rays included.
\item \textbf{Check only the inner part of the strip.} The check needs two
  measured neighbours on each side of a pixel, so it is applied to the
  pixels up to """ + str(BAND - 2) + r""" grid steps from the centre line
  (Fig.~\ref{fig:stages}b, darker blue).
\item \textbf{Compute a ``step value'' for every checked pixel}
  (Fig.~\ref{fig:stages}d). Along each gate direction:
  \begin{itemize}
  \item take the difference between the signal at this pixel and at the
    next one;
  \item compare it with the \emph{typical} difference around it, the median
    of the differences over 5 pixels: this follows the smooth slope of the
    sensor signal;
  \item the step value is how much the difference departs from that
    typical difference, divided by the typical size of a difference.
  \end{itemize}
  A smooth background, even a steep one, gives a small step value; a charge
  transition is a sudden jump and gives a large one. The larger of the two
  directions is used. The formula, the reason for each of its parts and a
  worked example are in the section \emph{The local edge detector in
  detail} below.
\item \textbf{Keep the pixels with the strongest steps.} A pixel is kept as
  a line pixel if its step value is among the strongest
  """ + f"{v['keep_pct']:.0f}" + r"""\,\% of the checked pixels (the
  threshold is the """ + f"{100 * v['q']:.0f}" + r"""th percentile). This
  percentage was chosen on the 75 validation devices, never on the test
  devices.
\end{enumerate}
Nothing is fitted: the result is the set of kept pixels (Fig.~\ref{fig:stages}e, red).
Where the U-Net drew a line but the signal shows no jump, nothing is kept,
and that predicted line is dropped (orange).

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{stages.png}
\caption{The stages on one test device; grey lines mark the pixel grid.
(a) U-Net lines (light blue) and their one-pixel centre line (dark blue).
(b) The measured strip and the checked region. (c) The signal measured in
step 2; the red segment is the cut of Fig.~\ref{fig:cut}. (d) The step value of every
checked pixel. (e) Kept pixels (red), predicted lines dropped for lack of a
step (orange), and the true lines (grey). (f) The final lines scored
against the true lines at the 1-pixel tolerance of F1@1: red, a found
pixel with a true line within 1 pixel; black, a found pixel with none;
teal, a true-line pixel with no found pixel within 1 pixel; grey, a
true-line pixel that was found. On this device precision
""" + f"{v['s_t']['precision']:.3f}" + r""", recall
""" + f"{v['s_t']['recall']:.3f}" + r""".}
\label{fig:stages}\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.85\textwidth]{cut.png}
\caption{One cut across the plane (the red segment of
Fig.~\ref{fig:stages}c), along $V_2$. Top: the measured signal $Z_k$; it
varies smoothly and jumps at each transition (grey bands). Middle: the
jump to the next pixel, $d_k$ (dots, drawn between pixels $k$ and $k+1$),
its running median over 5 jumps, $b_k$ (line), and $\pm s$, the typical
size of a jump on this device (dotted). Bottom: the step value $r_k$ of
every checked pixel on the cut; red bars are above the threshold and are
kept. A, B, C: the pixels of Table~\ref{tab:example}.}
\label{fig:cut}\end{figure}

\section*{The local edge detector in detail}
The step value is computed for every pixel from the signal of a few
neighbours along each gate direction. Nothing is fitted and no model of the
device is used. Figure~\ref{fig:cut} shows each quantity on one cut, and
Table~\ref{tab:example} gives the numbers at three pixels of that cut.

\paragraph{The formula.} Along one direction ($V_2$; $V_1$ is the same with
columns instead of rows), let $Z_k$ be the signal at pixel $k$.
\begin{enumerate}
\item The jump to the next pixel: $d_k = Z_{k+1} - Z_k$.
\item The local background slope: $b_k = \operatorname{median}(d_{k-2},
  d_{k-1}, d_k, d_{k+1}, d_{k+2})$.
\item The typical jump: $s$, the median of $|d|$ over every pair of
  neighbouring measured pixels of the device. It is one number per device
  and direction; here $s = """ + f"{1e3 * v['s0']:.3f}" + r"""\times10^{-3}$
  along $V_2$ and $""" + f"{1e3 * v['s1']:.3f}" + r"""\times10^{-3}$ along
  $V_1$ (signal scaled to the range 0 to 1).
\end{enumerate}
\begin{equation*}
  r_k = \frac{|d_k - b_k|}{|b_k| + s}.
\end{equation*}
The step value of a pixel is the larger of its $r_k$ along $V_2$ and along
$V_1$, and the pixel is kept if that value is above the threshold.

\paragraph{The numerator, $|d_k - b_k|$: remove the background.} The
sensor signal falls smoothly across the plane (Fig.~\ref{fig:cut}, top), so
its jumps $d_k$ change slowly from pixel to pixel and $b_k$ follows them
(middle). At a transition $d_k$ is one outlier among the five, and the
median ignores it: at pixel A, $d_k$ is large and $b_k$ stays on the
background, so $r_k$ is large. One pixel later, at B, the same jump is
still inside the window, but the median still ignores it and $r_k$ stays
small, so this step is marked at one pixel only.""" + ("" if not extra else
    " This is not always so: on the same cut, row" + ("s " if len(extra) > 1
    else " ") + ", ".join(str(kk) for kk, _ in extra) + ", next to a true "
    "line, " + ("are" if len(extra) > 1 else "is") + " kept as well, with "
    "step value " + ", ".join(f"{x:.3f}" for _, x in extra) + ", just "
    "above the threshold (Fig.~\\ref{fig:cut}, bottom).") + r""" A median of five ignores
up to two outliers. A mean would not: with a mean of the same five jumps,
B would get """ + f"{ex_n['rm']:.3f}" + r""" instead of
""" + f"{ex_n['r0']:.3f}" + r""", far above the threshold
(""" + f"{thr:.3f}" + r"""), and the step would be marked at B as well as
at A. Where the jumps rise or fall
steadily over the five pixels, as on a smooth bend of the background,
$d_k$ is itself the median and $r_k$ is exactly zero (pixel C).

\paragraph{$|b_k|$ in the denominator: a relative step.} Where the
background itself is steep, $|b_k|$ is large, and a jump must be large
compared with the local slope to count. The same jump therefore gives a
smaller step value on a steep background than on a flat one.

\paragraph{$s$ in the denominator: no division by zero.} Where the
background is flat, $b_k \approx 0$, and without $s$ any small difference
$d_k - b_k$ would give a very large $r_k$. With $s$, the step value is
measured in units of a typical jump of this device. The simulations here
are noise-free, so $s$ is the typical jump of the smooth background, not a
noise level. With measurement noise, the jump between neighbouring pixels
would be dominated by noise, $s$ would come close to the noise level, and
$r_k$ would behave like a signal-to-noise ratio; that case is not tested
here.

\paragraph{No unit and no offset.} Adding a constant to the signal does not
change any $d_k$. Multiplying the signal by a constant multiplies $d_k$,
$b_k$ and $s$ alike, and the ratio $r_k$ is unchanged. The step value
therefore does not depend on the sensor's units or offset; the scaling to 0
to 1 only makes the numbers readable.

\paragraph{Two directions.} A transition line that crosses the cut gives a
jump along $V_2$; a line almost parallel to $V_2$ gives a jump only along
$V_1$. Taking the larger of the two finds lines of any orientation. At
pixel A the $V_1$ value is """ + f"{ex_t['r1']:.3f}" + r""": this line is
seen along $V_2$ alone.

\paragraph{Which pixels it uses.} $b_k$ needs $d_{k-2}, \dots, d_{k+2}$,
that is, the signal at pixels $k-2$ to $k+3$: two pixels before and three
after, along each direction. This is why only the pixels up to
""" + str(BAND - 2) + r""" grid steps from the centre line are checked
inside the """ + str(BAND) + r"""-step strip: a checked pixel never uses an
unmeasured one.

\paragraph{Where the step is marked.} $r_k$ describes the jump between
pixels $k$ and $k+1$ and is given to pixel $k$. The true line is drawn at
one of these two pixels (at A, at pixel $k$), and the 1-pixel tolerance of
F1@1 covers either case.

\paragraph{The threshold.} Pixels are kept if their step value is among
the strongest """ + f"{v['keep_pct']:.0f}" + r"""\,\% of the checked pixels
of the device; this fraction is the one number chosen on the validation
devices. The threshold value itself therefore differs from device to
device (""" + f"{thr:.3f}" + r""" here).

\begin{table}[H]\centering\small
\caption{The step value at the three pixels marked on
Fig.~\ref{fig:cut}: A, the strongest step on the cut; B, the next pixel,
whose five jumps still contain A's jump; C, a background pixel (the median
step value among the checked pixels not kept). Signal and jumps in units
of $10^{-3}$ of the signal range; in the five jumps, $d_k$ itself is in
bold.}
\label{tab:example}
\begin{tabular}{lrrr}
\toprule
 & A & B & C \\
\midrule
""" + ex_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\paragraph{What the detector cannot do.} Three or more jumps within five
neighbouring pixels, as where several lines cross, move the median away
from the background. A step spread over several pixels gives several
smaller jumps and a smaller $r_k$ at each; on the cut of
Fig.~\ref{fig:cut} every transition is a jump between two neighbouring
pixels. A steep background lowers the step value of the transitions on
it.

\section*{Why this improves the result}
The U-Net alone marks some pixels as line where there is none (precision
""" + f"{float(v['u']['precision']):.3f}" + r"""). The local check removes
them, because the measured signal does not jump there: precision rises to
""" + f"{v['p']:.3f}" + r""" and F1@1 from
""" + f"{float(v['u']['f1']):.3f}" + r""" to """ + f"{v['f1']:.3f}" + r"""
(50 test devices, Table~III of the manuscript).

\section*{The raster scan at the same cost}
The same number of measurements can be spent on a regular raster instead:
""" + f"{v['k']} $\\times$ {v['k']}" + r""" points,
""" + f"{float(v['ras']['measured']):.1f}" + r"""\,\% of the plane, the
smallest square raster with at least as many points as step~1 plus step~2
use on average (""" + f"{v['measured']:.1f}" + r"""\,\%). The raster is
interpolated linearly to the full grid, and the same step value is computed
in every pixel; the strongest
""" + f"{100 * (1 - v['q_r']):.0f}" + r"""\,\% are kept (the threshold,
and the choice of interpolating rather than working on the raster's own
grid, were chosen on the validation devices). Fig.~\ref{fig:raster} shows it on the same
device. The raster misses parts of the shallow lines (teal), and marks
short false lines in the upper right (black), where the signal rises
steeply: there the kinks of the linear interpolation at the raster points
give a step value of their own (the dot pattern in
Fig.~\ref{fig:raster}c).

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{raster.png}
\caption{The raster scan at the cost of step~2, on the same device.
(a) The measured raster points. (b) The raster, interpolated to the full
grid. (c) The step value of every pixel. (d) The raster's lines scored
against the true lines (colours as in Fig.~\ref{fig:stages}f): precision
""" + f"{v['s_r']['precision']:.3f}" + r""", recall
""" + f"{v['s_r']['recall']:.3f}" + r""". (e) The two-step result of
Fig.~\ref{fig:stages}f, for comparison. (f) The pixels measured by the two steps
(""" + f"{v['dev_meas']:.1f}" + r"""\,\% on this device; raster
""" + f"{v['dev_meas_r']:.1f}" + r"""\,\%).}
\label{fig:raster}\end{figure}

Over the 50 test devices the raster reaches F1@1
""" + f"{float(v['ras']['f1']):.3f}" + r""" (precision
""" + f"{float(v['ras']['precision']):.3f}" + r""", recall
""" + f"{float(v['ras']['recall']):.3f}" + r"""), against
""" + f"{v['f1']:.3f}" + r""" for the two steps, at about the same cost.
The full scan, every pixel with the same step value, reaches
""" + f"{float(v['full']['f1']):.3f}" + r""".

\section*{What it cannot do}
\begin{itemize}
\item It only looks inside the strip, so a line that the U-Net missed
  entirely is not found.
\item The strip is measured pixel by pixel here; in an experiment, short
  sweeps across each predicted line would cost less.
\item The signal is noise-free in these simulations; with noise, small
  jumps would be harder to tell from the background.
\end{itemize}
\end{document}
"""
    with open(os.path.join(BUILD, "Step2_explained.tex"), "w",
              encoding="utf-8") as f:
        f.write(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "Step2_explained.tex"], cwd=BUILD,
                       capture_output=True)
    os.replace(os.path.join(BUILD, "Step2_explained.pdf"),
               os.path.join(OUT, "Step2_explained.pdf"))
    print("wrote", os.path.join(OUT, "Step2_explained.pdf"), "q", q,
          "thr", round(thr, 3))


if __name__ == "__main__":
    main()
