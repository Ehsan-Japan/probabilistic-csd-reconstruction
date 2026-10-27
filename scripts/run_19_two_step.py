"""
Two steps: the U-Net locates the lines, then the device is measured only
around them and the lines are found there.

    python scripts/run_19_two_step.py

Step 1  the U-Net (8 rays x 60 points) predicts the transition lines.
Step 2  local sweeps: measure every pixel within BAND grid steps of the
        binarized U-Net output, and test every pixel of that strip: it is
        a transition if the measured signal steps there, by the 3-pixel
        step detector (step3_map: pixels k-1, k, k+1), taken along each
        gate direction in which both neighbours were measured, so that the
        strip can be as narrow as +-2 and every swept pixel is tested.
        History (2026-10-25): the strip was first taken around the one-pixel
        centre lines of the output (two_step/_centre_line_version/,
        run_20_centre_line.py), then around the output with the 6-pixel
        detector of charge_states.step_residual_map and +-3 ... +-6
        (two_step/_band5_6pixel_version/, two_step6 below).

Compared, on the 50 test devices, with
        the U-Net alone, the full scan (every pixel, the 6-pixel step
        detector, which is the better one on a full image), and a coarse
        raster scan (k x k points, the 6-pixel step detector) at the cost
        of each U-Net setting: k is the smallest side with k^2 at least the
        U-Net setting's mean number of measured pixels, so the raster never
        gets fewer measurements than the method it is compared with.

Only measured pixels enter step 2: a direction counts for a pixel only if
its neighbours k-1 and k+1 along it were measured.  (Before the evening of
2026-10-25, only pixels within BAND-1 steps with both directions measured
were tested: two_step_inner, two_step/_band2_inner_check_version/.)

Score: line accuracy F1@1, and the fraction of the plane measured.  The
detector's one threshold is chosen on the 75 validation devices by F1@1,
separately for every method and band; the test devices only report.

Writes results/<run>/two_step/: table.csv, table.md, fig_example.png
(fig_f1_vs_measured.png: run_23_equal_seeds.py).
"""
import csv
import os

import numpy as np
from scipy import ndimage

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from csdrecon.ml import grid_train
from csdrecon.ml.grid_metrics import tolerant_f1
from csdrecon.ml.ray_peaks import load_grid
from csdrecon.study import charge_states as C
from run_7_virtual_gates import model_dir, split_arrays
from run_16_charge_states import local
from run_18_equal_cost import coarse_grid, coarse_image, native_line_map

# ══ settings ══════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
CELL = "8_rays_60_points_500_samples"
BANDS = (0, 1, 2, 3)   # half-width of the swept band, px; 0 sweeps only the
#                        pixels of the binarized U-Net output (2-5: see
#                        two_step/_band2to5_version/)
QS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98)
#      detector thresholds scanned (quantiles); lines cover ~1-17% of pixels
EXAMPLE = "sample_223"
FIG_BAND = 2
# ══════════════════════════════════════════════════════════════════════════

RUN_DIR = os.path.join(ROOT, "results", RUN)
OUT = os.path.join(RUN_DIR, "two_step")


def load(net, thr, split):
    X, Y, s = split_arrays(os.path.join(RUN_DIR, CELL), split)
    d = [local(x) for x in s]
    return {"X": X, "Y": Y > 0.5, "B": grid_train.predict(net, X) > thr,
            "Z": [load_grid(x)[2] for x in d],
            "names": [os.path.basename(x) for x in d]}


def dilate(m, k):
    # iterations=0 would mean "until nothing changes" in scipy
    return ndimage.binary_dilation(m, iterations=k) if k > 0 else m.copy()


def steps(Z, where, q, measured=None):
    """Pixels of `where` at which the signal steps: the step residual above
    its q-quantile over `where`.  With `measured`, the residual uses only
    the measured pixels (see charge_states.step_residual_map)."""
    R = C.step_residual_map(Z, 5, True, measured)
    v = R[where]
    return where & (R > np.quantile(v, q)) if v.size else where


def inside(measured):
    """Pixels whose step value uses measured pixels only: the difference to
    the next pixel and the 5-wide running median of differences use pixels
    k-2 ... k+3 along each axis.  Beyond the frame there is no pixel, and
    nothing is used."""
    ok = np.ones_like(measured)
    for ax in (0, 1):
        n = measured.shape[ax]
        for s in range(-2, 4):
            sh = np.ones_like(measured)
            src, dst = [slice(None)] * 2, [slice(None)] * 2
            if s >= 0:
                src[ax], dst[ax] = slice(s, n), slice(0, n - s)
            else:
                src[ax], dst[ax] = slice(0, n + s), slice(-s, n)
            sh[tuple(dst)] = measured[tuple(src)]      # sh[k] = measured[k+s]
            ok &= sh
    return ok


def _neighbour_measured(measured, ax, sft):
    """sh[k] = measured[k + sft] along axis ax; beyond the frame there is no
    pixel and nothing is used, so it counts as measured."""
    n = measured.shape[ax]
    sh = np.ones_like(measured)
    src, dst = [slice(None)] * 2, [slice(None)] * 2
    if sft > 0:
        src[ax], dst[ax] = slice(sft, n), slice(0, n - sft)
    else:
        src[ax], dst[ax] = slice(0, n + sft), slice(-sft, n)
    sh[tuple(dst)] = measured[tuple(src)]
    return sh


def step3_map(Z, measured=None, per_direction=False):
    """The 3-pixel step value: along each axis, the change to the next
    pixel d_k = Z[k+1] - Z[k] compared with the change just before it,
    r_k = |d_k - d_{k-1}| / (|d_{k-1}| + s), s the median |d| over the
    measured pixel pairs; the larger of the two axes.  Uses pixels k-1, k,
    k+1.  As step_residual_map, the signal is scaled over the measured
    pixels first (which does not change r).

    With per_direction, a direction counts for a pixel only if its pixels
    k-1 and k+1 along it were measured; returns (step value, mask of the
    pixels with at least one such direction)."""
    Z = np.asarray(Z, dtype=float)
    ref = Z[measured] if measured is not None and measured.any() else Z
    Z = (Z - ref.min()) / (np.ptp(ref) or 1.0)
    out = np.zeros(Z.shape)
    usable = np.zeros(Z.shape, bool)
    for ax in (0, 1):
        d = np.diff(Z, axis=ax)                       # d[k] = Z[k+1] - Z[k]
        if measured is not None:
            both = ((measured[1:] & measured[:-1]) if ax == 0 else
                    (measured[:, 1:] & measured[:, :-1]))
            s = (np.median(np.abs(d[both])) if both.any() else
                 np.median(np.abs(d)))
        else:
            s = np.median(np.abs(d))
        prev = np.empty_like(d)                       # prev[k] = d[k-1]
        if ax == 0:
            prev[1:], prev[0] = d[:-1], d[0]
        else:
            prev[:, 1:], prev[:, 0] = d[:, :-1], d[:, 0]
        r = np.pad(np.abs(d - prev) / (np.abs(prev) + s),
                   [(0, 1), (0, 0)] if ax == 0 else [(0, 0), (0, 1)])
        if per_direction and measured is not None:
            ok = (measured & _neighbour_measured(measured, ax, -1)
                  & _neighbour_measured(measured, ax, 1))
            r = np.where(ok, r, 0.0)
            usable |= ok
        out = np.maximum(out, r)
    return (out, usable) if per_direction else out


def inside3(measured):
    """Pixels whose 3-pixel step value uses measured pixels only: k-1, k,
    k+1 along each axis (beyond the frame nothing is used)."""
    ok = measured.copy()
    for ax in (0, 1):
        for sft in (-1, 1):
            ok &= _neighbour_measured(measured, ax, sft)
    return ok


def steps3(Z, where, q, measured=None):
    """As steps, with the 3-pixel step value."""
    R = step3_map(Z, measured)
    v = R[where]
    return where & (R > np.quantile(v, q)) if v.size else where


def sweep_lines(Z, strip, q, measured):
    """Step 2: every pixel of the strip is tested, along each direction in
    which its two neighbours were measured; the larger step value counts.
    Returns the kept pixels, the step value and the tested pixels."""
    R, usable = step3_map(Z, measured, per_direction=True)
    tested = strip & usable
    v = R[tested]
    kept = tested & (R > np.quantile(v, q)) if v.size else tested
    return kept, R, tested


def unet_only(D, i, q):
    return D["B"][i], float(D["X"][i, 1].mean())


def two_step(D, i, q, band):
    B = D["B"][i]
    strip = dilate(B, band)
    measured = strip | (D["X"][i, 1] > 0.5)
    # every pixel of the strip is tested, with measured neighbours only
    kept, _, _ = sweep_lines(D["Z"][i], strip, q, measured)
    return kept, float(measured.mean())


def two_step_inner(D, i, q, band):
    """The version before: only pixels within band-1 steps were tested."""
    B = D["B"][i]
    measured = dilate(B, band) | (D["X"][i, 1] > 0.5)
    checked = dilate(B, max(band - 1, 0)) & inside3(measured)
    return steps3(D["Z"][i], checked, q, measured), float(measured.mean())


def two_step6(D, i, q, band):
    """The earlier step 2: the 6-pixel detector, checked within band-2."""
    B = D["B"][i]
    measured = dilate(B, band) | (D["X"][i, 1] > 0.5)
    checked = dilate(B, max(band - 2, 0)) & inside(measured)
    return steps(D["Z"][i], checked, q, measured), float(measured.mean())


def full_scan(D, i, q):
    return steps(D["Z"][i], np.ones_like(D["Y"][i]), q), 1.0


def coarse_scan(D, i, q, k, how="native"):
    """k x k points; steps found on the coarse grid itself ("native"), or
    on the coarse image upsampled to the full grid ("upsampled")."""
    n = k * k
    Zg = coarse_grid(D["Z"][i], n)
    if how == "native":
        B = native_line_map(Zg, ("steps", 5, 1, q, 0, 0), D["Y"][i].shape)
    else:
        B = steps(coarse_image(D["Z"][i], n, 1)[0],
                  np.ones_like(D["Y"][i]), q)
    return B, Zg.size / D["Y"][i].size


def run(D, fn, q, *a):
    out = [fn(D, i, q, *a) for i in range(len(D["Y"]))]
    m = [tolerant_f1(B, D["Y"][i], 1.0) for i, (B, _) in enumerate(out)]
    return {"measured": 100 * float(np.mean([c for _, c in out])),
            "precision": float(np.mean([x["precision"] for x in m])),
            "recall": float(np.mean([x["recall"] for x in m])),
            "f1": float(np.mean([x["f1"] for x in m]))}


def choose(val, fn, *a):
    return max(QS, key=lambda q: run(val, fn, q, *a)["f1"])


ZOOM = (slice(14, 40), slice(14, 40))   # rows, cols of the magnified panel
RAY_C, BAND_C, CENTRE_C, FOUND_C, TRUE_C = (
    "#111111", "#cfe3f5", "#2a6ab0", "#c00000", "#9a9a9a")


def fig_example(test, q, band, path):
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch, Rectangle
    i = test["names"].index(EXAMPLE)
    Y, X = test["Y"][i], test["X"][i]
    B = test["B"][i]
    rays = X[1] > 0.5
    D, _ = two_step(test, i, q, band)

    def rgb(*layers):
        img = np.ones(Y.shape + (3,))
        for mask, col in layers:
            img[mask] = to_rgb(col)
        return img
    strip = dilate(B, band)               # step 2 alone, without the rays

    def grid(a, rows, cols, lw):
        """The pixel grid: one line between every two pixels."""
        for k in range(cols.start, cols.stop + 1):
            a.axvline(k - 0.5, color="#c8c8c8", lw=lw, zorder=3)
        for k in range(rows.start, rows.stop + 1):
            a.axhline(k - 0.5, color="#c8c8c8", lw=lw, zorder=3)

    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.2),
                           gridspec_kw=dict(wspace=0.06, hspace=0.16,
                                            left=0.005, right=0.995,
                                            top=0.955, bottom=0.075))
    ax = ax.ravel()
    H, W = Y.shape
    full = (slice(0, H), slice(0, W))
    panels = [
        (rgb((rays, RAY_C)), "(a) rays measured"),
        (rgb((B, CENTRE_C)), "(b) U-Net prediction"),
        (rgb((strip, BAND_C)),
         f"(c) strip of ±{band} px measured"),
        (None, "(d) magnified region of (c)"),
        (rgb((Y, TRUE_C), (D, FOUND_C)), "(e) lines found vs true lines"),
    ]
    for a, (img, title) in zip(ax, panels):
        if img is not None:
            a.imshow(img, origin="lower", interpolation="nearest")
            grid(a, *full, lw=0.12)
        a.set_title(title, fontsize=8)
    r, c = ZOOM
    ax[2].add_patch(Rectangle((c.start - 0.5, r.start - 0.5),
                              c.stop - c.start, r.stop - r.start, fill=False,
                              ec=FOUND_C, lw=1.0, zorder=4))
    z = rgb((Y, TRUE_C), (D, FOUND_C))[r, c]
    ax[3].imshow(z, origin="lower", interpolation="nearest",
                 extent=(c.start - 0.5, c.stop - 0.5, r.start - 0.5,
                         r.stop - 0.5))
    grid(ax[3], r, c, lw=0.35)
    for a in ax[:5]:
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.6)
    ax[5].axis("off")
    ax[5].legend(handles=[Patch(color=RAY_C, label="ray pixel"),
                          Patch(color=CENTRE_C, label="U-Net line"),
                          Patch(color=FOUND_C, label="line found"),
                          Patch(color=TRUE_C, label="true line")],
                 loc="center", frameon=False, fontsize=8,
                 handlelength=1.4, labelspacing=0.9)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def fig_curve(rows, path):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, a = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    g = lambda m: [r for r in rows if r["method"] == m]          # noqa
    two = g("U-Net, then local check")
    co = g("coarse scan")
    # one point per U-Net setting it is matched to; no line, since nothing
    # was measured between the costs
    a.plot([r["measured"] for r in co], [r["f1"] for r in co], "s",
           color="#8c8c8c", ms=4, label="raster scan (same cost)")
    a.plot([r["measured"] for r in two], [r["f1"] for r in two], "o-",
           color="#c00000", ms=4, label="U-Net, then local check")
    for r in two:                       # which margin each point is
        a.annotate(r["setting"].replace("band ", "").replace(" px", ""),
                   (r["measured"], r["f1"]), textcoords="offset points",
                   xytext=(-4, 5), ha="right", fontsize=6.5, color="#c00000")
    u = g("U-Net only")[0]
    a.plot(u["measured"], u["f1"], "o", color="#2a6ab0", ms=6,
           label="U-Net only")
    f = g("full scan")[0]
    a.plot(f["measured"], f["f1"], "*", color="#111111", ms=9,
           label="full scan")
    a.set_xlabel("fraction of the plane measured (%)")
    a.set_ylabel("line accuracy F1@1")
    a.set_xlim(0, 105)
    a.set_ylim(0.3, 1.0)
    a.grid(ls=":", lw=0.6, color="#aaaaaa")
    a.spines[["top", "right"]].set_visible(False)
    a.legend(fontsize=7, loc="lower right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    net, ck = grid_train.load(os.path.join(RUN_DIR, CELL,
                                           model_dir(RUN_DIR, CELL),
                                           "unet.pt"))
    thr = float(ck["threshold"])
    val, test = load(net, thr, "validation"), load(net, thr, "test")
    rows = [{"method": "U-Net only", "setting": "",
             **run(test, unet_only, None)}]
    for b in BANDS:
        q = choose(val, two_step, b)
        rows.append({"method": "U-Net, then local check",
                     "setting": f"band ±{b} px", "q": q,
                     **run(test, two_step, q, b)})
        print(rows[-1], flush=True)
    q = choose(val, full_scan)
    rows.append({"method": "full scan", "setting": "", "q": q,
                 **run(test, full_scan, q)})
    n_px = test["Y"][0].size
    for ref in [r for r in rows if r["method"] != "full scan"]:
        # a raster at the cost of this U-Net setting, never cheaper; it gets
        # the better of its two detectors, chosen on validation together
        # with the threshold
        k = int(np.ceil(np.sqrt(ref["measured"] / 100 * n_px)))
        _, q, how = max(((run(val, coarse_scan, q, k, how)["f1"], q, how)
                         for how in ("native", "upsampled") for q in QS))
        rows.append({"method": "coarse scan", "setting": f"{k} x {k} ({how})",
                     "matched": ref["setting"] or ref["method"], "q": q,
                     **run(test, coarse_scan, q, k, how)})
        print(rows[-1], flush=True)
    keys = ["method", "setting", "matched", "measured", "precision",
            "recall", "f1", "q"]
    with open(os.path.join(OUT, "table.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "table.md"), "w", encoding="utf-8") as f:
        f.write("| method | setting | measured (%) | precision | recall | "
                "F1@1 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['setting']} | "
                    f"{r['measured']:.1f} | {r['precision']:.3f} | "
                    f"{r['recall']:.3f} | {r['f1']:.3f} |\n")
    qb = next(r["q"] for r in rows if r["setting"] == f"band ±{FIG_BAND} px")
    fig_example(test, qb, FIG_BAND, os.path.join(OUT, "fig_example.png"))
    # fig_f1_vs_measured.png (Fig. 17) is drawn by run_23_equal_seeds.py,
    # with the equally spaced seeds as the baseline of the local sweeps
    print("wrote", OUT)


if __name__ == "__main__":
    main()
