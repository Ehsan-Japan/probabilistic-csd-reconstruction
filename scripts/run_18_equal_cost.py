"""
The conventional route at EQUAL measurement cost.

    python scripts/run_18_equal_cost.py

The question (Prof. Fujita's): what does predicting the lines directly with
the U-Net gain over the conventional route of reconstructing the image and
then detecting the lines?  run_16/17 compared the U-Net with the
conventional route on the FULL image, which measures 20-60x more points.
Here the conventional route gets exactly the U-Net's measurement budget,
for every one of the 15 budgets:

  rays + interpolation   the SAME ray measurements the U-Net sees (channel 0
                         of X at the visited pixels), interpolated to the
                         full grid, then the step detector
  coarse raster scan     rays x points measurements spent on a regular
                         k x k grid instead (k = floor(sqrt(rays*points))),
                         upsampled to the full grid, then the step detector

Both then go through the same label_states and the same scoring as the
U-Net, with and without the same line fit (the setting run_17 chose).

Fairness: each route's detector setting (interpolation, and the step
detector's knobs) is chosen on the validation devices, by mean few-electron
balanced accuracy over three budgets (4x40, 6x50, 8x60), exactly as the
U-Net's threshold and the line fit were chosen on validation.  The test
devices only report.

Writes results/<run>/equal_cost/: scan_<route>.csv, per_device.csv,
summary.csv/.md/.tex, figures/.
"""
import csv
import os

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from csdrecon.ml.ray_peaks import load_grid
from csdrecon.study import charge_states as C
from run_7_virtual_gates import cells, split_arrays
from run_16_charge_states import local
from run_17_few_electron import NMAX

# ══ settings ══════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
SELECT_ON = ["4_rays_40_points_500_samples", "6_rays_50_points_500_samples",
             "8_rays_60_points_500_samples"]     # budgets that choose knobs
FIGURE_CELL = "8_rays_60_points_500_samples"
EXAMPLES = ["sample_223", "sample_550", "sample_11"]
# the step detector's knobs scanned for both routes
STEP_GRID = [("steps", w, 1, q, 0, dil) for w in (5, 7)
             for q in (0.60, 0.70, 0.80, 0.90, 0.95)
             for dil in (0, 1)]
ORDERS = {"rays": ("nearest", "linear", "cubic"),   # griddata method
          "coarse": (0, 1, 3),                        # zoom spline order
          "native": ("native",)}                      # no upsampling
# ══════════════════════════════════════════════════════════════════════════

ROUTES = {"rays": "rays + interpolation",
          "coarse": "coarse scan, upsampled",
          "native": "coarse scan, steps on the coarse grid"}


# ── the two equal-cost images ─────────────────────────────────────────────

def rays_image(X, method):
    """The ray measurements the U-Net sees, interpolated to the full grid.
    Outside the rays' convex hull (where linear/cubic give nothing) the
    nearest measured value is used."""
    vis = X[1] > 0.5
    r, c = np.nonzero(vis)
    vals = X[0][vis]
    gr, gc = np.mgrid[0:X.shape[1], 0:X.shape[2]]
    Z = griddata((r, c), vals, (gr, gc), method=method)
    if np.isnan(Z).any():
        Zn = griddata((r, c), vals, (gr, gc), method="nearest")
        Z = np.where(np.isnan(Z), Zn, Z)
    return Z


def coarse_image(Z_full, n_points, order):
    """n_points spent on a regular k x k grid (the simulator's exact values
    at those voltages), upsampled back to the full grid."""
    H, W = Z_full.shape
    k = int(np.floor(np.sqrt(n_points)))
    ri = np.rint(np.linspace(0, H - 1, k)).astype(int)
    ci = np.rint(np.linspace(0, W - 1, k)).astype(int)
    Zc = Z_full[np.ix_(ri, ci)]
    Z = zoom(Zc, (H / k, W / k), order=order, mode="nearest")
    return Z[:H, :W], k


def coarse_grid(Z_full, n_points):
    """The k x k coarse scan itself, not upsampled."""
    H, W = Z_full.shape
    k = int(np.floor(np.sqrt(n_points)))
    ri = np.rint(np.linspace(0, H - 1, k)).astype(int)
    ci = np.rint(np.linspace(0, W - 1, k)).astype(int)
    return Z_full[np.ix_(ri, ci)]


def native_line_map(Zc, knob, shape):
    """Steps found on the coarse grid itself, where a transition is one jump
    between two neighbouring measurements; the flagged coarse cells are then
    mapped onto the full grid and thinned to one-pixel lines."""
    from skimage.morphology import skeletonize
    _, w, rel, q, _, dil = knob
    R = C.step_residual_map(Zc, w, bool(rel))
    E = R > np.quantile(R, q)
    H, W = shape
    k = Zc.shape[0]
    ri = np.minimum((np.arange(H) * k) // H, k - 1)
    ci = np.minimum((np.arange(W) * k) // W, k - 1)
    B = skeletonize(E[np.ix_(ri, ci)])
    if dil:
        from scipy.ndimage import binary_dilation
        B = binary_dilation(B, iterations=dil)
    return B


def line_map(route, img, knob, shape):
    if route == "native":
        return native_line_map(img, knob, shape)
    return C.dense_line_map(img, knob)


# ── one split ─────────────────────────────────────────────────────────────

def load(run_dir, split, names_only=None):
    cl = cells(run_dir)
    _, Y, samples = split_arrays(os.path.join(run_dir, cl[0][0]), split)
    dirs = [local(s) for s in samples]
    names = [os.path.basename(d) for d in dirs]
    info = [C.read_device(d) for d in dirs]
    ctx = {"Y": Y.astype(bool), "dirs": dirs, "names": names, "info": info,
           "states": [C.load_states(d) for d in dirs],
           "px": [C.pixel_mV(d, Y[i].shape) for i, d in enumerate(info)],
           "Zfull": [load_grid(d)[2] for d in dirs], "X": {}, "cells": cl}
    for name, rays, points in cl:
        if names_only and name not in names_only:
            continue
        X, _, sc = split_arrays(os.path.join(run_dir, name), split)
        assert [os.path.basename(local(s)) for s in sc] == names, name
        ctx["X"][name] = (X, rays, points)
    return ctx


def images(ctx, name, route, order):
    X, rays, points = ctx["X"][name]
    out = []
    for i in range(len(ctx["Y"])):
        if route == "rays":
            out.append(rays_image(X[i], order))
        elif route == "native":
            out.append(coarse_grid(ctx["Zfull"][i], rays * points))
        else:
            out.append(coarse_image(ctx["Zfull"][i], rays * points,
                                    order)[0])
    return out


def score(ctx, i, B):
    res = C.label_states(B, *ctx["px"][i])
    s = C.score(res, ctx["states"][i], ctx["Y"][i], *ctx["px"][i])
    f = C.score_few(res, ctx["states"][i], ctx["Y"][i], NMAX)
    return {"acc_rel": s["acc_rel"], "state_recall": s["state_recall"],
            **f}, res


def saved_choice(route, out):
    """The setting a previous run chose, if its scan covered this grid."""
    path = os.path.join(out, f"scan_{route}.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(STEP_GRID) * len(ORDERS[route]):
        return None
    r = next(r for r in rows if r["chosen"] == "1")
    k = next(k for k in STEP_GRID if C.dense_knob(k) == r["detector"])
    o = r["interpolation"]
    o = int(o) if route == "coarse" else o
    print(f"  {route}: reusing the saved validation choice {o} + "
          f"{r['detector']}", flush=True)
    return o, k


def choose(ctx, route, out):
    """Interpolation x step-detector setting with the best mean balanced
    few-electron accuracy on the validation devices, over SELECT_ON."""
    saved = saved_choice(route, out)
    if saved:
        return saved
    scan = []
    for order in ORDERS[route]:
        imgs = {n: images(ctx, n, route, order) for n in SELECT_ON}
        for k in STEP_GRID:
            v = [score(ctx, i, line_map(route, imgs[n][i], k,
                                        ctx["Y"][i].shape))[0]
                 ["bal_few"] for n in SELECT_ON
                 for i in range(len(ctx["Y"]))]
            scan.append((float(np.nanmean(v)), order, k))
        print(f"    {route} order {order}: best so far "
              f"{100 * max(scan)[0]:.1f} %", flush=True)
    best = max(scan, key=lambda t: t[0])
    with open(os.path.join(out, f"scan_{route}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interpolation", "detector", "bal_few_mean_validation",
                    "chosen"])
        for v, o, k in sorted(scan, key=lambda t: -t[0]):
            w.writerow([o, C.dense_knob(k), f"{v:.4f}",
                        int((v, o, k) == best)])
    print(f"  {route}: chosen {best[1]} + {C.dense_knob(best[2])} "
          f"({100 * best[0]:.1f} % on validation)", flush=True)
    return best[1], best[2]


def fit_knob(run_dir):
    with open(os.path.join(run_dir, "few_electron", "fit_scan.csv")) as f:
        txt = next(r["setting"] for r in csv.DictReader(f)
                   if r["chosen"] == "1")
    return next(k for k in C.FIT_GRID if C.fit_knob(k) == txt)


# ── tables ────────────────────────────────────────────────────────────────

KEYS = ("bal_few", "land_11", "acc_few", "state_recall")


def summarise(rows):
    keys = []
    for r in rows:
        k = (r["method"], r["rays"], r["points"])
        if k not in keys:
            keys.append(k)
    out = []
    for m, ra, pa in keys:
        rs = [r for r in rows if (r["method"], r["rays"], r["points"])
              == (m, ra, pa)]
        s = {"method": m, "rays": ra, "points": pa, "n": len(rs),
             "coverage_%": 100 * float(np.mean([r["coverage"]
                                                for r in rs]))}
        for k in KEYS:
            v = np.array([r[k] for r in rs], float)
            v = v[np.isfinite(v)]
            s[k] = (100 * v.mean(), 100 * np.median(v), len(v))
        out.append(s)
    return out


def write_tables(summ, out, caption):
    with open(os.path.join(out, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rays", "points", "coverage_%", "n"]
                   + [f"{k}_{q}" for k in KEYS
                      for q in ("mean", "median", "n")])
        for s in summ:
            w.writerow([s["method"], s["rays"], s["points"],
                        f"{s['coverage_%']:.4f}", s["n"]]
                       + [x for k in KEYS for x in s[k]])
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(caption + "\n\n| method | rays × points | grid measured % | "
                "balanced few % | (1,1) landing % | pixel acc. few % | "
                "whole-window recall % |\n|---|---|---|---|---|---|---|\n")
        for s in summ:
            b = f"{s['rays']} × {s['points']}" if s["rays"] != "" else "—"
            f.write(f"| {s['method']} | {b} | {s['coverage_%']:.2f} | "
                    + " | ".join(f"{s[k][0]:.1f} [{s[k][1]:.1f}]"
                                 for k in KEYS) + " |\n")
        f.write("\nmean [median] over the test devices.\n")


# ── figures ───────────────────────────────────────────────────────────────

INK, MUTED = "#111111", "#555555"
STYLE = {"U-Net": ("#2a78d6", "o"), "U-Net + line fit": ("#1baf7a", "s"),
         "rays + interpolation": ("#b07aa1", "^"),
         "rays + interpolation + line fit": ("#7b3fbf", "v"),
         "coarse scan, upsampled": ("#f0a35e", "<"),
         "coarse scan, upsampled + line fit": ("#d1622b", ">"),
         "coarse scan, steps on the coarse grid": ("#9c9c9c", "d"),
         "coarse scan, steps on the coarse grid + line fit": ("#555555",
                                                              "D")}


def fig_curves(summ, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for a, k, lab in ((ax[0], "bal_few", "balanced accuracy, few-e (%)"),
                      (ax[1], "land_11", "(1,1) landing (%)")):
        for m, (col, mk) in STYLE.items():
            ss = [s for s in summ if s["method"] == m]
            a.plot([s["coverage_%"] for s in ss], [s[k][0] for s in ss], mk,
                   ms=3.5, color=col, label=m,
                   mfc=col if "fit" in m else "none", mew=0.9)
        for m, col, mk in (("dense, no ML (full image)", "#eb6834", "D"),):
            s = next(s for s in summ if s["method"] == m)
            a.plot(100, s[k][0], mk, ms=5, color=col, label=m)
        b = next(s for s in summ if s["method"] == "all (0,0)")
        a.axhline(b[k][0], color=MUTED, ls="--", lw=0.8, label="all (0,0)")
        a.set_xscale("log")
        t = [1, 2, 5, 10, 20, 50, 100]
        a.set_xticks(t)
        a.set_xticklabels([str(v) for v in t])
        a.set_xlim(1, 140)
        a.set_ylim(-2, 104)
        a.set_xlabel("grid measured (%)", fontsize=7)
        a.set_ylabel(lab, fontsize=7)
        a.tick_params(labelsize=6)
        a.grid(alpha=0.25, lw=0.6)
    h, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="outside lower center", ncol=3, fontsize=5.8,
               frameon=False)
    fig.savefig(os.path.join(out, "equal_cost_curves.png"), dpi=300)
    fig.savefig(os.path.join(out, "equal_cost_curves.pdf"))
    plt.close(fig)


def fig_examples(ctx, unet_maps, chosen, fk, out):
    """Per device: the three images, their line maps and labels."""
    _, rays, points = ctx["X"][FIGURE_CELL]
    fig, axs = plt.subplots(len(EXAMPLES), 6,
                            figsize=(7.2, 1.55 * len(EXAMPLES) + 0.5),
                            gridspec_kw=dict(wspace=0.06, hspace=0.55,
                                             left=0.065, right=0.995,
                                             top=0.9, bottom=0.06))
    titles = ["(a) U-Net map + fit", "(b) its labels",
              "(c) rays interpolated", "(d) its labels",
              "(e) coarse scan", "(f) its labels"]
    for k, n in enumerate(EXAMPLES):
        i = ctx["names"].index(n)
        ext = ctx["info"][i]["voltage_window"]
        kw = dict(origin="lower", extent=ext, interpolation="nearest")
        ax = axs[k]
        Bu = C.fit_lines(unet_maps[n], *fk)
        o_r, k_r = chosen["rays"]
        Zr = rays_image(ctx["X"][FIGURE_CELL][0][i], o_r)
        Br = C.fit_lines(C.dense_line_map(Zr, k_r), *fk)
        _, k_c = chosen["native"]
        Zg = coarse_grid(ctx["Zfull"][i], rays * points)
        kk = Zg.shape[0]
        Zc = coarse_image(ctx["Zfull"][i], rays * points, 0)[0]
        Bc = C.fit_lines(native_line_map(Zg, k_c, Zc.shape), *fk)
        txt = []
        for c, (img, B, cm) in enumerate(((None, Bu, None),
                                          (Zr, Br, "viridis"),
                                          (Zc, Bc, "viridis"))):
            a = ax[2 * c]
            if img is None:
                a.imshow(B, cmap=ListedColormap(["white", INK]), vmin=0,
                         vmax=1, **kw)
            else:
                a.imshow(img, cmap=cm, **kw)
                a.contour(B, [0.5], colors="white", linewidths=0.35,
                          origin="lower", extent=ext)
            sc, res = score(ctx, i, B)
            ax[2 * c + 1].imshow(C._state_rgb(res["labels"]), **kw)
            C._write_labels(ax[2 * c + 1], res["labels"], ext, fs=3.6)
            txt.append(f"{100 * sc['bal_few']:.0f} %")
        for c, a in enumerate(ax):
            C._map_axes(a, c == 0)
            if k == 0:
                a.set_title(titles[c], fontsize=6, pad=11)
        ax[0].text(0, 1.04, f"{n}: balanced few-e accuracy  U-Net+fit "
                   f"{txt[0]},  rays interpolated+fit {txt[1]},  coarse "
                   f"{kk}$\\times${kk}+fit {txt[2]}",
                   transform=ax[0].transAxes, fontsize=5.8, va="bottom")
    fig.savefig(os.path.join(out, "equal_cost_examples.png"), dpi=300)
    fig.savefig(os.path.join(out, "equal_cost_examples.pdf"))
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    run_dir = os.path.join(ROOT, "results", RUN)
    out = os.path.join(run_dir, "equal_cost")
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(RUN, flush=True)
    val = load(run_dir, "validation", names_only=SELECT_ON)
    chosen = {r: choose(val, r, out) for r in ROUTES}
    del val
    fk = fit_knob(run_dir)

    test = load(run_dir, "test")
    rows, unet_maps = [], {}
    prev = {(r["method"], r["rays"], r["points"], r["device"]): r
            for r in csv.DictReader(open(os.path.join(
                run_dir, "few_electron", "per_device.csv")))}
    n = len(test["Y"])
    keyf = lambda r: {k: float(r[k]) for k in KEYS}           # noqa: E731
    for i, dn in enumerate(test["names"]):
        for m, out_m in (("all (0,0)", "all (0,0)"),
                         ("dense, no ML", "dense, no ML (full image)")):
            r = prev[(m, "", "", dn)]
            rows.append({"method": out_m, "rays": "", "points": "",
                         "coverage": float(r["coverage"]), "device": dn,
                         **keyf(r)})
    for name, rays, points in test["cells"]:
        for m in ("U-Net", "U-Net + line fit"):
            for dn in test["names"]:
                r = prev[(m, str(rays), str(points), dn)]
                rows.append({"method": m, "rays": rays, "points": points,
                             "coverage": float(r["coverage"]), "device": dn,
                             **keyf(r)})
        X = test["X"][name][0]
        cov = X[:, 1].reshape(n, -1).mean(1)
        for route, label in ROUTES.items():
            o, k = chosen[route]
            imgs = images(test, name, route, o)
            for i in range(n):
                B = line_map(route, imgs[i], k, test["Y"][i].shape)
                c = (float(cov[i]) if route == "rays"
                     else int(np.sqrt(rays * points)) ** 2 / B.size)
                for suffix, M in (("", B), (" + line fit",
                                            C.fit_lines(B, *fk))):
                    rows.append({"method": label + suffix, "rays": rays,
                                 "points": points, "coverage": c,
                                 "device": test["names"][i],
                                 **score(test, i, M)[0]})
        print(f"  test {name} done", flush=True)
    with open(os.path.join(out, "per_device.csv"), "w", newline="") as f:
        fields = ["method", "rays", "points", "coverage", "device"] \
            + list(KEYS)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows)
    write_tables(summ, out,
                 f"Equal measurement cost, few-electron region (n1, n2 <= "
                 f"{NMAX}), test split ({n} devices), {RUN}.  Settings "
                 f"chosen on validation: rays {chosen['rays'][0]} + "
                 f"{C.dense_knob(chosen['rays'][1])}; coarse upsampled "
                 f"order {chosen['coarse'][0]} + "
                 f"{C.dense_knob(chosen['coarse'][1])}; coarse native "
                 f"{C.dense_knob(chosen['native'][1])}; line fit "
                 f"{C.fit_knob(fk)}.")
    fig_curves(summ, fig_dir)
    from csdrecon.ml import grid_train
    from run_7_virtual_gates import model_dir
    net, ck = grid_train.load(os.path.join(run_dir, FIGURE_CELL,
                                           model_dir(run_dir, FIGURE_CELL),
                                           "unet.pt"))
    P = grid_train.predict(net, test["X"][FIGURE_CELL][0]) > \
        float(ck["threshold"])
    for e in EXAMPLES:
        unet_maps[e] = P[test["names"].index(e)]
    fig_examples(test, unet_maps, chosen, fk, fig_dir)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
