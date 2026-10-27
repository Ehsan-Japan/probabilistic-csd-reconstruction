"""
The few-electron region, and straight-line repair of the U-Net maps.

    python scripts/run_17_few_electron.py

Two changes against run_16:

1.  Only the few-electron region is scored: pixels whose TRUE state has
    n1 <= NMAX and n2 <= NMAX.  Labels are absolute (the top-right corner
    is (0,0) on every device), because an experiment needs the occupation
    itself.  Metrics: charge_states.score_few (pixel accuracy, balanced
    state accuracy, (1,1) landing).

2.  A new method, U-Net + line fit: charge_states.fit_lines on each U-Net
    binary map before label_states.  Its knobs are chosen on the validation
    devices, by mean balanced few-electron accuracy over every U-Net, and
    applied unchanged to the test devices.

Rows: all (0,0) (baseline), dense no ML (the setting run_16 chose), full
CSD, and for every U-Net budget: U-Net, U-Net + line fit.

Writes results/<run>/few_electron/: fit_scan.csv, per_device.csv,
summary.md/.csv/.tex, figures/.
"""
import csv
import os

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from csdrecon.ml import grid_train
from csdrecon.ml.ray_peaks import load_grid
from csdrecon.study import charge_states as C
from run_7_virtual_gates import cells, model_dir, split_arrays
from run_16_charge_states import local

# ══ settings ══════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
NMAX = 3                         # few-electron region: n1, n2 <= NMAX
FIGURE_UNET = "8_rays_60_points_500_samples"
EXAMPLES = ["sample_223", "sample_550", "sample_11"]
# ══════════════════════════════════════════════════════════════════════════

FEW = ("acc_few", "bal_few", "land_11")
GLOBAL = ("acc_rel", "state_recall")
FIT = "U-Net + line fit"


def load_split(run_dir, split):
    cl = cells(run_dir)
    _, Y, samples = split_arrays(os.path.join(run_dir, cl[0][0]), split)
    dirs = [local(s) for s in samples]
    names = [os.path.basename(d) for d in dirs]
    info = [C.read_device(d) for d in dirs]
    ctx = {"Y": Y.astype(bool), "dirs": dirs, "names": names, "info": info,
           "states": [C.load_states(d) for d in dirs],
           "px": [C.pixel_mV(d, Y[i].shape) for i, d in enumerate(info)],
           "unet": {}, "cov": {}, "cells": cl}
    for name, rays, points in cl:
        X, _, sc = split_arrays(os.path.join(run_dir, name), split)
        assert [os.path.basename(local(s)) for s in sc] == names, name
        net, ck = grid_train.load(os.path.join(run_dir, name,
                                               model_dir(run_dir, name),
                                               "unet.pt"))
        ctx["unet"][name] = grid_train.predict(net, X) > float(ck["threshold"])
        ctx["cov"][name] = X[:, 1].reshape(len(X), -1).mean(1)
    return ctx


def evaluate(ctx, i, B):
    """Every metric of one binary map of device i."""
    res = C.label_states(B, *ctx["px"][i])
    s = C.score(res, ctx["states"][i], ctx["Y"][i], *ctx["px"][i])
    f = C.score_few(res, ctx["states"][i], ctx["Y"][i], NMAX)
    return {**{k: s[k] for k in GLOBAL}, **f}, res


def choose_fit(ctx, out):
    """The line-fit knob with the best mean balanced few-electron accuracy
    over every U-Net and every validation device."""
    scan = []
    for k in C.FIT_GRID:
        v = [evaluate(ctx, i, C.fit_lines(ctx["unet"][n][i], *k))[0]
             ["bal_few"] for n in ctx["unet"] for i in range(len(ctx["Y"]))]
        scan.append((float(np.nanmean(v)), k))
        print(f"    {C.fit_knob(k):45s} bal_few {100 * scan[-1][0]:.1f} %")
    plain = np.nanmean([evaluate(ctx, i, ctx["unet"][n][i])[0]["bal_few"]
                        for n in ctx["unet"] for i in range(len(ctx["Y"]))])
    best = max(scan)
    with open(os.path.join(out, "fit_scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "bal_few_mean_validation", "chosen"])
        w.writerow(["no line fit (U-Net map as is)", f"{plain:.4f}", 0])
        for v, k in sorted(scan, reverse=True):
            w.writerow([C.fit_knob(k), f"{v:.4f}", int(k == best[1])])
    print(f"  line fit chosen on validation: {C.fit_knob(best[1])} "
          f"({100 * best[0]:.1f} % vs {100 * plain:.1f} % without)")
    return best[1]


def dense_knob(out16):
    with open(os.path.join(out16, "dense_scan.csv"), newline="") as f:
        txt = next(r["setting"] for r in csv.DictReader(f)
                   if r["chosen"] == "1")
    return next(k for k in C.DENSE_GRID if C.dense_knob(k) == txt)


def test_rows(ctx, fit_knob, dknob):
    rows, keep = [], {}
    n = len(ctx["Y"])
    zero = np.zeros_like(ctx["Y"][0])
    for i in range(n):
        base = {"labels": np.zeros(ctx["Y"][i].shape + (2,), int),
                "regions": np.ones(ctx["Y"][i].shape, int), "n_regions": 1,
                "consistency": np.nan, "n_checked": 0, "unreached_px": 0}
        f = C.score_few(base, ctx["states"][i], ctx["Y"][i], NMAX)
        s = C.score(base, ctx["states"][i], ctx["Y"][i], *ctx["px"][i])
        common = {"device": ctx["names"][i],
                  "n_states": len(np.unique(C.state_code(ctx["states"][i])))}
        rows.append({"method": "all (0,0)", "rays": "", "points": "",
                     "coverage": 0.0, **common,
                     **{k: s[k] for k in GLOBAL}, **f})
        dn = C.dense_line_map(load_grid(ctx["dirs"][i])[2], dknob)
        for m, B in (("full CSD", ctx["Y"][i]), ("dense, no ML", dn)):
            rows.append({"method": m, "rays": "", "points": "",
                         "coverage": 1.0, **common,
                         **evaluate(ctx, i, B)[0]})
    for name, rays, points in ctx["cells"]:
        for i in range(n):
            common = {"device": ctx["names"][i], "rays": rays,
                      "points": points,
                      "coverage": float(ctx["cov"][name][i]),
                      "n_states": len(np.unique(
                          C.state_code(ctx["states"][i])))}
            B = ctx["unet"][name][i]
            F = C.fit_lines(B, *fit_knob)
            rows.append({"method": "U-Net", **common,
                         **evaluate(ctx, i, B)[0]})
            rows.append({"method": FIT, **common, **evaluate(ctx, i, F)[0]})
            if name == FIGURE_UNET:
                keep[ctx["names"][i]] = (B, F)
    del zero
    return rows, keep


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
             "coverage_%": 100 * float(np.mean([r["coverage"] for r in rs]))}
        for k in FEW + GLOBAL:
            v = np.array([r[k] for r in rs], float)
            v = v[np.isfinite(v)]
            s[k] = (100 * v.mean(), 100 * np.median(v), len(v))
        out.append(s)
    return out


HEAD = [("pixel acc. few (%)", "acc_few"), ("balanced few (%)", "bal_few"),
        ("(1,1) landing (%)", "land_11"),
        ("whole-window acc. (%)", "acc_rel"),
        ("whole-window recall (%)", "state_recall")]


def write_tables(summ, out, caption):
    budget = lambda s: (f"{s['rays']} × {s['points']}"           # noqa: E731
                        if s["rays"] != "" else "—")
    cell = lambda t: f"{t[0]:.1f} [{t[1]:.1f}]"                   # noqa: E731
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(caption + "\n\n| method | rays × points | coverage % | "
                + " | ".join(h for h, _ in HEAD) + " |\n")
        f.write("|" + "---|" * (3 + len(HEAD)) + "\n")
        for s in summ:
            f.write(f"| {s['method']} | {budget(s)} | "
                    f"{s['coverage_%']:.2f} | "
                    + " | ".join(cell(s[k]) for _, k in HEAD) + " |\n")
        f.write("\nmean [median] over devices; (1,1) landing over the "
                f"devices whose window contains (1,1).  Few-electron region: "
                f"true n1, n2 <= {NMAX}, absolute labels.  Whole-window "
                "columns: run_16's relative accuracy and state recall.\n")
    with open(os.path.join(out, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rays", "points", "coverage_%", "n"]
                   + [f"{k}_{q}" for _, k in HEAD
                      for q in ("mean", "median", "n")])
        for s in summ:
            w.writerow([s["method"], s["rays"], s["points"],
                        f"{s['coverage_%']:.4f}", s["n"]]
                       + [x for _, k in HEAD for x in s[k]])
    with open(os.path.join(out, "summary.tex"), "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write(r"\begin{tabular}{llr" + "r" * len(HEAD) + "}\n\\toprule\n")
        f.write(r"method & rays $\times$ pts & coverage (\%) & "
                r"acc.\ few (\%) & balanced few (\%) & (1,1) landing (\%) & "
                r"whole acc.\ (\%) & whole recall (\%) \\" + "\n\\midrule\n")
        prev = None
        for s in summ:
            grp = "u" if s["rays"] != "" else s["method"]
            if prev is not None and grp != prev and grp == "u":
                f.write("\\midrule\n")
            prev = grp
            b = (f"{s['rays']} $\\times$ {s['points']}" if s["rays"] != ""
                 else "--")
            f.write(" & ".join([s["method"], b, f"{s['coverage_%']:.2f}"]
                               + [cell(s[k]) for _, k in HEAD]) + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ── figures ───────────────────────────────────────────────────────────────

INK, MUTED = "#111111", "#555555"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"


def fig_examples(ctx, keep, out):
    fig, axs = plt.subplots(len(EXAMPLES), 5,
                            figsize=(7.2, 1.9 * len(EXAMPLES) + 0.4),
                            gridspec_kw=dict(wspace=0.08, hspace=0.5,
                                             left=0.075, right=0.995,
                                             top=0.9, bottom=0.07))
    titles = ["(a) U-Net line map", "(b) labels from (a)",
              "(c) after line fit", "(d) labels from (c)",
              f"(e) truth; $n_1,n_2\\leq{NMAX}$ outlined"]
    bw = ListedColormap(["white", INK])
    for k, n in enumerate(EXAMPLES):
        i = ctx["names"].index(n)
        B, F = keep[n]
        ext = ctx["info"][i]["voltage_window"]
        cs = ctx["states"][i]
        kw = dict(origin="lower", extent=ext, interpolation="nearest")
        ax = axs[k]
        txt = []
        for c, M in ((0, B), (2, F)):
            sc, res = evaluate(ctx, i, M)
            ax[c].imshow(M, cmap=bw, vmin=0, vmax=1, **kw)
            ax[c + 1].imshow(C._state_rgb(res["labels"]), **kw)
            C._write_labels(ax[c + 1], res["labels"], ext, fs=4)
            txt.append(f"balanced few {100 * sc['bal_few']:.0f} %, landing "
                       f"{'yes' if sc['land_11'] == 1 else 'no'}")
        ax[4].imshow(C._state_rgb(cs), **kw)
        C._write_labels(ax[4], cs, ext, fs=4)
        roi = (cs[..., 0] <= NMAX) & (cs[..., 1] <= NMAX)
        ax[4].contour(roi, [0.5], colors=INK, linewidths=0.8,
                      origin="lower", extent=ext)
        for c, a in enumerate(ax):
            C._map_axes(a, c == 0)
            if k == 0:
                a.set_title(titles[c], fontsize=6.5, pad=12)
        ax[0].text(0, 1.03, f"{n}.  U-Net: {txt[0]};   with line fit: "
                   f"{txt[1]}", transform=ax[0].transAxes, fontsize=6.3,
                   va="bottom")
    fig.savefig(os.path.join(out, "fig_fit_examples.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_fit_examples.pdf"))
    plt.close(fig)


def fig_coverage(summ, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for a, k, lab in ((ax[0], "acc_few", "pixel accuracy, few-e (%)"),
                      (ax[1], "bal_few", "balanced accuracy, few-e (%)"),
                      (ax[2], "land_11", "(1,1) landing (%)")):
        for m, col, mk in (("U-Net", BLUE, "o"), (FIT, GREEN, "s")):
            ss = [s for s in summ if s["method"] == m]
            a.plot([s["coverage_%"] for s in ss], [s[k][0] for s in ss], mk,
                   ms=3.5, color=col, label=m)
        for m, col, mk in (("dense, no ML", ORANGE, "D"),
                           ("full CSD", INK, "*")):
            s = next(s for s in summ if s["method"] == m)
            a.plot(100, s[k][0], mk, ms=6, color=col, label=m)
        b = next(s for s in summ if s["method"] == "all (0,0)")
        a.axhline(b[k][0], color=MUTED, ls="--", lw=0.8,
                  label="all (0,0)")
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
    ax[2].legend(fontsize=5.5, frameon=False, loc="center right")
    fig.savefig(os.path.join(out, "fig_few_vs_coverage.png"), dpi=300)
    fig.savefig(os.path.join(out, "fig_few_vs_coverage.pdf"))
    plt.close(fig)


def main():
    run_dir = os.path.join(ROOT, "results", RUN)
    out = os.path.join(run_dir, "few_electron")
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(RUN)
    val = load_split(run_dir, "validation")
    fit_knob = choose_fit(val, out)
    del val
    test = load_split(run_dir, "test")
    rows, keep = test_rows(test, fit_knob,
                           dense_knob(os.path.join(run_dir, "charge_states")))
    with open(os.path.join(out, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows)
    write_tables(summ, out,
                 f"Few-electron region (n1, n2 <= {NMAX}), test split "
                 f"({len(test['Y'])} devices), {RUN}.  Line fit: "
                 f"{C.fit_knob(fit_knob)} (chosen on validation).")
    fig_examples(test, keep, fig_dir)
    fig_coverage(summ, fig_dir)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
