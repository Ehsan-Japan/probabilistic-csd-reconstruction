# -*- coding: utf-8 -*-
"""
make_eval.py -- figures, tables and numbers for evaluation_method.pdf: how
the accuracy of the U-Net is evaluated, from the line map to the charge
states.

    python docs/charge_states/make_eval.py
    pdflatex evaluation.tex  (twice, in the output folder)

Line-level metrics are recomputed here, on the test devices, from the very
checkpoints the charge-state studies used (model_dir: the retrained seed
where a cell was retrained), so every number in the note comes from one
model per budget.  Charge-state numbers are read from run_16 / run_17.
"""
import csv
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from csdrecon.ml import grid_train                      # noqa: E402
from csdrecon.ml.grid_metrics import (coverage_at_tau,   # noqa: E402
                                      evaluate, tolerant_f1)
from csdrecon.study import charge_states as C           # noqa: E402
from run_7_virtual_gates import cells, model_dir        # noqa: E402
from run_17_few_electron import NMAX                    # noqa: E402

RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
CELL = "8_rays_60_points_500_samples"
DEVICE = "sample_223"
RUN_DIR = os.path.join(ROOT, "results", RUN)
OUT = os.path.join(RUN_DIR, "evaluation_note")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")
INK, MUTED = "#111111", "#555555"
TP, FP, FN = "#1baf7a", "#d62728", "#2a78d6"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5})


def name_of(s):
    return os.path.basename(str(s).replace("\\", "/"))


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── line-level metrics, every U-Net, recomputed from the used checkpoints ──
def line_metrics():
    rows, keep = [], None
    for name, rays, points in cells(RUN_DIR):
        z = np.load(os.path.join(RUN_DIR, name, "test.npz"),
                    allow_pickle=True)
        md = model_dir(RUN_DIR, name)
        net, ck = grid_train.load(os.path.join(RUN_DIR, name, md,
                                               "unet.pt"))
        thr = float(ck["threshold"])
        prob = grid_train.predict(net, z["X"])
        B = prob > thr
        Y = z["Y"] > 0.5
        m = evaluate(B, Y)
        f1 = [tolerant_f1(B[i], Y[i], 1.0)["f1"] for i in range(len(Y))]
        rows.append({"cell": name, "rays": rays, "points": points,
                     "model": md, "threshold": thr,
                     "coverage": float(z["X"][:, 1].mean()),
                     "cov1": float(np.mean([coverage_at_tau(x, 1)
                                            for x in z["X"][:, 1]])),
                     "pred_frac": float(B.mean()),
                     "true_frac": float(Y.mean()),
                     "f1_1_median": float(np.median(f1)), **m})
        if name == CELL:
            names = [name_of(s) for s in z["samples"]]
            i = names.index(DEVICE)
            keep = {"prob": prob[i], "B": B[i], "Y": Y[i], "thr": thr,
                    "X": z["X"][i]}
    with open(os.path.join(OUT, "line_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, keep


def line_table(rows):
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"rays $\times$ pts & checkpoint & threshold & coverage (\%) & "
             r"F1@0 & F1@1 & precision@1 & recall@1 & predicted line px "
             r"(\%) \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join([
            f"{r['rays']} $\\times$ {r['points']}",
            r["model"].replace("_", r"\_"), f"{r['threshold']:g}",
            f"{100 * r['coverage']:.2f}", f"{r['f1@0']:.3f}",
            f"{r['f1@1']:.3f}", f"{r['precision@1']:.3f}",
            f"{r['recall@1']:.3f}", f"{100 * r['pred_frac']:.1f}"])
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "line_table.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── fig 1: what tolerant F1 counts ─────────────────────────────────────────
def fig_line(k, dev):
    Y, B = k["Y"], k["B"]
    ext = dev["voltage_window"]
    kw = dict(origin="lower", extent=ext, interpolation="nearest")
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.15),
                           gridspec_kw=dict(wspace=0.08, left=0.07,
                                            right=0.995, top=0.86,
                                            bottom=0.2))
    ax[0].imshow(k["X"][1] > 0.5, cmap=ListedColormap(["white", INK]),
                 vmin=0, vmax=1, **kw)
    ax[0].set_title(f"(a) measured pixels ({100 * k['X'][1].mean():.2f} %)")
    ax[1].imshow(k["prob"], cmap="viridis", vmin=0, vmax=1, **kw)
    ax[1].set_title(f"(b) U-Net $P$; $B=P>{k['thr']:g}$")
    for a, tau, lab in ((ax[2], 0, "(c)"), (ax[3], 1, "(d)")):
        dT = distance_transform_edt(~Y)
        dP = distance_transform_edt(~B)
        img = np.ones(Y.shape + (3,))
        hit = B & (dT <= tau)
        miss_p = B & (dT > tau)
        miss_t = Y & (dP > tau)
        img[hit] = matplotlib.colors.to_rgb(TP)
        img[miss_p] = matplotlib.colors.to_rgb(FP)
        img[miss_t] = matplotlib.colors.to_rgb(FN)
        a.imshow(img, **kw)
        m = tolerant_f1(B, Y, tau)
        a.set_title(f"{lab} $\\tau$={tau}: P {m['precision']:.2f}, "
                    f"R {m['recall']:.2f}, F1 {m['f1']:.2f}")
    for c, a in enumerate(ax):
        C._map_axes(a, c == 0)
    fig.legend(handles=[Patch(color=TP, label="predicted, true line within τ"),
                        Patch(color=FP, label="predicted, no true line "
                              "within τ (false alarm)"),
                        Patch(color=FN, label="true line, nothing predicted "
                              "within τ (miss)")],
               loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.savefig(os.path.join(FIG, "eval_line.pdf"))
    fig.savefig(os.path.join(FIG, "eval_line.png"), dpi=200)
    plt.close(fig)
    return {t: tolerant_f1(B, Y, t) for t in (0, 1)}


# ── fig 2: the charge-state metrics on one device ──────────────────────────
def fig_states(k, dev, cs, fit_knob):
    Y = k["Y"]
    dx, dy = C.pixel_mV(dev, Y.shape)
    ext = dev["voltage_window"]
    kw = dict(origin="lower", extent=ext, interpolation="nearest")
    B = C.fit_lines(k["B"], *fit_knob)
    res = C.label_states(B, dx, dy)
    L = res["labels"]
    sc = C.score(res, cs, Y, dx, dy)
    few = C.score_few(res, cs, Y, NMAX)
    m = C.keep_mask(Y)
    roi = (cs[..., 0] <= NMAX) & (cs[..., 1] <= NMAX)
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.62], wspace=0.1,
                          hspace=0.55, left=0.08, right=0.99, top=0.92,
                          bottom=0.1)
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax.append(fig.add_subplot(gs[1, 1:3]))
    ax[0].imshow(C._state_rgb(cs), **kw)
    C._write_labels(ax[0], cs, ext, fs=4)
    ax[0].set_title("(a) truth")
    ax[1].imshow(C._state_rgb(L), **kw)
    C._write_labels(ax[1], L, ext, fs=4)
    ax[1].set_title("(b) predicted labels")
    img = np.ones(Y.shape + (3,))
    right = (L == cs).all(-1)
    img[m & right] = 0.80
    img[m & ~right] = matplotlib.colors.to_rgb(FP)
    ax[2].imshow(img, **kw)
    ax[2].contour(roi, [0.5], colors=INK, linewidths=0.6, origin="lower",
                  extent=ext)
    ax[2].set_title("(c) right / wrong")
    ax[3].imshow(C._state_rgb(cs) * 0.45 + 0.55, **kw)
    p11 = (L[..., 0] == 1) & (L[..., 1] == 1)
    if p11.any():
        ax[3].contour(p11, [0.5], colors=INK, linewidths=0.8,
                      origin="lower", extent=ext)
        d = distance_transform_edt(np.pad(p11, 1))[1:-1, 1:-1]
        r, c = np.unravel_index(np.argmax(d), d.shape)
        x = ext[0] + c * dx
        y = ext[2] + r * dy
        ok = (cs[r, c] == (1, 1)).all()
        ax[3].plot(x, y, "*", ms=9, color="#f2c200" if ok else FP,
                   mec=INK, mew=0.5)
        ax[3].set_title(f"(d) landing: {'in' if ok else 'not in'} (1,1)")
    else:
        ax[3].set_title("(d) landing: no (1,1) predicted")
    for i in range(4):
        C._map_axes(ax[i], i == 0)
    # per-state accuracy in the few-electron region
    sel = m & roi
    tcode = C.state_code(cs)
    st = np.unique(tcode[sel])
    accs = [right[sel & (tcode == s)].mean() for s in st]
    labs = [f"{s // 1000},{s % 1000}" for s in st]
    ax[4].bar(range(len(st)), [100 * a for a in accs], color="#999999")
    ax[4].set_xticks(range(len(st)))
    ax[4].set_xticklabels(labs, fontsize=5.5)
    ax[4].axhline(100 * few["bal_few"], color=INK, lw=0.8, ls="--")
    ax[4].set_ylim(0, 100)
    ax[4].set_xlabel(f"true state $(n_1,n_2)$, $n_1,n_2\\leq{NMAX}$",
                     fontsize=6)
    ax[4].set_ylabel("accuracy (%)", fontsize=6)
    ax[4].set_title(f"(e) per-state accuracy; balanced = their mean "
                    f"(dashed) = {100 * few['bal_few']:.0f} %")
    ax[4].tick_params(labelsize=5.5)
    fig.savefig(os.path.join(FIG, "eval_states.pdf"))
    fig.savefig(os.path.join(FIG, "eval_states.png"), dpi=200)
    plt.close(fig)
    return sc, few, float(m.mean()), int(len(st))


# ── uncertainty: bootstrap over the test devices ───────────────────────────
def bootstrap(values, n=10000, seed=0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v))).mean(1)
    return 100 * v.mean(), 100 * np.percentile(means, 2.5), \
        100 * np.percentile(means, 97.5)


def ci_table():
    rows = read(os.path.join(RUN_DIR, "few_electron", "per_device.csv"))
    r_, p_ = CELL.split("_")[0], CELL.split("_")[2]
    sel = [("all (0,0)", "", ""), ("dense, no ML", "", ""),
           ("U-Net", r_, p_), ("U-Net + line fit", r_, p_)]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"method & balanced few (\%) & (1,1) landing (\%) & "
             r"whole-window recall (\%) \\", r"\midrule"]
    out = {}
    for m, ra, pa in sel:
        rs = [r for r in rows if r["method"] == m and r["rays"] == ra
              and r["points"] == pa]
        cells_ = []
        for k in ("bal_few", "land_11", "state_recall"):
            mu, lo, hi = bootstrap([float(r[k]) for r in rs])
            cells_.append(f"{mu:.1f} [{lo:.1f}, {hi:.1f}]")
            out[(m, k)] = (mu, lo, hi)
        lab = m + (f" {ra}$\\times${pa}" if ra else "")
        lines.append(" & ".join([lab] + cells_) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "ci_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def main():
    rows, k = line_metrics()
    line_table(rows)
    dev = C.read_device(os.path.join(POOL, DEVICE))
    cs = C.load_states(os.path.join(POOL, DEVICE))
    lm = fig_line(k, dev)
    scan = read(os.path.join(RUN_DIR, "few_electron", "fit_scan.csv"))
    txt = next(r["setting"] for r in scan if r["chosen"] == "1")
    fit_knob = next(kk for kk in C.FIT_GRID if C.fit_knob(kk) == txt)
    sc, few, scored, nst = fig_states(k, dev, cs, fit_knob)
    ci = ci_table()

    with open(os.path.join(RUN_DIR, CELL, "config.json")) as f:
        cfg = json.load(f)
    zt = np.load(os.path.join(RUN_DIR, CELL, "train.npz"), allow_pickle=True)
    n_train = len(zt["Y"])
    n_val = max(1, int(grid_train.VAL_FRACTION * n_train))
    n_test = len(np.load(os.path.join(RUN_DIR, CELL, "test.npz"),
                         allow_pickle=True)["Y"])
    me = next(r for r in rows if r["cell"] == CELL)
    dscan = read(os.path.join(RUN_DIR, "charge_states", "dense_scan.csv"))
    v = {
        "Run": RUN.replace("_", r"\_"), "Device": DEVICE.replace("_", r"\_"),
        "Bud": "8 $\\times$ 60", "NTrain": str(n_train),
        "NVal": str(n_val), "NFit": str(n_train - n_val),
        "NTest": str(n_test), "NPool": str(n_train + n_test),
        "ValFrac": f"{100 * grid_train.VAL_FRACTION:.0f}",
        "Thresholds": ", ".join(f"{t:g}" for t in grid_train.THRESHOLDS),
        "Thr": f"{k['thr']:g}", "Nmax": str(NMAX),
        "LPzero": f"{lm[0]['precision']:.2f}",
        "LRzero": f"{lm[0]['recall']:.2f}", "LFzero": f"{lm[0]['f1']:.2f}",
        "LPone": f"{lm[1]['precision']:.2f}",
        "LRone": f"{lm[1]['recall']:.2f}", "LFone": f"{lm[1]['f1']:.2f}",
        "MeFone": f"{me['f1@1']:.3f}",
        "MeFoneMed": f"{me['f1_1_median']:.3f}",
        "MePred": f"{100 * me['pred_frac']:.1f}",
        "MeTrue": f"{100 * me['true_frac']:.1f}",
        "MeCov": f"{100 * me['coverage']:.2f}",
        "MeCovOne": f"{100 * me['cov1']:.1f}",
        "MeAcc": f"{me['accuracy@0']:.3f}",
        "NoLineAcc": f"{100 * (1 - me['true_frac']):.1f}",
        "DevAccRel": f"{100 * sc['acc_rel']:.1f}",
        "DevRec": f"{100 * sc['state_recall']:.0f}",
        "DevBal": f"{100 * few['bal_few']:.0f}",
        "DevAccFew": f"{100 * few['acc_few']:.1f}",
        "DevLand": "yes" if few["land_11"] == 1 else "no",
        "DevScored": f"{100 * scored:.1f}", "DevNSt": str(nst),
        "NDense": str(len(dscan)), "NFitSet": str(len(scan) - 1),
        "FitKnob": txt,
        "Model": me["model"].replace("_", r"\_"),
    }
    for (m, kk), (mu, lo, hi) in ci.items():
        tag = {"all (0,0)": "Base", "dense, no ML": "Dense",
               "U-Net": "U", "U-Net + line fit": "F"}[m]
        kt = {"bal_few": "Bal", "land_11": "Land",
              "state_recall": "Rec"}[kk]
        v[f"{tag}{kt}"] = f"{mu:.1f}"
        v[f"{tag}{kt}Lo"] = f"{lo:.1f}"
        v[f"{tag}{kt}Hi"] = f"{hi:.1f}"
    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as f:
        f.write("% written by make_eval.py -- do not edit\n")
        for kk, x in v.items():
            f.write(f"\\newcommand{{\\v{kk}}}{{{x}}}\n")
    shutil.copy2(os.path.join(HERE, "evaluation.tex"),
                 os.path.join(OUT, "evaluation.tex"))
    for kk in sorted(v):
        print(f"  {kk:10s} {v[kk]}")


if __name__ == "__main__":
    main()
