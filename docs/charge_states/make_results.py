# -*- coding: utf-8 -*-
"""
make_results.py -- tables, one figure and the numbers for
charge_states_results.pdf.

    python scripts/run_16_charge_states.py           # the study
    python docs/charge_states/make_results.py        # this note's inputs
    pdflatex results.tex  (twice, in the output folder)

Reads only what run_16 wrote (summary*.csv, per_device.csv, figures) and the
device pool; every number in the note is written here into values.tex.
"""
import csv
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from csdrecon.study import charge_states as C          # noqa: E402

RUNS = {"A": ("4-5-6-7-8_rays_40-50-60_points_500_samples", (8, 60)),
        "B": ("4-6-8-12-16-24_rays_40-60-100_points_500_samples", (8, 40))}
# a sweep whose results folder is gone is left out, not re-made
RUNS = {k: r for k, r in RUNS.items()
        if os.path.isfile(os.path.join(ROOT, "results", r[0], "charge_states",
                                       "summary.csv"))}
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")
OUT = os.path.join(ROOT, "results", RUNS["A"][0], "charge_states",
                   "results_note")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
INK, MUTED, BLUE, ORANGE = "#111111", "#555555", "#2a78d6", "#eb6834"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def study(run):
    return os.path.join(ROOT, "results", run, "charge_states")


def baseline(run, split):
    """Per device: fraction of scored pixels whose true state is (0,0)."""
    rows = [r for r in read(os.path.join(study(run), "per_device.csv"))
            if r["split"] == split and r["method"] == "full CSD"]
    out = []
    for r in rows:
        d = os.path.join(POOL, r["device"])
        s = C.load_states(d)
        y = np.load(os.path.join(d, "numpy", "simulation",
                                 "ground_truth_labels.npy")) > 0.5
        m = C.keep_mask(y)
        out.append(((s[..., 0] == 0) & (s[..., 1] == 0))[m].mean())
    return np.array(out)


def per_device(run, split, rays, points):
    return [r for r in read(os.path.join(study(run), "per_device.csv"))
            if r["split"] == split and r["method"] == "U-Net"
            and r["rays"] == str(rays) and r["points"] == str(points)]


COLS = [("acc_rel", "pixel acc.\\ (\\%)", 1, 1),
        ("state_recall", "state recall (\\%)", 1, 1),
        ("state_precision", "state prec.\\ (\\%)", 1, 1),
        ("iou_11", "(1,1) IoU", 1, 2),
        ("consistency", "consistency", 1, 2)]


def compact_table(run, split, base):
    """A readable LaTeX table: mean [median] per column, test or val."""
    name = "summary.csv" if split == "test" else "summary_validation.csv"
    rows = read(os.path.join(study(run), name))
    head = (["method", r"rays $\times$ pts", r"coverage (\%)"]
            + [c[1] for c in COLS])
    lines = [r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}",
             r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    lines.append(" & ".join(
        ["all (0,0)", "--", "0"]
        + [f"{100 * base.mean():.1f} [{100 * np.median(base):.1f}]"]
        + ["--"] * (len(COLS) - 1)) + r" \\")
    lines.append(r"\midrule")
    first_unet = True
    for r in rows:
        unet = r["method"] == "U-Net"
        cells = [r["method"],
                 f"{r['rays']} $\\times$ {r['points']}" if unet else "--",
                 f"{float(r['coverage_%']):.2f}"]
        for k, _, _, d in COLS:
            cells.append(f"{float(r[k + '_mean']):.{d}f} "
                         f"[{float(r[k + '_median']):.{d}f}]")
        if unet and first_unet:
            lines.append(r"\midrule")
            first_unet = False
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


DENSE = "dense, no ML"
BINS = ((0, 12), (13, 20), (21, 40), (41, 10 ** 6))


def dense_choice(run):
    """(knob tuple, its text) the study chose on the validation devices."""
    rows = read(os.path.join(study(run), "dense_scan.csv"))
    txt = next(r["setting"] for r in rows if r["chosen"] == "1")
    knob = next(k for k in C.DENSE_GRID if C.dense_knob(k) == txt)
    return knob, txt


def density_table(run, ra, pa, bmap):
    """U-Net and dense accuracy by the number of states in the window."""
    rows = [r for r in read(os.path.join(study(run), "per_device.csv"))
            if r["split"] == "test"]
    un = [r for r in rows if r["method"] == "U-Net"
          and r["rays"] == str(ra) and r["points"] == str(pa)]
    dn = {r["device"]: r for r in rows if r["method"] == DENSE}
    head = ["states in window", "devices", r"all (0,0) acc.\ (\%)",
            f"U-Net {ra}$\\times${pa} acc.\\ (\\%)",
            r"U-Net recall (\%)", r"dense acc.\ (\%)",
            r"dense recall (\%)"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]

    def m(vals):
        return 100 * float(np.mean(vals))
    for lo, hi in BINS:
        sel = [r for r in un if lo <= float(r["n_states"]) <= hi]
        if not sel:
            continue
        lab = f"{lo}--{hi}" if hi < 10 ** 6 else f"$>${lo - 1}"
        lines.append(" & ".join([
            lab, str(len(sel)),
            f"{m([bmap[r['device']] for r in sel]):.1f}",
            f"{m([float(r['acc_rel']) for r in sel]):.1f}",
            f"{m([float(r['state_recall']) for r in sel]):.1f}",
            f"{m([float(dn[r['device']]['acc_rel']) for r in sel]):.1f}",
            f"{m([float(dn[r['device']]['state_recall']) for r in sel]):.1f}",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def fig_dense(run):
    """What the dense detector sees and draws, three test devices."""
    from csdrecon.ml.ray_peaks import load_grid
    from matplotlib.colors import ListedColormap
    knob, _ = dense_choice(run)
    names = ["sample_550", "sample_223", "sample_11"]
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6),
                            gridspec_kw=dict(wspace=0.08, hspace=0.42,
                                             left=0.08, right=0.995,
                                             top=0.93, bottom=0.07))
    titles = ["(a) dense sensor image", "(b) step residual",
              "(c) line map from (b)", "(d) labels from (c)"]
    for i, n in enumerate(names):
        d = os.path.join(POOL, n)
        Z = load_grid(d)[2]
        dev = C.read_device(d)
        cs = C.load_states(d)
        y = np.load(os.path.join(d, "numpy", "simulation",
                                 "ground_truth_labels.npy")) > 0.5
        dx, dy = C.pixel_mV(dev, y.shape)
        B = C.dense_line_map(Z, knob)
        res = C.label_states(B, dx, dy)
        sc = C.score(res, cs, y, dx, dy)
        ext = dev["voltage_window"]
        kw = dict(origin="lower", extent=ext, interpolation="nearest")
        ax = axs[i]
        ax[0].imshow(Z, cmap="viridis", **kw)
        R = C.step_residual_map(Z, knob[1], bool(knob[2]))
        ax[1].imshow(R, cmap="magma", vmax=np.quantile(R, 0.98), **kw)
        ax[2].imshow(B, cmap=ListedColormap(["white", INK]), vmin=0,
                     vmax=1, **kw)
        L = res["labels"]
        sh = C.best_shift(L, cs, C.keep_mask(y))
        Ls = np.where(L == -999, -999, L + sh)
        ax[3].imshow(C._state_rgb(Ls), **kw)
        C._write_labels(ax[3], Ls, ext, fs=4)
        for c, a in enumerate(ax):
            C._map_axes(a, c == 0)
            if i == 0:
                a.set_title(titles[c], fontsize=7, pad=12)
        ax[0].text(0, 1.03, f"{n}: accuracy {100 * sc['acc_rel']:.1f} %, "
                   f"state recall {100 * sc['state_recall']:.0f} %",
                   transform=ax[0].transAxes, fontsize=6.3, va="bottom")
    fig.savefig(os.path.join(FIG, "dense_examples.pdf"))
    fig.savefig(os.path.join(FIG, "dense_examples.png"), dpi=200)
    plt.close(fig)


def fig_both(bases):
    """Pixel accuracy and state recall vs coverage, both sweeps, baseline."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for key, col, mk in (("A", BLUE, "o"), ("B", ORANGE, "s")):
        if key not in RUNS:
            continue
        run = RUNS[key][0]
        rows = [r for r in read(os.path.join(study(run), "summary.csv"))
                if r["method"] == "U-Net"]
        x = [float(r["coverage_%"]) for r in rows]
        for a, k in ((ax[0], "acc_rel"), (ax[1], "state_recall")):
            a.plot(x, [float(r[k + "_mean"]) for r in rows], mk, ms=4,
                   color=col, label=f"U-Net (mean), one point per budget")
    dn = [r for r in read(os.path.join(study(RUNS["A"][0]), "summary.csv"))
          if r["method"] == DENSE][0]
    for a, k in ((ax[0], "acc_rel"), (ax[1], "state_recall")):
        a.plot(100, 100, "*", ms=9, color=INK, label="full CSD $Y$")
        a.plot(100, float(dn[k + "_mean"]), "D", ms=5, color=ORANGE,
               label="dense sensor image, no ML (mean)")
    b = bases["A"]
    ax[0].axhline(100 * b.mean(), color=MUTED, ls="--", lw=0.9)
    ax[0].text(12, 100 * b.mean() - 6, "label everything (0,0)",
               fontsize=6.5, color=MUTED)
    for a, lab in ((ax[0], "pixel accuracy, relative (%)"),
                   (ax[1], "state recall (%)")):
        a.set_xscale("log")
        t = [1, 2, 5, 10, 20, 50, 100]
        a.set_xticks(t)
        a.set_xticklabels([str(v) for v in t])
        a.set_xlim(1, 140)
        a.set_ylim(0, 104)
        a.set_xlabel("fraction of the grid measured (%)", fontsize=8)
        a.set_ylabel(lab, fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.6)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "both_sweeps.pdf"))
    fig.savefig(os.path.join(FIG, "both_sweeps.png"), dpi=200)
    plt.close(fig)


def main():
    bases = {k: baseline(RUNS[k][0], "test") for k in RUNS}
    bval = {k: baseline(RUNS[k][0], "validation") for k in RUNS}
    v = {"BaseMean": f"{100 * bases['A'].mean():.1f}",
         "BaseStd": f"{100 * bases['A'].std():.1f}",
         "BaseMed": f"{100 * np.median(bases['A']):.1f}"}
    for key, (run, (ra, pa)) in RUNS.items():
        with open(os.path.join(OUT, f"table_{key}_test.tex"), "w",
                  encoding="utf-8") as f:
            f.write(compact_table(run, "test", bases[key]))
        with open(os.path.join(OUT, f"table_{key}_val.tex"), "w",
                  encoding="utf-8") as f:
            f.write(compact_table(run, "validation", bval[key]))
        S = read(os.path.join(study(run), "summary.csv"))
        full = S[0]
        dn = next(r for r in S if r["method"] == DENSE)
        un = sorted([r for r in S if r["method"] == "U-Net"],
                    key=lambda r: float(r["coverage_%"]))
        best = max(un, key=lambda r: float(r["acc_rel_mean"]))
        lo, hi = un[0], un[-1]
        pd = per_device(run, "test", ra, pa)
        acc = np.array([float(r["acc_rel"]) for r in pd])
        b = bases[key]
        names = [r["device"] for r in pd]
        # beats the baseline: the per-device baseline is in the same order
        full_rows = [r for r in read(os.path.join(study(run),
                                                  "per_device.csv"))
                     if r["split"] == "test" and r["method"] == "full CSD"]
        border = [r["device"] for r in full_rows]
        bmap = dict(zip(border, b))
        beat = sum(a > bmap[n] for a, n in zip(acc, names))
        g = lambda r, k: float(r[k])                        # noqa: E731
        v.update({
            f"{key}Run": run.replace("_", r"\_"),
            f"{key}N": str(len(un)),
            f"{key}FullAcc": f"{g(full, 'acc_rel_mean'):.1f}",
            f"{key}FullRec": f"{g(full, 'state_recall_mean'):.1f}",
            f"{key}LoBud": f"{lo['rays']} $\\times$ {lo['points']}",
            f"{key}LoCov": f"{g(lo, 'coverage_%'):.2f}",
            f"{key}LoAcc": f"{g(lo, 'acc_rel_mean'):.1f}",
            f"{key}LoRec": f"{g(lo, 'state_recall_mean'):.1f}",
            f"{key}HiBud": f"{hi['rays']} $\\times$ {hi['points']}",
            f"{key}HiCov": f"{g(hi, 'coverage_%'):.2f}",
            f"{key}HiAcc": f"{g(hi, 'acc_rel_mean'):.1f}",
            f"{key}HiAccMed": f"{g(hi, 'acc_rel_median'):.1f}",
            f"{key}HiRec": f"{g(hi, 'state_recall_mean'):.1f}",
            f"{key}HiRecMed": f"{g(hi, 'state_recall_median'):.1f}",
            f"{key}HiIoU": f"{g(hi, 'iou_11_mean'):.2f}",
            f"{key}HiIoUMed": f"{g(hi, 'iou_11_median'):.2f}",
            f"{key}HiCon": f"{g(hi, 'consistency_mean'):.2f}",
            f"{key}BestBud": f"{best['rays']} $\\times$ {best['points']}",
            f"{key}BestAcc": f"{g(best, 'acc_rel_mean'):.1f}",
            f"{key}FigBud": f"{ra} $\\times$ {pa}",
            f"{key}Beat": str(int(beat)),
            f"{key}NDev": str(len(acc)),
            f"{key}AccMin": f"{100 * acc.min():.1f}",
            f"{key}AccMax": f"{100 * acc.max():.1f}",
            f"{key}DnAcc": f"{g(dn, 'acc_rel_mean'):.1f}",
            f"{key}DnAccMed": f"{g(dn, 'acc_rel_median'):.1f}",
            f"{key}DnRec": f"{g(dn, 'state_recall_mean'):.1f}",
            f"{key}DnRecMed": f"{g(dn, 'state_recall_median'):.1f}",
            f"{key}DnIoU": f"{g(dn, 'iou_11_mean'):.2f}",
            f"{key}DnCon": f"{g(dn, 'consistency_mean'):.2f}",
            f"{key}DnKnob": dense_choice(run)[1].replace("_", r"\_"),
            f"{key}DnNSet": str(len(C.DENSE_GRID)),
        })
        with open(os.path.join(OUT, f"table_{key}_density.tex"), "w",
                  encoding="utf-8") as f:
            f.write(density_table(run, ra, pa, bmap))
        fig_dense(run)
        for n in ("fig2_labelling_examples", "fig4_failures"):
            shutil.copy2(os.path.join(study(run), "figures", n + ".pdf"),
                         os.path.join(FIG, f"{key}_{n}.pdf"))
    shutil.copy2(os.path.join(study(RUNS["A"][0]), "figures",
                              "fig1_device_variation.pdf"),
                 os.path.join(FIG, "fig1_device_variation.pdf"))
    fig_both(bases)
    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as f:
        f.write("% written by make_results.py -- do not edit\n")
        for k, x in v.items():
            f.write(f"\\newcommand{{\\v{k}}}{{{x}}}\n")
    for k in sorted(v):
        print(f"  {k:12s} {v[k]}")


if __name__ == "__main__":
    main()
