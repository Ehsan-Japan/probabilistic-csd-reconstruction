# -*- coding: utf-8 -*-
"""
make_report.py -- the inputs of charge_states_report.pdf, the full account
of the charge-state work: the question, what was done and why, the results,
what they mean, and whether they are a contribution.

    python docs/charge_states/make_report.py
    pdflatex report.tex  (twice, in the output folder)

Reads what run_16, run_17, make_note, make_results, make_eval and
fig_fit_steps wrote; computes one new figure (line-level F1 against
charge-state accuracy, per device).  Every number in the report is written
here into values.tex.
"""
import csv
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

from csdrecon.ml import grid_train                      # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1         # noqa: E402
from csdrecon.study import charge_states as C           # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402
from run_17_few_electron import NMAX                    # noqa: E402
from make_eval import bootstrap                         # noqa: E402

RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
CELL = "8_rays_60_points_500_samples"
RA, PA = "8", "60"
RUN_DIR = os.path.join(ROOT, "results", RUN)
OUT = os.path.join(RUN_DIR, "report_note")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")
BLUE, GREEN, INK, MUTED = "#2a78d6", "#1baf7a", "#111111", "#555555"


def read(*p):
    with open(os.path.join(RUN_DIR, *p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1_vs_states():
    """Per test device: F1@1 of the U-Net map against the few-electron
    balanced accuracy of the labels from it (and after the line fit)."""
    z = np.load(os.path.join(RUN_DIR, CELL, "test.npz"), allow_pickle=True)
    names = [os.path.basename(str(s).replace("\\", "/"))
             for s in z["samples"]]
    net, ck = grid_train.load(os.path.join(RUN_DIR, CELL,
                                           model_dir(RUN_DIR, CELL),
                                           "unet.pt"))
    B = grid_train.predict(net, z["X"]) > float(ck["threshold"])
    f1 = {n: tolerant_f1(B[i], z["Y"][i] > 0.5, 1.0)["f1"]
          for i, n in enumerate(names)}
    rows = read("few_electron", "per_device.csv")
    pick = lambda m: {r["device"]: float(r["bal_few"]) for r in rows  # noqa
                      if r["method"] == m and r["rays"] == RA
                      and r["points"] == PA}
    u, fit = pick("U-Net"), pick("U-Net + line fit")
    x = np.array([f1[n] for n in names])
    yu = np.array([u[n] for n in names])
    yf = np.array([fit[n] for n in names])
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    ax.plot(x, 100 * yu, "o", ms=3.5, color=BLUE, mfc="none", mew=0.8,
            label="U-Net map")
    ax.plot(x, 100 * yf, "s", ms=3.2, color=GREEN, mfc="none", mew=0.8,
            label="after line fit")
    ax.set_xlabel("line-level F1@1 of the U-Net map", fontsize=7)
    ax.set_ylabel("balanced accuracy, few-e (%)", fontsize=7)
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "f1_vs_states.pdf"))
    fig.savefig(os.path.join(FIG, "f1_vs_states.png"), dpi=250)
    plt.close(fig)
    hi = x >= 0.85
    return {"RhoU": f"{spearmanr(x, yu)[0]:.2f}",
            "RhoF": f"{spearmanr(x, yf)[0]:.2f}",
            "FoneMin": f"{x.min():.2f}", "FoneMax": f"{x.max():.2f}",
            "NHiFone": str(int(hi.sum())),
            "HiFoneBalMin": f"{100 * yu[hi].min():.0f}",
            "HiFoneBalMax": f"{100 * yu[hi].max():.0f}"}


def compact_table(S):
    """The rows the report discusses, from run_17's summary."""
    keep = [("all (0,0)", "", ""), ("dense, no ML", "", ""),
            ("full CSD", "", ""), ("U-Net", "4", "40"),
            ("U-Net + line fit", "4", "40"), ("U-Net", RA, PA),
            ("U-Net + line fit", RA, PA)]
    by = {(r["method"], r["rays"], r["points"]): r for r in S}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"method & rays $\times$ pts & grid measured (\%) & "
             r"balanced acc.\ (\%) & (1,1) landing (\%) & "
             r"whole-window recall (\%) \\", r"\midrule"]
    for i, k in enumerate(keep):
        r = by[k]
        if i in (3,):
            lines.append(r"\midrule")
        lines.append(" & ".join([
            k[0], f"{k[1]} $\\times$ {k[2]}" if k[1] else "--",
            f"{float(r['coverage_%']):.2f}",
            f"{float(r['bal_few_mean']):.1f}",
            f"{float(r['land_11_mean']):.1f}",
            f"{float(r['state_recall_mean']):.1f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table_main.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return by


COMPETITORS = ("rays + interpolation", "rays + interpolation + line fit",
               "coarse scan, upsampled", "coarse scan, upsampled + line fit",
               "coarse scan, steps on the coarse grid",
               "coarse scan, steps on the coarse grid + line fit")


def equal_cost(v):
    """Numbers of the equal-cost comparison (run_18), and a paired
    bootstrap of U-Net+fit minus the best competitor at every budget."""
    S = read("equal_cost", "summary.csv")
    rows = read("equal_cost", "per_device.csv")
    by = {(r["method"], r["rays"], r["points"]): r for r in S}
    budgets = sorted({(r["rays"], r["points"]) for r in S if r["rays"]},
                     key=lambda b: int(b[0]) * int(b[1]))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"rays $\times$ pts & U-Net & U-Net + fit & best competitor "
             r"& which & difference [95\,\% CI] \\", r"\midrule"]
    wins = raw = 0
    for ra, pa in budgets:
        comp = max(COMPETITORS, key=lambda m: float(
            by[(m, ra, pa)]["bal_few_mean"]))
        dev = lambda m: {r["device"]: float(r["bal_few"])    # noqa: E731
                         for r in rows if r["method"] == m
                         and r["rays"] == ra and r["points"] == pa}
        a, b = dev("U-Net + line fit"), dev(comp)
        d = [a[k] - b[k] for k in a if np.isfinite(a[k])
             and np.isfinite(b[k])]
        mu, lo, hi = bootstrap(d)
        wins += lo > 0
        raw += (float(by[("U-Net", ra, pa)]["bal_few_mean"])
                > float(by[(comp, ra, pa)]["bal_few_mean"]))
        short = (comp.replace("coarse scan, steps on the coarse grid",
                              "coarse, native")
                 .replace("coarse scan, upsampled", "coarse, upsampled")
                 .replace("rays + interpolation", "rays interp."))
        lines.append(" & ".join([
            f"{ra} $\\times$ {pa}",
            f"{float(by[('U-Net', ra, pa)]['bal_few_mean']):.1f}",
            f"{float(by[('U-Net + line fit', ra, pa)]['bal_few_mean']):.1f}",
            f"{float(by[(comp, ra, pa)]['bal_few_mean']):.1f}", short,
            f"{mu:+.1f} [{lo:+.1f}, {hi:+.1f}]"]) + r" \\")
        if (ra, pa) == (RA, PA):
            v.update({"EqComp": short, "EqCompBal": f"{float(by[(comp, ra, pa)]['bal_few_mean']):.1f}",
                      "EqDiff": f"{mu:.1f}", "EqDiffLo": f"{lo:.1f}",
                      "EqDiffHi": f"{hi:.1f}"})
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table_equal.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    v.update({"EqWins": str(int(wins)), "EqNBud": str(len(budgets)),
              "EqRawWins": str(int(raw))})
    g = lambda m, k: f"{float(by[(m, RA, PA)][k]):.1f}"            # noqa
    for tag, m in (("EqRays", "rays + interpolation"),
                   ("EqRaysFit", "rays + interpolation + line fit"),
                   ("EqUp", "coarse scan, upsampled"),
                   ("EqNat", "coarse scan, steps on the coarse grid"),
                   ("EqNatFit",
                    "coarse scan, steps on the coarse grid + line fit")):
        v[f"{tag}Bal"] = g(m, "bal_few_mean")
        v[f"{tag}Land"] = g(m, "land_11_mean")
    v["EqK"] = str(int(np.floor(np.sqrt(int(RA) * int(PA)))))
    v["EqNPts"] = str(int(RA) * int(PA))
    with open(os.path.join(RUN_DIR, "equal_cost", "summary.md"),
              encoding="utf-8") as f:
        cap = f.readline()
    v["EqChosen"] = cap.split("Settings chosen on validation: ")[1] \
        .rstrip(".\n").replace("_", r"\_")
    for n in ("equal_cost_curves.pdf", "equal_cost_examples.pdf"):
        shutil.copy2(os.path.join(RUN_DIR, "equal_cost", "figures", n),
                     os.path.join(FIG, n))


def main():
    v = {"Run": RUN.replace("_", r"\_"), "Nmax": NMAX}
    equal_cost(v)
    v.update(f1_vs_states())

    S = read("few_electron", "summary.csv")
    by = compact_table(S)
    g = lambda k, c: float(by[k][c])                              # noqa
    U, F = ("U-Net", RA, PA), ("U-Net + line fit", RA, PA)
    U4, F4 = ("U-Net", "4", "40"), ("U-Net + line fit", "4", "40")
    D, B0 = ("dense, no ML", "", ""), ("all (0,0)", "", "")
    for tag, k in (("U", U), ("F", F), ("Ufour", U4), ("Ffour", F4),
                   ("D", D), ("Base", B0)):
        v[f"{tag}Bal"] = f"{g(k, 'bal_few_mean'):.1f}"
        v[f"{tag}Land"] = f"{g(k, 'land_11_mean'):.1f}"
        v[f"{tag}Rec"] = f"{g(k, 'state_recall_mean'):.1f}"
        v[f"{tag}Cov"] = f"{g(k, 'coverage_%'):.2f}"
    un = [r for r in S if r["method"] == "U-Net"]
    fi = {(r["rays"], r["points"]): r for r in S
          if r["method"] == "U-Net + line fit"}
    gains = [float(fi[(r["rays"], r["points"])]["bal_few_mean"])
             - float(r["bal_few_mean"]) for r in un]
    v.update({"GainMin": f"{min(gains):.0f}", "GainMax": f"{max(gains):.0f}",
              "NBud": str(len(un)),
              "BestLand": f"{max(float(r['land_11_mean']) for r in fi.values()):.0f}",
              "CovLo": f"{min(float(r['coverage_%']) for r in un):.2f}",
              "CovHi": f"{max(float(r['coverage_%']) for r in un):.2f}",
              "Fold": f"{100 / g(U, 'coverage_%'):.0f}"})

    # confidence intervals from the per-device rows
    rows = read("few_electron", "per_device.csv")
    for tag, (m, ra, pa) in (("U", U), ("F", F), ("D", D)):
        vals = [float(r["bal_few"]) for r in rows if r["method"] == m
                and r["rays"] == ra and r["points"] == pa]
        mu, lo, hi = bootstrap(vals)
        v[f"{tag}BalLo"], v[f"{tag}BalHi"] = f"{lo:.1f}", f"{hi:.1f}"

    # whole-window numbers and the device variation, from run_16
    S16 = read("charge_states", "summary.csv")
    b16 = {(r["method"], r["rays"], r["points"]): r for r in S16}
    v["UAccRel"] = f"{float(b16[('U-Net', RA, PA)]['acc_rel_mean']):.1f}"
    v["DAccRel"] = f"{float(b16[('dense, no ML', '', '')]['acc_rel_mean']):.1f}"
    var = [r for r in read("charge_states", "per_device.csv")
           if r["split"] == "test" and r["method"] == "full CSD"]
    ns = np.array([float(r["n_states"]) for r in var])
    v.update({"NStMin": f"{ns.min():.0f}", "NStMax": f"{ns.max():.0f}",
              "NStMed": f"{np.median(ns):.0f}", "NTest": str(len(var))})

    # line level, from make_eval's recomputation with the used checkpoints
    lm = read("evaluation_note", "line_metrics.csv")
    me = next(r for r in lm if r["cell"] == CELL)
    f1s = [float(r["f1@1"]) for r in lm]
    v.update({"LineFone": f"{float(me['f1@1']):.2f}",
              "LineFoneMin": f"{min(f1s):.2f}",
              "LineFoneMax": f"{max(f1s):.2f}"})

    scan = read("charge_states", "dense_scan.csv")
    v["NDense"] = str(len(scan))
    fscan = read("few_electron", "fit_scan.csv")
    v["NFitSet"] = str(len(fscan) - 1)
    v["FitKnob"] = next(r["setting"] for r in fscan if r["chosen"] == "1")

    # figures made by the other notes
    copies = {
        "step1_regions.pdf": ("charge_states", "method_note", "figures"),
        "step3_tree_labels.pdf": ("charge_states", "method_note", "figures"),
        "dense_examples.pdf": ("charge_states", "results_note", "figures"),
        "fit_steps.pdf": ("few_electron", "figures"),
        "fig_few_vs_coverage.pdf": ("few_electron", "figures"),
        "fig_fit_examples.pdf": ("few_electron", "figures"),
        "eval_states.pdf": ("evaluation_note", "figures"),
    }
    for n, p in copies.items():
        shutil.copy2(os.path.join(RUN_DIR, *p, n), os.path.join(FIG, n))
    shutil.copy2(os.path.join(RUN_DIR, "charge_states", "results_note",
                              "table_A_density.tex"),
                 os.path.join(OUT, "table_density.tex"))
    shutil.copy2(os.path.join(HERE, "report.tex"),
                 os.path.join(OUT, "report.tex"))
    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as f:
        f.write("% written by make_report.py -- do not edit\n")
        for k, x in v.items():
            f.write(f"\\newcommand{{\\v{k}}}{{{x}}}\n")
    for k in sorted(v):
        print(f"  {k:12s} {v[k]}")


if __name__ == "__main__":
    main()
