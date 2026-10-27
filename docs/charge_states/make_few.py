# -*- coding: utf-8 -*-
"""
make_few.py -- values.tex and the copied table/figures for few_electron.pdf.

    python scripts/run_17_few_electron.py        # the study
    python docs/charge_states/make_few.py        # this note's inputs
    pdflatex few_electron.tex  (twice, in the output folder)

Every number in the note is read here from run_17's files.
"""
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_17_few_electron import NMAX, RUN, FIGURE_UNET   # noqa: E402

STUDY = os.path.join(ROOT, "results", RUN, "few_electron")
OUT = os.path.join(STUDY, "note")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)


def read(name):
    with open(os.path.join(STUDY, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    S = read("summary.csv")
    scan = read("fit_scan.csv")
    by = {(r["method"], r["rays"], r["points"]): r for r in S}
    un = sorted([r for r in S if r["method"] == "U-Net"],
                key=lambda r: float(r["coverage_%"]))
    fit = {(r["rays"], r["points"]): r for r in S
           if r["method"] == "U-Net + line fit"}
    ra, pa = FIGURE_UNET.split("_")[0], FIGURE_UNET.split("_")[2]
    best_u = max(un, key=lambda r: float(r["bal_few_mean"]))
    best_f = max(fit.values(), key=lambda r: float(r["bal_few_mean"]))
    gains = [float(fit[(r["rays"], r["points"])]["bal_few_mean"])
             - float(r["bal_few_mean"]) for r in un]
    lgain = [float(fit[(r["rays"], r["points"])]["land_11_mean"])
             - float(r["land_11_mean"]) for r in un]
    f = lambda r, k: f"{float(r[k]):.1f}"                     # noqa: E731
    chosen = next(r for r in scan if r["chosen"] == "1")
    plain = next(r for r in scan if r["setting"].startswith("no line fit"))
    v = {
        "Nmax": str(NMAX), "Run": RUN.replace("_", r"\_"),
        "NDev": by[("full CSD", "", "")]["n"],
        "NLand": by[("full CSD", "", "")]["land_11_n"],
        "FitKnob": chosen["setting"],
        "FitVal": f"{100 * float(chosen['bal_few_mean_validation']):.1f}",
        "PlainVal": f"{100 * float(plain['bal_few_mean_validation']):.1f}",
        "NFit": str(len(scan) - 1),
    }
    for tag, key in (("Base", ("all (0,0)", "", "")),
                     ("Dense", ("dense, no ML", "", "")),
                     ("Full", ("full CSD", "", "")),
                     ("U", ("U-Net", ra, pa)),
                     ("F", ("U-Net + line fit", ra, pa))):
        r = by[key]
        for k, n in (("acc_few", "Acc"), ("bal_few", "Bal"),
                     ("land_11", "Land")):
            v[f"{tag}{n}"] = f(r, k + "_mean")
            v[f"{tag}{n}Med"] = f(r, k + "_median")
    v.update({
        "Bud": f"{ra} $\\times$ {pa}",
        "Cov": f"{float(by[('U-Net', ra, pa)]['coverage_%']):.2f}",
        "LoCov": f"{float(un[0]['coverage_%']):.2f}",
        "HiCov": f"{float(un[-1]['coverage_%']):.2f}",
        "GainMin": f"{min(gains):+.1f}", "GainMax": f"{max(gains):+.1f}",
        "GainPos": str(sum(g > 0 for g in gains)), "NBud": str(len(gains)),
        "LGainMin": f"{min(lgain):+.1f}", "LGainMax": f"{max(lgain):+.1f}",
        "BestU": f"{best_u['rays']} $\\times$ {best_u['points']}",
        "BestUBal": f(best_u, "bal_few_mean"),
        "BestF": f"{best_f['rays']} $\\times$ {best_f['points']}",
        "BestFBal": f(best_f, "bal_few_mean"),
        "BestFLand": f(best_f, "land_11_mean"),
    })
    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as fh:
        fh.write("% written by make_few.py -- do not edit\n")
        for k, x in v.items():
            fh.write(f"\\newcommand{{\\v{k}}}{{{x}}}\n")
    shutil.copy2(os.path.join(STUDY, "summary.tex"),
                 os.path.join(OUT, "summary.tex"))
    for n in ("fig_fit_examples", "fig_few_vs_coverage"):
        shutil.copy2(os.path.join(STUDY, "figures", n + ".pdf"),
                     os.path.join(OUT, "figures", n + ".pdf"))
    dn = os.path.join(ROOT, "results", RUN, "charge_states", "results_note",
                      "figures", "dense_examples.pdf")
    if os.path.isfile(dn):
        shutil.copy2(dn, os.path.join(OUT, "figures", "dense_examples.pdf"))
    for k in sorted(v):
        print(f"  {k:10s} {v[k]}")


if __name__ == "__main__":
    main()
