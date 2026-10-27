# -*- coding: utf-8 -*-
"""
make_evidence.py -- the numbers for evidence.pdf: what was done, and the
evidence behind every claim, with the file each number comes from.

    python docs/charge_states/make_report.py      # first (its values)
    python docs/charge_states/make_evidence.py
    pdflatex evidence.tex  (twice, in the output folder)

Every number is read from the study files or from make_report's values;
the test-suite result comes from running pytest here.
"""
import csv
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
RUN_DIR = os.path.join(ROOT, "results", RUN)
OUT = os.path.join(RUN_DIR, "evidence_note")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)


def read(*p):
    with open(os.path.join(RUN_DIR, *p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    # everything make_report computed
    with open(os.path.join(RUN_DIR, "report_note", "values.tex"),
              encoding="utf-8") as f:
        rep = f.read()
    v = {}

    # the labelling on the true line maps (run_16, whole window, test)
    s16 = {r["method"]: r for r in read("charge_states", "summary.csv")
           if r["method"] in ("full CSD", "dense, no ML")}
    v["FullAccRel"] = f"{float(s16['full CSD']['acc_rel_mean']):.1f}"
    v["FullRecall"] = f"{float(s16['full CSD']['state_recall_mean']):.1f}"
    v["FullCons"] = f"{float(s16['full CSD']['consistency_mean']):.2f}"
    sv = {r["method"]: r for r in read("charge_states",
                                       "summary_validation.csv")}
    v["FullAccRelVal"] = f"{float(sv['full CSD']['acc_rel_mean']):.1f}"
    s17 = {(r["method"], r["rays"], r["points"]): r
           for r in read("few_electron", "summary.csv")}
    v["FullBalFew"] = f"{float(s17[('full CSD', '', '')]['bal_few_mean']):.1f}"
    v["FullLandFew"] = f"{float(s17[('full CSD', '', '')]['land_11_mean']):.1f}"

    # Step 0 sign convention, from the per-device variation columns
    var = [r for r in read("charge_states", "per_device.csv")
           if r["method"] == "full CSD"]
    for split in ("test", "validation"):
        rs = [r for r in var if r["split"] == split]
        tag = "T" if split == "test" else "V"
        v[f"N{tag}"] = str(len(rs))
        v[f"Corner{tag}"] = str(sum(int(float(r["corner_is_00"]))
                                    for r in rs))
        v[f"SignOne{tag}"] = str(sum(int(float(r["n1_grows_to_neg_V1"]))
                                     for r in rs))
        v[f"SignTwo{tag}"] = str(sum(int(float(r["n2_grows_to_neg_V2"]))
                                     for r in rs))

    # scans: how many settings, what was chosen
    v["NDenseSet"] = str(len(read("charge_states", "dense_scan.csv")))
    fs = read("few_electron", "fit_scan.csv")
    v["FitValWith"] = f"{100 * float(next(r for r in fs if r['chosen'] == '1')['bal_few_mean_validation']):.1f}"
    v["FitValWithout"] = f"{100 * float(next(r for r in fs if r['setting'].startswith('no line'))['bal_few_mean_validation']):.1f}"
    for route in ("rays", "coarse", "native"):
        sc = read("equal_cost", f"scan_{route}.csv")
        best = next(r for r in sc if r["chosen"] == "1")
        v[f"Scan{route.capitalize()}N"] = str(len(sc))
        v[f"Scan{route.capitalize()}Best"] = \
            f"{100 * float(best['bal_few_mean_validation']):.1f}"

    # the test suite, run now
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) passed", p.stdout)
    f_ = re.search(r"(\d+) failed", p.stdout)
    v["TestsPassed"] = m.group(1) if m else "0"
    v["TestsFailed"] = f_.group(1) if f_ else "0"
    tc = open(os.path.join(ROOT, "tests", "test_charge_states.py"),
              encoding="utf-8").read().count("\ndef test_")
    v["TestsCharge"] = str(tc)

    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as f:
        f.write("% written by make_evidence.py -- do not edit\n")
        f.write(rep.split("\n", 1)[1])          # make_report's values
        for k, x in v.items():
            f.write(f"\\newcommand{{\\v{k}}}{{{x}}}\n")
    for n in ("equal_cost_curves.pdf", "f1_vs_states.pdf"):
        shutil.copy2(os.path.join(RUN_DIR, "report_note", "figures", n),
                     os.path.join(OUT, "figures", n))
    shutil.copy2(os.path.join(RUN_DIR, "report_note", "table_equal.tex"),
                 os.path.join(OUT, "table_equal.tex"))
    shutil.copy2(os.path.join(HERE, "evidence.tex"),
                 os.path.join(OUT, "evidence.tex"))
    for k in sorted(v):
        print(f"  {k:16s} {v[k]}")


if __name__ == "__main__":
    main()
