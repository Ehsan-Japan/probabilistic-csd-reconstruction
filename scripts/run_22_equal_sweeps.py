"""
Equally spaced sampling, without the U-Net: the same local sweeps and the
same 3-pixel step detector as step 2 of run_19, but the sweeps run along
every s-th row and every s-th column of the plane instead of around the
binarized U-Net output.

    python scripts/run_19_two_step.py          # table.csv (the U-Net side)
    python scripts/run_22_equal_sweeps.py

The k x k raster of run_19 cannot be equally spaced on the 100 x 100 grid
of the simulation (61 points over 100 pixels fall 1 or 2 pixels apart), and
its points have no measured neighbours, so the local detector cannot run on
it.  Here the period s is an integer, the lines are equally spaced and
centred on the plane, and a pixel on a row is tested along the row (on a
column along the column, at a crossing along both): run_19.sweep_lines
with the measured lines as both the strip and the measured pixels.

For each U-Net setting of run_19, s is the largest period whose lines cover
at least that setting's mean fraction of the plane, so the sampling never
gets fewer measurements.  The detector threshold is chosen on the 75
validation devices by F1@1; the 50 test devices only report.

Writes results/<run>/two_step/equal_sweeps.csv.  (Until 2026-10-25 this
was the baseline of the paper and drew Fig. 17; the baseline is now the
equally spaced seeds of run_23_equal_seeds.py, which draws Fig. 17.)
"""
import csv
import os

import numpy as np

import run_19_two_step as R
from csdrecon.ml import grid_train
from run_7_virtual_gates import model_dir


def lines(shape, s):
    """Every s-th row and column, the set centred on the plane."""
    m = np.zeros(shape, bool)
    for ax, n in enumerate(shape):
        k = (n - 1) // s + 1                       # lines that fit
        idx = (n - 1 - (k - 1) * s) // 2 + s * np.arange(k)
        if ax == 0:
            m[idx, :] = True
        else:
            m[:, idx] = True
    return m


def equal_sweeps(D, i, q, s):
    measured = lines(D["Y"][i].shape, s)
    kept, _, _ = R.sweep_lines(D["Z"][i], measured, q, measured)
    return kept, float(measured.mean())


def main():
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr = float(ck["threshold"])
    val, test = R.load(net, thr, "validation"), R.load(net, thr, "test")
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        refs = [r for r in csv.DictReader(f)
                if r["method"] in ("U-Net only", "U-Net, then local check")]
    shape = test["Y"][0].shape
    rows = []
    for ref in refs:
        need = float(ref["measured"]) / 100
        s = max(s for s in range(1, shape[0])
                if lines(shape, s).mean() >= need)
        q = R.choose(val, equal_sweeps, s)
        rows.append({"matched": ref["setting"] or ref["method"],
                     "matched_measured": float(ref["measured"]),
                     "matched_f1": float(ref["f1"]), "period": s, "q": q,
                     **R.run(test, equal_sweeps, q, s)})
        print(rows[-1], flush=True)
    keys = ["matched", "matched_measured", "matched_f1", "period",
            "measured", "precision", "recall", "f1", "q"]
    path = os.path.join(R.OUT, "equal_sweeps.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
