"""
A narrower local measurement: a strip of +-BAND grid steps around the
binarized U-Net output with a step detector that looks only one pixel each
way, so that the strip can be +-2 and the checked pixels +-1.

    python scripts/run_19_two_step.py        # the 6-pixel detector rows
    python scripts/run_21_narrow_strip.py

The detector of run_19 (charge_states.step_residual_map) compares the
change to the next pixel, d_k = Z_{k+1} - Z_k, with the median of the five
changes d_{k-2} ... d_{k+2}, and uses pixels k-2 ... k+3.  The 3-pixel
detector here compares it with the change just before, d_{k-1}:

    r_k = |d_k - d_{k-1}| / (|d_{k-1}| + s),   pixels k-1, k, k+1,

with s the median |d| over the measured pixel pairs of the device along
that direction, and the larger of the two directions kept, as in run_19.
A pixel is checked if it lies within BAND-1 steps of the binarized output
and k-1, k, k+1 were all measured along both directions.

Compared, on the 50 test devices, for every band, with the same threshold
rule as run_19 (quantile chosen on the 75 validation devices by F1@1); the
full scan with the 3-pixel detector is included for reference.

Writes results/<run>/two_step/narrow_strip.csv and narrow_strip.md.
"""
import csv
import os

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)

import run_19_two_step as R
from csdrecon.ml import grid_train
from run_7_virtual_gates import model_dir

BANDS3 = (2, 3, 4, 5)


step3_map, inside3, steps3 = R.step3_map, R.inside3, R.steps3


def narrow_step(D, i, q, band):
    B = D["B"][i]
    measured = R.dilate(B, band) | (D["X"][i, 1] > 0.5)
    checked = R.dilate(B, max(band - 1, 0)) & inside3(measured)
    return steps3(D["Z"][i], checked, q, measured), float(measured.mean())


def full3(D, i, q):
    return steps3(D["Z"][i], np.ones_like(D["Y"][i]), q), 1.0


def main():
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr = float(ck["threshold"])
    val, test = R.load(net, thr, "validation"), R.load(net, thr, "test")
    rows = []
    for b in BANDS3:
        q = R.choose(val, narrow_step, b)
        rows.append({"detector": "3-pixel", "band": b, "q": q,
                     **R.run(test, narrow_step, q, b)})
        print(rows[-1], flush=True)
    q = R.choose(val, full3)
    rows.append({"detector": "3-pixel", "band": "full scan", "q": q,
                 **R.run(test, full3, q)})
    print(rows[-1], flush=True)
    keys = ["detector", "band", "measured", "precision", "recall", "f1", "q"]
    with open(os.path.join(R.OUT, "narrow_strip.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(R.OUT, "narrow_strip.md"), "w",
              encoding="utf-8") as f:
        f.write("| detector | band | measured (%) | precision | recall | "
                "F1@1 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['detector']} | {r['band']} | {r['measured']:.1f} "
                    f"| {r['precision']:.3f} | {r['recall']:.3f} | "
                    f"{r['f1']:.3f} |\n")
    print("wrote", R.OUT)


if __name__ == "__main__":
    main()
