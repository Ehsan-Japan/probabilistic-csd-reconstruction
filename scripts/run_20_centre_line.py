"""
Is a centre line needed?  Step 2 of run_19 measures a strip of +-BAND
grid steps around the binarized U-Net prediction itself.  The alternative,
used until 2026-10-25, is to thin the prediction first to its one-pixel
centre lines (skeletonize) and measure the strip around those: it is then
narrower by the width of the band.  On the 50 test devices the two trace
the same accuracy against cost, so run_19 no longer uses the centre line.

    python scripts/run_19_two_step.py        # the binarized-band rows
    python scripts/run_20_centre_line.py

Both variants use the same step detector, the same rule for which pixels
are checked (within BAND-2 steps, and every pixel the step value uses
measured), and a detector threshold chosen on the 75 validation devices by
F1@1, separately for every variant and band; the test devices only report.

Writes results/<run>/two_step/centre_line.csv and centre_line.md.
"""
import csv
import os

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)

from skimage.morphology import skeletonize

import run_19_two_step as R
from csdrecon.ml import grid_train
from run_7_virtual_gates import model_dir


def centre_step(D, i, q, band):
    """Step 2 with the centre line: the strip around the one-pixel centre
    lines of the binarized U-Net output."""
    B = skeletonize(D["B"][i])
    measured = R.dilate(B, band) | (D["X"][i, 1] > 0.5)
    checked = R.dilate(B, max(band - 2, 0)) & R.inside(measured)
    return R.steps(D["Z"][i], checked, q, measured), float(measured.mean())


def main():
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr = float(ck["threshold"])
    val, test = R.load(net, thr, "validation"), R.load(net, thr, "test")
    rows = []
    for b in R.BANDS:
        for name, fn in (("centre line", centre_step),
                         ("binarized band", R.two_step6)):
            q = R.choose(val, fn, b)
            rows.append({"strip around": name, "band": b, "q": q,
                         **R.run(test, fn, q, b)})
            print(rows[-1], flush=True)
    # the binarized-band rows must be run_19's 6-pixel version
    with open(os.path.join(R.OUT, "_band5_6pixel_version", "table.csv"),
              newline="",
              encoding="utf-8") as f:
        ref = {r["setting"]: r for r in csv.DictReader(f)}
    for r in rows:
        if r["strip around"] == "binarized band":
            o = ref[f"band ±{r['band']} px"]
            assert abs(float(o["f1"]) - r["f1"]) < 1e-9, (o, r)
            assert abs(float(o["measured"]) - r["measured"]) < 1e-9
    keys = ["strip around", "band", "measured", "precision", "recall",
            "f1", "q"]
    with open(os.path.join(R.OUT, "centre_line.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(R.OUT, "centre_line.md"), "w",
              encoding="utf-8") as f:
        f.write("| strip around | band | measured (%) | precision | recall "
                "| F1@1 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['strip around']} | ±{r['band']} | "
                    f"{r['measured']:.1f} | {r['precision']:.3f} | "
                    f"{r['recall']:.3f} | {r['f1']:.3f} |\n")
    print("wrote", R.OUT)


if __name__ == "__main__":
    main()
