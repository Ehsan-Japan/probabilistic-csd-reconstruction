"""
Equally spaced seeds, without the U-Net: exactly step 2 of run_19 -- every
pixel within BAND grid steps of a seed pixel is swept and tested by the same
3-pixel step detector -- but the seeds are an equally spaced square lattice
of points instead of the binarized U-Net output.

    python scripts/run_19_two_step.py          # table.csv (the U-Net side)
    python scripts/run_23_equal_seeds.py

This is the like-for-like baseline of the local sweeps: the same strip
(dilate(seeds, BAND), a diamond of 2*BAND*(BAND+1)+1 pixels around each
seed), the same measured pixels (the strip only: no rays, which the U-Net
needs and this does not), the same detector, run_19.sweep_lines.  The
equally spaced rows and columns of run_22 were the baseline before.

For each margin of run_19, the lattice period s is the largest whose strip
covers at least that setting's mean fraction of the plane, so the seeds
never get fewer measurements.  The detector threshold is chosen on the 75
validation devices by F1@1; the 50 test devices only report.

Writes results/<run>/two_step/equal_seeds.csv and fig_f1_vs_measured.png
(Fig. 17: accuracy against cost, every method; with --plot, only the figure
from the csv files already written).
"""
import csv
import os
import sys

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt

import run_19_two_step as R
from csdrecon.ml import grid_train
from run_7_virtual_gates import model_dir


def seeds(shape, s):
    """An s-periodic square lattice of points, centred on the plane."""
    m = np.zeros(shape, bool)
    idx = []
    for n in shape:
        k = (n - 1) // s + 1
        idx.append((n - 1 - (k - 1) * s) // 2 + s * np.arange(k))
    m[np.ix_(*idx)] = True
    return m


def strip_of(shape, s, band):
    return R.dilate(seeds(shape, s), band)


def equal_seeds(D, i, q, s, band):
    strip = strip_of(D["Y"][i].shape, s, band)
    kept, _, _ = R.sweep_lines(D["Z"][i], strip, q, strip)
    return kept, float(strip.mean())


def fig_curve(path):
    """Line accuracy against the fraction of the plane measured: the U-Net
    alone, U-Net + local sweeps at each margin, the equally spaced seeds
    matched to each (labelled with their spacing)."""
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(R.OUT, "equal_seeds.csv"), newline="",
              encoding="utf-8") as f:
        eq = list(csv.DictReader(f))
    xy = lambda rs: ([float(r["measured"]) for r in rs],          # noqa
                     [float(r["f1"]) for r in rs])
    # (user, 2026-10-27) no +-0 margin in the paper: the sweeps around the
    # predicted pixels alone, and isolated sampling points that cannot be
    # tested, are left out of the figure
    eq = [r for r in eq if r["band"] != "0"]
    rows = [r for r in rows if r["setting"] != "band ±0 px"]
    g = lambda m: [r for r in rows if r["method"] == m]          # noqa
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    fig, a = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    # one point per U-Net setting it is matched to; no line, since nothing
    # was measured between the costs
    # (user, 2026-10-27) no raster: the figure shows the effect of the
    # U-Net guidance alone, same sweeps and detector on both sides
    a.plot(*xy(eq), "s", color="#8c8c8c", ms=4,
           label="Uniform-grid local sweeps")
    for r in eq:                        # its margin, as for the red points
        a.annotate(f"±{r['band']} px",
                   (float(r["measured"]), float(r["f1"])),
                   textcoords="offset points", ha="left", fontsize=6.5,
                   color="#6e6e6e",
                   # at F1@1 = 0 the label goes above, clear of the axis
                   **(dict(xytext=(5, 3), va="bottom")
                      if float(r["f1"]) < 0.05 else
                      dict(xytext=(5, -2), va="top")))
    two = g("U-Net, then local check")
    a.plot(*xy(two), "o-", color="#c00000", ms=4,
           label="U-Net-guided local sweeps (Step 2)")
    for r in two:                       # which margin each point is
        a.annotate(r["setting"].replace("band ", ""),
                   (float(r["measured"]), float(r["f1"])),
                   textcoords="offset points", xytext=(-4, 5), ha="right",
                   fontsize=6.5, color="#c00000")
    a.plot(*xy(g("U-Net only")), "o", color="#2a6ab0", ms=6,
           label="U-Net prediction (Step 1)")
    a.set_xlabel("Fraction of plane measured (%)")
    a.set_ylabel("F1@1")
    a.set_xlim(0, 60)
    a.set_ylim(-0.03, 1.0)
    a.grid(ls=":", lw=0.6, color="#aaaaaa")
    a.spines[["top", "right"]].set_visible(False)
    # the empty lower-right corner: no point or label under it; legend
    # in the manuscript's vocabulary (user, 2026-10-27)
    # Step 1, Step 2, then the two baselines
    h, lab = a.get_legend_handles_labels()
    a.legend(h[::-1], lab[::-1], fontsize=7, loc="lower right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    if "--plot" in sys.argv:
        fig_curve(os.path.join(R.OUT, "fig_f1_vs_measured.png"))
        return
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr = float(ck["threshold"])
    val, test = R.load(net, thr, "validation"), R.load(net, thr, "test")
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        refs = [r for r in csv.DictReader(f)
                if r["method"] == "U-Net, then local check"]
    shape = test["Y"][0].shape
    rows = []
    for ref in refs:
        band = int(ref["setting"].split("±")[1].split()[0])
        need = float(ref["measured"]) / 100
        s = max(s for s in range(1, shape[0])
                if strip_of(shape, s, band).mean() >= need)
        q = R.choose(val, equal_seeds, s, band)
        rows.append({"matched": ref["setting"], "band": band,
                     "matched_measured": float(ref["measured"]),
                     "matched_f1": float(ref["f1"]), "period": s, "q": q,
                     **R.run(test, equal_seeds, q, s, band)})
        print(rows[-1], flush=True)
    keys = ["matched", "band", "matched_measured", "matched_f1", "period",
            "measured", "precision", "recall", "f1", "q"]
    path = os.path.join(R.OUT, "equal_seeds.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fig_curve(os.path.join(R.OUT, "fig_f1_vs_measured.png"))
    print("wrote", path)


if __name__ == "__main__":
    main()
