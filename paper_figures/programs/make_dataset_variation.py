"""
make_dataset_variation.py -- how much the simulated devices differ, for the
training and the test devices side by side (a figure for Sec. 3.1.2).

    python paper_figures/programs/make_dataset_variation.py

Reads the device pool and the split stored in train.npz / test.npz of the
8 x 60 budget (every budget uses the same split).  Writes, into
method_docs_ver2/3_methods/figures/:
    fig_dataset_variation.png      the figure
    dataset_variation.json         the numbers the text quotes
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.colors import ListedColormap              # noqa: E402

from csdrecon.study import charge_states as C            # noqa: E402

RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
CELL = "8_rays_60_points_500_samples"
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                   "figures")
TRAIN_C, TEST_C = "#2a6ab0", "#c00000"
plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444"})


def devices(split):
    z = np.load(os.path.join(ROOT, "results", RUN, CELL, f"{split}.npz"),
                allow_pickle=True)
    return [os.path.join(POOL, os.path.basename(str(s).replace("\\", "/")))
            for s in z["samples"]]


def describe(dirs):
    rows = []
    for d in dirs:
        cs = C.load_states(d)
        dev = C.read_device(d)
        y = np.load(os.path.join(d, "numpy", "simulation",
                                 "ground_truth_labels.npy")) > 0.5
        v = C.device_variation(cs, dev)
        v["line_px_%"] = 100.0 * float(y.mean())
        v["dir"] = d
        rows.append(v)
    return rows


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def summary(rows):
    out = {"n": len(rows)}
    for k in ("n_states", "d1g1", "d2g2", "d1d2", "line_px_%"):
        x = col(rows, k)
        out[k] = {"median": float(np.median(x)), "min": float(x.min()),
                  "max": float(x.max())}
    return out


def hist(ax, tr, te, k, bins, xlabel, tag):
    a, b = col(tr, k), col(te, k)
    w_tr = np.ones_like(a) / len(a) * 100
    w_te = np.ones_like(b) / len(b) * 100
    ax.hist(a, bins=bins, weights=w_tr, color=TRAIN_C, alpha=0.45,
            label=f"training ({len(a)})")
    ax.hist(b, bins=bins, weights=w_te, histtype="step", color=TEST_C,
            lw=1.4, label=f"test ({len(b)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("devices (%)")
    ax.tick_params(labelsize=7)
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontweight="bold",
            fontsize=8)


def pick_examples(rows, n=3):
    """n devices spread over the number of charge states in the window."""
    order = np.argsort(col(rows, "n_states"))
    return [rows[i] for i in order[np.linspace(0, len(order) - 1, n)
                                   .round().astype(int)]]


def split_and_examples(tr, te):
    """(user, 2026-10-26) one figure for Sec. 3.1.2, without histograms:
    (a) the split of the devices, (b) three training and three test
    devices. Replaces ver2's Fig. 4 and the histogram figure."""
    from csdrecon.ml import grid_train
    n_tr, n_te = len(tr), len(te)
    n_val = max(1, int(grid_train.VAL_FRACTION * n_tr))
    n_fit = n_tr - n_val
    # (user, 2026-10-27) the split itself is in the text; the figure
    # shows only the example devices: their charge-sensor signal (top, in
    # the colours of the former one-device figure) above their true
    # transition lines (bottom); no titles, the frame colour tells
    # training (blue) from test (red)
    fig = plt.figure(figsize=(15.49 / 2.54, 2.04))
    gs = fig.add_gridspec(2, 6, left=0.01, right=0.99, top=0.99,
                          bottom=0.01, wspace=0.08, hspace=0.06)
    ex = [("training", r) for r in pick_examples(tr)] + \
         [("test", r) for r in pick_examples(te)]
    bw = ListedColormap(["white", "#111111"])
    for k, (lab, r) in enumerate(ex):
        sim = os.path.join(r["dir"], "numpy", "simulation")
        from csdrecon.ml import ray_peaks
        _ux, _uy, z = ray_peaks.load_grid(r["dir"])   # (V1, V2, z) rows
        y = np.load(os.path.join(sim, "ground_truth_labels.npy")) > 0.5
        for row, (img, cmap) in enumerate(((z, "hot"), (y, bw))):
            a = fig.add_subplot(gs[row, k])
            a.imshow(img, cmap=cmap, origin="lower", interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color(TRAIN_C if lab == "training" else TEST_C)
                sp.set_linewidth(1.4)
    fig.savefig(os.path.join(OUT, "fig_dataset_split.png"), dpi=400)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, te = describe(devices("train")), describe(devices("test"))
    split_and_examples(tr, te)
    fig = plt.figure(figsize=(15.49 / 2.54, 3.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.78], hspace=0.62,
                          left=0.08, right=0.99, top=0.94, bottom=0.01)
    top = gs[0].subgridspec(1, 4, wspace=0.45)
    ax = [fig.add_subplot(top[i]) for i in range(4)]
    hist(ax[0], tr, te, "n_states", np.linspace(0, 110, 23),
         "charge states in window", "(a)")
    hist(ax[1], tr, te, "d1g1", np.linspace(0.8, 6.0, 14),
         "$c_{d1g1}$ (cell size, dot 1)", "(b)")
    hist(ax[2], tr, te, "d1d2", np.linspace(0.2, 2.0, 10),
         "$c_{d1d2}$ (interdot)", "(c)")
    hist(ax[3], tr, te, "line_px_%", np.linspace(0, 25, 26),
         "line pixels (%)", "(d)")
    ax[3].set_ylim(0, ax[3].get_ylim()[1] * 1.45)   # room for the legend
    ax[3].legend(fontsize=6.5, frameon=False, loc="upper right")
    bot = gs[1].subgridspec(1, 6, wspace=0.08)
    ex = [("training", r) for r in pick_examples(tr)] + \
         [("test", r) for r in pick_examples(te)]
    bw = ListedColormap(["white", "#111111"])
    for k, (lab, r) in enumerate(ex):
        a = fig.add_subplot(bot[k])
        y = np.load(os.path.join(r["dir"], "numpy", "simulation",
                                 "ground_truth_labels.npy")) > 0.5
        a.imshow(y, cmap=bw, origin="lower", interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(TRAIN_C if lab == "training" else TEST_C)
            s.set_linewidth(1.4)
        a.set_title(f"{lab}: {int(r['n_states'])} states", fontsize=7,
                    color=TRAIN_C if lab == "training" else TEST_C)
        if k == 0:
            a.text(-0.16, 1.02, "(e)", transform=a.transAxes,
                   fontweight="bold", fontsize=8, ha="right", va="bottom")
    fig.savefig(os.path.join(OUT, "fig_dataset_variation.png"), dpi=400)
    plt.close(fig)
    stats = {"train": summary(tr), "test": summary(te)}
    with open(os.path.join(OUT, "dataset_variation.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
