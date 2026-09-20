# -*- coding: utf-8 -*-
"""
make_threshold_figure.py -- how the validation split picks the threshold.

Saying "the cut is chosen out of sample" is an assertion; this draws it.
Mean F1@1 over the VALIDATION devices for every candidate threshold, with
the one the checkpoint actually stores ringed.

The split is not re-invented here.  grid_train carves it out of the training
devices with default_rng(SEED) BEFORE training, and _validation_curve()
replays exactly that, so the curve is the one the choice was really made on
-- 75 devices held out of the 500, the test 50 never touched.

Writes threshold_validation.png/.pdf and the numbers it plotted to
threshold_validation.json, so a caption can quote them.

    python paper_figures/make_threshold_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _run import CFG_DIR, CONFIG, ROOT                         # noqa: E402
from csdrecon.ml import grid_train                             # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1               # noqa: E402
from csdrecon.study.dataset import load_split                  # noqa: E402

INK, MUT, RED = "#111111", "#5a606a", "#C00000"


def validation_curve(net, cfg_dir=CFG_DIR):
    """
    Re-derive the exact validation split the training used and score every
    candidate threshold on it.  grid_train carves it out with
    default_rng(SEED) before training, so this is reproducible.
    """
    Xtr, Ytr, _ = load_split(os.path.join(cfg_dir, "train.npz"))
    rng = np.random.default_rng(grid_train.SEED)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, int(grid_train.VAL_FRACTION * len(Xtr)))
    vi = idx[:n_val]
    pv, Yv = grid_train.predict(net, Xtr[vi]), Ytr[vi]
    cand = np.array(grid_train.THRESHOLDS)
    f1 = np.array([np.mean([tolerant_f1(pv[j] > t, Yv[j], 1.0)["f1"]
                            for j in range(len(Yv))]) for t in cand])
    return cand, f1, len(vi)


def main():
    net, ck = grid_train.load(os.path.join(CFG_DIR, "model", "unet.pt"))
    thr = float(ck["threshold"])
    cand, f1, n_val = validation_curve(net)
    best = int(np.argmin(np.abs(cand - thr)))
    print("  %s" % CONFIG)
    print("  %d validation devices; stored threshold %.2f, F1@1 %.3f"
          % (n_val, thr, f1[best]))

    fig = plt.figure(figsize=(2.9, 2.05))
    ax = fig.add_axes([0.20, 0.21, 0.77, 0.75])
    ax.plot(cand, f1, "-o", color=INK, linewidth=1.6, markersize=3.6,
            zorder=3)
    ax.plot([cand[best]], [f1[best]], "o", color=RED, markersize=11,
            markerfacecolor="none", markeredgewidth=1.8, zorder=4)
    ax.annotate("%.1f" % cand[best], (cand[best], f1[best]),
                textcoords="offset points", xytext=(2, -14), ha="center",
                fontsize=9, color=RED, fontweight="bold")
    ax.set_xlabel("threshold on $P$", fontsize=8.5, color=INK, labelpad=2)
    ax.set_ylabel("F1@1, validation", fontsize=8.5, color=INK, labelpad=2)
    ax.tick_params(labelsize=7.5, colors=INK, length=3, width=0.7)
    ax.grid(True, color="#9a9a9a", linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.85)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
        ax.spines[sp].set_linewidth(0.8)
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        p = os.path.join(HERE, "threshold_validation" + ext)
        fig.savefig(p, facecolor="white", **kw)
        print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)

    meta = os.path.join(HERE, "threshold_validation.json")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"configuration": CONFIG, "n_val": int(n_val),
                   "threshold": thr, "f1": round(float(f1[best]), 3),
                   "candidates": [float(c) for c in cand],
                   "curve": [round(float(v), 4) for v in f1]}, fh, indent=2)
    print("  ->", os.path.relpath(meta, ROOT))


if __name__ == "__main__":
    main()
