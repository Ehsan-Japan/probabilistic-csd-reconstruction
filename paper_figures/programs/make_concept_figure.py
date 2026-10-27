"""
make_concept_figure.py -- the conceptual figure of the conceptual version
of the manuscript (user, 2026-10-26): what a generative reconstruction
decides, against what the two-step method decides.

    python paper_figures/programs/make_concept_figure.py

(a) Generative reconstruction, Ref. 12: sparse measurement -> generative
    model trained on measured diagrams -> an estimate of the signal at every
    pixel -> edge and ridge detectors -> lines.  The three images are the
    crops of Fig. 3 of Ref. 12 made by make_pipeline_comparison_ai.py
    --ref12 (deck_figures/pipeline_comparison/tiles/h_*.png).
(b) This work: rays -> network trained on simulations -> candidate regions
    (the predicted lines) -> local sweeps on the device -> a step in the
    sensor signal -> verified lines.  The images are the device of
    make_pipeline_figure.py (same device, threshold and margin).
The two steps carry the colour code of Figs. 1 and 8: indigo for the
prediction, amber for the verification.  Writes fig_concept.png into
method_docs_ver2/conceptual/figures/.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.colors import to_rgb                     # noqa: E402
from matplotlib.patches import FancyBboxPatch            # noqa: E402

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.ml.ray_peaks import CH_VISITED             # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402
import make_pipeline_figure as PF                        # noqa: E402

TILES = os.path.join(ROOT, "paper_figures", "deck_figures",
                     "pipeline_comparison", "tiles")
OUT = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "conceptual",
                   "figures")
GEN_C, GEN_BG = "#555555", "#f2f2f2"            # the generative row
plt.rcParams.update({"font.family": "Arial", "font.size": 7})


def device():
    """The arrays of the pipeline figure's device, computed as there."""
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        q = float(next(r for r in csv.DictReader(f)
                       if r["setting"] == f"band ±{PF.BAND} px")["q"])
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    test = R.load(net, float(ck["threshold"]), "test")
    i = test["names"].index(PF.DEVICE)
    Y, B, Z, X = test["Y"][i], test["B"][i], test["Z"][i], test["X"][i]
    rays = X[CH_VISITED] > 0.5
    strip = R.dilate(B, PF.BAND)
    measured = strip | rays
    kept, _r, _t = R.sweep_lines(Z, strip, q, measured)
    assert np.array_equal(kept, R.two_step(test, i, q, PF.BAND)[0])
    return Z, B, rays, strip, kept


def rgb(shape, *layers):
    im = np.ones(shape + (3,))
    for m, c in layers:
        im[m] = to_rgb(c)
    return im


def main(ver6=False):
    # ver6=True (user, 2026-10-27): Fig. 1 of Manuscript_ver6, in its
    # vocabulary (Step 1: Line prediction, Step 2: Line verification, step
    # detection); written to 3_methods/figures/fig_two_formulations.png,
    # leaving the conceptual version's fig_concept.png as it is
    L = dict(pred="predicted lines", op0="U-Net trained\non simulations",
             op2="step\ndetection", s1="(b) Step 1: Line prediction",
             s2="Step 2: Line verification on the device") if ver6 else \
        dict(pred="candidate regions\n(predicted lines)",
             op0="network trained\non simulations\n(exact labels)",
             op2="step in the\nsensor signal", s1="(b) Step 1: Prediction",
             s2="Step 2: Verification on the device")
    os.makedirs(OUT, exist_ok=True)
    Z, B, rays, strip, kept = device()
    sh = B.shape
    raw = np.ma.masked_where(~rays, Z)

    W, H = 15.49 / 2.54, 3.05
    t, g = 0.66, 0.76                    # tile size; gap holding an operation
    x0 = 0.16
    yb, ya = 0.42, 0.42 + t + 0.78       # bottom of the tiles of (b), (a)
    fig = plt.figure(figsize=(W, H))

    def tile(k, y):
        a = fig.add_axes(((x0 + k * (t + g)) / W, y / H, t / W, t / H))
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.6)
        return a

    def under(a, text):
        a.text(0.5, -0.06, text, transform=a.transAxes, ha="center",
               va="top", fontsize=6.3, linespacing=1.15)

    def op(k, y, text, c):
        """An operation in the gap after tile k, with its arrow."""
        xa, xb = x0 + k * (t + g) + t, x0 + (k + 1) * (t + g)
        ym = y + t / 2
        fig.add_artist(matplotlib.patches.FancyArrowPatch(
            (xa / W + 0.003, ym / H), (xb / W - 0.003, ym / H),
            arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333",
            transform=fig.transFigure))
        fig.text((xa + xb) / 2 / W, (ym + 0.06) / H, text, ha="center",
                 va="bottom", fontsize=5.6, color=c, linespacing=1.1)

    def band(xl, xr, y, c, bg, label):
        fig.add_artist(FancyBboxPatch(
            (xl / W, (y - 0.30) / H), (xr - xl) / W, (t + 0.52) / H,
            boxstyle="round,pad=0,rounding_size=0.012", fc=bg, ec=c, lw=0.8,
            zorder=-1, transform=fig.transFigure))
        fig.text((xl + 0.06) / W, (y + t + 0.13) / H, label, ha="left",
                 va="center", fontsize=6.8, fontweight="bold", color=c)

    # (a) generative reconstruction, Ref. 12
    for k, (name, text) in enumerate((
            ("h_masked", "sparse measurement"),
            ("h_diffusion", "estimate of the signal\nat every pixel"),
            ("h_ridges", "lines found\nin the estimate"))):
        a = tile(k, ya)
        a.imshow(plt.imread(os.path.join(TILES, name + ".png")))
        under(a, text)
    op(0, ya, "generative model\ntrained on measured,\nannotated diagrams",
       GEN_C)
    op(1, ya, "edge and ridge\ndetectors", GEN_C)
    band(0.06, x0 + 2 * (t + g) + t + 0.10, ya, GEN_C, GEN_BG,
         "(a) Generative reconstruction (Ref. 12): every pixel is estimated")
    fig.text((x0 + 3 * (t + g) + t / 2) / W, (ya + t / 2) / H,
             "lines decided\nin an estimated\nimage", ha="center",
             va="center", fontsize=6.6, color=GEN_C, style="normal",
             linespacing=1.15)

    # (b) this work: prediction, then verification on the device
    a = tile(0, yb)
    a.imshow(np.full(sh + (3,), 0.88), origin="lower")
    a.imshow(raw, origin="lower", cmap="hot", interpolation="nearest",
             vmin=float(Z[rays].min()), vmax=float(Z[rays].max()))
    under(a, "rays")
    a = tile(1, yb)
    a.imshow(rgb(sh, (B, PF.LINE_C)), origin="lower",
             interpolation="nearest")
    under(a, L["pred"])
    a = tile(2, yb)
    a.imshow(rgb(sh, (strip, PF.CHECK_C), (rays, PF.RAY_C)), origin="lower",
             interpolation="nearest")
    under(a, "swept pixels")
    a = tile(3, yb)
    a.imshow(rgb(sh, (kept, PF.KEPT_C)), origin="lower",
             interpolation="nearest")
    under(a, "verified lines")
    op(0, yb, L["op0"], PF.PRED_C)
    op(1, yb, "local sweeps\non the device", PF.VER_C)
    op(2, yb, L["op2"], PF.VER_C)
    xs = x0 + 1 * (t + g) + t + 0.08            # where verification begins
    band(0.06, xs - 0.04, yb, PF.PRED_C, PF.PRED_BG,
         L["s1"])
    band(xs, x0 + 3 * (t + g) + t + 0.10, yb, PF.VER_C, PF.VER_BG,
         L["s2"])
    fig.text((x0 + 3 * (t + g) + t + 0.10 + W - 0.04) / 2 / W,
             (yb + t / 2) / H, "lines decided\nby the device's\ncharge sensor",
             ha="center", va="center", fontsize=6.6, color=PF.VER_C,
             fontweight="bold", linespacing=1.15)
    out = os.path.join(ROOT, "paper_figures", "method_docs_ver2", "3_methods",
                       "figures", "fig_two_formulations.png") if ver6 \
        else os.path.join(OUT, "fig_concept.png")
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(ver6="--ver6" in sys.argv)
