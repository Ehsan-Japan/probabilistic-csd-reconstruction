# -*- coding: utf-8 -*-
"""
paper_figures/make_figures.py — the manuscript / talk figures, each drawn
from the real arrays in the device pool rather than mocked up.

    fig_measurement_panel       sensor | truth | what the network is shown
    fig_network_input           the TWO maps the U-Net is actually given
    fig_unet                    a compact schematic of the network
    fig_model_flow              input | network | output, in one picture
    panel_channel1/2            the same maps as standalone panels
    panel_probability           the probability map on its own
    panel_charge_sensor         raw sensor, and with its background removed
    p2l_1 ... p2l_9             probability map -> threshold -> tolerance
    fig_probability_to_lines    the same story as one composite
    fig_data_split              where the validation devices come from
    fig_tau_metrics             F1 / precision / recall against tolerance
    results_f1_vs_coverage      the headline chart

These are the NEAT figures: one point per panel.  The dense diagnostic
gallery lives in results/<run>/figures/ and is not what goes in the paper.

Run from the project root:   python paper_figures/make_figures.py

Every figure is written as .png (600 dpi, for slides) and .pdf (vector, for
the manuscript).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _run                                                    # noqa: E402
from _run import (CFG_DIR, CONFIG, DEVICE, DEVICE_F1, N_POINTS, N_RAYS,
                  N_TEST, ROOT, RUN_DIR, TEST_MEAN_F1,
                  THRESHOLD)                                   # noqa: E402
from csdrecon.ml import grid_train, ray_peaks                  # noqa: E402
from csdrecon.ml.grid_metrics import tolerant_f1               # noqa: E402

INK = "#111111"
MUT = "#5a606a"

# Error colours for the reconstruction panels, taken from the deck's own
# theme (teal / red / plum) rather than green and orange.
HIT_HEX, MISS_HEX, FALSE_HEX = "#1B587C", "#9F2936", "#7A5C99"
HIT_RGB = (0.106, 0.345, 0.486)
MISS_RGB = (0.624, 0.161, 0.212)
FALSE_RGB = (0.478, 0.361, 0.600)

# The two-colour scheme of the probability-to-lines panels.
J_TRUTH, J_PRED, J_BAND = "#000000", "#C00000", "#D9D9D9"
J_TRUTH_RGB = (0.000, 0.000, 0.000)
J_PRED_RGB = (0.753, 0.000, 0.000)
J_BAND_RGB = (0.851, 0.851, 0.851)
J_CMAP = "magma"                     # black = P 0, yellow = P 1
J_PTITLE = "#7d2b3a"                 # probability-panel title
J_GRID = "#9a9a9a"

# Ray count is ORDERED, so the series run light-blue -> blue -> navy ->
# black -> red: still a ramp, but each step changes hue as well as
# lightness, and each carries its own marker.  Four greys were
# indistinguishable on a projector, and marker shape survives greyscale
# printing and the common colour-vision deficiencies.
STYLE = {4: ("#9ECAE1", "o"), 5: ("#4292C6", "s"), 6: ("#08519C", "^"),
         7: ("#000000", "D"), 8: (J_PRED, "o")}

TAUS = (0, 1, 2, 3)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.edgecolor": "#999999",
    "savefig.facecolor": "white",
})

# Titles on the p2l panels can be left OFF, so a deck can carry them as real
# text boxes the author may delete.  The blank title keeps the same number
# of lines, so the reserved space -- and with it the figure's proportions --
# does not move.  Blanked panels get their own filenames.
P2L_TITLES_OFF = os.environ.get("P2L_TITLES") == "off"
P2L_TITLES = {}

_MODEL = None


def net():
    """The trained checkpoint for the headline budget, loaded once."""
    global _MODEL
    if _MODEL is None:
        _MODEL = grid_train.load(os.path.join(CFG_DIR, "model", "unet.pt"))
    return _MODEL


def save(fig, name):
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        p = os.path.join(HERE, name + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)


def pick_case():
    """(channels, truth, probability) for the picked held-out device."""
    m = ray_peaks.measure(DEVICE, N_RAYS, N_POINTS)
    _ux, _uy, Z = ray_peaks.load_grid(DEVICE)
    ch = ray_peaks.to_channels(m, Z.shape)
    truth = ray_peaks.load_ground_truth(DEVICE)
    prob = grid_train.predict(net()[0], ch[None, :ray_peaks.NET_CHANNELS])[0]
    return ch, truth, prob


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")


# -- figure 1: the two input maps -----------------------------------------
def fig_network_input(compact=False):
    m = ray_peaks.measure(DEVICE, N_RAYS, N_POINTS)
    _ux, _uy, Z = ray_peaks.load_grid(DEVICE)
    ch = ray_peaks.to_channels(m, Z.shape)
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    cov = 100.0 * vis.mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))

    # channel 1 - the measured value, shown only where a ray passed
    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    ax = axes[0]
    im = ax.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("channel 1   —   measured sensor signal", fontsize=11,
                 color=INK, pad=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("normalised charge-sensor signal", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("grey = never measured", fontsize=9, color=MUT)

    # channel 2 - where we looked at all
    ax = axes[1]
    ax.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title("channel 2   —   visited mask", fontsize=11, color=INK,
                 pad=9)
    ax.set_xlabel("black = a ray passed here  (%.1f %% of the grid)" % cov,
                  fontsize=9, color=MUT)

    for ax in axes:
        blank(ax)

    if not compact:
        # the manuscript version carries its own title and caption; the
        # slide version does not, because the slide already says both
        fig.suptitle("What the network is given — %d rays × %d "
                     "points, one held-out device" % (N_RAYS, N_POINTS),
                     fontsize=12.5, color=INK, y=1.0)
        fig.text(0.5, -0.045,
                 "Channel 2 is what separates “measured here, and the "
                 "signal was low” from “never looked here”.  "
                 "Without it a zero in channel 1 is ambiguous everywhere.",
                 ha="center", fontsize=9.5, color=MUT)
    fig.tight_layout()
    save(fig, "fig_network_input_slide" if compact else "fig_network_input")


# -- figure 2: compact U-Net schematic ------------------------------------
ENC = "#4f81a8"
BOT = "#3d4a5c"
DEC = "#c07a4a"
IO = "#5f8a5f"

_BLOCK_FS = 1.0          # set by draw_unet so sub-captions scale with fs


def _block(ax, x, y, w, h, color, label, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=0, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y - 0.055, sub, ha="center", va="top",
                fontsize=8.5 * _BLOCK_FS, color=MUT)


def _arrow(ax, p, q, color, style="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=style,
                                 shrinkA=1, shrinkB=1, zorder=1))


def draw_unet(ax, fs=1.0, io_labels=True):
    """Draw the U-Net schematic into an existing axes.  fs scales the fonts;
    io_labels=False drops the input/output captions, for when the figure
    already shows the real input and output either side."""
    global _BLOCK_FS
    _BLOCK_FS = fs
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.32, 1.25)
    ax.axis("off")

    lvl = [(0.72, 0.85), (0.56, 0.60), (0.40, 0.35)]      # h, y per depth
    enc_x = [1.35, 2.35, 3.35]
    dec_x = [5.30, 6.30, 7.30]
    widths = ["32", "64", "128"]
    grids = ["100²", "50²", "25²"]

    _block(ax, 0.15, 0.30, 0.55, 0.72, IO, "2", "input")
    for i, (x, w, g) in enumerate(zip(enc_x, widths, grids)):
        h, yt = lvl[i]
        _block(ax, x, yt - h, 0.72, h, ENC, w, g)
    _block(ax, 4.30, 0.00, 0.72, 0.28, BOT, "256", "12²")
    for i, (x, w, g) in enumerate(zip(dec_x, reversed(widths),
                                      reversed(grids))):
        h, yt = lvl[2 - i]
        _block(ax, x, yt - h, 0.72, h, DEC, w, g)
    _block(ax, 8.40, 0.30, 0.55, 0.72, IO, "1", "output")

    # down / up path
    _arrow(ax, (0.72, 0.66), (1.33, 0.66), INK)
    for i in range(2):
        h0, y0 = lvl[i]
        h1, y1 = lvl[i + 1]
        _arrow(ax, (enc_x[i] + 0.72, y0 - h0 / 2),
               (enc_x[i + 1], y1 - h1 / 2), ENC)
    h, y = lvl[2]
    _arrow(ax, (enc_x[2] + 0.72, y - h / 2), (4.30, 0.14), ENC)
    _arrow(ax, (5.02, 0.14), (dec_x[0], y - h / 2), DEC)
    for i in range(2):
        h0, y0 = lvl[2 - i]
        h1, y1 = lvl[1 - i]
        _arrow(ax, (dec_x[i] + 0.72, y0 - h0 / 2),
               (dec_x[i + 1], y1 - h1 / 2), DEC)
    _arrow(ax, (dec_x[2] + 0.72, 0.66), (8.38, 0.66), INK)

    # skip connections - the deepest one is routed clear of the bottleneck
    skip_y = [lvl[0][1] - 0.06, lvl[1][1] - 0.06, 0.33]
    for i in range(3):
        _arrow(ax, (enc_x[i] + 0.72, skip_y[i]),
               (dec_x[2 - i], skip_y[i]), "#9aa2ac", style=(0, (4, 3)),
               lw=1.2)
    ax.text(4.66, lvl[0][1] + 0.02, "skip connections",
            ha="center", fontsize=8.5 * fs, color=MUT, style="italic")

    if io_labels:
        ax.text(0.42, 0.16, "channel 1  ray signal\nchannel 2  visited mask",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
        ax.text(8.68, 0.16, "P(transition line)\nper pixel",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
    ax.text(2.43, 1.16, "encoder", fontsize=10.5 * fs, color=ENC,
            fontweight="bold", ha="center")
    ax.text(6.38, 1.16, "decoder", fontsize=10.5 * fs, color=DEC,
            fontweight="bold", ha="center")
    ax.text(4.66, -0.14, "bottleneck", fontsize=9.5 * fs, color=BOT,
            fontweight="bold", ha="center", va="top")
    return ax


def fig_unet():
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    draw_unet(ax)
    save(fig, "fig_unet")


# -- figure 3: the probability map, cut and scored ------------------------
# The ladder is the chosen threshold and one deliberately too STRICT, so the
# pair shows what the cut costs when it is set too high.  The chosen value
# comes from the checkpoint, never from a number typed here.
#
# Not a too-LOOSE partner: threshold_validation.png shows the validation
# curve is flat from 0.3 to 0.8, so a 0.4 / 0.7 pair is two near-identical
# pictures.  It only falls off past 0.9, which is where the difference is.
LADDER_HIGH = float(os.environ.get("CSD_LADDER_HIGH", 0.9))


def ladder():
    other = LADDER_HIGH if THRESHOLD < LADDER_HIGH else 0.4
    return tuple(sorted((other, THRESHOLD)))


def fig_probability_to_lines():
    from scipy.ndimage import distance_transform_edt

    _ch, Yt, p = pick_case()
    truth = Yt > 0.5
    print("  %s   threshold %g   F1@1 %.3f (test mean %.3f)"
          % (os.path.basename(DEVICE), THRESHOLD, DEVICE_F1, TEST_MEAN_F1))

    shown = (0, 1, 3)
    d_true_by_tau = {}
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.40, wspace=0.12,
                          left=0.035, right=0.99, top=0.88, bottom=0.075)

    def err_rgb(pred, tau):
        """teal = found within tau, red = missed line, plum = false line."""
        if tau not in d_true_by_tau:
            d_true_by_tau[tau] = (distance_transform_edt(~truth)
                                  if truth.any()
                                  else np.full(truth.shape, np.inf))
        dt = d_true_by_tau[tau]
        dp = (distance_transform_edt(~pred) if pred.any()
              else np.full(pred.shape, np.inf))
        rgb = np.ones(truth.shape + (3,))
        rgb[pred & (dt > tau)] = FALSE_RGB
        rgb[truth & (dp > tau)] = MISS_RGB
        rgb[pred & (dt <= tau)] = HIT_RGB
        return rgb

    # -- row 1: the output, and the map cut two ways ----------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(p, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title("U-Net output\nP(transition line)", fontsize=11.5,
                 color=J_PTITLE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=6.5)
    blank(ax)

    for k, t in enumerate(ladder()):
        pred = p > t
        m = tolerant_f1(pred, Yt, 1.0)
        ax = fig.add_subplot(gs[0, 1 + k])
        ax.imshow(err_rgb(pred, 1.0), origin="lower", interpolation="nearest")
        chosen = abs(t - THRESHOLD) < 1e-9
        ax.set_title("P > %g%s\nF1@1 %.3f"
                     % (t, "  (chosen)" if chosen else "", m["f1"]),
                     fontsize=11.5, color="#c0392b" if chosen else INK,
                     fontweight="bold" if chosen else "normal")
        blank(ax)
        if chosen:
            for sp in ax.spines.values():
                sp.set_color("#c0392b")
                sp.set_linewidth(2.2)

    # -- row 2: ONE prediction, scored at three tolerances ----------------
    pred = p > THRESHOLD
    for k, tau in enumerate(shown):
        m = tolerant_f1(pred, Yt, float(tau))
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(err_rgb(pred, float(tau)), origin="lower",
                  interpolation="nearest")
        head = {0: "τ = 0   strict"}.get(tau, "τ = %d" % tau)
        ax.set_title("%s\nF1@%d %.3f" % (head, tau, m["f1"]), fontsize=11.5,
                     color="#1f5fa8" if tau == 1 else INK,
                     fontweight="bold" if tau == 1 else "normal")
        blank(ax)
        if tau == 1:
            for sp in ax.spines.values():
                sp.set_color("#1f5fa8")
                sp.set_linewidth(2.2)

    fig.text(0.5, 0.975, "cutting the probability map at two thresholds",
             ha="center", fontsize=10.5, color=MUT)
    fig.text(0.5, 0.492, "the SAME prediction (P > %g), scored at three "
             "tolerances τ — a predicted pixel counts as correct "
             "when a true line lies within τ pixels" % THRESHOLD,
             ha="center", fontsize=10.5, color=MUT)
    for _x, _t, _c in (
            (0.22, "found  — a predicted line that really is there",
             HIT_HEX),
            (0.53, "missed  — a real line the model did not draw",
             MISS_HEX),
            (0.83, "false  — a line drawn where there is none",
             FALSE_HEX)):
        fig.text(_x, 0.018, "■  " + _t, ha="center", fontsize=10.5,
                 color=_c, fontweight="bold")

    save(fig, "fig_probability_to_lines")


# -- figure 4: the whole flow - input | network | output ------------------
def fig_model_flow():
    ch, _Yt, p = pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    fig = plt.figure(figsize=(13.0, 3.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 3.45, 1.25],
                          wspace=0.30, hspace=0.42,
                          left=0.02, right=0.985, top=0.86, bottom=0.06)

    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
               vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("channel 1 · measured signal", fontsize=9.5, pad=5)
    blank(ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax2.set_title("channel 2 · visited mask (%.1f %%)"
                  % (100 * vis.mean()), fontsize=9.5, pad=5)
    blank(ax2)

    axn = fig.add_subplot(gs[:, 1])
    draw_unet(axn, fs=0.92, io_labels=False)

    axo = fig.add_subplot(gs[:, 2])
    im = axo.imshow(p, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
                    interpolation="nearest")
    axo.set_title("P(transition line) per pixel", fontsize=10.5,
                  color=J_PTITLE, pad=6)
    fig.colorbar(im, ax=axo, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=7.5)
    blank(axo)

    def farrow(x0, y0, x1, y1):
        fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=15,
                                       linewidth=1.8, color="#444444"))
    b1, b2 = ax1.get_position(), ax2.get_position()
    bn, bo = axn.get_position(), axo.get_position()
    farrow(b1.x1 + 0.004, b1.y0 + b1.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(b2.x1 + 0.004, b2.y0 + b2.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(bn.x1 - 0.010, bn.y0 + bn.height * 0.62,
           bo.x0 - 0.006, bo.y0 + bo.height * 0.5)

    fig.text(0.5, 0.955, "Two measured maps in, one probability map out",
             ha="center", fontsize=12.5, color=INK)
    save(fig, "fig_model_flow")


# -- figure 5: the same three maps, each as its own standalone panel ------
# Separate files so a slide can lay them out itself, with its own arrows,
# instead of being handed one merged image.
def fig_panels():
    ch, _Yt, prob = pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    def panel(draw, title, name, colour=INK, cbar=False):
        fig, ax = plt.subplots(figsize=(2.7, 2.95))
        im = draw(ax)
        ax.set_title(title, fontsize=11, color=colour, pad=7)
        blank(ax)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        save(fig, name)

    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    panel(lambda ax: ax.imshow(np.where(vis > 0.5, sig, np.nan),
                               origin="lower", cmap=cm, vmin=0, vmax=1,
                               interpolation="nearest"),
          "channel 1\nmeasured sensor signal", "panel_channel1")
    panel(lambda ax: ax.imshow(1.0 - vis, origin="lower", cmap="gray",
                               vmin=0, vmax=1, interpolation="nearest"),
          "channel 2\nvisited mask (%.1f %% of the grid)"
          % (100 * vis.mean()), "panel_channel2")
    panel(lambda ax: ax.imshow(prob, origin="lower", cmap=J_CMAP, vmin=0,
                               vmax=1, interpolation="nearest"),
          "output\nP(transition line) per pixel", "panel_probability",
          colour=J_PTITLE, cbar=True)


# -- figure 6: the charge sensor, raw and with its background removed -----
# The raw sensor signal is a large smooth gradient with the transition lines
# riding on it as a small modulation - which is why the raw map looks like a
# featureless wash.  Showing both is the honest way to present it, and it is
# also the reason the whole problem is hard.
def fig_charge_sensor(device=DEVICE):
    from scipy.ndimage import gaussian_filter

    ux, uy, Z = ray_peaks.load_grid(device)
    gy, gx = np.gradient(Z)
    g = gx + gy
    detail = g - gaussian_filter(g, 3.0)                 # high-pass
    lim = np.percentile(np.abs(detail), 99.0)

    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))

    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="hot", extent=ext,
                   aspect="auto", interpolation="nearest")
    ax.set_title("as measured\na large smooth background", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=8)

    ax = axes[1]
    im = ax.imshow(detail, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=ext, aspect="auto", interpolation="nearest")
    ax.set_title("same data, slow background removed\nthe transition "
                 "lines appear", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=8)

    for ax in axes:
        ax.set_xlabel("$V_1$ (mV)", fontsize=9.5)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#999999")
    axes[0].set_ylabel("$V_2$ (mV)", fontsize=9.5)

    fig.tight_layout()
    save(fig, "panel_charge_sensor")


# -- figure 7: the measurement story for the picked device ----------------
def fig_measurement_panel():
    """charge sensor | ground truth | what the network is shown."""
    ux, uy, Z = ray_peaks.load_grid(DEVICE)
    m = ray_peaks.measure(DEVICE, N_RAYS, N_POINTS)
    ch = ray_peaks.to_channels(m, Z.shape)
    truth = ray_peaks.load_ground_truth(DEVICE) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))

    axes[0].imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                   interpolation="nearest")
    axes[0].set_title("charge sensor", fontsize=11)

    axes[1].imshow(1 - truth, origin="lower", cmap="gray", extent=ext,
                   aspect="auto", interpolation="nearest")
    axes[1].set_title("stability diagram\n(ground truth)", fontsize=11)

    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    vis = ch[ray_peaks.CH_VISITED]
    axes[2].imshow(np.where(vis > 0.5, ch[ray_peaks.CH_SIGNAL], np.nan),
                   origin="lower", cmap=cm, vmin=0, vmax=1, extent=ext,
                   aspect="auto", interpolation="nearest")
    axes[2].set_title("%d rays × %d points\nwhat the network is shown  "
                      "(%.1f %% of the grid)"
                      % (N_RAYS, N_POINTS, 100 * vis.mean()), fontsize=11)

    for ax in axes:
        blank(ax)
    fig.tight_layout()
    save(fig, "fig_measurement_panel")


# -- figure 8: probability -> lines, one file per panel -------------------
# TWO colours, plus a tolerance band.  The panels answer one question - how
# close is the model's output to the truth - so they carry exactly the two
# things being compared:
#
#     black  the ground truth transition line (1 pixel wide)
#     red    the model's output after thresholding
#     grey   the tolerance band: every pixel within tau of the truth
#
# The band is what makes tau visible.  Note that tau is symmetric: for
# PRECISION the truth is dilated (is this red pixel near a true line?), for
# RECALL the prediction is dilated (is this true pixel near a red one?).
# The band drawn here is the precision side, which is the visible one.
def _p2l_name(stem):
    return stem + "_notitle" if P2L_TITLES_OFF else stem


def _p2l_title(stem, text, colour, size, bold=False):
    """Record a panel title, and blank it when a deck supplies it."""
    P2L_TITLES[stem] = {"text": text.replace("$", ""), "colour": colour,
                        "size": size, "bold": bold}
    if not P2L_TITLES_OFF:
        return text
    # blanks of the SAME line count: an empty string has no height at all,
    # and matplotlib then reclaims the gap the box has to sit in
    return "\n".join([" "] * (text.count("\n") + 1))


def _zoom_window(truth, size=40):
    """Top-left corner of the size x size crop holding the most truth
    pixels, so the zoom row lands on lines rather than on empty diagram
    whatever device is picked."""
    from scipy.ndimage import uniform_filter
    H, W = truth.shape
    dens = uniform_filter(truth.astype(float), size, mode="constant")
    half = size // 2
    inner = dens[half:H - half, half:W - half]
    r, c = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return r, c


def fig_probability_panels(taus=(0, 1, 3)):
    from scipy.ndimage import distance_transform_edt

    _ch, Yt, p = pick_case()
    truth = Yt > 0.5
    d_true = distance_transform_edt(~truth)

    ux, uy, _Z = ray_peaks.load_grid(DEVICE)
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    def overlay(pred, tau):
        """grey band (truth +/- tau), the red output, then black truth."""
        rgb = np.ones(truth.shape + (3,))
        rgb[d_true <= tau] = J_BAND_RGB
        rgb[pred] = J_PRED_RGB
        rgb[truth] = J_TRUTH_RGB
        return rgb

    def axes_style(ax):
        ax.set_xlabel("$V_1$ (mV)", fontsize=9, color=INK, labelpad=2)
        ax.set_ylabel("$V_2$ (mV)", fontsize=9, color=INK, labelpad=2)
        ax.tick_params(labelsize=7.5, length=3, width=0.7, direction="out",
                       colors=INK)
        ax.set_xticks(np.round(np.linspace(ext[0], ext[1], 5), 2))
        ax.set_yticks(np.round(np.linspace(ext[2], ext[3], 5), 2))
        ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
                alpha=0.85)
        ax.set_axisbelow(False)           # the grid sits ON TOP of the image
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.8)

    def panel(name, draw, title, colour=INK, boxed=None, cbar=False):
        fig, ax = plt.subplots(figsize=(2.55, 2.85))
        im = draw(ax)
        ax.set_title(_p2l_title(name, title, colour, 10.5, bool(boxed)),
                     fontsize=10.5, color=colour, pad=6,
                     fontweight="bold" if boxed else "normal")
        axes_style(ax)
        if boxed:
            for sp in ax.spines.values():
                sp.set_color(boxed)
                sp.set_linewidth(2.2)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)
            cb.outline.set_linewidth(0.6)
        fig.tight_layout()
        save(fig, _p2l_name(name))

    def show(ax, rgb):
        return ax.imshow(rgb, origin="lower", extent=ext, aspect="auto",
                         interpolation="nearest")

    # 1 - the raw probability map
    panel("p2l_1_probability",
          lambda ax: ax.imshow(p, origin="lower", cmap=J_CMAP, vmin=0,
                               vmax=1, extent=ext, aspect="auto",
                               interpolation="nearest"),
          "U-Net output\n$P$(transition line)", colour=J_PTITLE, cbar=True)

    # 2, 3 - the same map cut at two thresholds.
    #
    # Two versions of each.  The tau = 1 pair is the manuscript's.  The
    # tau = 0 pair is for a talk, whose first row is about WHERE to cut and
    # so must show no tolerance band: tau has not been introduced yet at
    # that point.  Their titles say "exact match" rather than "F1@0" for the
    # same reason.
    for k, t in enumerate(ladder()):
        pred_t = p > t
        m = tolerant_f1(pred_t, Yt, 1.0)
        m0 = tolerant_f1(pred_t, Yt, 0.0)
        chosen = abs(t - THRESHOLD) < 1e-9
        head = "$P$ > %g%s" % (t, "  (chosen)" if chosen else "")
        stem = ("p2l_%d_threshold_%g" % (2 + k, t)).replace(".", "p")
        panel(stem,
              lambda ax, pr=pred_t: show(ax, overlay(pr, 1.0)),
              head + "\nF1@1 = %.3f" % m["f1"],
              colour=J_PRED if chosen else INK,
              boxed=J_PRED if chosen else None)
        panel(stem + "_tau0",
              lambda ax, pr=pred_t: show(ax, overlay(pr, 0.0)),
              head + "\nF1 = %.3f  (exact match)" % m0["f1"],
              colour=J_PRED if chosen else INK,
              boxed=J_PRED if chosen else None)
        print("    P > %g:  F1@1 = %.3f   exact-match F1 = %.3f"
              % (t, m["f1"], m0["f1"]))

    # 4, 5, 6 - ONE prediction, three tolerances: only the band changes
    pred = p > THRESHOLD
    for k, tau in enumerate(taus):
        m = tolerant_f1(pred, Yt, float(tau))
        head = ("τ = 0  (no band)" if tau == 0
                else "τ = %d" % tau)
        panel("p2l_%d_tau%d" % (4 + k, tau),
              lambda ax, tt=tau: show(ax, overlay(pred, float(tt))),
              "%s\nF1@%d = %.3f" % (head, tau, m["f1"]),
              colour=J_PRED if tau == 1 else INK,
              boxed=J_PRED if tau == 1 else None)

    # 7 - the tolerance curve: F1, and the precision / recall behind it
    row = next(r for r in _run.comparison_rows()
               if r["configuration"] == CONFIG)
    f1s = [float(row["f1@%d" % t]) for t in TAUS]
    prs = [float(row["precision@%d" % t]) for t in TAUS]
    rcs = [float(row["recall@%d" % t]) for t in TAUS]

    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    ax.plot(TAUS, prs, "--s", color="#777777", linewidth=1.0, markersize=3.6,
            label="precision", zorder=2)
    ax.plot(TAUS, rcs, ":^", color="#777777", linewidth=1.0, markersize=3.8,
            label="recall", zorder=2)
    ax.plot(TAUS, f1s, "-o", color=J_TRUTH, linewidth=1.8, markersize=5,
            label="F1", zorder=3)
    ax.plot([1], [f1s[1]], "o", color=J_PRED, markersize=12,
            markerfacecolor="none", markeredgewidth=1.8, zorder=4)
    for x, y in zip(TAUS, f1s):
        off, ha = {0: ((13, -4), "left"),
                   1: ((0, 16), "center")}.get(x, ((0, -16), "center"))
        ax.annotate("%.3f" % y, (x, y), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8.5,
                    color=J_PRED if x == 1 else INK)
    ax.set_xticks(list(TAUS))
    ax.set_ylim(min(min(f1s), min(prs), min(rcs)) - 0.08, 1.02)
    ax.set_xlabel("tolerance τ  (pixels)", fontsize=9.5, color=INK)
    ax.set_ylabel("score on %d held-out devices" % N_TEST, fontsize=9.5,
                  color=INK)
    ax.set_title(_p2l_title("p2l_7_tolerance_curve",
                            "τ = 0 → 1 is the big jump;\n"
                            "beyond that the curve flattens", INK, 10),
                 fontsize=10, color=INK, pad=6)
    ax.tick_params(labelsize=8.5, colors=INK)
    ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right",
              handlelength=2.2, borderpad=0.2, labelspacing=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
        ax.spines[sp].set_linewidth(0.8)
    fig.tight_layout()
    save(fig, _p2l_name("p2l_7_tolerance_curve"))

    # 8 - the colour key
    key = ((0.000, "ground truth — the real transition line",
            J_TRUTH, J_TRUTH),
           (0.360, "model output — after the P > %g cut" % THRESHOLD,
            J_PRED, J_PRED),
           (0.700, "τ band — within τ pixels of the truth",
            "#D9D9D9", INK))
    fig, ax = plt.subplots(figsize=(9.0, 0.40))
    ax.axis("off")
    for x, txt, swatch, tcol in key:
        ax.add_patch(plt.Rectangle((x, 0.34), 0.013, 0.34, facecolor=swatch,
                                   edgecolor="#888888", linewidth=0.6,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.020, 0.5, txt, transform=ax.transAxes, va="center",
                fontsize=9.0, color=tcol, fontweight="bold")
    save(fig, "p2l_8_legend")

    # 9 - the point of the whole story, zoomed until pixels are visible:
    #     the band grows with tau, the black and the red never move
    r0, c0 = _zoom_window(truth, 40)
    size = 40
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 3.00))
    for ax, tau in zip(axes, TAUS):
        ax.imshow(overlay(pred, float(tau))[r0:r0 + size, c0:c0 + size],
                  origin="lower", interpolation="nearest")
        m = tolerant_f1(pred, Yt, float(tau))
        ax.set_title("τ = %d\nP %.2f   R %.2f   F1 %.2f"
                     % (tau, m["precision"], m["recall"], m["f1"]),
                     fontsize=11, color=J_PRED if tau == 1 else INK,
                     fontweight="bold" if tau == 1 else "normal", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(J_PRED if tau == 1 else "#777777")
            sp.set_linewidth(1.8 if tau == 1 else 0.8)
    fig.suptitle(_p2l_title("p2l_9_tau_zoom",
                            "Same truth, same output — only the "
                            "tolerance band grows", INK, 14),
                 fontsize=14, color=INK, y=0.985)
    # the colour key rides inside this figure, so a slide does not need a
    # separate legend strip and the panels can be larger
    fig.tight_layout(rect=(0, 0.105, 1, 0.945))
    for x, txt, swatch, tcol in key:
        fig.patches.append(plt.Rectangle(
            (0.030 + x * 0.955, 0.030), 0.0125, 0.042, facecolor=swatch,
            edgecolor="#888888", linewidth=0.6,
            transform=fig.transFigure, figure=fig))
        fig.text(0.050 + x * 0.955, 0.051, txt, va="center", fontsize=10.5,
                 color=tcol, fontweight="bold")
    save(fig, _p2l_name("p2l_9_tau_zoom"))

    with open(os.path.join(HERE, "p2l_titles.json"), "w",
              encoding="utf-8") as fh:
        json.dump(P2L_TITLES, fh, ensure_ascii=False, indent=2)
    print("  titles %s -> p2l_titles.json"
          % ("blanked" if P2L_TITLES_OFF else "kept"))
    print("  threshold %g" % THRESHOLD)
    print("  tau:        " + "  ".join("%d" % t for t in TAUS))
    print("  F1:         " + "  ".join("%.3f" % v for v in f1s))
    print("  precision:  " + "  ".join("%.3f" % v for v in prs))
    print("  recall:     " + "  ".join("%.3f" % v for v in rcs))


# -- figure 9: the headline results chart ---------------------------------
# The gallery version (figures/02_f1_vs_coverage.png) labels all fifteen
# points and the labels collide.  Here the budgets are GROUPED BY RAY COUNT
# - one line each - so the reading "more rays beats more points at the same
# coverage" is the shape of the plot rather than something the caption has
# to assert.
#
# The runs that did not converge are NOT silently dropped: they are drawn as
# open grey circles, off the lines, and named in the legend.  A reader can
# see that points are missing from the trend, and why.
def fig_results_f1_vs_coverage():
    ok = _run.converged()
    good, bad = {}, []
    for r in _run.comparison_rows():
        pt = (100.0 * float(r["coverage"]), float(r["f1@1"]),
              int(r["n_points"]))
        if ok[r["configuration"]]:
            good.setdefault(int(r["n_rays"]), []).append(pt)
        else:
            bad.append(pt)
    for v in good.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    best_n = max(good)
    for n in sorted(good):
        x = [c for c, _f, _p in good[n]]
        y = [f for _c, f, _p in good[n]]
        colour, marker = STYLE[n]
        best = n == best_n
        ax.plot(x, y, marker=marker, linestyle="-", color=colour,
                linewidth=2.2 if best else 1.5,
                markersize=6.5 if best else 5.0,
                markeredgecolor="white" if best else colour,
                markeredgewidth=0.8 if best else 0.6,
                zorder=4 if best else 2, label="%d rays" % n)
    if bad:
        ax.plot([c for c, _f, _p in bad], [f for _c, f, _p in bad], "o",
                markerfacecolor="none", markeredgecolor="#9a9a9a",
                markersize=6.0, markeredgewidth=1.1, linestyle="none",
                zorder=3, label="did not converge")

    # the headline point, read off the run rather than typed in
    row = next(r for r in _run.comparison_rows()
               if r["configuration"] == CONFIG)
    hx, hy = 100.0 * float(row["coverage"]), float(row["f1@1"])
    ax.annotate("%d × %d\nF1 %.3f at %.1f %%"
                % (N_RAYS, N_POINTS, hy, hx),
                xy=(hx, hy), xytext=(hx - 0.55, hy + 0.045), fontsize=9,
                color=J_PRED, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=J_PRED, linewidth=0.9))

    xs = ([c for v in good.values() for c, _f, _p in v]
          + [c for c, _f, _p in bad])
    ys = ([f for v in good.values() for _c, f, _p in v]
          + [f for _c, f, _p in bad])
    ax.set_xlabel("fraction of the grid measured  (%)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("F1 @ 1 px tolerance", fontsize=10.5, color=INK)
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.45)
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.10)
    ax.tick_params(labelsize=9, colors=INK)
    ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.85)
    ax.set_axisbelow(True)
    # Centre right: the lower right is where the runs that did not
    # converge land and the upper left is where the 4/5/6-ray lines run, so
    # a box in either corner hides real points.
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92,
              edgecolor="#cccccc", loc="center right", handlelength=1.9,
              labelspacing=0.35, borderpad=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
        ax.spines[sp].set_linewidth(0.8)
    fig.tight_layout()
    save(fig, "results_f1_vs_coverage")
    print("  headline %s: F1@1 %.3f at %.1f %% coverage" % (CONFIG, hy, hx))
    print("  %d of %d runs excluded from the trend lines"
          % (len(bad), len(ok)))


# -- figure 10: where the validation devices come from --------------------
# The dataset table says 500 / 50 and the validation split is invisible in
# it, which is the one thing a reviewer will ask about: the threshold is a
# fitted quantity, so WHERE it was fitted has to be on the figure.
#   550 devices -> 500 training pool + 50 test          (split_seed 12345)
#   500 pool    -> 425 fit weights + 75 validation      (default_rng(0))
SPLIT_LABEL = []
SPLIT_LABEL_OFF = os.environ.get("SPLIT_LABEL") == "off"


def fig_data_split():
    cfg = _run.cfg_json()
    n_pool, n_test = int(cfg["n_train"]), int(cfg["n_test"])
    n_total = n_pool + n_test
    n_val = max(1, int(grid_train.VAL_FRACTION * n_pool))
    n_fit = n_pool - n_val

    pool_c, test_c = "#c8c8c8", "#1a1a1a"
    fit_c, val_c = "#e2e2e2", J_PRED

    # Only the bars are drawn here.  The heading, the colour key and the
    # footnote can be real text boxes on a slide, so they can be edited or
    # deleted without regenerating a figure.
    fig, ax = plt.subplots(figsize=(7.4, 1.30))
    ax.set_xlim(0, n_total)
    ax.set_ylim(0.22, 0.88)
    ax.axis("off")

    def bar(x0, w, y, h, fc):
        ax.add_patch(plt.Rectangle((x0, y), w, h, facecolor=fc,
                                   edgecolor="#444444", linewidth=0.9))

    # -- level 1: the split that is stored with the device pool -----------
    bar(0, n_pool, 0.63, 0.22, pool_c)
    bar(n_pool, n_test, 0.63, 0.22, test_c)
    ax.text(n_pool / 2, 0.74, "%d training devices" % n_pool, ha="center",
            va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(n_pool + n_test / 2, 0.74, "%d\ntest" % n_test, ha="center",
            va="center", fontsize=9, color="white", fontweight="bold",
            linespacing=1.2)

    # -- the carve-out, before any training -------------------------------
    for x in (0, n_pool):
        ax.plot([x, x], [0.63, 0.52], color="#999999", linewidth=0.8,
                linestyle=(0, (2, 2)))
    ax.annotate("", xy=(n_pool / 2, 0.50), xytext=(n_pool / 2, 0.61),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                linewidth=1.1))
    SPLIT_LABEL[:] = [n_pool / 2 + 10, 0.555]

    # -- level 2: inside the training devices -----------------------------
    bar(0, n_fit, 0.26, 0.22, fit_c)
    bar(n_fit, n_val, 0.26, 0.22, val_c)
    ax.text(n_fit / 2, 0.37, "%d  fit the weights" % n_fit, ha="center",
            va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(n_fit + n_val / 2, 0.37, "%d\nvalidation" % n_val, ha="center",
            va="center", fontsize=9, color="white", fontweight="bold",
            linespacing=1.2)

    fig.tight_layout(pad=0.2)
    if SPLIT_LABEL_OFF:
        fig.canvas.draw()
        px = ax.transData.transform(tuple(SPLIT_LABEL))
        fx, fy = fig.transFigure.inverted().transform(px)
        with open(os.path.join(HERE, "data_split_label.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"x": float(fx), "y": float(fy)}, fh, indent=2)
        print("    label slot at x=%.3f y=%.3f of the figure" % (fx, fy))
        save(fig, "fig_data_split_notext")
    else:
        ax.text(SPLIT_LABEL[0], SPLIT_LABEL[1],
                "%d %% carved out, before any training"
                % round(100 * grid_train.VAL_FRACTION),
                fontsize=8.8, color=MUT, va="center")
        save(fig, "fig_data_split")


# -- figure 11: every metric against tolerance, three budgets -------------
# The gallery version (figures/40_tau_all_metrics.png) draws all fifteen
# budgets and four metrics: sixteen indistinguishable lines per panel.  This
# keeps the smallest budget, a middle one and the best - all three of them
# runs that converged - and drops the pixel accuracy panel: accuracy is high
# whatever the model does, which is a point to make out loud, not a panel to
# squint at.
def _three_budgets():
    """Smallest, middle and best coverage among the CONVERGED runs."""
    ok = _run.converged()
    rows = sorted((r for r in _run.comparison_rows()
                   if ok[r["configuration"]]),
                  key=lambda r: float(r["coverage"]))
    pick = [rows[0], rows[len(rows) // 2], rows[-1]]
    return [(int(r["n_rays"]), int(r["n_points"])) for r in pick]


def fig_tau_metrics(budgets=None):
    budgets = budgets or _three_budgets()
    rows = {(int(r["n_rays"]), int(r["n_points"])): r
            for r in _run.comparison_rows()}
    metrics = (("f1", "F1"), ("precision", "precision"), ("recall", "recall"))
    best = max(budgets, key=lambda b: float(rows[b]["coverage"]))

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.15), sharey=True)
    lo = 1.0
    for ax, (key, label) in zip(axes, metrics):
        for budget in budgets:
            r = rows[budget]
            y = [float(r["%s@%d" % (key, t)]) for t in TAUS]
            lo = min(lo, min(y))
            colour, marker = STYLE[budget[0]]
            is_best = budget == best
            ax.plot(TAUS, y, marker=marker, linestyle="-", color=colour,
                    linewidth=2.0 if is_best else 1.4,
                    markersize=5.5 if is_best else 4.5,
                    markeredgecolor="white" if is_best else colour,
                    markeredgewidth=0.8 if is_best else 0.6,
                    zorder=4 if is_best else 2,
                    label="%d × %d   (%.1f %%)"
                          % (budget[0], budget[1],
                             100 * float(r["coverage"])))
        ax.set_title(label, fontsize=10.5, color=INK, pad=5, loc="left")
        ax.set_xlabel("tolerance τ  (pixels)", fontsize=9.5, color=INK)
        ax.set_xticks(list(TAUS))
        ax.tick_params(labelsize=8.5, colors=INK, labelleft=True)
        ax.set_ylabel("score on %d\nheld-out devices" % N_TEST, fontsize=8.5,
                      color=INK, linespacing=1.3)
        ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
                alpha=0.85)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#444444")
            ax.spines[sp].set_linewidth(0.8)
    axes[0].set_ylim(lo - 0.06, 1.02)

    axes[0].legend(fontsize=8, frameon=True, framealpha=0.92,
                   edgecolor="#cccccc", loc="lower right", handlelength=1.8,
                   labelspacing=0.3, borderpad=0.3,
                   title="rays × points  (coverage)", title_fontsize=8)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    save(fig, "fig_tau_metrics")
    print("  budgets: " + ", ".join("%d x %d" % b for b in budgets))


def main():
    print("run:     ", os.path.basename(RUN_DIR))
    print("budget:  ", CONFIG)
    print("device:  ", os.path.basename(DEVICE),
          " F1@1 %.3f  (test mean %.3f over %d devices)"
          % (DEVICE_F1, TEST_MEAN_F1, N_TEST))
    print("figures ->", HERE)
    fig_measurement_panel()
    fig_network_input()
    fig_network_input(compact=True)
    fig_unet()
    fig_model_flow()
    fig_panels()
    fig_charge_sensor()
    fig_probability_to_lines()
    fig_probability_panels()
    fig_results_f1_vs_coverage()
    fig_tau_metrics()
    fig_data_split()


if __name__ == "__main__":
    main()


# -- figure 12: the budget ladder, for the Results slide ------------------
# The gallery draws all fifteen budgets; the deck's old chart drew them as
# three series of five.  Both are too busy to read from the back of a room,
# and in THIS run the 60-points/ray series cannot carry a line at all --
# three of its five trainings failed.
#
# So: ONE ladder (40 points per ray, all five converged, no gaps), the best
# budget overall marked, and ONE same-coverage pair that demonstrates "more
# rays beat more points per ray" instead of asserting it.  Seven marks.
LADDER_POINTS = int(os.environ.get("CSD_LADDER_POINTS", 40))


def fig_budget_ladder():
    ok = _run.converged()
    rows = {(int(r["n_rays"]), int(r["n_points"])): r
            for r in _run.comparison_rows()}

    def pt(n_rays, n_points):
        r = rows[(n_rays, n_points)]
        return 100.0 * float(r["coverage"]), float(r["f1@1"])

    line = sorted(((n, LADDER_POINTS) for n in (4, 5, 6, 7, 8)
                   if ok["%d_rays_%d_points_500_samples"
                         % (n, LADDER_POINTS)]),
                  key=lambda b: pt(*b)[0])
    xs = [pt(*b)[0] for b in line]
    ys = [pt(*b)[1] for b in line]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(xs, ys, "-o", color="#08519C", markersize=6.0, linewidth=2.0,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            label="%d points per ray" % LADDER_POINTS)
    for (n, _p), x, y in zip(line, xs, ys):
        ax.annotate("%d" % n, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", va="bottom", fontsize=8,
                    color="#08519C", zorder=5)

    # the best budget that converged, wherever it sits
    best_cfg = max((r for r in _run.comparison_rows()
                    if ok[r["configuration"]]),
                   key=lambda r: float(r["f1@1"]))
    bn, bp = int(best_cfg["n_rays"]), int(best_cfg["n_points"])
    bx, by = pt(bn, bp)
    ax.plot([bx], [by], "o", color=J_PRED, markersize=7.0,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.plot([bx], [by], "o", markersize=14, markerfacecolor="none",
            markeredgecolor=J_PRED, markeredgewidth=1.4, zorder=5)
    ax.annotate("best:  %d × %d\nF1@1 %.3f at %.1f %%"
                % (bn, bp, by, bx), xy=(bx, by),
                xytext=(bx - 0.18, by + 0.030), fontsize=8.5,
                color=J_PRED, fontweight="bold", ha="right", va="bottom")

    # one same-coverage pair: the "more rays" claim, drawn rather than said
    pair = None
    for n_rays, n_points in ((4, 60), (4, 50)):
        if not ok["%d_rays_%d_points_500_samples" % (n_rays, n_points)]:
            continue
        px, py = pt(n_rays, n_points)
        near = min(line, key=lambda b: abs(pt(*b)[0] - px))
        nx, ny = pt(*near)
        if abs(nx - px) < 0.10 and ny > py:
            pair = (n_rays, n_points, px, py, near[0], nx, ny)
            break
    if pair:
        n_rays, n_points, px, py, m_rays, nx, ny = pair
        ax.plot([px], [py], "o", markerfacecolor="white",
                markeredgecolor="#777777", markersize=6.0,
                markeredgewidth=1.3, zorder=4)
        ax.annotate("", xy=(nx, ny - 0.008), xytext=(px, py + 0.008),
                    arrowprops=dict(arrowstyle="-|>", color="#444444",
                                    linewidth=1.1, shrinkA=3, shrinkB=3))
        ax.annotate("%d × %d" % (n_rays, n_points), (px, py),
                    textcoords="offset points", xytext=(0, -13),
                    ha="center", va="top", fontsize=8, color="#555555")
        ax.text(px + 0.08, (py + ny) / 2,
                "same coverage,\n%d more rays:  +%.3f"
                % (m_rays - n_rays, ny - py),
                fontsize=8, color="#444444", va="center", ha="left")

    ax.set_xlabel("fraction of the grid actually measured  (%)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("F1@1 on %d held-out devices" % N_TEST, fontsize=9.5,
                  color=INK)
    ax.set_xlim(1.3, 4.5)
    ax.set_ylim(0.62, 0.86)
    ax.grid(True, linestyle=":", linewidth=0.7, color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK, length=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
    leg = ax.legend(fontsize=8.5, frameon=True, loc="lower right",
                    handlelength=2.0)
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.30)
    n_bad = sum(1 for v in ok.values() if not v)
    ax.text(0.0, -0.26,
            "the number beside each point is the ray count.\n"
            "%d of the %d budgets did not converge and are not shown."
            % (n_bad, len(ok)),
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            ha="left", va="top")
    save(fig, "results_budget_ladder")
    print("  ladder: %s" % ", ".join("%d x %d" % b for b in line))
    print("  best:   %d x %d  F1@1 %.3f at %.1f %%" % (bn, bp, by, bx))
