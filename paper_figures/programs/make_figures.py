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

Every figure is written as .png at 600 dpi.  Set WITH_PDF = True below to
get a vector .pdf beside each one.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Write a vector .pdf beside every .png.  Off: the PNGs are 600 dpi
# and the PDFs doubled the folder for nothing.
WITH_PDF = False

HERE = os.path.dirname(os.path.abspath(__file__))   # programs/
sys.path.insert(0, HERE)

import _figures                                                # noqa: E402
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
        _MODEL = grid_train.load(os.path.join(_run.model_dir(), "unet.pt"))
    return _MODEL


def save(fig, name, pdf=False):
    # PNG only.  The vector twin was written for a journal that wants
    # one; set WITH_PDF = True to get it back.  _figures decides WHERE the
    # file goes, so the folder layout lives in one place.
    for ext, kw in ((".png", {"dpi": 600}),) + (
            ((".pdf", {}),) if (WITH_PDF or pdf) else ()):
        p = _figures.out_path(name + ext)
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
    """The two input channels.  The paper version (compact=False) carries
    no text beyond the colour bar and a panel letter inside each axes --
    the caption says what each channel is (2026-09-26).  The slide version
    keeps its titles and notes, because a slide has no caption."""
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
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("normalised charge-sensor signal", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # channel 2 - where we looked at all
    ax = axes[1]
    ax.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")

    for ax in axes:
        blank(ax)

    if compact:
        axes[0].set_title("channel 1   —   measured sensor signal",
                          fontsize=11, color=INK, pad=9)
        axes[0].set_xlabel("grey = never measured", fontsize=9, color=MUT)
        axes[1].set_title("channel 2   —   visited mask", fontsize=11,
                          color=INK, pad=9)
        axes[1].set_xlabel("black = a ray passed here  (%.1f %% of the grid)"
                           % cov, fontsize=9, color=MUT)
    else:
        # an empty colour bar keeps the two panels the same size
        fig.colorbar(im, ax=axes[1], fraction=0.046,
                     pad=0.03).ax.set_visible(False)
        # top left: the rays all leave from the top-right corner
        for ax, letter in zip(axes, "ab"):
            _panel_letter(ax, letter, x=0.03, ha="left")
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
            fontsize=11 * _BLOCK_FS, color="white", fontweight="bold",
            zorder=3)
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
    """Input | network | output, in one picture, for the paper (Fig. 6).

    The two input channels are drawn as two overlapping squares -- channel
    1 in front, channel 2 behind, offset up and right -- because they are
    one two-channel tensor, and the stack keeps the figure narrow enough
    to read at text width.  No titles over the panels: the caption says
    what each part is (2026-09-26).  Laid out in inches so the squares
    stay square."""
    ch, _Yt, p = pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    # inputs and output kept small so the network gets the room
    side, off, oside = 1.7, 0.35, 1.8
    nx, ny, nw, nh = 2.25, 0.0, 5.4, 3.0
    ox = nx + nw * 0.895 + 0.38
    W, H = ox + oside + 0.72, 3.0          # figure, inches
    fig = plt.figure(figsize=(W, H))

    def axes_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    # -- the input: two squares, channel 2 behind channel 1 ---------------
    x0, y0 = 0.08, (H - side - off) / 2
    back = axes_in(x0 + off, y0 + off, side, side, zorder=1)
    back.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
                interpolation="nearest")
    front = axes_in(x0, y0, side, side, zorder=2)
    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    front.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
                 vmin=0, vmax=1, interpolation="nearest")
    for ax, tag in ((front, "1"), (back, "2")):
        blank(ax)
        for sp in ax.spines.values():
            sp.set_color("#555555")
            sp.set_linewidth(0.9)
        # which channel: a small tag inside the square, not a title
        ax.text(0.04, 0.96 if ax is back else 0.04, tag,
                transform=ax.transAxes, ha="left",
                va="top" if ax is back else "bottom", fontsize=9,
                fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#999999", lw=0.5))

    # -- the network -------------------------------------------------------
    axn = axes_in(nx, ny, nw, nh)
    draw_unet(axn, fs=0.9, io_labels=False)

    # -- the output --------------------------------------------------------
    oy = (H - oside) / 2
    axo = axes_in(ox, oy, oside, oside)
    im = axo.imshow(p, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
                    interpolation="nearest")
    blank(axo)
    cax = axes_in(ox + oside + 0.08, oy, 0.1, oside)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("P(transition line)", fontsize=7.5)

    # -- short arrows, from edge to edge ------------------------------------
    def to_fig(ax, xd, yd):
        return fig.transFigure.inverted().transform(
            ax.transData.transform((xd, yd)))

    def arrow(p0, p1):
        fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9,
                                       linewidth=1.1, color="#444444"))
    yin = to_fig(axn, 0.15, 0.66)[1]              # the "2" block, mid-height
    arrow(((x0 + off + side + 0.03) / W, yin), (to_fig(axn, 0.13, 0.66)[0],
                                                 yin))
    yout = to_fig(axn, 8.95, 0.66)[1]
    arrow((to_fig(axn, 9.0, 0.66)[0], yout), ((ox - 0.04) / W, yout))

    save(fig, "fig_model_flow")


# -- figure 5: the same three maps, each as its own standalone panel ------
# Separate files so a slide can lay them out itself, with its own arrows,
# instead of being handed one merged image.
def fig_panels():
    ch, _Yt, prob = pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    def panel(draw, note, name, colour=INK, cbar=False):
        """One standalone map.  `note` goes UNDER it: these are separate
        files whose names already say which map they are, and the deck
        places its own title box above each one."""
        fig, ax = plt.subplots(figsize=(2.7, 2.95))
        im = draw(ax)
        blank(ax)
        if note:
            ax.set_xlabel(note, fontsize=8.5, color=MUT)
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
          "", "panel_channel1")
    panel(lambda ax: ax.imshow(1.0 - vis, origin="lower", cmap="gray",
                               vmin=0, vmax=1, interpolation="nearest"),
          "%.1f %% of the grid" % (100 * vis.mean()), "panel_channel2")
    panel(lambda ax: ax.imshow(prob, origin="lower", cmap=J_CMAP, vmin=0,
                               vmax=1, interpolation="nearest"),
          "", "panel_probability", colour=J_PTITLE, cbar=True)


# -- figure 6: the charge sensor, and the lines the simulator knows -------
# Three panels: what the sensor reads, the same reading with its slow
# background taken off, and the transition lines themselves.
#
# THE DEVICE IS NOT `DEVICE`.  This figure is the only one that shows a
# stability diagram for its own sake rather than as the input to a
# reconstruction, and sample_157 -- the device every other figure follows
# -- is a poor one to show: its sensor response is concentrated in one
# corner, so the honeycomb is invisible on a linear scale.  SENSOR_DEVICE
# is the third device of the held-out split, whose response spreads across
# the window; the steps are legible in the raw signal, with no stretching
# of the colour scale.
SENSOR_DEVICE = "sample_10"
EDGE = 2                        # px trimmed: the gradient is undefined there


def _sensor_dir(name=None):
    """The device folder for this figure, beside the one DEVICE lives in."""
    return os.path.join(os.path.dirname(DEVICE), name or SENSOR_DEVICE)


def _raw_grid(sample_dir):
    """(ux, uy, Z) with Z in the simulator's own units, NOT normalised.

    ray_peaks.load_grid rescales to [0, 1] because the pipeline does; the
    colour bar here is meant to carry the signal the simulator produced,
    so the same file is read without that step."""
    path = os.path.join(sample_dir, "numpy", "simulation",
                        ray_peaks.SENSOR_GRID)
    data = np.load(path)
    ux, uy = np.unique(data[:, 0]), np.unique(data[:, 1])
    return ux, uy, data[:, 2].reshape(len(uy), len(ux)).astype(np.float64)


def _panel_letter(ax, letter, x=0.97, y=0.97, ha="right", va="top"):
    """The panel letter, inside the axes -- no title above a panel."""
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, ha=ha, va=va,
            fontsize=11, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999",
                      lw=0.6))


def _sensor_colorbar(fig, ax, im, Z):
    """Colour bar for the raw sensor signal, its exponent in the label."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    exp = int(np.floor(np.log10(np.abs(Z).max())))
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _pos: "%.1f" % (v / 10.0 ** exp)))
    cb.set_label(r"charge sensor signal z ($\times 10^{%d}$)" % exp,
                 fontsize=8.5)
    return cb


def fig_inputs(device=None):
    """One held-out device, from measurement to target (2026-09-26).

    (a) the charge-sensor map a dense scan would record, (b) the
    ground-truth transition lines -- the U-Net's target -- and the two
    channels the U-Net is actually given at the headline budget: (c) the
    normalised sensor signal on the sampled pixels and (d) the sampling
    mask.  Replaces the separate charge-sensor, ray-fan and input-channel
    figures.

    The device is SENSOR_DEVICE (sample_10), a held-out TEST device, not
    the picked DEVICE of the prediction figures: in raw sensor units the
    background swamps the steps on most devices, and sample_10 is one
    whose steps are visible (5th of 50 test devices by line-to-background
    gradient ratio).  Its F1@1 is 0.825 against the test mean 0.796."""
    dev = _sensor_dir(device)
    ux, uy, Z = _raw_grid(dev)
    truth = ray_peaks.load_ground_truth(dev) > 0.5
    _u, _v, Zn = ray_peaks.load_grid(dev)
    m = ray_peaks.measure(dev, N_RAYS, N_POINTS)
    ch = ray_peaks.to_channels(m, Zn.shape)
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.6))
    (a, b), (c, d) = axs

    im = a.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                  interpolation="nearest")
    _sensor_colorbar(fig, a, im, Z)

    b.imshow(1 - truth, origin="lower", cmap="gray", vmin=0, vmax=1,
             extent=ext, aspect="auto", interpolation="nearest")

    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#e6e6e6")
    imc = c.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
                   vmin=0, vmax=1, extent=ext, aspect="auto",
                   interpolation="nearest")
    cb = fig.colorbar(imc, ax=c, fraction=0.046, pad=0.03)
    cb.set_label("normalised signal", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    d.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
             extent=ext, aspect="auto", interpolation="nearest")
    for ax in (b, d):                         # same width as (a) and (c)
        fig.colorbar(imc, ax=ax, fraction=0.046,
                     pad=0.03).ax.set_visible(False)

    # each letter in a corner that is empty in that panel
    corner = {"a": (0.03, "left"), "b": (0.97, "right"),
              "c": (0.03, "left"), "d": (0.03, "left")}
    for ax, letter in zip((a, b, c, d), "abcd"):
        ax.set_xlabel("$V_1$ (mV)", fontsize=9, color=INK)
        ax.set_ylabel("$V_2$ (mV)", fontsize=9, color=INK)
        ax.tick_params(labelsize=7.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.8)
        _panel_letter(ax, letter, x=corner[letter][0], ha=corner[letter][1])
    fig.tight_layout()
    save(fig, "fig_inputs")


def prob_map_stats():
    """What the probability map says beyond its thresholded lines.

    Written to prob_map_stats.json for the Results and Conclusions text
    (2026-09-27): on every held-out device of the headline budget, how
    much of the plane is confidently off a line, how much is uncertain and
    where that uncertainty lies, and how many of the thresholded errors
    carry an intermediate probability.  Calibration is NOT measured."""
    from scipy.ndimage import distance_transform_edt
    from csdrecon.study.dataset import load_split
    model, ck = net()
    X, Y, _ = load_split(os.path.join(CFG_DIR, "test.npz"))
    P = grid_train.predict(model, X[:, :ray_peaks.NET_CHANNELS])
    Y = Y > 0.5
    mid = (P > 0.1) & (P < 0.9)
    near = sum(int((distance_transform_edt(~y)[m] <= 2).sum())
               for y, m in zip(Y, mid))
    B = P > float(ck["threshold"])
    fp, fn = B & ~Y, ~B & Y
    out = {"n_devices": int(len(P)), "threshold": float(ck["threshold"]),
           "confident_off_pct": 100.0 * float((P <= 0.1).mean()),
           "uncertain_pct": 100.0 * float(mid.mean()),
           "uncertain_near_line_pct": 100.0 * near / float(mid.sum()),
           "fp_uncertain_pct": 100.0 * float((mid & fp).sum() / fp.sum()),
           "fn_uncertain_pct": 100.0 * float((mid & fn).sum() / fn.sum())}
    with open(_figures.out_path("prob_map_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("  prob-map stats:", out)
    return out


def fig_charge_sensor(device=None):
    """(a) the raw charge-sensor signal, (b) the transition lines.

    No titles over the panels: what each one is goes in the caption, and
    the panel letter sits inside the axes, top left.  The differential
    panel that used to sit between them was dropped (2026-09-26); it is
    still drawn in panel_charge_sensor_big_b."""
    sample = _sensor_dir(device)
    ux, uy, Z = _raw_grid(sample)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))

    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="hot", extent=ext,
                   aspect="auto", interpolation="nearest")
    _sensor_colorbar(fig, ax, im, Z)

    ax = axes[1]
    ax.imshow(1 - truth, origin="lower", cmap="gray", vmin=0, vmax=1,
              extent=ext, aspect="auto", interpolation="nearest")
    # No colour bar: the panel has two values, and the caption says which.
    # An empty one keeps the two panels the same size.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.set_visible(False)

    # top right: the corner where the lines of this device do not run
    for ax, letter in zip(axes, "ab"):
        _panel_letter(ax, letter)
        ax.set_xlabel("$V_1$ (mV)", fontsize=9.5)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#999999")
    axes[0].set_ylabel("$V_2$ (mV)", fontsize=9.5)

    fig.tight_layout()
    save(fig, "panel_charge_sensor")


# -- figure 6b: the same three panels, with (b) given the room -----------
# Panel (b) is where the honeycomb actually reads, and at one third of a
# 11.4-inch row the steps are a few pixels wide.  This variant keeps the
# same data and the same colour scales and simply re-apportions the space:
# (b) fills a tall left column, (a) and (c) sit stacked beside it as
# thumbnails.  Same device, same figure content -- only the layout differs.
def fig_charge_sensor_big_b(device=None):
    sample = _sensor_dir(device)
    ux, uy, Z = _raw_grid(sample)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    gy, gx = np.gradient(Z)
    # Panel (b) is the DIFFERENTIAL sensor output: the derivative of
    # the same signal along the window diagonal, which is the
    # direction that crosses both families of transition lines.  It
    # is what a lock-in measurement returns, so the panel shows a
    # quantity an experiment produces rather than a processing step.
    # It used to be that derivative with a Gaussian high-pass taken
    # off it; the high-pass flattened the slow background better but
    # was not itself a measurable.
    detail = gx + gy

    e = EDGE
    detail = detail[e:-e, e:-e]
    lim = np.percentile(np.abs(detail), 99.0)

    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    dx = (ux.max() - ux.min()) / Z.shape[1]
    dy = (uy.max() - uy.min()) / Z.shape[0]
    ext_cut = [ux.min() + e * dx, ux.max() - e * dx,
               uy.min() + e * dy, uy.max() - e * dy]

    fig = plt.figure(figsize=(10.6, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0],
                          wspace=0.28, hspace=0.30)

    ax_b = fig.add_subplot(gs[:, 0])
    im = ax_b.imshow(detail, origin="lower", cmap="RdBu_r", vmin=-lim,
                     vmax=lim, extent=ext_cut, aspect="auto",
                     interpolation="nearest")
    ax_b.set_title("(b)  differential sensor output", fontsize=13)
    cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=9)

    ax_a = fig.add_subplot(gs[0, 1])
    im = ax_a.imshow(Z, origin="lower", cmap="hot", extent=ext,
                     aspect="auto", interpolation="nearest")
    ax_a.set_title("(a)  charge sensor output", fontsize=10)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("charge sensor signal z", fontsize=7.5)

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.imshow(1 - truth, origin="lower", cmap="gray", vmin=0, vmax=1,
                extent=ext, aspect="auto", interpolation="nearest")
    ax_c.set_title("(c)  transition lines", fontsize=10)

    for ax, fs in ((ax_b, 12), (ax_a, 9), (ax_c, 9)):
        ax.set_xlabel("$V_1$ (mV)", fontsize=fs)
        ax.set_ylabel("$V_2$ (mV)", fontsize=fs)
        ax.tick_params(labelsize=fs - 2)
        for sp in ax.spines.values():
            sp.set_color("#999999")

    save(fig, "panel_charge_sensor_big_b")


# -- figure 0 tiles: the pictures the Illustrator figure places -----------
# Figure 1 is drawn in Illustrator (docs/figures/build_overview.jsx), but
# the DATA in it is not drawn there: these four tiles are written from the
# run and placed into the layout.  Each is square, frameless and
# axis-free, because in the finished figure it sits inside a circle and
# its label is set in Illustrator.
#
# One device for all four, the same one the ray figures use, and it is in
# the held-out split -- so the probability tile is a held-out result.
TILE_PX = 3.2                   # inches; 600 dpi -> ~1900 px
TILE_LINE = "#00B0F0"           # exact transition lines over `hot`
TILE_RAY = "#FFFFFF"            # the rays


def _tile(name):
    fig, ax = plt.subplots(figsize=(TILE_PX, TILE_PX))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def _tile_save(fig, name):
    fig.subplots_adjust(0, 0, 1, 1)
    p = _figures.out_path(name + ".png")
    fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.0,
                facecolor="white")
    print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)


def fig_overview_tiles():
    import matplotlib.patheffects as pe

    sample = _sensor_dir()
    ux, uy, Z = _raw_grid(sample)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    m = ray_peaks.measure(sample, N_RAYS, N_POINTS)
    _gx, _gy, Zn = ray_peaks.load_grid(sample)
    ch = ray_peaks.to_channels(m, Zn.shape)
    prob = grid_train.predict(net()[0],
                              ch[None, :ray_peaks.NET_CHANNELS])[0]

    def _lines(ax):
        ax.contour(np.linspace(ux.min(), ux.max(), truth.shape[1]),
                   np.linspace(uy.min(), uy.max(), truth.shape[0]),
                   truth.astype(float), levels=[0.5], colors=[TILE_LINE],
                   linewidths=1.6)

    # 1. the simulated diagram, with the lines that label it
    fig, ax = _tile("diagram")
    ax.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
              interpolation="nearest")
    _lines(ax)
    _tile_save(fig, "tile_diagram")

    # 2. the same diagram, and the rays that are all that is measured
    fig, ax = _tile("rays")
    ax.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
              interpolation="nearest")
    stroke = [pe.withStroke(linewidth=3.0, foreground="#202020")]
    for a in ray_peaks.fan_angles(N_RAYS):
        pts = ray_peaks.ray_polyline(a, N_POINTS, ux, uy)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=TILE_RAY, lw=1.4,
                path_effects=stroke)
        ax.plot(pts[:, 0], pts[:, 1], ".", color=TILE_RAY, ms=2.2,
                path_effects=stroke)
    ax.set_xlim(ux.min(), ux.max())
    ax.set_ylim(uy.min(), uy.max())
    _tile_save(fig, "tile_rays")

    # 3. the two channels, the whole of what the network is given
    vis = ch[ray_peaks.CH_VISITED] > 0.5
    sig = np.where(vis, ch[ray_peaks.CH_SIGNAL], np.nan)
    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#efefef")
    fig, axes = plt.subplots(1, 2, figsize=(2 * TILE_PX, TILE_PX))
    for a, data, cmap in ((axes[0], sig, cm),
                          (axes[1], np.where(vis, 0.0, 1.0), "gray")):
        a.imshow(data, origin="lower", cmap=cmap, vmin=0, vmax=1,
                 interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#999999")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.06)
    _tile_save(fig, "tile_channels")

    # 4. what the network returns for that device
    fig, ax = _tile("prob")
    ax.imshow(prob, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
              extent=ext, aspect="auto", interpolation="nearest")
    _tile_save(fig, "tile_probability")


# -- figure 0: the whole method, in four panels ---------------------------
# Everything in it is drawn from the run: the diagram and its exact lines
# come from the device pool, the fan from ray_peaks, the channels and the
# probability map from the trained network through pick_case().  Nothing
# is an illustration of what the method would do.
#
# Two things the figure is careful NOT to say.  Every device in it is
# simulated, so no panel shows a measured diagram; the intended
# application to a fabricated device is a grey box behind a dashed arrow
# out of the trained network, and the caption says it is not demonstrated
# here.  And the word "twin" is not used: the capacitances are drawn at
# random to make an ensemble, which is not a model calibrated to one
# device.
#
# No callout text inside the panels.  A panel carries its letter, its
# axes and its data; what each panel MEANS is the caption's job, which is
# what a journal figure does and a slide does not.
#
# PRX Quantum: 17.8 cm for a double-column figure, every label 7 pt or
# more at print size, one sans-serif face, no shadows, no gradients.
OVERVIEW_W = 17.8 / 2.54        # double-column width, in inches
OVERVIEW_LINE = "#00B0F0"       # the exact transition lines, over `hot`
OVERVIEW_RAY = "#FFFFFF"        # the rays: white, outlined, legible on both
OVERVIEW_GREY = "#8c8c8c"       # the part that is future work
OVERVIEW_TICK = 7               # pt
OVERVIEW_AXIS = 7.5             # pt
OVERVIEW_PANEL = 8.5            # pt, the (a) (b) (c) (d)


def _panel_label(ax, text):
    ax.text(0.0, 1.015, text, transform=ax.transAxes,
            fontsize=OVERVIEW_PANEL, fontweight="bold", va="bottom",
            ha="left")


def _map_axes(ax, ux, uy, unit="mV"):
    ax.set_xlabel("$V_1$ (%s)" % unit, fontsize=OVERVIEW_AXIS, labelpad=1.5)
    ax.set_ylabel("$V_2$ (%s)" % unit, fontsize=OVERVIEW_AXIS, labelpad=1.5)
    ax.tick_params(labelsize=OVERVIEW_TICK, length=2, pad=1.5)
    ax.set_xlim(ux.min(), ux.max())
    ax.set_ylim(uy.min(), uy.max())
    for sp in ax.spines.values():
        sp.set_color("#999999")


def _lines_over(ax, ux, uy, mask, lw=0.9):
    """The exact transition lines, as a contour so they stay one line wide."""
    ax.contour(np.linspace(ux.min(), ux.max(), mask.shape[1]),
               np.linspace(uy.min(), uy.max(), mask.shape[0]),
               mask.astype(float), levels=[0.5], colors=[OVERVIEW_LINE],
               linewidths=lw)


def fig_overview():
    """Fig. 1 of the manuscript: simulation, measurement, network, result."""
    import matplotlib.patheffects as pe

    sample = _sensor_dir()
    ux, uy, Z = _raw_grid(sample)
    truth_s = ray_peaks.load_ground_truth(sample) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    # ONE device for the whole figure.  Panels (a) and (b) show the
    # diagram and the measurement; (c) and (d) show what the network makes
    # of that same measurement.  Four panels of three different devices
    # would be four unrelated pictures.  sample_10 is in the held-out
    # split, so (d) is a genuine held-out result.
    m = ray_peaks.measure(sample, N_RAYS, N_POINTS)
    _gx, _gy, Zn = ray_peaks.load_grid(sample)          # normalised, as fed
    ch = ray_peaks.to_channels(m, Zn.shape)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    prob = grid_train.predict(net()[0],
                              ch[None, :ray_peaks.NET_CHANNELS])[0]
    dx, dy = ux, uy

    fig, axes = plt.subplots(2, 2, figsize=(OVERVIEW_W, 0.80 * OVERVIEW_W))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) a simulated device, and the lines that are exact ────────────
    ax_a.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                interpolation="nearest")
    _lines_over(ax_a, ux, uy, truth_s)
    _map_axes(ax_a, ux, uy)
    _panel_label(ax_a, "(a)")

    # ── (b) the sparse measurement: a fan, not a grid ───────────────────
    ax_b.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                interpolation="nearest")
    stroke = [pe.withStroke(linewidth=2.0, foreground="#202020")]
    for a in ray_peaks.fan_angles(N_RAYS):
        pts = ray_peaks.ray_polyline(a, N_POINTS, ux, uy)
        ax_b.plot(pts[:, 0], pts[:, 1], "-", color=OVERVIEW_RAY, lw=0.9,
                  zorder=3, path_effects=stroke)
        # the rays are SAMPLED, not swept continuously; the dots are the
        # measurement and the line is only there to follow them
        ax_b.plot(pts[:, 0], pts[:, 1], ".", color=OVERVIEW_RAY, ms=1.5,
                  zorder=4, path_effects=stroke)
    ax_b.plot([ux.max()], [uy.max()], "o", color=OVERVIEW_RAY, ms=3.5,
              zorder=5, path_effects=stroke)
    _map_axes(ax_b, ux, uy)
    _panel_label(ax_b, "(b)")

    # ── (c) two channels in, one probability map out ────────────────────
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    _panel_label(ax_c, "(c)")

    vis = ch[ray_peaks.CH_VISITED] > 0.5
    sig = np.where(vis, ch[ray_peaks.CH_SIGNAL], np.nan)
    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#efefef")

    def _chip(rect, data, cmap, label, **kw):
        a = ax_c.inset_axes(rect)
        a.imshow(data, origin="lower", cmap=cmap, interpolation="nearest",
                 **kw)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#999999")
        a.set_title(label, fontsize=OVERVIEW_TICK, pad=2.5)
        return a

    _chip([0.02, 0.56, 0.30, 0.40], sig, cm, "channel 1  signal",
          vmin=0, vmax=1)
    _chip([0.02, 0.06, 0.30, 0.40], np.where(vis, 0.0, 1.0), "gray",
          "channel 2  sampled mask", vmin=0, vmax=1)
    arrow = dict(arrowstyle="-|>", color="#444444", lw=0.9)
    ax_c.annotate("", xy=(0.46, 0.50), xytext=(0.34, 0.50),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=arrow)
    ax_c.add_patch(plt.Rectangle((0.46, 0.41), 0.15, 0.18,
                                 transform=ax_c.transAxes, facecolor="white",
                                 edgecolor="#444444", lw=0.9, zorder=4))
    ax_c.text(0.535, 0.50, "U-Net", transform=ax_c.transAxes, ha="center",
              va="center", fontsize=OVERVIEW_AXIS, zorder=5)
    ax_c.annotate("", xy=(0.70, 0.50), xytext=(0.62, 0.50),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=arrow)
    _chip([0.70, 0.30, 0.30, 0.40], prob, J_CMAP, "probability per pixel",
          vmin=0, vmax=1)

    # the intended application, drawn as what it is: not done here
    ax_c.annotate("", xy=(0.85, 0.17), xytext=(0.85, 0.28),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color=OVERVIEW_GREY,
                                  lw=0.9, ls="--"))
    ax_c.text(0.85, 0.13, "measured device\n(not demonstrated here)",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=OVERVIEW_TICK, color=OVERVIEW_GREY,
              bbox=dict(boxstyle="round,pad=0.30", facecolor="#f4f4f4",
                        edgecolor=OVERVIEW_GREY, lw=0.8))

    # ── (d) the result on a held-out device ─────────────────────────────
    ax_d.imshow(prob, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
                extent=ext, aspect="auto", interpolation="nearest")
    _lines_over(ax_d, dx, dy, truth, lw=0.8)
    _map_axes(ax_d, dx, dy)
    _panel_label(ax_d, "(d)")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30, wspace=0.22)
    save(fig, "fig_overview", pdf=True)


# -- figure 6b: the ray fan, and the traces it returns --------------------
# This is the pair of plots the RBC project already draws, kept in its
# conventions so the paper and the lab code show the same thing the same
# way (src/dqd/visualization/plotter.py):
#
#   (a) plot_rays_with_peaks   -- the sensor map in `hot`, the rays over
#                                 it, the detected peaks as black crosses
#   (b) plot_current_vs_distance -- every ray's trace against NORMALISED
#                                 distance along the ray, 0 to 1, coloured
#                                 by angle from a viridis ramp, with the
#                                 angle legend and the dashed grid
#
# What is plotted is this repository's own measurement, not a second
# implementation of it: the traces are Measurement.traces from
# ray_peaks.measure and the peaks are what scipy find_peaks returns on
# them, which is the call measure() itself makes.  ray_processor.run in
# the RBC project does the same thing -- nearest-cell lookup along the
# polyline -- so the two agree by construction.
FAN_CMAP = "viridis"            # ray colour by angle, as plotter.py does
FAN_PEAK = "#000000"            # peak marker, as plotter.py does
# The fan is DRAWN at the smallest budget of the sweep, not at the headline
# one.  Eight rays leaving a single corner are eight lines within 80
# degrees of one another, and at the size this sits on the page they are
# a fan of touching strokes; four are four rays.  The scheme is what the
# figure has to show, and the scheme does not change with n_rays.  The
# text and the caption both read this number back from the sidecar, so
# changing it here changes what they say.
FAN_RAYS = 4


def fig_ray_fan(n_rays=None, n_points=None):
    from scipy.signal import find_peaks

    n_rays = n_rays or FAN_RAYS
    n_points = n_points or N_POINTS
    sample = _sensor_dir()
    ux, uy, Z = _raw_grid(sample)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    angles = ray_peaks.fan_angles(n_rays)

    # the measurement itself, not a re-implementation of it
    m = ray_peaks.measure(sample, n_rays, n_points)
    colours = plt.get_cmap(FAN_CMAP)(np.linspace(0, 1, len(angles)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # -- (a) the rays over the sensor map, with the peaks -----------------
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                   interpolation="nearest")
    _sensor_colorbar(fig, ax, im, Z)
    for k, a in enumerate(angles):
        pts = ray_peaks.ray_polyline(a, n_points, ux, uy)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=colours[k], lw=1.4,
                zorder=3)
        idx = find_peaks(m.traces[k])[0]
        if len(idx):
            ax.plot(pts[idx, 0], pts[idx, 1], "x", color=FAN_PEAK,
                    ms=5, mew=1.2, zorder=5)
    ax.set_xlim(ux.min(), ux.max())
    ax.set_ylim(uy.min(), uy.max())
    _panel_letter(ax, "a", x=0.03, y=0.03, ha="left", va="bottom")
    ax.set_xlabel("$V_1$ (mV)", fontsize=9.5)
    ax.set_ylabel("$V_2$ (mV)", fontsize=9.5)

    # -- (b) every trace against normalised distance ----------------------
    ax = axes[1]
    dist = np.linspace(0, 1, n_points)
    for k, a in enumerate(angles):
        ax.plot(dist, m.traces[k], "-o", color=colours[k], lw=1.3, ms=2.6,
                label="%g\u00b0" % a)
        idx = find_peaks(m.traces[k])[0]
        if len(idx):
            ax.plot(dist[idx], m.traces[k][idx], "x", color=FAN_PEAK,
                    ms=5, mew=1.2, zorder=5)
    ax.set_xlim(0, 1)
    _panel_letter(ax, "b", x=0.03, y=0.03, ha="left", va="bottom")
    ax.set_xlabel("normalised distance along the ray", fontsize=9.5)
    ax.set_ylabel("normalised charge-sensor signal", fontsize=9.5)
    ax.grid(True, which="both", ls="--", lw=0.5, color="#d8d8d8")
    ax.set_axisbelow(True)
    ax.legend(title=r"$\theta_k$", fontsize=8, title_fontsize=8.5,
              frameon=False, ncol=2, loc="upper right")

    for ax in axes:
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#999999")

    # The caption quotes how many measured points land on a transition
    # line and how many peaks the detector reports.  They are counted
    # HERE, from the same measurement the figure draws, and written out
    # for make_method_docs to read -- so the caption cannot carry a number
    # this figure does not show.
    on_line = peaks_found = 0
    for k, a in enumerate(angles):
        q = ray_peaks.ray_polyline(a, n_points, ux, uy)
        r, c = ray_peaks.voltage_to_pixel(q[:, 0], q[:, 1], ux, uy)
        on_line += int(truth[r, c].sum())
        peaks_found += len(find_peaks(m.traces[k])[0])
    with open(_figures.out_path("ray_fan_counts.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_rays": int(n_rays), "n_points": int(n_points),
                   "angles": [float(a) for a in angles],
                   "on_line": on_line, "peaks_found": peaks_found,
                   "measurements": int(n_rays * n_points)}, fh, indent=1)

    fig.tight_layout()
    save(fig, "fig_ray_fan")


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


def fig_threshold_panels():
    """Choosing and applying the threshold, as ONE figure (2026-09-26).

    (a) the U-Net's probability map for the held-out device, (b) the
    validation curve the threshold is chosen on, (c) that map cut at the
    stored threshold and (d) cut too high.  It replaces four separate
    figures.  No titles: the caption says what each panel is, and the
    panel letter sits inside the axes.  The cuts show the prediction (red)
    over the truth (black) with NO tolerance band -- tau is only defined
    in the next subsection.  The curve is read from
    threshold_validation.json, which make_threshold_figure.py writes from
    the real validation split."""
    with open(_figures.find("threshold_validation.json"),
              encoding="utf-8") as fh:
        tv = json.load(fh)
    _ch, Yt, p = pick_case()
    truth = Yt > 0.5
    ux, uy, _Z = ray_peaks.load_grid(DEVICE)
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    lo, hi = ladder()

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.6))
    # (a) the continuous map first, then (b) the curve the cut is
    # chosen on, then (c), (d) the cuts -- the order of the processing
    # (swapped 2026-09-26).  `a` is still the curve and `b` the map.
    (b, a), (c, d) = axs

    # (a) the validation curve
    cand, f1 = np.array(tv["candidates"]), np.array(tv["curve"])
    best = int(np.argmin(np.abs(cand - tv["threshold"])))
    a.plot(cand, f1, "-o", color=INK, linewidth=1.6, markersize=3.6,
           zorder=3)
    a.plot([cand[best]], [f1[best]], "o", color=J_PRED, markersize=11,
           markerfacecolor="none", markeredgewidth=1.8, zorder=4)
    a.annotate("%.1f" % cand[best], (cand[best], f1[best]),
               textcoords="offset points", xytext=(0, -16), ha="center",
               fontsize=9, color=J_PRED, fontweight="bold")
    a.set_xlabel("threshold on $P$", fontsize=9, color=INK)
    a.set_ylabel("F1@1, validation", fontsize=9, color=INK)
    a.set_ylim(-0.02, 1.0)
    a.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
           alpha=0.85)
    a.set_axisbelow(True)

    # (b) the probability map
    im = b.imshow(p, origin="lower", cmap=J_CMAP, vmin=0, vmax=1,
                  extent=ext, aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03)
    cb.set_label("$P$(transition line)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # (c), (d) the map cut at the stored threshold and too high
    def cut(ax, t):
        rgb = np.ones(truth.shape + (3,))
        rgb[p > t] = J_PRED_RGB
        rgb[truth] = J_TRUTH_RGB
        ax.imshow(rgb, origin="lower", extent=ext, aspect="auto",
                  interpolation="nearest")
    cut(c, lo if abs(lo - THRESHOLD) < 1e-9 else hi)
    cut(d, hi if abs(lo - THRESHOLD) < 1e-9 else lo)
    for ax in (a, c, d):                     # same width as (b)
        fig.colorbar(im, ax=ax, fraction=0.046,
                     pad=0.03).ax.set_visible(False)

    for ax in (b, c, d):
        ax.set_xlabel("$V_1$ (mV)", fontsize=9, color=INK)
        ax.set_ylabel("$V_2$ (mV)", fontsize=9, color=INK)
    for ax, letter in zip((b, a, c, d), "abcd"):
        ax.tick_params(labelsize=7.5, colors=INK)
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.8)
        # top right: empty in all four panels
        _panel_letter(ax, letter)
    fig.tight_layout()
    save(fig, "fig_threshold_panels")


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
    # No title (2026-09-26): what the curve shows is said in the caption.
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
    # NO TITLE TEXT, but the band it used to occupy is still reserved:
    # slide 12 of the SSDM deck places its own caption text box there,
    # and without the gap that box lands on the panel headings.  A blank
    # suptitle is what holds the space open -- an empty string has no
    # height at all and matplotlib reclaims the gap.
    fig.suptitle(" ", fontsize=14, y=0.985)
    # the colour key rides inside this figure, so a slide needs no
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

    with open(_figures.out_path("p2l_titles.json"), "w",
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
# A single axes carrying all fifteen budgets labels every point and the
# labels collide.  Here the budgets are GROUPED BY RAY COUNT
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
        with open(_figures.out_path("data_split_label.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"x": float(fx), "y": float(fy)}, fh, indent=2)
        print("    label slot at x=%.3f y=%.3f of the figure" % (fx, fy))
        save(fig, "fig_data_split_notext")
    else:
        save(fig, "fig_data_split")


# -- figure 11: every metric against tolerance, three budgets -------------
# The gallery version (figures/40_tau_all_metrics.png) draws every budget
# in the sweep against four metrics, which is unreadable at fifteen.  This
# keeps the smallest budget, a middle one and the best - all three of them
# runs that converged.  Pixel accuracy IS drawn, first, so that its
# flatness is visible next to the metrics that move.
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
    # Accuracy FIRST, and only so it can be dismissed: transition lines
    # are ~7 % of the diagram, so predicting no line anywhere already
    # scores ~0.93.  Putting it beside the three metrics that do mean
    # something shows, rather than asserts, why it is not the headline.
    #
    # coverage@tau LAST, right of F1: it is not a score at all but the
    # PRICE of the tolerance the scores are read at -- the fraction of
    # the plane within tau px of a measured pixel.  It shares the y axis
    # because it is a fraction of the same diagram, so the reader can see
    # directly that F1 climbs while the area it is judged over climbs too.
    # 2026-09-26: coverage@tau is no longer a panel (the text quotes it),
    # and the four scores are a 2 x 2 grid with the panel letter inside
    # each axes instead of the note that ran along the bottom.
    metrics = (("accuracy", "pixel accuracy"), ("precision", "precision"),
               ("recall", "recall"), ("f1", "F1"))
    best = max(budgets, key=lambda b: float(rows[b]["coverage"]))

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), sharey=True)
    axes = axes.ravel()
    lo = 1.0
    for ax, (key, label), letter in zip(axes, metrics, "abcd"):
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
                    label="%d × %d" % budget)
        ax.set_xlabel("tolerance τ  (pixels)", fontsize=9.5, color=INK)
        ax.set_xticks(list(TAUS))
        ax.tick_params(labelsize=8.5, colors=INK, labelleft=True)
        # the metric moves onto the y axis: without it the panels are
        # indistinguishable, and a y label is not a title
        ax.set_ylabel("%s\nof the plane" % label if key == "coverage"
                      else "%s on %d\nheld-out devices" % (label, N_TEST),
                      fontsize=8.5, color=INK, linespacing=1.3)
        # A key on EVERY panel: these are read one at a time, cropped
        # into a slide or a caption, and a panel that has been separated
        # from the first one must still say which line is which.
        ax.legend(fontsize=7, frameon=True, framealpha=0.92,
                  edgecolor="#cccccc", handlelength=1.6,
                  labelspacing=0.25, borderpad=0.28,
                  title="rays × points", title_fontsize=7,
                  loc="lower right")
        # bottom left: empty in all four panels
        _panel_letter(ax, letter, x=0.03, y=0.03, ha="left", va="bottom")
        ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
                alpha=0.85)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#444444")
            ax.spines[sp].set_linewidth(0.8)
    # the floor drops to 0 so coverage@0 (~1.6 %) is not clipped off the
    # bottom of the shared axis
    axes[0].set_ylim(min(0.0, lo - 0.06), 1.02)

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.2)
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
    fig_threshold_panels()
    fig_inputs()
    prob_map_stats()
    fig_results_f1_vs_coverage()
    fig_tau_metrics()
    fig_budget_ladder()
    fig_data_split()



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

    # One line per ray length (2026-09-27): 40, 50 and 60 points per ray,
    # each through its CONVERGED budgets only; a budget that did not
    # converge is left out and the line joins its neighbours.
    ladders = {40: ("#08519C", "o", (0, 8), "bottom"),
               50: ("#D9711A", "s", (0, -10), "top"),
               60: ("#2E8B57", "^", (9, 0), "center")}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    line = []
    for pts, (colour, marker, off, va) in ladders.items():
        ladder = sorted(((n, pts) for n in (4, 5, 6, 7, 8)
                         if ok.get("%d_rays_%d_points_500_samples"
                                   % (n, pts))),
                        key=lambda b: pt(*b)[0])
        if not ladder:
            continue
        line += ladder
        xs = [pt(*b)[0] for b in ladder]
        ys = [pt(*b)[1] for b in ladder]
        ax.plot(xs, ys, linestyle="none", marker=marker, color=colour,
                markersize=5.5, markeredgecolor="white",
                markeredgewidth=0.7, zorder=4,
                label="%d points per ray" % pts)
        # solid between neighbouring ray counts; DASHED where a budget in
        # between did not converge, so no trend is drawn through a point
        # that does not exist
        for (n0, _a), (n1, _b), x0, x1, y0, y1 in zip(
                ladder, ladder[1:], xs, xs[1:], ys, ys[1:]):
            gap = n1 != n0 + 1
            ax.plot([x0, x1], [y0, y1], color=colour,
                    linewidth=1.3 if gap else 1.8,
                    linestyle=(0, (4, 3)) if gap else "-",
                    alpha=0.8 if gap else 1.0, zorder=3)
        for (n, _p), x, y in zip(ladder, xs, ys):
            ax.annotate("%d" % n, (x, y), textcoords="offset points",
                        xytext=off, ha="center" if off[0] == 0 else "left",
                        va=va, fontsize=7.5, color=colour, zorder=5)

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
                # to the right of the ring (2026-09-27), level with it
                xytext=(bx + 0.14, by), fontsize=8.5,
                color=J_PRED, fontweight="bold", ha="left", va="center")

    # The same-coverage open marker (4 x 60), its arrow and note were
    # removed (your call, 2026-09-27); the finding is stated in the text.

    ax.set_xlabel("fraction of the grid actually measured  (%)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("F1@1 on %d held-out devices" % N_TEST, fontsize=9.5,
                  color=INK)
    ax.set_xlim(1.3, 5.0)          # room for the best-budget label
    ax.set_ylim(0.62, 0.86)
    ax.grid(True, linestyle=":", linewidth=0.7, color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK, length=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, marker=m, linewidth=1.8,
                      markersize=5.5, markeredgecolor="white",
                      markeredgewidth=0.7, label="%d points per ray" % n)
               for n, (c, m, _o, _v) in ladders.items()]
    leg = ax.legend(handles=handles, fontsize=8.5, frameon=True,
                    loc="lower right", handlelength=2.0)
    leg.get_frame().set_edgecolor("#bbbbbb")
    leg.get_frame().set_linewidth(0.6)
    # no note under the plot (2026-09-27): the caption says what the
    # numbers beside the points are
    fig.tight_layout(pad=0.4)
    save(fig, "results_budget_ladder")
    print("  ladder: %s" % ", ".join("%d x %d" % b for b in line))
    print("  best:   %d x %d  F1@1 %.3f at %.1f %%" % (bn, bp, by, bx))


# -- figure 13: what the tolerance buys, and what it costs ---------------
# F1@tau can only rise with tau, so a high one proves nothing by itself.
# claim@tau is the price: dilate the model's OWN PREDICTION by tau and
# measure what fraction of the plane it then covers -- scoring at tau credits
# a predicted pixel for a whole disc of radius tau, so the model is in effect
# asserting "a line passes somewhere in here" across that disc.  At tau = 3
# that is ~44 % of the plane, and a score read off it says very little.
#
# Per device, then averaged over the held-out set, with the spread as
# horizontal bars: line density varies several-fold across the test devices,
# so the area a prediction covers does too.
#
# NOT coverage@tau, which dilates the VISITED MASK: that is the measurement
# geometry alone, identical whatever the model predicts, so it cannot price
# the score.
if __name__ == "__main__":
    main()
