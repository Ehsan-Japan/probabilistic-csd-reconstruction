"""
make_fig6_model_flow.py -- Fig. 6 of Manuscript_ver6 (network input, U-Net,
output) in the look of Fig. 1 (user, 2026-10-27).

    python paper_figures/programs/make_fig6_model_flow.py

The arrays are those of make_figures.fig_model_flow -- the picked test
device, its two input channels (8 rays x 60 points) and the trained
network's probability map (make_figures.pick_case) -- only the drawing
changes: white blocks with coloured borders as in Fig. 1 (encoder dark
blue, decoder blue, bottleneck and input/output slate), channel 1 in greys
on white, and the probability map from white to the blue of Fig. 1.
Writes method_docs_ver2/3_methods/figures/fig_model_flow.png; the
manuscript generator swaps it in for Fig. 6.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import make_figures as MF                                  # noqa: E402
from csdrecon.ml import ray_peaks                          # noqa: E402

OUT = os.path.join(MF.ROOT, "paper_figures", "method_docs_ver2",
                   "3_methods", "figures", "fig_model_flow.png")
# the colours of Fig. 1
INK, MUTED = "#111827", "#4b5563"
ENC, DEC, BOT, IO = "#1e40af", "#2563eb", "#4a5568", "#4a5568"
GUIDE = LinearSegmentedColormap.from_list("w2b", ["#ffffff", "#2563eb",
                                                  "#1e3a8a"])
plt.rcParams.update({"font.family": "Arial"})


def block(ax, x, y, w, h, colour, label, sub=None, fs=1.0):
    """a white rounded block with a coloured border and dark bold text"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                linewidth=1.4, edgecolor=colour,
                                facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10 * fs, color=INK, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y - 0.055, sub, ha="center", va="top",
                fontsize=8 * fs, color=MUTED)


def arrow(ax, p, q, colour, style="-", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=colour, linestyle=style,
                                 shrinkA=1, shrinkB=1, zorder=1))


def draw_unet(ax, fs=0.9):
    """the geometry of make_figures.draw_unet, drawn as in Fig. 1"""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.32, 1.25)
    ax.axis("off")
    lvl = [(0.72, 0.85), (0.56, 0.60), (0.40, 0.35)]      # h, y per depth
    enc_x = [1.35, 2.35, 3.35]
    dec_x = [5.30, 6.30, 7.30]
    widths = ["32", "64", "128"]
    grids = ["100²", "50²", "25²"]
    block(ax, 0.15, 0.30, 0.55, 0.72, IO, "2", "input", fs)
    for i, (x, w, g) in enumerate(zip(enc_x, widths, grids)):
        h, yt = lvl[i]
        block(ax, x, yt - h, 0.72, h, ENC, w, g, fs)
    block(ax, 4.30, 0.00, 0.72, 0.28, BOT, "256", "12²", fs)
    for i, (x, w, g) in enumerate(zip(dec_x, reversed(widths),
                                      reversed(grids))):
        h, yt = lvl[2 - i]
        block(ax, x, yt - h, 0.72, h, DEC, w, g, fs)
    block(ax, 8.40, 0.30, 0.55, 0.72, IO, "1", "output", fs)
    arrow(ax, (0.72, 0.66), (1.33, 0.66), INK)
    for i in range(2):
        h0, y0 = lvl[i]
        h1, y1 = lvl[i + 1]
        arrow(ax, (enc_x[i] + 0.72, y0 - h0 / 2),
              (enc_x[i + 1], y1 - h1 / 2), ENC)
    h, y = lvl[2]
    arrow(ax, (enc_x[2] + 0.72, y - h / 2), (4.30, 0.14), ENC)
    arrow(ax, (5.02, 0.14), (dec_x[0], y - h / 2), DEC)
    for i in range(2):
        h0, y0 = lvl[2 - i]
        h1, y1 = lvl[1 - i]
        arrow(ax, (dec_x[i] + 0.72, y0 - h0 / 2),
              (dec_x[i + 1], y1 - h1 / 2), DEC)
    arrow(ax, (dec_x[2] + 0.72, 0.66), (8.38, 0.66), INK)
    skip_y = [lvl[0][1] - 0.06, lvl[1][1] - 0.06, 0.33]
    for i in range(3):
        arrow(ax, (enc_x[i] + 0.72, skip_y[i]), (dec_x[2 - i], skip_y[i]),
              "#9ca3af", style=(0, (4, 3)), lw=1.1)
    ax.text(4.66, lvl[0][1] + 0.02, "skip connections", ha="center",
            fontsize=8 * fs, color=MUTED, style="italic")
    ax.text(2.43, 1.16, "encoder", fontsize=10 * fs, color=ENC,
            fontweight="bold", ha="center")
    ax.text(6.38, 1.16, "decoder", fontsize=10 * fs, color=DEC,
            fontweight="bold", ha="center")
    ax.text(4.66, -0.14, "bottleneck", fontsize=9 * fs, color=BOT,
            fontweight="bold", ha="center", va="top")


def main():
    ch, _Yt, p = MF.pick_case()
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    side, off, oside = 1.7, 0.35, 1.8           # as in fig_model_flow
    nx, ny, nw, nh = 2.25, 0.0, 5.4, 3.0
    ox = nx + nw * 0.895 + 0.38
    W, H = ox + oside + 0.72, 3.0
    fig = plt.figure(figsize=(W, H))

    def axes_in(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    def frame(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.9)

    # the input: channel 2 (where a ray passed) behind channel 1 (the
    # normalised signal there), in greys on white
    x0, y0 = 0.08, (H - side - off) / 2
    back = axes_in(x0 + off, y0 + off, side, side, zorder=1)
    back.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
                interpolation="nearest")
    front = axes_in(x0, y0, side, side, zorder=2)
    cm = plt.get_cmap("Greys").copy()
    cm.set_bad("#ffffff")
    front.imshow(np.where(vis > 0.5, 0.25 + 0.75 * sig, np.nan),
                 origin="lower", cmap=cm, vmin=0, vmax=1,
                 interpolation="nearest")
    for ax, tag in ((front, "1"), (back, "2")):
        frame(ax)
        ax.text(0.04, 0.96 if ax is back else 0.04, tag,
                transform=ax.transAxes, ha="left",
                va="top" if ax is back else "bottom", fontsize=9,
                fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#9ca3af", lw=0.5))

    axn = axes_in(nx, ny, nw, nh)
    draw_unet(axn)

    # the output: P(transition line), white to the blue of Fig. 1
    oy = (H - oside) / 2
    axo = axes_in(ox, oy, oside, oside)
    im = axo.imshow(p, origin="lower", cmap=GUIDE, vmin=0, vmax=1,
                    interpolation="nearest")
    frame(axo)
    cax = axes_in(ox + oside + 0.08, oy, 0.1, oside)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    cb.set_label("P(transition line)", fontsize=7.5)

    def to_fig(ax, xd, yd):
        return fig.transFigure.inverted().transform(
            ax.transData.transform((xd, yd)))

    def fig_arrow(p0, p1):
        fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9,
                                       linewidth=1.2, color=INK))
    yin = to_fig(axn, 0.15, 0.66)[1]
    fig_arrow(((x0 + off + side + 0.03) / W, yin),
              (to_fig(axn, 0.13, 0.66)[0], yin))
    yout = to_fig(axn, 8.95, 0.66)[1]
    fig_arrow((to_fig(axn, 9.0, 0.66)[0], yout), ((ox - 0.04) / W, yout))
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
