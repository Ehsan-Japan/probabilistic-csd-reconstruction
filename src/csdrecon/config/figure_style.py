"""
figure_style.py — one place that decides how big every saved figure is.

For a paper the images must be directly comparable, which needs three things
identical from figure to figure: the canvas size in inches and the dpi, the
position of the plotting box inside it, and the data-to-inches scale (equal
aspect plus the same voltage limits, so 1 mV is the same number of millimetres
everywhere).

Nothing here saves with bbox_inches="tight", which crops the canvas to
whatever the labels needed and is why no two images used to come out the same
size; the fixed axes rectangle leaves the room instead.

    fig, ax, cax = new_map_figure(with_colorbar=True)
    save_figure(fig, out_path)

Multi-panel figures use new_figure(ncols=2), which widens the canvas so each
panel keeps its physical size.  Set the size once per program with
set_figure_style().
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from . import log

# 12 x 12 in matches what summary_total.png / summary_peaks_only.png always
# used.  Do NOT shrink it: fonts, legends and markers are sized in POINTS,
# so a smaller canvas does not scale them down -- it makes them look huge
# relative to the frame.  Changing this value changes how big the legend and
# the markers appear in every image.
DEFAULT_WIDTH_IN = 12.0
DEFAULT_HEIGHT_IN = 12.0
DEFAULT_DPI = 300

# Plotting box inside the canvas, as (left, bottom, width, height) fractions,
# identical in every figure.  The right-hand strip is reserved for a colorbar
# whether or not one is drawn, so figures with and without one share the box.
# Width and height are equal, so on a square canvas the box is square in
# inches and 1 mV is the same length on both axes.
AXES_RECT = (0.100, 0.100, 0.750, 0.750)
CBAR_RECT = (0.870, 0.100, 0.025, 0.750)


# ----------------------------------------------------------------------
# The one house style, taken from summary.png
# ----------------------------------------------------------------------
#
# Every figure drawing the binary transition map uses these: white background,
# black transition cells, a visible black cell grid.  Sensor-signal figures
# keep the "hot" colormap, being a continuous measurement, but share the cell
# grid and the marker styles, so the family reads as one set.
GT_CMAP = "gray_r"                 # 0 -> white, 1 -> black
GT_EDGECOLOR = "k"
GT_LINEWIDTH = 0.5

# Overlay markers, identical everywhere.
MARKER_SCANNED = dict(color="blue", s=10, alpha=0.5)
MARKER_PEAK = dict(marker="x", color="red", s=50, linewidths=1)
MARKER_CENTRE = dict(color="lime", marker="*", s=200, edgecolors="black")

LABEL_SCANNED = "Measured Points"
LABEL_PEAK = "Detected Transitions"


def draw_ground_truth_map(ax, x_edges, y_edges, binary) -> None:
    """Draw a binary transition map in the shared house style."""
    ax.pcolormesh(x_edges, y_edges, binary, cmap=GT_CMAP,
                  edgecolors=GT_EDGECOLOR, linewidth=GT_LINEWIDTH,
                  vmin=0, vmax=1)


@dataclass
class FigureStyle:
    """Canvas geometry shared by every saved figure."""

    width_in: float = DEFAULT_WIDTH_IN
    height_in: float = DEFAULT_HEIGHT_IN
    dpi: int = DEFAULT_DPI

    def size(self, ncols: int = 1, nrows: int = 1) -> Tuple[float, float]:
        """Canvas size for an ncols x nrows panel grid, in inches."""
        return (self.width_in * ncols, self.height_in * nrows)


_ACTIVE = FigureStyle()


def set_figure_style(
    width_in: Optional[float] = None,
    height_in: Optional[float] = None,
    dpi: Optional[int] = None,
) -> FigureStyle:
    """Set the canvas geometry used by every figure. None = leave unchanged."""
    if width_in is not None:
        _ACTIVE.width_in = width_in
    if height_in is not None:
        _ACTIVE.height_in = height_in
    if dpi is not None:
        _ACTIVE.dpi = dpi
    return _ACTIVE


def get_figure_style() -> FigureStyle:
    """The active FigureStyle object."""
    return _ACTIVE


def figure_size(ncols: int = 1, nrows: int = 1) -> Tuple[float, float]:
    """Canvas size in inches — pass straight to ``figsize=``."""
    return _ACTIVE.size(ncols, nrows)


def figure_dpi() -> int:
    """The shared output dpi."""
    return _ACTIVE.dpi


def new_map_figure(with_colorbar: bool = False):
    """
    Figure for a voltage-space map: fixed canvas, fixed axes rectangle.

    Returns ``(fig, ax, cax)``.  ``cax`` is the colorbar axes when
    ``with_colorbar`` is True, else None — but the space it would occupy is
    reserved either way, so the main axes box never moves.
    """
    fig = plt.figure(figsize=figure_size())
    ax = fig.add_axes(AXES_RECT)
    cax = fig.add_axes(CBAR_RECT) if with_colorbar else None
    return fig, ax, cax


def new_figure(ncols: int = 1, nrows: int = 1, **kwargs):
    """
    Figure for anything that is not a voltage map (line plots, 3-D surfaces,
    multi-panel figures).  Same canvas size per panel; ``constrained_layout``
    keeps the labels inside the canvas instead of cropping it away.

    Returns whatever ``plt.subplots`` returns: ``(fig, ax)`` or ``(fig, axes)``.
    """
    kwargs.setdefault("figsize", figure_size(ncols, nrows))
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, **kwargs)


def apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax) -> None:
    """
    Give a voltage map the shared labels, limits and scale.

    Equal aspect + the same limits in every figure is what makes 1 mV the
    same physical distance across all of them.
    """
    from .axis_labels import x_label, y_label

    ax.set_xlabel(x_label())
    ax.set_ylabel(y_label())
    ax.set_xlim(vxmin, vxmax)
    ax.set_ylim(vymin, vymax)
    ax.set_aspect("equal", adjustable="box")


# ----------------------------------------------------------------------
# Legend-free copies
# ----------------------------------------------------------------------
#
# The paper's legends are drawn by hand, so every sample-level figure is also
# saved without its legend into <sample>/figures_no_legend/; the normal figure
# is untouched.  DatasetPipeline sets the directory; None disables copying.
_NO_LEGEND_DIR: Optional[str] = None

NO_LEGEND_DIRNAME = "figures_no_legend"


def set_no_legend_dir(path: Optional[str]) -> None:
    """
    Collect legend-free copies of sample-level figures in ``path``.

    Pass None to switch the copies off.  Figures under ``cropped_results/``
    (the per-peak crops) are skipped — there are hundreds per sample and they
    are not paper figures.
    """
    global _NO_LEGEND_DIR
    _NO_LEGEND_DIR = path


def _no_legend_path(path: str) -> Optional[str]:
    """
    Destination for the legend-free copy of ``path``, or None to skip it.

    Only sample-level figures qualify: the sample root itself and one level
    below it (images/), never the per-peak cropped_results tree.  Names from a
    sub-directory are flattened ("images/a.jpg" -> "images__a.jpg")
    so nothing collides in the single output folder.
    """
    if not _NO_LEGEND_DIR:
        return None
    sample_root = os.path.dirname(os.path.abspath(_NO_LEGEND_DIR))
    try:
        rel = os.path.relpath(os.path.abspath(path), sample_root)
    except ValueError:                      # different drive on Windows
        return None
    parts = rel.split(os.sep)
    if parts[0] in (os.pardir, "cropped_results", NO_LEGEND_DIRNAME):
        return None
    if len(parts) > 2:                      # deeper than <sample>/<dir>/file
        return None
    return os.path.join(_NO_LEGEND_DIR, "__".join(parts))


def _strip_legends(fig) -> int:
    """Remove every legend from the figure, in place.  Returns how many."""
    removed = 0
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
            removed += 1
    for leg in list(getattr(fig, "legends", [])):
        leg.remove()
        removed += 1
    return removed


def save_figure(fig, path: str, dpi: Optional[int] = None) -> None:
    """Save at the shared dpi, without bbox_inches="tight".

    Cropping to the ink is what made the old images different sizes; the fixed
    axes rectangle already leaves room for the labels.

    Sample-level figures with a legend are saved a second time, legend
    removed, into set_no_legend_dir()'s directory — after the normal figure,
    so the original is unaffected.  Figures without a legend get no copy.
    """
    dpi = dpi or _ACTIVE.dpi
    fig.savefig(path, dpi=dpi)

    copy_path = _no_legend_path(path)
    if copy_path:
        try:
            if _strip_legends(fig):
                os.makedirs(os.path.dirname(copy_path), exist_ok=True)
                fig.savefig(copy_path, dpi=dpi)
        except Exception as exc:            # never let a copy break the run
            log.detail(f"[figure_style] no-legend copy failed for {path}: {exc}")

    plt.close(fig)
