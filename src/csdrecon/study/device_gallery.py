"""
device_gallery.py — contact sheets: every device in a split, on one page.

The per-device figures in device_figures.py are one device per file, at
paper resolution.  That is the wrong thing for the question "what did the
simulator actually produce?", which is asked about a whole split at once and
is answered by looking at all of it quickly.

This module answers that question instead.  One page shows many devices side
by side, and for each device the panels that say what kind of data it is:

    charge_sensor            the raw simulated sensor signal, as simulated
    charge_sensor_gradient   the same data with the smooth gate background
                             differenced away — where the honeycomb is
                             actually visible
    stability_diagram        the binary ground truth the network is scored
                             against

    results/<run>/<config>/figures/gallery/
        train_page_01.png ... train_page_20.png
        test_page_01.png
        index.txt            which device is on which page

Nothing here changes the data, the model or any score: it only writes .png
files, so it is safe to run at any time, on a finished run or a half-built
one.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import log
from .device_figures import _edges, _extent, draw_truth, load_device

# The panels drawn for each device, in order, with the column heading each
# one gets on the page.
PANELS: List[tuple] = [
    ("charge_sensor", "sensor"),
    ("charge_sensor_gradient", "gradient"),
    ("stability_diagram", "ground truth"),
]

DEFAULT_PANELS = tuple(name for name, _ in PANELS)
GALLERY_DIRNAME = "gallery"


def _panel(ax, kind: str, ux, uy, Z, gt, cell_grid: bool) -> None:
    """One thumbnail.  No colorbar, no ticks — this is a page to scan."""
    if kind == "stability_diagram":
        x_edges, y_edges = _edges(ux, uy)
        draw_truth(ax, x_edges, y_edges, gt, cell_grid=cell_grid)
    else:
        field = Z
        if kind == "charge_sensor_gradient":
            field = np.gradient(Z, axis=0) + np.gradient(Z, axis=1)
        ax.imshow(field, extent=_extent(ux, uy), origin="lower",
                  aspect="auto", cmap="hot")
    # The window is square in voltage, so the thumbnail is square: a
    # stretched honeycomb reads as a different device.
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")
        s.set_linewidth(0.5)


def render_split(sample_dirs: Sequence[str], out_dir: str, split: str,
                 panels: Sequence[str] = DEFAULT_PANELS,
                 per_page: int = 15, columns: int = 3,
                 dpi: int = 150, cell_grid: bool = False,
                 thumb_in: float = 1.6) -> List[str]:
    """
    Contact sheets for one split.  Returns the files written.

    per_page devices per page, `columns` devices per row; each device takes
    len(panels) adjacent cells.  The device's own folder name is printed
    above it, so anything odd on the page can be opened directly.
    """
    panels = [p for p in panels if p in DEFAULT_PANELS]
    if not sample_dirs or not panels:
        return []

    headings = dict(PANELS)
    rows_per_page = int(np.ceil(per_page / columns))
    n_pages = int(np.ceil(len(sample_dirs) / per_page))
    os.makedirs(out_dir, exist_ok=True)
    log.detail(f"  {split}: {len(sample_dirs)} devices x {len(panels)} panels "
               f"-> {n_pages} page(s)")

    written, index = [], []
    for page in range(n_pages):
        here = list(enumerate(sample_dirs, 1))[page * per_page:(page + 1) * per_page]
        n_rows = int(np.ceil(len(here) / columns))
        fig, axes = plt.subplots(
            n_rows, columns * len(panels), squeeze=False,
            figsize=(thumb_in * columns * len(panels) + 0.6,
                     thumb_in * n_rows + 0.9))
        for ax in axes.ravel():
            ax.axis("off")

        for k, (number, sdir) in enumerate(here):
            r, c = divmod(k, columns)
            try:
                ux, uy, Z, gt = load_device(sdir)
            except Exception as exc:                      # a device that the
                log.warn(f"    [skip] sample_{number}: {exc}")   # pool lost
                continue
            for j, kind in enumerate(panels):
                ax = axes[r][c * len(panels) + j]
                ax.axis("on")
                _panel(ax, kind, ux, uy, Z, gt, cell_grid)
                if j == 0:
                    ax.set_ylabel(f"sample_{number}", fontsize=7,
                                  rotation=0, ha="right", va="center",
                                  labelpad=4)
                if r == 0:
                    ax.set_title(headings[kind], fontsize=7, pad=3)
            index.append(f"page {page + 1:>3}   sample_{number}   "
                         f"{os.path.abspath(sdir)}")

        fig.suptitle(f"{split} devices — page {page + 1} of {n_pages} "
                     f"({len(here)} of {len(sample_dirs)})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(out_dir, f"{split}_page_{page + 1:02d}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    with open(os.path.join(out_dir, "index.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(index) + "\n")
    return written


def render_config(cfg, splits: Optional[Sequence[str]] = None,
                  panels: Sequence[str] = DEFAULT_PANELS,
                  per_page: int = 15, columns: int = 3,
                  dpi: int = 150, cell_grid: bool = False) -> List[str]:
    """
    Contact sheets for a configuration's train and test devices.

    The sample folders come out of the configuration's own train.npz /
    test.npz, so the devices on the page are exactly the devices the model
    was trained and scored on — not a fresh listing of the pool that might
    include devices this configuration never used.
    """
    from .dataset import load_split

    out_dir = os.path.join(cfg.figures_dir, GALLERY_DIRNAME)
    # Rebuilt from scratch: a stale index.txt from an earlier call would
    # otherwise describe pages that no longer exist.
    index = os.path.join(out_dir, "index.txt")
    if os.path.isfile(index):
        os.remove(index)

    written = []
    for split, npz in (("train", cfg.train_npz), ("test", cfg.test_npz)):
        if splits and split not in splits:
            continue
        if not os.path.isfile(npz):
            log.warn(f"  [skip] {split}: {os.path.abspath(npz)} is missing")
            continue
        _, _, sample_dirs = load_split(npz)
        written += render_split(sample_dirs, out_dir, split, panels=panels,
                                per_page=per_page, columns=columns, dpi=dpi,
                                cell_grid=cell_grid)
    if written:
        log.say(f"  {len(written)} contact sheet(s) -> "
                f"{os.path.abspath(out_dir)}")
    return written
