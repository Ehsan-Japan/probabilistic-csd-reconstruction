"""
device_gallery.py — contact sheets: every device in a split, on one page.

The per-device figures in device_figures.py are one device per file, at
paper resolution.  That is the wrong thing for the question "what did the
simulator actually produce, and what did the network make of it?", which is
asked about a whole split at once and is answered by looking at all of it
quickly.

This module answers that question instead.  One page shows many devices side
by side, and for each device the panels that say what kind of data it is and
what came back out:

    charge_sensor            the raw simulated sensor signal, as simulated
    charge_sensor_gradient   the same data with the smooth gate background
                             differenced away — where the honeycomb is
                             actually visible
    stability_diagram        the binary ground truth the network is scored
                             against
    probability              the U-Net's reconstruction: one probability per
                             pixel, from this device's ray measurement alone
    prediction               that map cut at the stored threshold — the lines
                             the study actually reports

    results/<run>/<config>/figures/gallery/
        train_page_01.png ... train_page_20.png
        test_page_01.png
        index.txt            which device is on which page, and its F1@1

THE TRAIN PAGES ARE IN-SAMPLE.  The network was fitted on those devices, so
its reconstruction of them is not evidence of anything except that training
converged; the page says so in its own title.  Only the test pages are
held-out.  Devices are labelled by their POOL FOLDER NAME (sample_7), not by
their position in the split, because only the folder name identifies the
device.

Nothing here changes the data, the model or any score: it only writes .png
files, so it is safe to run at any time, on a finished run or a half-built
one.  With no checkpoint on disk the two network panels are skipped and the
rest of the page is drawn.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import log
from ..ml.grid_metrics import tolerant_f1
from .device_figures import _edges, _extent, draw_truth, load_device

# The panels drawn for each device, in order, with the column heading each
# one gets on the page.
PANELS: List[tuple] = [
    ("charge_sensor", "sensor"),
    ("charge_sensor_gradient", "gradient"),
    ("stability_diagram", "ground truth"),
    ("probability", "U-Net probability"),
    ("prediction", "U-Net lines"),
]

DEFAULT_PANELS = tuple(name for name, _ in PANELS)

# The panels that need a trained model; without one they are dropped.
NETWORK_PANELS = ("probability", "prediction")

GALLERY_DIRNAME = "gallery"

# The tolerance the per-device score printed under each device is read at —
# the study's headline tolerance.
LABEL_TAU = 1


def _panel(ax, kind: str, ux, uy, Z, gt, prob, threshold: float,
           cell_grid: bool) -> None:
    """One thumbnail.  No colorbar, no ticks — this is a page to scan."""
    if kind == "stability_diagram":
        x_edges, y_edges = _edges(ux, uy)
        draw_truth(ax, x_edges, y_edges, gt, cell_grid=cell_grid)
    elif kind == "prediction":
        x_edges, y_edges = _edges(ux, uy)
        draw_truth(ax, x_edges, y_edges, (prob > threshold).astype(float),
                   cell_grid=cell_grid)
    elif kind == "probability":
        # The same orientation and extent as every other panel, so a line on
        # the probability map sits over the line on the ground truth.
        ax.imshow(prob, extent=_extent(ux, uy), origin="lower",
                  aspect="auto", cmap="inferno", vmin=0, vmax=1)
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
                 prob: Optional[np.ndarray] = None,
                 threshold: float = 0.5,
                 in_sample: bool = False,
                 per_page: int = 10, columns: int = 2,
                 dpi: int = 150, cell_grid: bool = False,
                 thumb_in: float = 1.6) -> List[str]:
    """
    Contact sheets for one split.  Returns the files written.

    prob : (N, H, W) probability maps in the order of `sample_dirs`, or None
           to draw the device panels alone.

    per_page devices per page, `columns` devices per row; each device takes
    len(panels) adjacent cells.  The device's own folder name is printed
    beside it, with its F1@1 when a reconstruction is on the page, so
    anything odd can be opened directly.
    """
    panels = [p for p in panels if p in DEFAULT_PANELS]
    if prob is None:
        dropped = [p for p in panels if p in NETWORK_PANELS]
        if dropped:
            log.detail(f"  {split}: no prediction available — "
                       f"{', '.join(dropped)} skipped")
        panels = [p for p in panels if p not in NETWORK_PANELS]
    if not sample_dirs or not panels:
        return []

    headings = dict(PANELS)
    n_pages = int(np.ceil(len(sample_dirs) / per_page))
    os.makedirs(out_dir, exist_ok=True)
    log.detail(f"  {split}: {len(sample_dirs)} devices x {len(panels)} panels "
               f"-> {n_pages} page(s)")

    written, index = [], []
    for page in range(n_pages):
        here = list(enumerate(sample_dirs))[page * per_page:(page + 1) * per_page]
        n_rows = int(np.ceil(len(here) / columns))
        fig, axes = plt.subplots(
            n_rows, columns * len(panels), squeeze=False,
            figsize=(thumb_in * columns * len(panels) + 1.2,
                     thumb_in * n_rows + 1.0))
        for ax in axes.ravel():
            ax.axis("off")

        for k, (i, sdir) in enumerate(here):
            r, c = divmod(k, columns)
            name = os.path.basename(sdir)
            try:
                ux, uy, Z, gt = load_device(sdir)
            except Exception as exc:                      # a device the pool
                log.warn(f"    [skip] {name}: {exc}")     # no longer has
                continue

            p = prob[i] if prob is not None else None
            label = name
            if p is not None:
                f1 = tolerant_f1(p > threshold, gt, LABEL_TAU)["f1"]
                label = f"{name}\nF1@{LABEL_TAU} {f1:.2f}"
                index.append(f"page {page + 1:>3}   {name:<12} "
                             f"F1@{LABEL_TAU} {f1:.3f}   "
                             f"{os.path.abspath(sdir)}")
            else:
                index.append(f"page {page + 1:>3}   {name:<12} "
                             f"{'':<14}   {os.path.abspath(sdir)}")

            for j, kind in enumerate(panels):
                ax = axes[r][c * len(panels) + j]
                ax.axis("on")
                _panel(ax, kind, ux, uy, Z, gt, p, threshold, cell_grid)
                if j == 0:
                    ax.set_ylabel(label, fontsize=7, rotation=0,
                                  ha="right", va="center", labelpad=4)
                if r == 0:
                    ax.set_title(headings[kind], fontsize=7, pad=3)

        title = (f"{split} devices — page {page + 1} of {n_pages} "
                 f"({len(here)} of {len(sample_dirs)})")
        if prob is not None:
            title += (f"   ·   threshold {threshold:g}"
                      + ("   ·   IN-SAMPLE: the network was trained on these"
                         if in_sample else "   ·   held out"))
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(out_dir, f"{split}_page_{page + 1:02d}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    with open(os.path.join(out_dir, "index.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(index) + "\n")
    return written


def _prediction(cfg, X: np.ndarray):
    """
    (probability maps, threshold) for one split's measurements, or
    (None, 0.5) when this configuration has no trained model yet.

    The threshold is the one stored IN THE CHECKPOINT — chosen on a
    validation split carved out of the training devices — so the lines on
    the page are the lines the study reports, not a fresh guess.
    """
    from ..ml import grid_train

    if not os.path.isfile(cfg.checkpoint):
        return None, 0.5
    net, ck = grid_train.load(cfg.checkpoint)
    if (ck["n_rays"], ck["n_points"]) != (cfg.n_rays, cfg.n_points):
        log.warn(f"  [note] the checkpoint was trained at {ck['n_rays']} rays "
                 f"x {ck['n_points']} points, not {cfg.n_rays} x "
                 f"{cfg.n_points} — the network panels are skipped")
        return None, 0.5
    return grid_train.predict(net, X), float(ck["threshold"])


def render_config(cfg, splits: Optional[Sequence[str]] = None,
                  panels: Sequence[str] = DEFAULT_PANELS,
                  per_page: int = 10, columns: int = 2,
                  dpi: int = 150, cell_grid: bool = False) -> List[str]:
    """
    Contact sheets for a configuration's train and test devices.

    The sample folders and the measurements come out of the configuration's
    own train.npz / test.npz, so the devices on the page are exactly the
    devices the model was trained and scored on — and the reconstruction
    shown is the one produced from that device's own rays.
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
        X, _, sample_dirs = load_split(npz)
        prob, threshold = _prediction(cfg, X)
        written += render_split(sample_dirs, out_dir, split, panels=panels,
                                prob=prob, threshold=threshold,
                                in_sample=(split == "train"),
                                per_page=per_page, columns=columns, dpi=dpi,
                                cell_grid=cell_grid)
    if written:
        log.say(f"  {len(written)} contact sheet(s) -> "
                f"{os.path.abspath(out_dir)}")
    return written
