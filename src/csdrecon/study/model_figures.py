"""
model_figures.py — the comparison gallery: every way of putting the trained
models beside each other.

run_4 (and run_3's COMPARE_AFTER) writes all of these into

    results/<sweep>/figures/

and run_10 re-renders them one bundle at a time, a few models per plot.

They come from three sources, in increasing order of how much they tell you:

    comparison.csv          one row per model — the headline numbers
    <cfg>/evaluation/       per_device.csv, one row per held-out device —
                            the SPREAD behind each mean
    <cfg>/model/            history.json — how each model trained

These are DIAGNOSTICS: they are for reading a sweep, not for showing one.
The figures that go in the paper and on slides are built separately, in
paper_figures/, one point per panel.

DESIGN RULES THIS FILE FOLLOWS
Colour identifies the MODEL and nothing else, assigned in a fixed order and
held across every figure — so the 3-ray model is the same blue everywhere,
and adding a 6-ray run never repaints it.  One measured quantity per axis;
never two y-scales.  Sequential (magnitude) encodings use one hue, light to
dark; the diverging ones use blue-to-red through a neutral grey.  Marks are
thin, grids are hairlines a shade off the surface, and a legend is always
present when there is more than one model.

The categorical palette is the validated reference set, used unchanged and in
its documented order.  Scatter-type figures, where every pair of colours has
to be separable at once, are capped at three models or drawn as small
multiples in a single hue instead.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..config import log
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# ── palette ───────────────────────────────────────────────────────────────
# Categorical: identity.  Fixed order, never cycled, never assigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_SERIES = len(SERIES)
# Scatter and other forms where EVERY pair must separate at once: the first
# three slots are the validated all-pairs set.
MAX_SCATTER_SERIES = 3

# Sequential: magnitude.  One hue, light -> dark.
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("dqd_seq", SEQ_STEPS)
# Diverging: polarity, through a neutral grey midpoint.
DIV_CMAP = LinearSegmentedColormap.from_list(
    "dqd_div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f2020"])

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e2de"

LINE_W = 1.8
MARK_S = 6.0
FIG_DPI = 200

# The headline metric, named once.
TAU = 1
HEADLINE = f"f1@{TAU}"
HEADLINE_LABEL = "F1 @ 1 px tolerance"
TAUS = (0, 1, 2, 3)


def _rc():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": FIG_DPI,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10,
    })


# ── small helpers ─────────────────────────────────────────────────────────

def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "",
           grid_axis: str = "y"):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=10)
    ax.grid(True, axis=grid_axis, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ── the models ────────────────────────────────────────────────────────────

class Model:
    """
    One trained configuration, with everything the gallery needs already
    loaded: the summary row, the per-device table, the training history.

    ``color`` is fixed at construction from the model's position in the
    canonical order, so it identifies the model and not its rank.
    """

    def __init__(self, cfg, row: Dict, color: str):
        self.cfg = cfg
        self.row = row
        self.color = color
        self.name = row["configuration"]
        self.n_rays = int(row["n_rays"])
        self.n_points = int(row["n_points"])
        self.n_train = int(row["n_train"])
        self.per_device = _load_per_device(cfg)
        self.history = _load_history(cfg)

    @property
    def label(self) -> str:
        return f"{self.n_rays} rays x {self.n_points} pts"

    @property
    def long_label(self) -> str:
        return f"{self.n_rays} rays x {self.n_points} pts, {self.n_train} train"

    def get(self, key: str, default=np.nan) -> float:
        v = self.row.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def device(self, key: str) -> np.ndarray:
        """One per-device column as a float array."""
        if not self.per_device:
            return np.zeros(0)
        return np.array([float(r[key]) for r in self.per_device])


def _load_per_device(cfg) -> List[Dict]:
    import csv
    path = os.path.join(cfg.eval_dir, "per_device.csv")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_history(cfg) -> Optional[Dict]:
    path = os.path.join(cfg.model_dir, "history.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_models(configs, rows: Sequence[Dict]) -> List[Model]:
    """
    Pair each csv row with its configuration and give it a fixed colour.

    Colour is assigned by position in the list the user asked for, so it
    follows the model.  Past eight models the palette stops rather than
    inventing a ninth hue: the extra models stay in every table and in the
    single-hue figures, and are left out of the colour-coded ones.
    """
    by_name = {c.name: c for c in configs}
    models = []
    for i, row in enumerate(rows):
        cfg = by_name.get(row["configuration"])
        if cfg is None:
            continue
        models.append(Model(cfg, row, SERIES[i % MAX_SERIES]
                            if i < MAX_SERIES else MUTED))
    if len(models) > MAX_SERIES:
        log.detail(f"  [note] {len(models)} models but only {MAX_SERIES} "
                   f"distinguishable colours — the extra ones are drawn grey in "
                   f"colour-coded figures and appear in full in the tables")
    return models


# ══════════════════════════════════════════════════════════════════════════
#  A. Headline figures — from comparison.csv
# ══════════════════════════════════════════════════════════════════════════

def _panel_f1_vs_tolerance(ax, models):
    """How much of the error is sub-pixel misalignment rather than a miss."""
    for m in models:
        y = [m.get(f"f1@{t}") for t in TAUS]
        ax.plot(TAUS, y, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                label=m.label, zorder=3)
    _style(ax, "tolerance tau (pixels)", "F1 on held-out devices",
           "How much of the error is sub-pixel")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)
    if len(models) > 1:
        ax.legend(loc="lower right")


def fig_f1_vs_tolerance(models, out_dir):
    """How much of the error is sub-pixel misalignment rather than a miss."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _panel_f1_vs_tolerance(ax, models)
    fig.text(0.01, -0.02,
             "a predicted line pixel counts as correct if a true one lies "
             "within tau pixels", fontsize=8, color=MUTED)
    return _save(fig, out_dir, "03_f1_vs_tolerance")


def fig_train_size(models, out_dir):
    """F1 against training-set size, for models differing only in that."""
    groups: Dict[Tuple[int, int], List[Model]] = {}
    for m in models:
        groups.setdefault((m.n_rays, m.n_points), []).append(m)
    groups = {k: sorted(v, key=lambda m: m.n_train)
              for k, v in groups.items() if len({m.n_train for m in v}) > 1}
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, ((rays, points), ms) in enumerate(sorted(groups.items())):
        color = SERIES[i % MAX_SERIES]
        ax.plot([m.n_train for m in ms], [m.get(HEADLINE) for m in ms], "o-",
                color=color, lw=LINE_W, ms=MARK_S,
                label=f"{rays} rays x {points} pts", zorder=3)
    _style(ax, "training devices", HEADLINE_LABEL,
           "Would more training devices have helped?")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    if len(groups) > 1:
        ax.legend(loc="lower right")
    fig.text(0.01, -0.02,
             "read the curve, not the endpoint: still climbing = the result "
             "is data-limited", fontsize=8, color=MUTED)
    return _save(fig, out_dir, "11_f1_vs_training_size")


# ══════════════════════════════════════════════════════════════════════════
#  B. Tolerance — how accuracy shifts with tau, model by model
#
#  tau is not a knob on the model.  The network outputs a probability map,
#  the threshold turns it into a picture, and tau only decides which pixels
#  of that FIXED picture count as correct: a predicted line pixel scores if a
#  true one lies within tau pixels.  So a curve that climbs steeply from
#  tau = 0 to 1 means the lines are in the right place but a pixel off; a
#  curve that stays flat and low means they are simply not there.  Comparing
#  the SHAPE of that curve between models is the cleanest way to separate
#  "draws the lines slightly wrong" from "misses the lines".
# ══════════════════════════════════════════════════════════════════════════

TAU_METRICS = (("f1", "F1"), ("precision", "precision"),
               ("recall", "recall"), ("accuracy", "pixel accuracy"))


def fig_tau_all_metrics(models, out_dir):
    """Every tolerance-dependent metric against tau, one panel each."""
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9), sharey=True)
    for ax, (key, label) in zip(axes, TAU_METRICS):
        for m in models:
            y = [m.get(f"{key}@{t}") for t in TAUS]
            if not np.all(np.isfinite(y)):
                continue
            ax.plot(TAUS, y, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                    label=m.label, zorder=3)
        _style(ax, "tolerance tau (pixels)", "", label)
        ax.set_xticks(list(TAUS))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score on the held-out devices")
    if len(models) > 1:
        axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Every metric against tolerance, every model", x=0.02,
                 ha="left", fontsize=13, color=INK)
    fig.text(0.02, -0.03,
             "pixel accuracy is shown only to be dismissed: transition lines "
             "are a few percent of the diagram, so it is high no matter what "
             "the model does", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _save(fig, out_dir, "40_tau_all_metrics")


def fig_tolerance_price(models, out_dir):
    """
    What raising tau BUYS, against what it COSTS.

    F1@tau can only rise with tau, so quoting a high one proves nothing on
    its own.  claim@tau is the price: dilate the model's own prediction by
    tau and measure what fraction of the plane it then covers.  Scoring at
    tau credits a predicted pixel for a whole disc of radius tau, so the
    model is in effect asserting "a line passes somewhere in here" for every
    pixel of that disc.  Once the discs swallow half the diagram, a high
    F1@tau says very little.

    Computed PER DEVICE and averaged over the held-out set, with the spread
    shown: line density varies several-fold across the test devices, so the
    area a prediction covers does too, and a single pooled number would hide
    it.

    (a) the price against tau, mean with a +/-1 sd band.
    (b) the trade-off: score against price.  A curve climbing steeply at the
        left is buying accuracy cheaply; one that flattens is only widening
        the band.
    """
    if not all(np.isfinite([m.get(f"claim@{t}") for t in TAUS]).all()
               for m in models):
        return None              # scored before claim@tau existed — re-run run_3

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # Error BARS, not shaded bands: at five models the bands overlap into an
    # unreadable smear.  Each series is nudged a little along x so its bars
    # are legible -- the x positions are the integers either way.
    n = len(models)
    for i, m in enumerate(models):
        claim = np.array([100 * m.get(f"claim@{t}") for t in TAUS])
        sd = np.array([100 * m.get(f"claim@{t}_std", 0.0) for t in TAUS])
        f1 = [m.get(f"f1@{t}") for t in TAUS]
        dodge = 0.0 if n < 2 else 0.30 * (i / (n - 1) - 0.5)
        x = np.array(TAUS, dtype=float) + dodge
        if np.all(np.isfinite(sd)) and sd.any():
            axes[0].errorbar(x, claim, yerr=sd, fmt="none", ecolor=m.color,
                             elinewidth=1.0, capsize=2.5, capthick=1.0,
                             alpha=0.75, zorder=2)
        axes[0].plot(x, claim, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                     label=m.label, zorder=3)
        axes[1].plot(claim, f1, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                     label=m.label, zorder=3)

    # tau is what moves along the right-hand curve, so it is labelled there
    lead = max(models, key=lambda m: m.get("claim@3"))
    for t in TAUS:
        axes[1].annotate(f"tau {t}",
                         (100 * lead.get(f"claim@{t}"), lead.get(f"f1@{t}")),
                         textcoords="offset points", xytext=(4, -11),
                         fontsize=8, color=MUTED, zorder=4)

    _style(axes[0], "tolerance tau (pixels)",
           "fraction of the plane the prediction claims (%)",
           "(a) what the tolerance costs")
    axes[0].set_xticks(list(TAUS))
    axes[0].set_ylim(0, None)
    if len(models) > 1:
        axes[0].legend(loc="upper left", fontsize=8)

    _style(axes[1], "fraction of the plane the prediction claims (%)",
           "F1 @ tau", "(b) score against price")
    axes[1].set_ylim(0, 1)

    fig.text(0.02, -0.03,
             "price = the model's own prediction dilated by tau, per device "
             "then averaged over the held-out set; band is +/-1 sd. F1@tau "
             "rises with tau by construction, so a score is only evidence of "
             "recovery while the area it claims stays small.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "45_tolerance_price")


def fig_tau_normalised(models, out_dir):
    """
    The SHAPE of each model's tolerance curve, with its overall level divided
    out (every curve is scaled to its own F1 at tau = 3).

    This is the figure that separates the two questions.  Models whose curves
    lie on top of each other make the SAME KIND of error and differ only in
    how much of it; a curve that starts lower than the others is a model
    whose lines are placed worse, not just fewer.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in models:
        top = m.get(f"f1@{TAUS[-1]}")
        if not np.isfinite(top) or top <= 0:
            continue
        y = [m.get(f"f1@{t}") / top for t in TAUS]
        ax.plot(TAUS, y, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                label=m.label, zorder=3)
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=1)
    _style(ax, "tolerance tau (pixels)",
           f"F1 as a fraction of its own F1 at tau = {TAUS[-1]}",
           "The shape of the tolerance curve, level removed")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1.08)
    if len(models) > 1:
        ax.legend(loc="lower right")
    fig.text(0.01, -0.03,
             "curves on top of each other = the same kind of error, "
             "different amounts of it", fontsize=8, color=MUTED)
    return _save(fig, out_dir, "43_tau_shape")


def fig_tau_band(models, out_dir):
    """
    The tolerance curve per DEVICE, not just for the mean.

    Line: the median device.  Band: the middle half of them.  Overlapping
    bands mean the difference between two budgets is smaller than the
    difference between two devices at either budget — worth knowing before
    claiming one budget beats another.
    """
    ms = [m for m in models if len(m.per_device)]
    if not ms:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for m in ms:
        curves = np.stack([m.device(f"f1@{t}") for t in TAUS])   # (tau, dev)
        med = np.median(curves, axis=1)
        q1 = np.percentile(curves, 25, axis=1)
        q3 = np.percentile(curves, 75, axis=1)
        ax.fill_between(TAUS, q1, q3, color=m.color, alpha=0.16, lw=0,
                        zorder=2)
        ax.plot(TAUS, med, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                label=m.label, zorder=3)
    _style(ax, "tolerance tau (pixels)", "F1 on held-out devices",
           "Tolerance curve, device spread included")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)
    if len(ms) > 1:
        ax.legend(loc="lower right")
    fig.text(0.01, -0.03,
             "line: the median device.  band: the middle half of the devices",
             fontsize=8, color=MUTED)
    return _save(fig, out_dir, "44_tau_device_band")


# ══════════════════════════════════════════════════════════════════════════
#  C. Training figures — how each model got there
# ══════════════════════════════════════════════════════════════════════════

def fig_training_loss(models, out_dir):
    ms = [m for m in models if m.history]
    if not ms:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ms:
        y = m.history["train_loss"]
        ax.plot(np.arange(1, len(y) + 1), y, color=m.color, lw=LINE_W,
                label=m.label, zorder=3)
    _style(ax, "epoch", "training loss (weighted BCE + soft Dice)",
           "Training loss")
    ax.set_yscale("log")
    if len(ms) > 1:
        ax.legend(loc="upper right")
    return _save(fig, out_dir, "30_training_loss")


def fig_validation_f1(models, out_dir):
    ms = [m for m in models if m.history]
    if not ms:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ms:
        y = m.history["val_f1"]
        ax.plot(np.arange(1, len(y) + 1), y, color=m.color, lw=LINE_W,
                label=m.label, zorder=3)
        b = int(np.argmax(y))
        ax.plot([b + 1], [y[b]], "o", color=m.color, ms=MARK_S,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    _style(ax, "epoch", "validation F1 @ 1 px", "Validation accuracy while training")
    ax.set_ylim(0, 1)
    if len(ms) > 1:
        ax.legend(loc="lower right")
    fig.text(0.01, -0.03,
             "validation devices come from the TRAINING capacitance bands; "
             "the held-out numbers elsewhere are from the disjoint ones",
             fontsize=8, color=MUTED)
    return _save(fig, out_dir, "31_validation_f1")


# ══════════════════════════════════════════════════════════════════════════

GALLERY = [
    fig_f1_vs_tolerance, fig_train_size,
    fig_tau_all_metrics, fig_tolerance_price, fig_tau_normalised,
    fig_tau_band,
    fig_training_loss, fig_validation_f1,
]


def render_all(configs, rows: Sequence[Dict], out_dir: str) -> List[str]:
    """
    Draw every comparison figure that this set of models supports.

    A figure that needs something the sweep does not have — two
    training-set sizes for the learning curve — returns None and is
    reported as skipped rather than drawn empty.
    """
    if not rows:
        return []
    _rc()
    models = build_models(configs, rows)
    if not models:
        return []

    written, skipped = [], []
    for fn in GALLERY:
        name = fn.__name__.replace("fig_", "")
        try:
            path = fn(models, out_dir)
            if path:
                written.append(path)
            else:
                skipped.append((name, "not enough variation in the sweep"))
        except Exception as exc:
            skipped.append((name, f"error: {exc}"))
            plt.close("all")

    log.detail(f"  {len(written)} comparison figures -> {os.path.abspath(out_dir)}")
    for name, why in skipped:
        log.detail(f"    [skipped] {name}: {why}")
    return written
