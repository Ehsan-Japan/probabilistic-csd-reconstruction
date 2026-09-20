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
from matplotlib.lines import Line2D
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

def _style(ax, xlabel: str = "", ylabel: str = "", grid_axis: str = "y"):
    """Axis labels and the house grid.  NO TITLE, deliberately: the file name
    already says what the figure is, and a title repeating it only steals
    height and has to be cropped out again for a paper or a slide."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
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
    _style(ax, "tolerance tau (pixels)", "F1 on held-out devices")
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
    _style(ax, "training devices", HEADLINE_LABEL)
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
    """
    Every tolerance-dependent metric against tau -- ONE FILE EACH.

    Four panels side by side forced every axes into a quarter of the width,
    and a slide or a paper never wants all four at once anyway.  Separate
    files can be placed, cropped and captioned independently.
    """
    out = []
    for suffix, (key, label) in zip("abcd", TAU_METRICS):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        drew = False
        for m in models:
            y = [m.get(f"{key}@{t}") for t in TAUS]
            if not np.all(np.isfinite(y)):
                continue
            ax.plot(TAUS, y, "o-", color=m.color, lw=LINE_W, ms=MARK_S,
                    label=m.label, zorder=3)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        _style(ax, "tolerance tau (pixels)",
               "%s on the held-out devices" % label)
        ax.set_xticks(list(TAUS))
        ax.set_ylim(0, 1)
        if len(models) > 1:
            ax.legend(loc="lower right", fontsize=8)
        if key == "accuracy":
            fig.text(0.02, -0.03,
                     "pixel accuracy is high whatever the model does",
                     fontsize=8, color=MUTED)
        fig.tight_layout()
        out.append(_save(fig, out_dir, f"40{suffix}_tau_{key}"))
    return out


def fig_f1_and_coverage(models, out_dir):
    """
    The score on the left axis, the area it is read over on the right.

    40a_tau_f1 shows F1 climbing with tau and stops there, which flatters
    the result: F1 climbs partly because the tolerance keeps widening the
    strip of plane a predicted pixel is allowed to stand for.  coverage@tau
    is that strip -- the fraction of the diagram within tau pixels of
    something the rays actually touched -- and it belongs in the same
    picture.

    TWO SCALES, so each curve fills its own axis and the shapes can be
    compared.  The cost is that the VERTICAL GAP between a solid line and
    its dashed partner no longer means anything: the right axis can be
    stretched or squashed independently, so only the SHAPES are comparable
    here, not the distance between them.  41b draws the same two quantities
    on one shared 0-1 axis, where the gap is a real distance; read that one
    when the question is "how much of the score is just a wider ruler".

    Left axis, solid:  F1@tau.
    Right axis, dashed: coverage@tau, as a percentage of the plane.
    """
    if not all(np.isfinite([m.get(f"coverage@{t}") for t in TAUS]).all()
               for m in models):
        return None              # an older run recorded no coverage@tau

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    right = ax.twinx()
    for m in models:
        ax.plot(TAUS, [m.get(f"f1@{t}") for t in TAUS], "o-", color=m.color,
                lw=LINE_W, ms=MARK_S, label=m.label, zorder=3)
        right.plot(TAUS, [100 * m.get(f"coverage@{t}") for t in TAUS],
                   "o--", color=m.color, lw=LINE_W * 0.8, ms=MARK_S * 0.6,
                   alpha=0.75, zorder=2)

    _style(ax, "tolerance tau (pixels)", "F1 @ tau   (left axis, solid)")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)

    top = max(100 * m.get("coverage@%d" % TAUS[-1]) for m in models)
    right.set_ylim(0, top * 1.12)
    right.set_ylabel("coverage @ tau  (% of the plane;  right axis, dashed)")
    # the twin brings its own frame: keep the right spine, drop the rest, and
    # do NOT draw a second grid over the first
    right.grid(False)
    right.spines["top"].set_visible(False)
    right.spines["right"].set_visible(True)
    right.tick_params(axis="y")

    style_key = [Line2D([], [], color=INK_2, lw=LINE_W, marker="o",
                        ms=MARK_S, label="F1@tau  (left)"),
                 Line2D([], [], color=INK_2, lw=LINE_W * 0.8, ls="--",
                        marker="o", ms=MARK_S * 0.6,
                        label="coverage@tau  (right)")]
    first = ax.legend(handles=style_key, loc="upper left", fontsize=8,
                      frameon=True)
    ax.add_artist(first)
    if len(models) > 1:
        ax.legend(loc="lower right", fontsize=8)
    fig.text(0.02, -0.03,
             "two scales: compare the SHAPES, not the gap between a solid "
             "line and its dashed partner.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "41a_f1_and_coverage_twin_axis")


def fig_f1_and_coverage_shared(models, out_dir):
    """
    The same two quantities on ONE 0-1 axis.

    Both are fractions of the same plane, so they can share an axis, and
    then the vertical gap between a solid line and its dashed partner is a
    real distance: the part of the score that is not simply a wider ruler.
    Kept alongside the twin-axis version because that one, by construction,
    cannot show this.
    """
    if not all(np.isfinite([m.get(f"coverage@{t}") for t in TAUS]).all()
               for m in models):
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for m in models:
        ax.plot(TAUS, [m.get(f"f1@{t}") for t in TAUS], "o-", color=m.color,
                lw=LINE_W, ms=MARK_S, label=m.label, zorder=3)
        ax.plot(TAUS, [m.get(f"coverage@{t}") for t in TAUS], "o--",
                color=m.color, lw=LINE_W * 0.8, ms=MARK_S * 0.6,
                alpha=0.75, zorder=2)

    _style(ax, "tolerance tau (pixels)",
           "fraction of 1  (F1, or of the plane)")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)

    style_key = [Line2D([], [], color=INK_2, lw=LINE_W, marker="o",
                        ms=MARK_S, label="F1@tau"),
                 Line2D([], [], color=INK_2, lw=LINE_W * 0.8, ls="--",
                        marker="o", ms=MARK_S * 0.6,
                        label="coverage@tau  (the price)")]
    first = ax.legend(handles=style_key, loc="upper left", fontsize=8,
                      frameon=True)
    ax.add_artist(first)
    if len(models) > 1:
        ax.legend(loc="lower right", fontsize=8)
    fig.text(0.02, -0.03,
             "one shared scale, so the gap between a solid line and its "
             "dashed partner is a real distance.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "41b_f1_and_coverage_shared_axis")


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
           f"F1 as a fraction of its own F1 at tau = {TAUS[-1]}")
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
    _style(ax, "tolerance tau (pixels)", "F1 on held-out devices")
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
    _style(ax, "epoch", "training loss (weighted BCE + soft Dice)")
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
    _style(ax, "epoch", "validation F1 @ 1 px")
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
    fig_tau_all_metrics, fig_f1_and_coverage, fig_f1_and_coverage_shared,
    fig_tau_normalised,
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
            # a figure function returns one path, or SEVERAL when the thing
            # it draws belongs in separate files rather than as panels
            path = fn(models, out_dir)
            if isinstance(path, (list, tuple)):
                written.extend(p for p in path if p)
            elif path:
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
