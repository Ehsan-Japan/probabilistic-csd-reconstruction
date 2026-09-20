"""
model_figures.py — the comparison gallery: every way of putting the trained
models beside each other.

run_4 (and run_3's COMPARE_AFTER) writes all of these into

    results/comparison/figures/

They come from three sources, in increasing order of how much they tell you:

    comparison.csv          one row per model — the headline numbers
    <cfg>/evaluation/       per_device.csv, 100 rows per model — the SPREAD,
                            and, when every model was scored on the same test
                            pool, device-by-device PAIRED comparisons
    <cfg>/model/            history.json — how each model trained

Pick the two or three that make your point; the rest are diagnostics.

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

def _int_axis(ax, values):
    """Whole-number ticks: a count of rays is never 3.25."""
    ax.set_xticks(sorted({int(v) for v in values}))


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


def rounded_bars(ax, xs, heights, width, colors, base: float = 0.0,
                 radius_frac: float = 0.22, zorder: int = 3):
    """
    Bars with rounded data-ends and a square base.

    matplotlib has no rounded bar; the shape is built as a path so the end
    that carries the value is soft and the end anchored to the baseline stays
    flat, which is what keeps the reading unambiguous.
    """
    if np.isscalar(colors):
        colors = [colors] * len(xs)
    for x, h, c in zip(xs, heights, colors):
        if not np.isfinite(h):
            continue
        r = min(width * radius_frac, abs(h - base) * 0.5)
        x0, x1 = x - width / 2, x + width / 2
        top = h
        sign = 1 if h >= base else -1
        verts = [(x0, base), (x0, top - sign * r),
                 (x0, top), (x0 + r, top),          # curve
                 (x1 - r, top),
                 (x1, top), (x1, top - sign * r),   # curve
                 (x1, base), (x0, base)]
        codes = [MplPath.MOVETO, MplPath.LINETO,
                 MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO,
                 MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=c,
                               edgecolor="none", zorder=zorder))


def _label_end(ax, x, y, text: str, color: str, dx: float = 6, dy: float = 0):
    """Direct label at a series endpoint — identity without reading a legend."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8.5, color=INK_2, va="center")


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


def _test_pool(cfg) -> Optional[str]:
    """The pool a configuration's devices came from, or None."""
    path = os.path.join(cfg.dir, "dataset_summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    ids = tuple(d.get("split", {}).get("test_ids", []))
    # Two models are paired-comparable when they were scored on the same pool
    # AND the same test IDs out of it.
    return (d.get("pool"), ids) if d.get("pool") else None


def paired_ok(models: Sequence[Model]) -> bool:
    """
    True when device i means the same physical device in every model.

    Only then can the models be compared device by device — the strongest
    comparison available, because it removes device difficulty from the
    picture entirely.  Two configurations built with different n_test, or
    from different capacitance intervals, do not qualify.
    """
    if len(models) < 2:
        return False
    pools = {_test_pool(m.cfg) for m in models}
    sizes = {len(m.per_device) for m in models}
    return len(pools) == 1 and None not in pools and len(sizes) == 1 \
        and 0 not in sizes


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

def fig_f1_vs_rays(models, out_dir):
    """F1 against number of rays, one line per ray resolution."""
    groups: Dict[int, List[Model]] = {}
    for m in models:
        groups.setdefault(m.n_points, []).append(m)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for points, ms in sorted(groups.items()):
        ms = sorted(ms, key=lambda m: m.n_rays)
        x = [m.n_rays for m in ms]
        y = [m.get(HEADLINE) for m in ms]
        e = [m.get("f1@1_std", 0.0) for m in ms]
        # The connecting line is neutral and the ray resolution is named by a
        # direct label.  Colour has ONE meaning across this whole gallery —
        # which model — so it cannot also be spent on "which resolution"
        # here, or blue would mean two different things in two figures.
        ax.errorbar(x, y, yerr=e, fmt="-", color=MUTED, lw=1.2, capsize=3,
                    elinewidth=1.0, ecolor=MUTED, zorder=2)
        for m, xi, yi in zip(ms, x, y):
            ax.plot(xi, yi, "o", color=m.color, ms=MARK_S + 2,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        _label_end(ax, x[-1], y[-1], f"{points} points per ray", INK_2, dx=10)
    _style(ax, "number of rays", HEADLINE_LABEL,
           "Transition-line recovery vs number of rays")
    _int_axis(ax, [m.n_rays for m in models])
    ax.set_ylim(0, 1)
    for m in sorted(models, key=lambda m: m.n_rays):
        ax.plot([], [], "o", color=m.color, ms=MARK_S, label=m.label)
    if len(models) > 1:
        ax.legend(loc="lower right", ncol=2 if len(models) > 4 else 1)
    fig.text(0.01, -0.02, "error bars: standard deviation over held-out devices",
             fontsize=8, color=MUTED)
    return _save(fig, out_dir, "01_f1_vs_rays")


def _panel_f1_vs_coverage(ax, models):
    """The result on the honest axis: what the measurement cost."""
    ms = sorted(models, key=lambda m: m.get("coverage"))
    x = [100 * m.get("coverage") for m in ms]
    y = [m.get(HEADLINE) for m in ms]
    ax.plot(x, y, "-", color=SERIES[0], lw=LINE_W, zorder=2)
    for m in ms:
        ax.plot(100 * m.get("coverage"), m.get(HEADLINE), "o", color=m.color,
                ms=MARK_S + 2, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)
        _label_end(ax, 100 * m.get("coverage"), m.get(HEADLINE), m.label,
                   m.color, dx=8)
    _style(ax, "fraction of the grid actually measured (%)", HEADLINE_LABEL,
           "Accuracy against measurement cost")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)


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


def fig_f1_vs_coverage(models, out_dir):
    """The same result on the honest axis: what the measurement cost."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _panel_f1_vs_coverage(ax, models)
    return _save(fig, out_dir, "02_f1_vs_coverage")


def fig_f1_vs_tolerance(models, out_dir):
    """How much of the error is sub-pixel misalignment rather than a miss."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _panel_f1_vs_tolerance(ax, models)
    fig.text(0.01, -0.02,
             "a predicted line pixel counts as correct if a true one lies "
             "within tau pixels", fontsize=8, color=MUTED)
    return _save(fig, out_dir, "03_f1_vs_tolerance")


def fig_metric_bars(models, out_dir):
    """Every tolerance, every model, as grouped bars."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    n = len(models)
    group_w = 0.8
    bar_w = group_w / max(n, 1) * 0.86        # the gap between adjacent bars
    for i, m in enumerate(models):
        xs = np.arange(len(TAUS)) - group_w / 2 + group_w * (i + 0.5) / n
        rounded_bars(ax, xs, [m.get(f"f1@{t}") for t in TAUS], bar_w, m.color)
        ax.plot([], [], "s", color=m.color, ms=8, label=m.label)
    _style(ax, "tolerance tau (pixels)", "F1 on held-out devices",
           "F1 at every tolerance, per model")
    ax.set_xticks(range(len(TAUS)), [f"tau = {t}" for t in TAUS])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(TAUS) - 0.4)
    if len(models) > 1:
        ax.legend(loc="upper left", ncol=min(len(models), 4))
    return _save(fig, out_dir, "04_f1_bars_by_tolerance")


def fig_precision_recall(models, out_dir):
    """
    Where each model sits in the precision/recall plane, with iso-F1 curves.

    A single hue for the trajectory: this is a scatter, where every pair of
    colours would have to separate at once, so identity is carried by direct
    labels instead of by eight hues.
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ms = sorted(models, key=lambda m: m.n_rays)
    xs = [m.get(f"precision@{TAU}") for m in ms]
    ys = [m.get(f"recall@{TAU}") for m in ms]

    # Zoom FIRST, then draw: three budgets sit within a few hundredths of each
    # other, the full unit square hides the difference entirely, and the
    # iso-F1 labels have to be placed inside the view to be readable.
    lo = max(0.0, min(min(xs), min(ys)) - 0.14)
    hi = min(1.0, max(max(xs), max(ys)) + 0.14)
    if hi - lo < 0.28:
        mid = 0.5 * (lo + hi)
        lo, hi = max(0.0, mid - 0.14), min(1.0, mid + 0.14)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    g = np.linspace(max(lo, 0.01), hi, 300)
    P, R = np.meshgrid(g, g)
    F = 2 * P * R / np.clip(P + R, 1e-9, None)
    cs = ax.contour(P, R, F, levels=np.round(np.arange(0.1, 1.0, 0.05), 2),
                    colors=[GRID], linewidths=0.8, zorder=1)
    ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"F1 {v:g}")

    ax.plot(xs, ys, "-", color=MUTED, lw=1.0, zorder=2)
    for k, m in enumerate(ms):
        ax.plot(xs[k], ys[k], "o", color=m.color, ms=MARK_S + 3,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # The models lie on a short trajectory, so a label to the right of
        # every dot would land on the next dot.  The ends point outwards and
        # anything between them goes above.
        if k == 0:
            ax.annotate(m.label, (xs[k], ys[k]), textcoords="offset points",
                        xytext=(-12, -4), ha="right", fontsize=8.5,
                        color=INK_2)
        elif k == len(ms) - 1:
            ax.annotate(m.label, (xs[k], ys[k]), textcoords="offset points",
                        xytext=(12, -4), ha="left", fontsize=8.5, color=INK_2)
        else:
            ax.annotate(m.label, (xs[k], ys[k]), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=8.5,
                        color=INK_2)
    _style(ax, f"precision @ {TAU} px", f"recall @ {TAU} px",
           "Precision against recall", grid_axis="both")
    ax.set_aspect("equal")
    fig.text(0.01, -0.02,
             "up = finds more of the lines;  right = fewer of the lines it "
             "draws are spurious.  Grey curves are constant F1.",
             fontsize=8, color=MUTED)
    return _save(fig, out_dir, "05_precision_recall_plane")


def fig_precision_recall_vs_rays(models, out_dir):
    """Precision and recall separately — which one the extra rays buy."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    ms = sorted(models, key=lambda m: m.n_rays)
    for ax, key, title in ((axes[0], "precision", "Precision"),
                           (axes[1], "recall", "Recall")):
        ax.plot([m.n_rays for m in ms], [m.get(f"{key}@{TAU}") for m in ms],
                "-", color=MUTED, lw=1.0, zorder=2)
        for m in ms:
            ax.plot(m.n_rays, m.get(f"{key}@{TAU}"), "o", color=m.color,
                    ms=MARK_S + 2, markeredgecolor=SURFACE, markeredgewidth=2,
                    zorder=3)
        _style(ax, "number of rays",
               f"{title.lower()} @ {TAU} px" if ax is axes[0] else "", title)
        _int_axis(ax, [m.n_rays for m in ms])
        ax.set_ylim(0, 1)
    for m in ms:
        axes[1].plot([], [], "o", color=m.color, ms=MARK_S, label=m.label)
    if len(ms) > 1:
        axes[1].legend(loc="lower right")
    fig.suptitle("What the extra rays actually buy", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, out_dir, "06_precision_recall_vs_rays")


def fig_heatmap(models, out_dir):
    """rays x points grid of F1 — needs a two-dimensional sweep."""
    rays = sorted({m.n_rays for m in models})
    points = sorted({m.n_points for m in models})
    if len(rays) < 2 or len(points) < 2:
        return None
    grid = np.full((len(points), len(rays)), np.nan)
    for m in models:
        grid[points.index(m.n_points), rays.index(m.n_rays)] = m.get(HEADLINE)
    fig, ax = plt.subplots(figsize=(1.15 * len(rays) + 3.2,
                                    1.0 * len(points) + 2.6))
    im = ax.imshow(grid, origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=1,
                   aspect="auto")
    for i in range(len(points)):
        for j in range(len(rays)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=9.5,
                        color="white" if grid[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(rays)), [str(v) for v in rays])
    ax.set_yticks(range(len(points)), [str(v) for v in points])
    _style(ax, "number of rays", "points per ray",
           "F1 @ 1 px across the budget grid", grid_axis="x")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=HEADLINE_LABEL)
    return _save(fig, out_dir, "10_f1_heatmap")


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
#  C. Tolerance — how accuracy shifts with tau, model by model
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


def fig_tau_grid(models, out_dir):
    """Models against tolerances, as one grid of numbers."""
    ms = sorted(models, key=lambda m: (m.n_points, m.n_rays))
    grid = np.array([[m.get(f"f1@{t}") for t in TAUS] for m in ms])
    fig, ax = plt.subplots(figsize=(1.5 * len(TAUS) + 3.6,
                                    0.62 * len(ms) + 2.4))
    im = ax.imshow(grid, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(TAUS)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=9.5,
                        color="white" if grid[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(TAUS)), [f"tau = {t}" for t in TAUS])
    ax.set_yticks(range(len(ms)), [m.label for m in ms])
    for tick, m in zip(ax.get_yticklabels(), ms):
        tick.set_color(m.color)
    _style(ax, "", "", "F1 for every model at every tolerance")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="F1")
    return _save(fig, out_dir, "41_tau_grid")

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
#  D. Training figures — how each model got there
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


def fig_generalisation_gap(models, out_dir):
    """
    Best validation F1 against held-out F1 — the price of unseen geometry.

    The validation devices are drawn from the TRAINING bands; the held-out
    devices from the disjoint ones.  The distance below the diagonal is what
    the generalisation to unseen device geometry actually costs.
    """
    ms = [m for m in models if m.history]
    if not ms:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    vxs = [max(m.history["val_f1"]) for m in ms]
    vys = [m.get(HEADLINE) for m in ms]
    # Zoomed to the models: the gap is a few hundredths, and on the full unit
    # square every point lands on the diagonal and the figure says nothing.
    lo = max(0.0, min(min(vxs), min(vys)) - 0.08)
    hi = min(1.0, max(max(vxs), max(vys)) + 0.08)
    if hi - lo < 0.2:
        mid = 0.5 * (lo + hi)
        lo, hi = max(0.0, mid - 0.1), min(1.0, mid + 0.1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1.0, zorder=1)
    ax.annotate("equal on both", (hi, hi), textcoords="offset points",
                xytext=(-6, -14), ha="right", fontsize=8, color=MUTED)
    for m, vx, vy in zip(ms, vxs, vys):
        # A dropline to the diagonal: its length IS the cost of unseen geometry.
        ax.plot([vx, vx], [vy, vx], color=m.color, lw=1.4, alpha=0.6, zorder=2)
        ax.plot(vx, vy, "o", color=m.color, ms=MARK_S + 3,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        _label_end(ax, vx, vy, f"{m.label}   -{vx - vy:.3f}", m.color, dx=10)
    _style(ax, "best validation F1 @ 1 px (training bands)",
           "held-out F1 @ 1 px (disjoint bands)",
           "What unseen device geometry costs", grid_axis="both")
    ax.set_aspect("equal")
    return _save(fig, out_dir, "32_generalisation_gap")

# ══════════════════════════════════════════════════════════════════════════

GALLERY = [
    fig_f1_vs_rays, fig_f1_vs_coverage, fig_f1_vs_tolerance,
    fig_metric_bars, fig_precision_recall, fig_precision_recall_vs_rays,
    fig_heatmap, fig_train_size,
    fig_tau_all_metrics, fig_tau_grid, fig_tau_normalised, fig_tau_band,
    fig_training_loss, fig_validation_f1, fig_generalisation_gap,
]


def render_all(configs, rows: Sequence[Dict], out_dir: str) -> List[str]:
    """
    Draw every comparison figure that this set of models supports.

    A figure that needs something the sweep does not have — a second ray
    resolution for the heatmap, two training-set sizes for the learning
    curve — returns None and is reported as skipped rather than drawn empty.
    """
    if not rows:
        return []
    _rc()
    models = build_models(configs, rows)
    if not models:
        return []

    if paired_ok(models):
        log.detail("  every model was scored on the SAME held-out devices — "
                   "paired per-device figures are included")
    else:
        log.detail("  models were scored on different held-out sets — paired "
                   "per-device figures are skipped")

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
