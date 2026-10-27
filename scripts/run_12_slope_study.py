"""
Can the loading-line slopes a virtual-gate matrix needs be measured from the
rays?

    python scripts/run_12_slope_study.py

One trained configuration, four ways of getting points, ONE line fit
(csdrecon/study/slopes.py), scored against the slopes the capacitances give:

    network    the U-Net's predicted lines, from the rays        (the method)
    canny      Canny edges of the full dense scan, 100 % measured (convention)
    truth      the true line map                                  (best case)
    ray_steps  the steps on each ray, no network                  (needed?)

Every knob (Canny's smoothing and thresholds, the step threshold, the Hough
settings of each arm) is chosen on the VALIDATION devices — the slice of the
training devices grid_train held out to pick the epoch and threshold — by
the error that is reported.  The test devices are only scored.

Writes <cell>/slopes/: per_device.csv, summary.json, results.txt,
angle_error_ecdf.png and examples.png.
"""
import csv
import json
import os

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt

from csdrecon.config import log
from csdrecon.ml import grid_train
from csdrecon.ml.ray_peaks import load_grid, load_ground_truth
from csdrecon.study import slopes as S
from csdrecon.study.config import StudyConfig
from csdrecon.study.dataset import load_split

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The configuration whose network is tested: 8 rays x 60 points, 4.7 % of
# the plane.
CELL = os.path.join(ROOT, "results", "8_rays_60_points_500_samples",
                    "8_rays_60_points_500_samples")

# Which trained seed of that cell to use (0 = model/, k = model_seed<k>/).
# Seed 3: the best VALIDATION F1@1 of seeds 0-3 (0.583, 0.421, 0.780, 0.817);
# training at this budget is seed-fragile, so the seed is chosen on
# validation devices and the others are reported, not hidden.
TRAIN_SEED = 3

# The baselines' own knobs, scanned on the validation devices.
# Canny: (smoothing sigma, low and high gradient quantile, background
# subtraction width in px; 0 = none).
CANNY_GRID = [(s, lo, hi, bg) for s in (1.0, 2.0)
              for lo, hi in ((0.80, 0.90), (0.90, 0.95), (0.95, 0.98))
              for bg in (0.0, 3.0, 6.0, 12.0)]
# Ray steps: a jump k times the local background slope.
STEP_K = (0.5, 1.0, 1.5, 2.0, 3.0)

# How many test devices examples.png shows.
N_EXAMPLES = 4

# ══════════════════════════════════════════════════════════════════════════

ARMS = ("network", "canny", "truth", "ray_steps")
LABEL = {"network": "network (rays, 4.7 %)",
         "canny": "Canny (dense scan, 100 %)",
         "truth": "true line map (fit only)",
         "ray_steps": "ray steps, no network"}
# validated categorical slots 1-4, fixed order
COLOR = {"network": "#2a78d6", "canny": "#eb6834", "truth": "#1baf7a",
         "ray_steps": "#eda100"}
INK, MUTED = "#111111", "#555555"


def _cfg():
    cfg = StudyConfig.load(CELL)
    if TRAIN_SEED:
        from dataclasses import replace
        d = cfg._dir
        cfg = replace(cfg, train_seed=TRAIN_SEED)
        cfg._dir = d
    return cfg


def _validation_dirs(train_dirs):
    """The devices grid_train held out of fitting: same rng, same slice."""
    rng = np.random.default_rng(grid_train.SEED)
    idx = rng.permutation(len(train_dirs))
    n_val = max(1, int(grid_train.VAL_FRACTION * len(train_dirs)))
    return sorted(int(i) for i in idx[:n_val])


def _point_sets(dirs, X, net, threshold, n_rays, n_points):
    """Every arm's knob-dependent point sets for a list of devices."""
    prob = grid_train.predict(net, X)
    out = {"network": {"stored threshold": [S.mask_points(p > threshold)
                                            for p in prob]},
           "truth": {"-": [S.mask_points(load_ground_truth(d)) for d in dirs]}}
    grids = [load_grid(d)[2] for d in dirs]
    out["canny"] = {f"sigma={s:g} q=({lo:g},{hi:g}) background={bg:g}px":
                    [S.canny_points(Z, s, lo, hi, bg) for Z in grids]
                    for s, lo, hi, bg in CANNY_GRID}
    out["ray_steps"] = {f"k={k:g}": [S.ray_step_points(d, n_rays, n_points, k)
                                     for d in dirs] for k in STEP_K}
    return out, prob


def _truths(dirs):
    out = []
    for d in dirs:
        t = S.device_truth(d)
        with open(os.path.join(d, "device.json")) as f:
            cap = json.load(f)["capacitance"]
        t["angle_sensor"] = {k: S.slope_to_angle(v)
                             for k, v in S.sensor_slopes(cap).items()}
        out.append(t)
    return out


def _ecdf(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, fam, title in zip(axes, S.FAMILIES,
                              ("dot-1 lines (steep)", "dot-2 lines (shallow)")):
        for arm in ARMS:
            e = np.array([r[f"{arm}_{fam}_err"] for r in rows
                          if r[f"{fam}_present"]], float)
            n = len(e)
            e = np.sort(np.nan_to_num(e, nan=np.inf))
            y = np.arange(1, n + 1) / n
            ax.step(np.clip(e, 1e-3, 90), y, where="post", lw=2,
                    color=COLOR[arm], label=LABEL[arm])
        for t in (1.0,):
            ax.axvline(t, color=MUTED, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 90)
        ax.set_xlabel("|fitted - true| line angle (degrees)")
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("fraction of held-out devices")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Slope error on the held-out devices (a failed fit never "
                 "reaches the top)", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _examples(dirs, sets, truths, fits, out_path):
    n = min(N_EXAMPLES, len(dirs))
    fig, axes = plt.subplots(n, len(ARMS), figsize=(3 * len(ARMS), 3 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        t = truths[i]
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            xy = sets[arm][i]
            if len(xy):
                ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5 if arm != "ray_steps"
                        else 5, color=COLOR[arm])
            for fam in S.FAMILIES:
                if not t["present"][fam]:
                    continue
                for ang, style, c in ((t["angle"][fam], "-", INK),
                                      (fits[arm][i][fam], "--", "#e34948")):
                    if ang is None:
                        continue
                    u = np.array([np.cos(np.radians(ang)),
                                  np.sin(np.radians(ang))])
                    ctr = np.array([50.0, 50.0])
                    p = np.stack([ctr - 80 * u, ctr + 80 * u])
                    ax.plot(p[:, 0], p[:, 1], style, color=c, lw=1)
            ax.set_xlim(0, 99)
            ax.set_ylim(0, 99)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(LABEL[arm], fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(t["sample"], fontsize=8, color=MUTED)
    fig.suptitle("solid: true slope  ·  dashed: fitted (drawn through the "
                 "centre, direction only)", fontsize=9, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    cfg = _cfg()
    out_dir = os.path.join(cfg.dir, "slopes" + (f"_seed{TRAIN_SEED}"
                                                if TRAIN_SEED else ""))
    os.makedirs(out_dir, exist_ok=True)
    net, ck = grid_train.load(cfg.checkpoint)
    threshold = float(ck["threshold"])
    shape = (cfg.resolution, cfg.resolution)

    Xtr, _, train_dirs = load_split(cfg.train_npz)
    vi = _validation_dirs(train_dirs)
    val_dirs = [train_dirs[i] for i in vi]
    Xte, _, test_dirs = load_split(cfg.test_npz)
    log.say(f"slope study — {cfg.name}, train seed {TRAIN_SEED}, "
            f"{len(val_dirs)} validation / {len(test_dirs)} test devices")

    # 1. choose every knob on the validation devices
    v_sets, _ = _point_sets(val_dirs, Xtr[vi], net, threshold,
                            cfg.n_rays, cfg.n_points)
    v_truth = _truths(val_dirs)
    chosen = {}
    for arm in ARMS:
        chosen[arm] = S.choose(v_sets[arm], v_truth, shape)
        log.say(f"  {arm:<10} chose {chosen[arm]['knob']}, "
                f"{chosen[arm]['hough']}  (val score "
                f"{chosen[arm]['score']:.3f})")
    del v_sets

    # 2. score the test devices with those settings
    t_all, _ = _point_sets(test_dirs, Xte, net, threshold,
                           cfg.n_rays, cfg.n_points)
    t_truth = _truths(test_dirs)
    t_sets = {a: t_all[a][chosen[a]["knob"]] for a in ARMS}
    fits = {a: [S.fit_slopes(xy, shape, chosen[a]["hough"])
                for xy in t_sets[a]] for a in ARMS}

    rows = []
    for i, t in enumerate(t_truth):
        row = {"sample": t["sample"]}
        for fam in S.FAMILIES:
            row[f"{fam}_present"] = t["present"][fam]
            row[f"{fam}_true_angle"] = t["angle"][fam]
            row[f"{fam}_true_angle_sensor"] = t["angle_sensor"][fam]
            for a in ARMS:
                f = fits[a][i][fam]
                row[f"{a}_{fam}_angle"] = f
                row[f"{a}_{fam}_err"] = S.angle_error(f, t["angle"][fam])
                row[f"{a}_{fam}_err_sensor"] = S.angle_error(
                    f, t["angle_sensor"][fam])
                row[f"{a}_{fam}_dalpha"] = (
                    float("nan") if f is None else abs(
                        S.cross_coupling(fam, np.tan(np.radians(f)))
                        - S.cross_coupling(fam, t["slope"][fam])))
        for a in ARMS:
            row[f"{a}_points"] = int(len(t_sets[a][i]))
        rows.append(row)

    with open(os.path.join(out_dir, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"configuration": cfg.name, "train_seed": TRAIN_SEED,
               "network_threshold": threshold,
               "n_validation": len(val_dirs), "n_test": len(test_dirs),
               "chosen_on_validation": chosen, "arms": {}}
    for a in ARMS:
        summary["arms"][a] = {"mean_points_per_device": float(np.mean(
            [r[f"{a}_points"] for r in rows]))}
        for key, suffix in (("vs_dot_model", "err"),
                            ("vs_sensor_model", "err_sensor")):
            summary["arms"][a][key] = {}
            for fam in S.FAMILIES + ("both",):
                fams = S.FAMILIES if fam == "both" else (fam,)
                e = [r[f"{a}_{g}_{suffix}"] for r in rows for g in fams
                     if r[f"{g}_present"]]
                summary["arms"][a][key][fam] = S.summarise(e)
        d = [r[f"{a}_{g}_dalpha"] for r in rows for g in S.FAMILIES
             if r[f"{g}_present"]]
        d = np.asarray(d, float)
        summary["arms"][a]["cross_coupling_abs_error_median"] = (
            float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"SLOPE STUDY — {cfg.name}, train seed {TRAIN_SEED}",
             f"{len(test_dirs)} held-out devices; knobs chosen on "
             f"{len(val_dirs)} validation devices",
             "angle error |fitted - true| in degrees, true = capacitance "
             "model of the double dot; a failed fit is a miss", "",
             f"{'arm':<11}{'family':<7}{'n':>4}{'fail':>6}{'median':>9}"
             f"{'p90':>8}{'<=0.5':>7}{'<=1':>7}{'<=2':>7}"
             f"{'med vs sensor':>15}"]
    for a in ARMS:
        for fam in S.FAMILIES + ("both",):
            s = summary["arms"][a]["vs_dot_model"][fam]
            s2 = summary["arms"][a]["vs_sensor_model"][fam]
            lines.append(f"{a:<11}{fam:<7}{s['n']:>4}{s['n_failed']:>6}"
                         f"{s['median_deg']:>9.3f}{s['p90_deg']:>8.2f}"
                         f"{s['within_0.5deg']:>7.2f}{s['within_1deg']:>7.2f}"
                         f"{s['within_2deg']:>7.2f}{s2['median_deg']:>15.3f}")
        lines.append("")
    lines.append("median |d alpha| (virtual-gate cross-coupling): " + ", ".join(
        f"{a} {summary['arms'][a]['cross_coupling_abs_error_median']:.4f}"
        for a in ARMS))
    lines.append("mean points per device: " + ", ".join(
        f"{a} {summary['arms'][a]['mean_points_per_device']:.0f}"
        for a in ARMS))
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "results.txt"), "w",
              encoding="utf-8") as f:
        f.write(text + "\n")
    log.say("\n" + text)

    _ecdf(rows, os.path.join(out_dir, "angle_error_ecdf.png"))
    _examples(test_dirs, t_sets, t_truth, fits,
              os.path.join(out_dir, "examples.png"))
    log.say(f"\nwrote {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
