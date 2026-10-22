"""
Virtual gates from the measured slopes — before and after, per method.

    python scripts/run_13_virtual_gates.py

Reads what run_12_slope_study.py measured (<cell>/slopes_seed<k>/) and asks
the question the slopes are for: build the virtual-gate matrix from each
method's two slopes, apply it, and see how much crosstalk is left.

    virtual gates  U1 = V1 + a12 V2        a12 = -1 / slope(dot-1 lines)
                   U2 = a21 V1 + V2        a21 = -slope(dot-2 lines)

With the TRUE a12, a21 the dot-1 lines become exactly vertical and the
dot-2 lines exactly horizontal.  With a measured pair they keep a residual
tilt; the residual crosstalk is the off-diagonal of the lever-arm matrix
seen through the virtual gates, L A^-1, normalised by its diagonal:

    r12 = (L A^-1)_12 / (L A^-1)_11    (how much U2 still moves dot 1)
    r21 = (L A^-1)_21 / (L A^-1)_22    (how much U1 still moves dot 2)

Before virtualisation (A = 1) these are the true a12, a21.

Writes <run>/virtual_gates/: README.md (the section, every number written
by this program from the files it read), per_device.csv, summary.csv,
alpha_measured_vs_true.png, residual_crosstalk.png, overview.png and one
folder per test device with a before/after image for every method.
"""
import csv
import json
import os
import re

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from csdrecon.config import log
from csdrecon.ml import grid_train
from csdrecon.ml.ray_peaks import load_ground_truth
from csdrecon.study import slopes as S
from csdrecon.study.config import StudyConfig
from csdrecon.study.dataset import load_split
from run_12_slope_study import ARMS, COLOR, INK, MUTED

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "results", "8_rays_60_points_500_samples")
CELL = os.path.join(RUN, "8_rays_60_points_500_samples")
TRAIN_SEED = 3                       # the seed run_12 reported
OUT = os.path.join(RUN, "virtual_gates")

N_DEVICE_FOLDERS = 10                # test devices drawn one image per method
N_OVERVIEW = 6                       # rows of overview.png

# ══════════════════════════════════════════════════════════════════════════

NAME = {"network": "Network (8 rays)", "canny": "Canny (dense scan)",
        "truth": "True map (fit only)", "ray_steps": "Ray steps (no network)"}
WRONG = "#e34948"


# ── the physics ───────────────────────────────────────────────────────────

def alphas_from_angles(a1, a2):
    """(a12, a21) from the two fitted line angles in degrees, or None."""
    if a1 is None or a2 is None:
        return None
    return (S.cross_coupling("dot1", np.tan(np.radians(a1))),
            S.cross_coupling("dot2", np.tan(np.radians(a2))))


def vg_matrix(a12, a21):
    return np.array([[1.0, a12], [a21, 1.0]])


def residual_crosstalk(L2, alpha):
    """(r12, r21) left after virtualising with alpha; alpha None = no gates."""
    A = np.eye(2) if alpha is None else vg_matrix(*alpha)
    Lv = L2 @ np.linalg.inv(A)
    return float(Lv[0, 1] / Lv[0, 0]), float(Lv[1, 0] / Lv[1, 1])


def residual_tilt(L2, alpha):
    """Angles (deg) of the true dot-1 line from vertical and dot-2 from
    horizontal, in the virtual-gate frame."""
    A = vg_matrix(*alpha)
    d1 = A @ np.array([-L2[0, 1], L2[0, 0]])     # along a dot-1 line
    d2 = A @ np.array([-L2[1, 1], L2[1, 0]])     # along a dot-2 line
    t1 = np.degrees(np.arctan2(d1[0], d1[1]))
    t2 = np.degrees(np.arctan2(d2[1], d2[0]))
    wrap = lambda t: (t + 90.0) % 180.0 - 90.0   # noqa: E731
    return float(wrap(t1)), float(wrap(t2))


def to_virtual(gt, alpha, n=220):
    """The line map resampled on a (U1, U2) grid; NaN outside the window.

    Pixel units are voltage units here (square window, equal steps), so the
    matrix acts on (column, row) directly.
    """
    A = vg_matrix(*alpha)
    h, w = gt.shape
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    uc = corners @ A.T
    u1 = np.linspace(uc[:, 0].min(), uc[:, 0].max(), n)
    u2 = np.linspace(uc[:, 1].min(), uc[:, 1].max(), n)
    U1, U2 = np.meshgrid(u1, u2)
    V = np.linalg.solve(A, np.stack([U1.ravel(), U2.ravel()]))
    col, row = V[0].reshape(U1.shape), V[1].reshape(U1.shape)
    img = map_coordinates(gt.astype(float), [row, col], order=0, cval=np.nan)
    inside = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
    img[~inside] = np.nan
    return img, (u1[0], u1[-1], u2[0], u2[-1])


# ── the points each method used (the settings run_12 chose) ───────────────

def method_points(chosen, test_dirs, X, cfg):
    net, ck = grid_train.load(cfg.checkpoint)
    prob = grid_train.predict(net, X)
    out = {"network": [S.mask_points(p > float(ck["threshold"])) for p in prob],
           "truth": [S.mask_points(load_ground_truth(d)) for d in test_dirs]}
    m = re.match(r"sigma=([\d.]+) q=\(([\d.]+),([\d.]+)\) background=([\d.]+)px",
                 chosen["canny"]["knob"])
    s, lo, hi, bg = map(float, m.groups())
    from csdrecon.ml.ray_peaks import load_grid
    out["canny"] = [S.canny_points(load_grid(d)[2], s, lo, hi, bg)
                    for d in test_dirs]
    k = float(chosen["ray_steps"]["knob"].split("=")[1])
    out["ray_steps"] = [S.ray_step_points(d, cfg.n_rays, cfg.n_points, k)
                        for d in test_dirs]
    return out


# ── drawing ───────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def _dir_line(ax, ang, ctr, style, color, lw=1.2):
    u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    p = np.stack([ctr - 150 * u, ctr + 150 * u])
    ax.plot(p[:, 0], p[:, 1], style, color=color, lw=lw)


def draw_before(ax, gt, xy, arm, row, measured_txt):
    ax.imshow(gt, origin="lower", cmap="Greys", vmin=0, vmax=3,
              interpolation="nearest")
    if len(xy):
        ax.plot(xy[:, 0], xy[:, 1], ".", color=COLOR[arm],
                ms=5 if arm == "ray_steps" else 1.6)
    ctr = np.array([49.5, 49.5])
    for fam in S.FAMILIES:
        _dir_line(ax, row[f"{fam}_true_angle"], ctr, "-", INK, 1.0)
        if row[f"{arm}_{fam}_angle"] is not None:
            _dir_line(ax, row[f"{arm}_{fam}_angle"], ctr, "--", WRONG)
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 99)
    ax.set_title(f"before: physical gates P1, P2\n{measured_txt}",
                 fontsize=8, color=INK)
    _style(ax)


def draw_after(ax, gt, alpha, title, L2=None):
    if alpha is None:
        ax.text(0.5, 0.5, "no fit —\nno virtual gates", ha="center",
                va="center", transform=ax.transAxes, color=WRONG)
        ax.set_title(title, fontsize=8, color=INK)
        _style(ax)
        return
    img, ext = to_virtual(gt, alpha)
    cm = plt.get_cmap("Greys").copy()
    cm.set_bad("#f3f3f3")
    ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1.4,
              interpolation="nearest", aspect="auto")
    if L2 is not None:
        t1, t2 = residual_tilt(L2, alpha)
        ax.text(0.02, 0.02, f"dot-1 tilt {t1:+.2f}°\ndot-2 tilt {t2:+.2f}°",
                transform=ax.transAxes, fontsize=7, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_title(title, fontsize=8, color=INK)
    _style(ax)


# ── main ──────────────────────────────────────────────────────────────────

def _num(v):
    if v in ("", "None", None):
        return None
    if v in ("True", "False"):
        return v == "True"
    return float(v)


def main():
    from dataclasses import replace
    base = StudyConfig.load(CELL)
    cfg = replace(base, train_seed=TRAIN_SEED)
    cfg._dir = base._dir
    sdir = os.path.join(cfg.dir, f"slopes_seed{TRAIN_SEED}")
    with open(os.path.join(sdir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(sdir, "per_device.csv")) as f:
        rows = [{k: (v if k == "sample" else _num(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    X, _, test_dirs = load_split(cfg.test_npz)
    assert [os.path.basename(d) for d in test_dirs] == [r["sample"]
                                                        for r in rows]
    os.makedirs(OUT, exist_ok=True)
    pts = method_points(summary["chosen_on_validation"], test_dirs, X, cfg)
    n_pix = X.shape[2] * X.shape[3]
    coverage = X[:, 1].reshape(len(X), -1).mean(axis=1)

    # ── per device numbers ────────────────────────────────────────────────
    table, L2s = [], []
    for i, (d, r) in enumerate(zip(test_dirs, rows)):
        with open(os.path.join(d, "device.json")) as f:
            cap = json.load(f)["capacitance"]
        L2 = S.lever_arms(cap["Cdd"], cap["Cgd"])[:, :2]
        L2s.append(L2)
        true_alpha = (float(L2[0, 1] / L2[0, 0]), float(L2[1, 0] / L2[1, 1]))
        rec = {"sample": r["sample"], "rays_coverage_%": 100 * coverage[i],
               "true_a12": true_alpha[0], "true_a21": true_alpha[1]}
        b12, b21 = residual_crosstalk(L2, None)
        rec["before_r12"], rec["before_r21"] = b12, b21
        for arm in ARMS:
            al = alphas_from_angles(r[f"{arm}_dot1_angle"],
                                    r[f"{arm}_dot2_angle"])
            rec[f"{arm}_measured_%"] = {"network": 100 * coverage[i],
                                        "ray_steps": 100 * coverage[i],
                                        "canny": 100.0,
                                        "truth": float("nan")}[arm]
            rec[f"{arm}_points_fitted"] = len(pts[arm][i])
            if al is None:
                rec[f"{arm}_a12"] = rec[f"{arm}_a21"] = float("nan")
                rec[f"{arm}_r12"] = rec[f"{arm}_r21"] = float("nan")
            else:
                rec[f"{arm}_a12"], rec[f"{arm}_a21"] = al
                rec[f"{arm}_r12"], rec[f"{arm}_r21"] = residual_crosstalk(L2,
                                                                          al)
        table.append(rec)
    with open(os.path.join(OUT, "per_device.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    def col(key):
        return np.array([t[key] for t in table], float)

    def worst(arm):          # the larger residual of the two, per device
        return np.fmax(np.abs(col(f"{arm}_r12")), np.abs(col(f"{arm}_r21")))

    before = np.fmax(np.abs(col("before_r12")), np.abs(col("before_r21")))
    summ = []
    for arm in ARMS:
        r = worst(arm)
        ok = r[np.isfinite(r)]
        summ.append({
            "method": NAME[arm],
            "plane_measured_%": ("answer key" if arm == "truth" else
                                 f"{np.nanmean(col(f'{arm}_measured_%')):.2f}"),
            "measured_pixels": ("-" if arm == "truth" else
                                f"{np.nanmean(col(f'{arm}_measured_%')) * n_pix / 100:.0f}"),
            "points_fitted_mean": f"{col(f'{arm}_points_fitted').mean():.0f}",
            "no_virtual_gates": int((~np.isfinite(r)).sum()),
            "median_|a12_err|": f"{np.nanmedian(np.abs(col(f'{arm}_a12') - col('true_a12'))):.4f}",
            "median_|a21_err|": f"{np.nanmedian(np.abs(col(f'{arm}_a21') - col('true_a21'))):.4f}",
            "residual_crosstalk_median_%": f"{100 * np.median(ok):.2f}",
            "residual_crosstalk_p90_%": f"{100 * np.percentile(ok, 90):.2f}",
            "devices_below_1%": f"{np.mean(np.nan_to_num(r, nan=1) < 0.01):.2f}",
            "devices_below_5%": f"{np.mean(np.nan_to_num(r, nan=1) < 0.05):.2f}",
        })
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # ── figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, k, lab in zip(axes, ("a12", "a21"),
                          (r"$\alpha_{12}$ (P2 on dot 1)",
                           r"$\alpha_{21}$ (P1 on dot 2)")):
        t = col(f"true_{k}")
        lim = [0, max(0.75, np.nanmax(t) * 1.1)]
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        for arm in ARMS:
            ax.plot(t, col(f"{arm}_{k}"), "o", ms=4.5, mfc="none",
                    mec=COLOR[arm], mew=1.3, label=NAME[arm])
        ax.set_xlim(lim)
        ax.set_ylim(-0.2, lim[1] + 0.3)
        ax.set_xlabel(f"true {lab}")
        ax.set_ylabel(f"measured {lab}")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Virtual-gate coefficients: measured vs the capacitance "
                 "model, 50 held-out devices", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "alpha_measured_vs_true.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [("before (no virtual gates)", before, "#999999")] + \
             [(NAME[a], worst(a), COLOR[a]) for a in ARMS]
    for lab, r, c in series:
        n = len(r)
        e = np.sort(np.nan_to_num(r, nan=np.inf))
        ax.step(np.clip(100 * e, 1e-3, 1e3), np.arange(1, n + 1) / n,
                where="post", lw=2, color=c, label=lab)
    ax.axvline(1, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 200)
    ax.set_xlabel("residual crosstalk max(|r12|, |r21|)  (%)")
    ax.set_ylabel("fraction of held-out devices")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residual_crosstalk.png"), dpi=200)
    plt.close(fig)

    # overview: samples x (true gates + four methods), after panels only
    n = min(N_OVERVIEW, len(rows))
    fig, axes = plt.subplots(n, 6, figsize=(15, 2.6 * n))
    for i in range(n):
        gt = load_ground_truth(test_dirs[i])
        t = table[i]
        axes[i, 0].imshow(gt, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
                          interpolation="nearest")
        axes[i, 0].set_title("before: P1, P2" if i == 0 else "", fontsize=8)
        axes[i, 0].set_ylabel(t["sample"], fontsize=8, color=MUTED)
        _style(axes[i, 0])
        draw_after(axes[i, 1], gt, (t["true_a12"], t["true_a21"]),
                   "after: TRUE virtual gates" if i == 0 else "", L2s[i])
        for j, arm in enumerate(ARMS, 2):
            al = (None if not np.isfinite(t[f"{arm}_a12"])
                  else (t[f"{arm}_a12"], t[f"{arm}_a21"]))
            draw_after(axes[i, j], gt, al,
                       f"after: {NAME[arm]}" if i == 0 else "", L2s[i])
        axes[i, 0].set_ylabel(t["sample"], fontsize=8, color=MUTED)
    fig.suptitle("True line map in each method's virtual gates — right is "
                 "lines exactly vertical and horizontal", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "overview.png"), dpi=130)
    plt.close(fig)

    # one folder per device, one image per method
    for i in range(min(N_DEVICE_FOLDERS, len(rows))):
        t, r = table[i], rows[i]
        gt = load_ground_truth(test_dirs[i])
        ddir = os.path.join(OUT, t["sample"])
        os.makedirs(ddir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
        axes[0].imshow(gt, origin="lower", cmap="Greys", vmin=0, vmax=1.4,
                       interpolation="nearest")
        axes[0].set_title("before: physical gates P1, P2", fontsize=9)
        _style(axes[0])
        draw_after(axes[1], gt, (t["true_a12"], t["true_a21"]),
                   f"after: true virtual gates\n"
                   f"a12 = {t['true_a12']:.4f}, a21 = {t['true_a21']:.4f}",
                   L2s[i])
        fig.suptitle(f"{t['sample']} — reference (capacitance model)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(ddir, "0_true_virtual_gates.png"), dpi=150)
        plt.close(fig)
        for k, arm in enumerate(ARMS, 1):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
            meas = ("answer key, not a measurement" if arm == "truth" else
                    f"measured {t[f'{arm}_measured_%']:.2f} % of the plane, "
                    f"{t[f'{arm}_points_fitted']} points fitted")
            draw_before(axes[0], gt, pts[arm][i], arm, r, meas)
            al = (None if not np.isfinite(t[f"{arm}_a12"])
                  else (t[f"{arm}_a12"], t[f"{arm}_a21"]))
            ttl = "after: virtual gates from these slopes"
            if al is not None:
                ttl += (f"\na12 {al[0]:.4f} (true {t['true_a12']:.4f}), "
                        f"a21 {al[1]:.4f} (true {t['true_a21']:.4f})")
            draw_after(axes[1], gt, al, ttl, L2s[i])
            res = ("" if al is None else
                   f"   residual crosstalk {100 * abs(t[f'{arm}_r12']):.2f} % / "
                   f"{100 * abs(t[f'{arm}_r21']):.2f} %  (before "
                   f"{100 * abs(t['before_r12']):.1f} % / "
                   f"{100 * abs(t['before_r21']):.1f} %)")
            fig.suptitle(f"{t['sample']} — {NAME[arm]}{res}", fontsize=9)
            fig.text(0.5, 0.005, "solid black: true slope   dashed red: "
                     "fitted   coloured dots: what the fit was given",
                     ha="center", fontsize=7, color=MUTED)
            fig.tight_layout(rect=[0, 0.03, 1, 1])
            fig.savefig(os.path.join(ddir, f"{k}_{arm}.png"), dpi=150)
            plt.close(fig)

    write_readme(summ, summary, n_pix, before)
    log.say(f"wrote {os.path.abspath(OUT)}")


def write_readme(summ, summary, n_pix, before):
    ch = summary["chosen_on_validation"]
    head = ["method", "plane measured (%)", "pixels measured",
            "points fitted", "no virtual gates", "median error a12",
            "median error a21", "residual crosstalk, median (%)",
            "residual crosstalk, p90 (%)", "devices < 1 %", "devices < 5 %"]
    keys = list(summ[0])
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(s[k]) for k in keys) + " |" for s in summ]
    devs = sorted(d for d in os.listdir(OUT)
                  if os.path.isdir(os.path.join(OUT, d)))
    text = f"""# Virtual gates from the measured slopes

Written by `scripts/run_13_virtual_gates.py` from
`{os.path.basename(CELL)}/slopes_seed{TRAIN_SEED}/` (run_12). Every number
below is computed by that program; {summary['n_test']} held-out test devices,
network train seed {TRAIN_SEED}, knobs chosen on {summary['n_validation']}
validation devices.

## The question

A virtual-gate matrix needs the two cross-couplings

    a12 = -1 / slope(dot-1 lines)    a21 = -slope(dot-2 lines)
    U1 = P1 + a12 P2,   U2 = a21 P1 + P2

In the virtual gates (U1, U2) the dot-1 lines are vertical and the dot-2
lines horizontal, so each dot is driven by its own gate only. The true
a12, a21 come from the capacitance matrices of each simulated device.
What is left after virtualising with a *measured* pair is the residual
crosstalk, r12 = (L A⁻¹)₁₂/(L A⁻¹)₁₁ and r21 = (L A⁻¹)₂₁/(L A⁻¹)₂₂.
Before any virtual gates it equals the true a12, a21.

## How much was measured, and how good the gates are

The window is {int(np.sqrt(n_pix))} × {int(np.sqrt(n_pix))} pixels. Residual crosstalk per device is the worse
of |r12| and |r21|. A method that returns no slope for a family gives no
virtual gates; that counts as a miss in the "devices <" columns.
Without any virtual gates the crosstalk is the true a12, a21 themselves:
median {100 * np.median(before):.1f} %, 90th percentile
{100 * np.percentile(before, 90):.1f} %.

{chr(10).join(lines)}

Settings used (chosen on the validation devices, never the test ones):
network threshold {summary['network_threshold']}, Hough {ch['network']['hough']};
Canny {ch['canny']['knob']}, Hough {ch['canny']['hough']};
ray steps {ch['ray_steps']['knob']}, Hough {ch['ray_steps']['hough']};
true map Hough {ch['truth']['hough']}.

## Figures

* `alpha_measured_vs_true.png` — measured against true a12 and a21, every
  device, every method. On the diagonal is right.
* `residual_crosstalk.png` — what is left after virtualising, per device,
  with the "before" curve (no virtual gates) for scale.
* `overview.png` — the true line map of {N_OVERVIEW} devices in physical gates and in
  the virtual gates of each method. Exactly vertical and horizontal lines
  mean the gates are right; the tilt left is printed in each panel.
* one folder per device ({', '.join(devs)}):
  `0_true_virtual_gates.png` is the reference, then one before/after image
  per method — left, the points that method gave the fit and its fitted
  slopes over the true map; right, the true map in the virtual gates built
  from those slopes.

The "after" panels show the simulator's true line map, re-drawn in each
method's virtual gates, so that the only thing that differs between
panels is the virtual-gate matrix.
"""
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
