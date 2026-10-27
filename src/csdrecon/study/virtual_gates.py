"""
The virtual-gate matrix of a double dot, three ways.

    G_true   from the capacitances in device.json
    G_full   from the full CSD's binary line map Y
    G_unet   from the U-Net's binary map (prob > the checkpoint's threshold)

G_full and G_unet come out of ONE function, slopes_from_lines, applied to
both maps in exactly the same way.  It has no tuned parameter, so it works
for any number of honeycomb cells in the window:

    thin -> cut at crossings -> one direction per straight piece
         -> three families from the sign of the slope + Otsu on the angles
         -> length-weighted median angle per family

The three slopes (dot-1, dot-2, interdot) give

    G = [[1,        M12],
         [rho*M21,  rho]]          = alpha / alpha11

with alpha the 2x2 lever-arm matrix.  M12 and M21 need only the two dot
slopes; rho needs the interdot slope, which is often too short to find.
"""
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

FAMILIES = ("dot1", "dot2", "interdot")
MIN_PIECE_PX = 3        # a direction needs at least three pixels


# ── the truth ─────────────────────────────────────────────────────────────

def lever_arms(cap: Dict) -> np.ndarray:
    """alpha = inv(C) @ Cgd[:, :2] with C the dots' total-capacitance matrix."""
    Cdd = np.asarray(cap["Cdd"], float)
    Cgd = np.asarray(cap["Cgd"], float)
    d12 = Cdd[0, 1]
    C1 = Cdd[0, 0] + d12 + Cgd[0].sum()
    C2 = Cdd[1, 1] + d12 + Cgd[1].sum()
    C = np.array([[C1, -d12], [-d12, C2]])
    return np.linalg.inv(C) @ Cgd[:, :2]


def true_slopes(cap: Dict) -> Dict[str, float]:
    """dV_P2/dV_P1 of the three line families."""
    a = lever_arms(cap)
    return {"dot1": float(-a[0, 0] / a[0, 1]),
            "dot2": float(-a[1, 0] / a[1, 1]),
            "interdot": float(-(a[0, 0] - a[1, 0]) / (a[0, 1] - a[1, 1]))}


def gates_from_slopes(s1: Optional[float], s2: Optional[float],
                      s_int: Optional[float]) -> Dict[str, float]:
    """M12, M21, rho and G from the three slopes (nan where a slope is missing)."""
    nan = float("nan")
    M12 = -1.0 / s1 if s1 is not None and np.isfinite(s1) else nan
    M21 = -s2 if s2 is not None and np.isfinite(s2) else nan
    if all(v is not None and np.isfinite(v) for v in (s1, s2, s_int)):
        rho = -(s_int - s1) / (s1 * (s_int - s2))
    else:
        rho = nan
    G = np.array([[1.0, M12], [rho * M21, rho]])
    return {"M12": float(M12), "M21": float(M21), "rho": float(rho), "G": G}


def true_gates(cap: Dict) -> Dict:
    """Slopes, angles, M12, M21, rho and G from the capacitances."""
    s = true_slopes(cap)
    out = gates_from_slopes(s["dot1"], s["dot2"], s["interdot"])
    out["slopes"] = s
    out["angles"] = {f: slope_to_theta(v) for f, v in s.items()}
    return out


def slope_to_theta(s: float) -> float:
    """Line direction in degrees in [0, 180) from a slope dV2/dV1."""
    return float(np.degrees(np.arctan(s)) % 180.0)


# ── the fit ───────────────────────────────────────────────────────────────

_RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def _otsu_split(theta: np.ndarray, w: np.ndarray) -> float:
    """Length-weighted Otsu threshold on 1-D angles: the cut between two
    sorted neighbours that maximises the between-class variance."""
    o = np.argsort(theta)
    t, w = theta[o], w[o]
    best, cut = -1.0, 0.5 * (t[0] + t[-1])
    W = w.sum()
    for k in range(1, len(t)):
        if t[k] == t[k - 1]:
            continue
        w0, w1 = w[:k].sum(), w[k:].sum()
        m0 = (t[:k] * w[:k]).sum() / w0
        m1 = (t[k:] * w[k:]).sum() / w1
        var = w0 * w1 / W ** 2 * (m0 - m1) ** 2
        if var > best:
            best, cut = var, 0.5 * (t[k - 1] + t[k])
    return cut


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, cw = v[o], np.cumsum(w[o])
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def slopes_from_lines(binary_map: np.ndarray, dx: float, dy: float) -> Dict:
    """
    Three line slopes from a binary line map.

    binary_map is (H, W), row = V_P2 with row 0 lowest, column = V_P1; dx, dy
    are the pixel size in mV.  Returns the three angles (degrees, [0, 180))
    and slopes, M12, M21, rho, the number of pieces per family, the pieces
    themselves (pixel (row, col) arrays with their angle and family) and a
    `failed` flag per family.
    """
    skel = skeletonize(np.asarray(binary_map) > 0)
    nb = ndimage.convolve(skel.astype(int), _RING, mode="constant")
    cut = skel & (nb < 3)              # remove crossings and triple points
    lab, n = ndimage.label(cut, structure=np.ones((3, 3)))

    pieces: List[Dict] = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        r, c = np.nonzero(lab[sl] == k)
        r, c = r + sl[0].start, c + sl[1].start
        if len(r) < MIN_PIECE_PX:
            continue
        x = c * dx - (c * dx).mean()
        y = r * dy - (r * dy).mean()
        th = 0.5 * np.degrees(np.arctan2(2 * (x * y).sum(),
                                         (x * x).sum() - (y * y).sum()))
        pieces.append({"rows": r, "cols": c, "theta": float(th % 180.0),
                       "length": int(len(r)), "family": None})

    # three families: positive slope -> interdot; negative -> Otsu in two
    neg = [p for p in pieces if 90.0 < p["theta"] < 180.0]
    for p in pieces:
        if 0.0 < p["theta"] < 90.0:
            p["family"] = "interdot"
    if len(neg) >= 2:
        th = np.array([p["theta"] for p in neg])
        w = np.array([p["length"] for p in neg], float)
        split = _otsu_split(th, w)
        for p in neg:
            # steeper (closer to 90 degrees) is dot 1
            p["family"] = "dot1" if p["theta"] < split else "dot2"
    elif len(neg) == 1:
        # one piece cannot be split; the diagonal decides which dot it is
        neg[0]["family"] = "dot1" if neg[0]["theta"] < 135.0 else "dot2"

    out = {"pieces": pieces, "angles": {}, "slopes": {}, "n_pieces": {},
           "failed": {}}
    for f in FAMILIES:
        fp = [p for p in pieces if p["family"] == f]
        out["n_pieces"][f] = len(fp)
        out["failed"][f] = not fp
        if fp:
            a = _weighted_median(np.array([p["theta"] for p in fp]),
                                 np.array([p["length"] for p in fp], float))
            out["angles"][f] = a
            out["slopes"][f] = float(np.tan(np.radians(a)))
        else:
            out["angles"][f] = float("nan")
            out["slopes"][f] = float("nan")
    g = gates_from_slopes(out["slopes"]["dot1"], out["slopes"]["dot2"],
                          out["slopes"]["interdot"])
    out.update(g)
    return out


def pixel_size(voltage_window: Sequence[float], shape) -> tuple:
    """(dx, dy) in mV from [x_min, x_max, y_min, y_max] and the map shape."""
    x0, x1, y0, y1 = voltage_window
    H, W = shape
    return (x1 - x0) / (W - 1), (y1 - y0) / (H - 1)


def angle_error(fit: float, true: float) -> float:
    """fit - true in degrees, wrapped to (-90, 90]; nan if no fit."""
    if not np.isfinite(fit):
        return float("nan")
    d = (fit - true) % 180.0
    return float(d - 180.0 if d > 90.0 else d)


# ── one device, one method ────────────────────────────────────────────────

ERRORS = ("dtheta1", "dtheta2", "dtheta_int", "dM12", "dM21", "drho",
          "rel_M12", "rel_M21", "rel_rho")


def device_row(fit: Dict, truth: Dict) -> Dict:
    """Every per-device number of one fit against G_true."""
    row = {}
    for f, key in zip(FAMILIES, ("1", "2", "_int")):
        row[f"theta{key}_fit"] = fit["angles"][f]
        row[f"theta{key}_true"] = truth["angles"][f]
        row[f"dtheta{key}"] = angle_error(fit["angles"][f], truth["angles"][f])
        row[f"n_pieces_{f}"] = fit["n_pieces"][f]
        row[f"failed_{f}"] = int(fit["failed"][f])
    for m in ("M12", "M21", "rho"):
        row[f"{m}_fit"], row[f"{m}_true"] = fit[m], truth[m]
        row[f"d{m}"] = fit[m] - truth[m]
        row[f"rel_{m}"] = 100.0 * (fit[m] - truth[m]) / truth[m]
    return row


# ── statistics ────────────────────────────────────────────────────────────

def stat(values: Sequence[float]) -> Dict:
    """mean, std, median |.| and n over the finite values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": np.nan, "std": np.nan, "med_abs": np.nan, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "med_abs": float(np.median(np.abs(v))), "n": int(len(v))}


def summary_row(rows: Sequence[Dict]) -> Dict:
    """One table row from the per-device rows of one method."""
    out = {k: stat([r[k] for r in rows]) for k in ERRORS}
    out["n_devices"] = len(rows)
    out["coverage_%"] = 100.0 * float(np.mean([r["coverage"] for r in rows]))
    out["interdot_found"] = int(sum(1 - r["failed_interdot"] for r in rows))
    out["dot_failed"] = int(sum(max(r["failed_dot1"], r["failed_dot2"])
                                for r in rows))
    return out


def fmt(s: Dict, digits: int = 2) -> str:
    """mean ± std [median |.|]."""
    if not s["n"]:
        return "—"
    return (f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f} "
            f"[{s['med_abs']:.{digits}f}]")


TABLE_COLS = [("dθ1 (°)", "dtheta1"), ("dθ2 (°)", "dtheta2"),
              ("dθ_int (°)", "dtheta_int"), ("M12 rel. err. (%)", "rel_M12"),
              ("M21 rel. err. (%)", "rel_M21"), ("rho rel. err. (%)", "rel_rho")]


def table_lines(summaries: Sequence[Dict]) -> List[List[str]]:
    """Header + one list of cell strings per method."""
    head = (["method", "rays × points", "coverage %"]
            + [c for c, _ in TABLE_COLS] + ["interdot found", "failed"])
    body = []
    for s in summaries:
        body.append([s["method"], s["budget"], f"{s['coverage_%']:.2f}"]
                    + [fmt(s[k]) for _, k in TABLE_COLS]
                    + [f"{s['interdot_found']}/{s['n_devices']}",
                       f"{s['dot_failed']}/{s['n_devices']}"])
    return [head] + body


def write_tables(summaries: Sequence[Dict], out_base: str, caption: str):
    """out_base.md, .csv and .tex from the same table."""
    import csv
    lines = table_lines(summaries)
    with open(out_base + ".md", "w", encoding="utf-8") as f:
        f.write(caption + "\n\n")
        f.write("| " + " | ".join(lines[0]) + " |\n")
        f.write("|" + "---|" * len(lines[0]) + "\n")
        for l in lines[1:]:
            f.write("| " + " | ".join(l) + " |\n")
        f.write("\nEach error cell: mean ± std over the devices where that "
                "family was found, median |error| in brackets.  'failed' = "
                "devices where the dot-1 or the dot-2 family gave no piece.  "
                "rho is computed only on devices where the interdot family "
                "was found.\n")
    with open(out_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rays", "points", "coverage_%", "n_devices"]
                   + [f"{k}_{q}" for _, k in TABLE_COLS
                      for q in ("mean", "std", "med_abs", "n")]
                   + ["interdot_found", "dot_failed"])
        for s in summaries:
            w.writerow([s["method"], s["rays"], s["points"],
                        f"{s['coverage_%']:.4f}", s["n_devices"]]
                       + [s[k][q] for _, k in TABLE_COLS
                          for q in ("mean", "std", "med_abs", "n")]
                       + [s["interdot_found"], s["dot_failed"]])
    tex_head = ["method", r"rays $\times$ points", r"coverage (\%)",
                r"$\delta\theta_1$ ($^\circ$)", r"$\delta\theta_2$ ($^\circ$)",
                r"$\delta\theta_\mathrm{int}$ ($^\circ$)",
                r"$M_{12}$ (\%)", r"$M_{21}$ (\%)", r"$\rho$ (\%)",
                "interdot", "failed"]
    with open(out_base + ".tex", "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write(r"\begin{tabular}{l" + "c" * (len(tex_head) - 1) + "}\n"
                + r"\hline" + "\n")
        f.write(" & ".join(tex_head) + r" \\ \hline" + "\n")
        for l in lines[1:]:
            cells = [c.replace("±", r"$\pm$").replace("×", r"$\times$")
                     .replace("%", r"\%") for c in l]
            f.write(" & ".join(cells) + r" \\" + "\n")
        f.write(r"\hline" + "\n" + r"\end{tabular}" + "\n")


# ── figures ───────────────────────────────────────────────────────────────

INK, MUTED = "#111111", "#555555"
FAMILY_COLOR = {"dot1": "#7b3fbf", "dot2": "#eb6834", "interdot": "#1baf7a"}
UNASSIGNED = "#bbbbbb"     # pieces at exactly 0 or 90 degrees: no slope sign
FAMILY_LABEL = {"dot1": "dot 1", "dot2": "dot 2", "interdot": "interdot"}
FULL_COLOR, UNET_COLOR = "#111111", "#2a78d6"
EXTRA_COLOR, MISSED_COLOR = "#d62728", "#1f5fd6"


def _read_rows(path):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("split", "method", "device"):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = np.nan
    return rows


def _key(r):
    return (r["method"], r["rays"], r["points"])


def _methods(rows):
    keys = []
    for r in rows:
        if _key(r) not in keys:
            keys.append(_key(r))
    return keys


def _is_cell(r, cell):
    return (r["method"] == "U-Net" and np.isfinite(r["rays"]) and
            cell.startswith(f"{int(r['rays'])}_rays_{int(r['points'])}_points"))


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)


def _gtxt(g):
    """M12, M21, rho of one fit or truth; a dash where a family is missing."""
    v = [f"{g[k]:.3f}" if np.isfinite(g[k]) else "–"
         for k in ("M12", "M21", "rho")]
    return f"$M_{{12}}$ = {v[0]}, $M_{{21}}$ = {v[1]}, $\\rho$ = {v[2]}"


def _pieces_panel(ax, fit, truth, ext, dx, dy):
    """Fitted pieces coloured by family, the true slope dashed through each."""
    x0, y0 = ext[0], ext[2]
    for p in fit["pieces"]:
        c = FAMILY_COLOR.get(p["family"], UNASSIGNED)
        ax.plot(x0 + p["cols"] * dx, y0 + p["rows"] * dy, "s", ms=0.9,
                mec="none", color=c)
    for f in FAMILIES:
        fp = [p for p in fit["pieces"] if p["family"] == f]
        if not fp:
            continue
        # through the longest piece of the family
        p = max(fp, key=lambda p: p["length"])
        mx, my = x0 + p["cols"].mean() * dx, y0 + p["rows"].mean() * dy
        s = truth["slopes"][f]
        u = np.array([1.0, s]) / np.hypot(1.0, s)
        t = np.linspace(-3, 3, 50)
        ax.plot(mx + t * u[0], my + t * u[1], "--", lw=0.6,
                color=FAMILY_COLOR[f])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])


def fig_examples(out_dir, rows, maps, cell, label):
    """Fig. 1: good / typical / bad test device by the U-Net's M12, M21 error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ur = [r for r in rows if r["split"] == "test" and _is_cell(r, cell)]
    err = np.array([0.5 * (abs(r["rel_M12"]) + abs(r["rel_M21"]))
                    for r in ur])
    ok = np.where(np.isfinite(err))[0]
    o = ok[np.argsort(err[ok])]
    pick = [("good", o[0]), ("typical", o[len(o) // 2]), ("bad", o[-1])]
    Y, P = maps["Y"], maps[cell]
    samples = [str(s) for s in maps["samples"]]
    names = [os.path.basename(s) for s in samples]

    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(6, 5, height_ratios=[1, 0.16] * 3, hspace=0.42,
                          wspace=0.10, left=0.085, right=0.995, top=0.93,
                          bottom=0.02)
    bw = ListedColormap(["white", INK])
    ov = ListedColormap(["white", INK, EXTRA_COLOR, MISSED_COLOR])
    titles = ["(a) full CSD", f"(b) U-Net {label}", "(c) overlay",
              "(d) fit on full CSD", "(e) fit on U-Net"]
    for k, (lab, j) in enumerate(pick):
        i = names.index(ur[j]["device"])
        with open(os.path.join(samples[i], "device.json")) as f:
            d = json.load(f)
        ext = d["voltage_window"]
        dx, dy = pixel_size(ext, Y[i].shape)
        truth = true_gates(d["capacitance"])
        ff = slopes_from_lines(Y[i], dx, dy)
        fu = slopes_from_lines(P[i], dx, dy)
        y, p = Y[i], P[i]
        o3 = np.zeros(y.shape, int)
        o3[y & p], o3[p & ~y], o3[y & ~p] = 1, 2, 3
        axs = [fig.add_subplot(gs[2 * k, c]) for c in range(5)]
        kw = dict(origin="lower", extent=ext, interpolation="nearest",
                  aspect="equal")
        axs[0].imshow(y, cmap=bw, vmin=0, vmax=1, **kw)
        axs[1].imshow(p, cmap=bw, vmin=0, vmax=1, **kw)
        axs[2].imshow(o3, cmap=ov, vmin=0, vmax=3, **kw)
        _pieces_panel(axs[3], ff, truth, ext, dx, dy)
        _pieces_panel(axs[4], fu, truth, ext, dx, dy)
        for c, a in enumerate(axs):
            a.set_aspect("equal")
            a.tick_params(labelsize=5, length=2, pad=1)
            if c:
                a.set_yticklabels([])
            a.set_xlabel("$V_\\mathrm{P1}$ (mV)", fontsize=5.5, labelpad=1)
            if k == 0:
                a.set_title(titles[c], fontsize=7)
        axs[0].set_ylabel(f"{lab} ({names[i]})\n$V_\\mathrm{{P2}}$ (mV)",
                          fontsize=6)
        ta = fig.add_subplot(gs[2 * k + 1, :])
        ta.axis("off")
        ta.text(0.5, 0.55,
                f"$G_\\mathrm{{true}}$: {_gtxt(truth)}        "
                f"$G_\\mathrm{{full}}$: {_gtxt(ff)}        "
                f"$G_\\mathrm{{U-Net}}$: {_gtxt(fu)}",
                fontsize=6.5, transform=ta.transAxes, ha="center",
                va="center")
    h = ([Patch(color=INK, label="matched"),
          Patch(color=EXTRA_COLOR, label="extra"),
          Patch(color=MISSED_COLOR, label="missed")]
         + [Line2D([], [], color=FAMILY_COLOR[f], marker="s", ls="none",
                   ms=3, label=FAMILY_LABEL[f]) for f in FAMILIES]
         + [Line2D([], [], color=UNASSIGNED, marker="s", ls="none", ms=3,
                   label="0° or 90° (no family)")]
         + [Line2D([], [], color=MUTED, ls="--", lw=0.8,
                   label="true slope")])
    fig.legend(handles=h, loc="upper center", ncol=8, fontsize=6,
               frameon=False, bbox_to_anchor=(0.54, 1.0))
    _save(fig, out_dir, "fig1_examples")
    plt.close(fig)
    return [(lab, ur[j]["device"], float(err[j])) for lab, j in pick]


def fig_parity(out_dir, rows, cell, label):
    """Fig. 2: fitted vs true M12, M21, rho; full CSD and one U-Net."""
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["split"] == "test"]
    full = [r for r in rs if r["method"] == "full CSD"]
    un = [r for r in rs if _is_cell(r, cell)]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    lab = {"M12": "$M_{12}$", "M21": "$M_{21}$", "rho": r"$\rho$"}
    for c, m in enumerate(("M12", "M21", "rho")):
        tv = np.array([r[f"{m}_true"] for r in full])
        lo, hi = tv.min(), tv.max()
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        for k, (rr, name, col) in enumerate(
                ((full, "full CSD", FULL_COLOR),
                 (un, f"U-Net {label}", UNET_COLOR))):
            a = ax[k, c]
            t = np.array([r[f"{m}_true"] for r in rr])
            f = np.array([r[f"{m}_fit"] for r in rr])
            ok = np.isfinite(f)
            a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, zorder=0)
            a.plot(t[ok], np.clip(f[ok], lo, hi), "o", ms=3, mfc="none",
                   mec=col, mew=0.7)
            n_off = int(((f[ok] < lo) | (f[ok] > hi)).sum())
            a.set_xlim(lo, hi)
            a.set_ylim(lo, hi)
            a.set_aspect("equal")
            s = stat(f[ok] - t[ok])
            txt = (f"{name}\nfit − true: {s['mean']:+.3f} ± {s['std']:.3f}\n"
                   f"n = {ok.sum()} of {len(f)}")
            if n_off:
                txt += f"; {n_off} clipped to the frame"
            a.text(0.03, 0.97, txt, transform=a.transAxes, va="top",
                   fontsize=6.3)
            a.set_xlabel(f"true {lab[m]}", fontsize=8)
            a.set_ylabel(f"fitted {lab[m]}", fontsize=8)
            a.tick_params(labelsize=7)
    _save(fig, out_dir, "fig2_parity")
    plt.close(fig)


def _label(key):
    return "full\nCSD" if key[0] == "full CSD" else \
        f"{int(key[1])}×{int(key[2])}"


def fig_angle_errors(out_dir, rows):
    """Fig. 3: distributions of dtheta1, dtheta2, dtheta_int per method."""
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["split"] == "test"]
    keys = _methods(rs)
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), constrained_layout=True,
                           sharex=True)
    names = {"dtheta1": r"$\delta\theta_1$ (°)",
             "dtheta2": r"$\delta\theta_2$ (°)",
             "dtheta_int": r"$\delta\theta_\mathrm{int}$ (°)"}
    for a, k in zip(ax, names):
        data = [np.array([r[k] for r in rs if _key(r) == key]) for key in keys]
        n_all = [len(d) for d in data]
        data = [d[np.isfinite(d)] for d in data]
        allv = np.concatenate(data)
        lim = float(np.ceil(max(np.percentile(np.abs(allv), 95), 1.0)))
        for x, d in enumerate(data):
            col = FULL_COLOR if keys[x][0] == "full CSD" else UNET_COLOR
            if len(d) >= 2 and np.ptp(d) > 0:
                v = a.violinplot(np.clip(d, -lim, lim), [x], widths=0.8,
                                 showextrema=False)
                for b in v["bodies"]:
                    b.set_facecolor(col)
                    b.set_edgecolor("none")
                    b.set_alpha(0.35)
            if len(d):
                a.plot([x - 0.28, x + 0.28], [np.median(d)] * 2, color=col,
                       lw=1.2)
            n_out = int((np.abs(d) > lim).sum())
            a.text(x, lim * 1.04, f"{len(d)}/{n_all[x]}"
                   + (f"\n{n_out} off" if n_out else ""),
                   ha="center", va="bottom", fontsize=5, color=MUTED)
        a.axhline(0, color=MUTED, lw=0.6, zorder=0)
        a.set_ylim(-lim, lim * 1.45)
        a.set_ylabel(names[k], fontsize=8)
        a.tick_params(labelsize=7)
    ax[-1].set_xticks(range(len(keys)))
    ax[-1].set_xticklabels([_label(k) for k in keys], fontsize=6.5)
    ax[-1].set_xlabel("method (U-Net: rays × points per ray, by coverage)",
                      fontsize=8)
    ax[0].set_title("bar: median; numbers: devices with that family found / "
                    "all; 'off': outside the axis (clipped)", fontsize=6.5,
                    color=MUTED)
    _save(fig, out_dir, "fig3_angle_errors")
    plt.close(fig)


def fig_vs_coverage(out_dir, rows):
    """Fig. 4: median relative error of M12 and M21 vs coverage."""
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["split"] == "test"]
    keys = _methods(rs)
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for m, col, mk in (("M12", "#2a78d6", "o"), ("M21", "#eb6834", "s")):
        cov, med, full = [], [], None
        for key in keys:
            rr = [r for r in rs if _key(r) == key]
            e = stat([r[f"rel_{m}"] for r in rr])["med_abs"]
            c = 100 * float(np.mean([r["coverage"] for r in rr]))
            if key[0] == "full CSD":
                full = (c, e)
            else:
                cov.append(c)
                med.append(e)
        o = np.argsort(cov)
        ax.plot(np.array(cov)[o], np.array(med)[o], mk, color=col, ms=4,
                label=f"$M_{{{m[1:]}}}$, U-Net")
        ax.plot(*full, mk, color=col, mfc="white", ms=6, mew=1.2,
                label=f"$M_{{{m[1:]}}}$, full CSD")
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("coverage of the grid (%)", fontsize=8)
    ax.set_ylabel("median |relative error| (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, frameon=False)
    _save(fig, out_dir, "fig4_error_vs_coverage")
    plt.close(fig)


def make_figures(run_dir: str, out_dir: str, cell: str) -> Dict:
    """All four figures from per_device.csv and the cached test maps."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows = _read_rows(os.path.join(out_dir, "per_device.csv"))
    maps = dict(np.load(os.path.join(out_dir, "_maps_test.npz")))
    parts = cell.split("_")
    label = f"{parts[0]}×{parts[2]}"
    picks = fig_examples(fig_dir, rows, maps, cell, label)
    fig_parity(fig_dir, rows, cell, label)
    fig_angle_errors(fig_dir, rows)
    fig_vs_coverage(fig_dir, rows)
    return {"examples": picks}
