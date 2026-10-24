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
        f.write("\begin{tabular}{l" + "c" * (len(tex_head) - 1) + "}\n\hline\n")
        f.write(" & ".join(tex_head) + r" \ \hline" + "\n")
        for l in lines[1:]:
            cells = [c.replace("±", r"$\pm$").replace("×", r"$\times$")
                     .replace("%", r"\%") for c in l]
            f.write(" & ".join(cells) + r" \\" + "\n")
        f.write("\hline\n\end{tabular}\n")
