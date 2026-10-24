"""
slopes.py — can the two loading-line slopes be measured from the rays?

To build virtual gates an experimenter needs the slope of each family of
loading lines in the stability diagram.  The simulator knows the
capacitances, so the true slopes are exact:

    lever arms   L = Cdd_maxwell^-1 @ Cgd      (2 x n_gates, qarray's own
                                                Maxwell conversion)
    dot-i lines  L[i, 0] dV1 + L[i, 1] dV2 = 0
                 slope dV2/dV1 = -L[i, 0] / L[i, 1]

dot-1 lines are the steep family, dot-2 lines the shallow one.  The same
fitting method is then run on four point sets per device and its angles are
compared with the true ones:

    network   pixels the U-Net calls a line (stored threshold) — the method
    canny     Canny edges of the FULL dense scan — the conventional route,
              100 % of the plane measured
    truth     the true line map — the best case: the error the fit alone adds
    ray_steps the steps found on each ray, no network — is the network needed?

The fit, identical in all four cases.  The point set is drawn onto the
pixel grid and a progressive probabilistic Hough transform finds straight
SEGMENTS in it; each segment's angle is then refined by a total-least-squares
fit to the points within one pixel of it.  Segments, not whole lines: when
the dots are strongly coupled the honeycomb breaks every loading line into
short pieces offset at each anticrossing, and a fit that treats a family as
long parallel lines measures the zig-zag of the chain instead of the slope
(tried first; median error ~3 degrees on the true maps).  Segments with a
negative slope are lead transitions (interdot ones slope up); they are split
into a steep and a shallow group by a length-weighted Otsu split of their
angles, and each family's angle is the length-weighted median of its group.
If only one group exists it is assigned by the -45 degree diagonal.  The true
slopes are never used to sort the segments.

Nothing here is tuned on the test devices.  The two baselines have knobs
(Canny's smoothing and thresholds, the step detector's threshold); they are
chosen on training devices by the same angle error that is reported, which
favours the baselines, not the network.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from ..ml.ray_peaks import (fan_angles, load_grid, load_ground_truth,
                            ray_polyline, voltage_to_pixel)

FAMILIES = ("dot1", "dot2")                 # steep, shallow
THETA = np.linspace(-np.pi / 2, np.pi / 2, 1800, endpoint=False)  # 0.1 deg
REFINE_PX = 1.0             # points within this of a segment refine its angle
ONE_FAMILY_DEG = 8.0        # two groups closer than this are one family
TRIM_DEG = 5.0      # segments this far from a family's median leave its fit
JOINT = True        # family angle = one shared TLS direction, not a median
MIN_FAMILY_PX = 15          # a family with fewer true line pixels is absent

# The Hough knobs every arm is scanned over on the validation devices.
HOUGH_GRID = [dict(threshold=t, line_length=l, line_gap=g)
              for t in (2, 5, 10) for l in (5, 10, 20) for g in (1, 3, 10, 30)]


# ── the truth ─────────────────────────────────────────────────────────────

def maxwell(Cdd: np.ndarray, Cgd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(Cdd_maxwell, -Cgd) exactly as qarray's convert_to_maxwell builds them.

    Re-implemented here so the tests do not need qarray; test_slopes.py
    checks the two agree when qarray is installed.
    """
    Cdd = np.array(Cdd, dtype=float)
    Cgd = np.array(Cgd, dtype=float)
    total = Cdd.sum(axis=1) + Cgd.sum(axis=1)
    off = Cdd.copy()
    np.fill_diagonal(off, 0.0)
    return np.diag(total) - off, -Cgd


def lever_arms(Cdd, Cgd) -> np.ndarray:
    """L = Cdd_maxwell^-1 @ Cgd: how far each gate moves each dot."""
    cdd, cgd = maxwell(Cdd, Cgd)
    return np.linalg.inv(cdd) @ cgd


def true_slopes(Cdd, Cgd) -> Dict[str, float]:
    """dV2/dV1 of the dot-1 and dot-2 loading lines in the (P1, P2) plane."""
    L = lever_arms(Cdd, Cgd)
    return {"dot1": float(-L[0, 0] / L[0, 1]),
            "dot2": float(-L[1, 0] / L[1, 1])}


def slope_to_angle(slope: float) -> float:
    """Line angle in degrees from the +V1 axis, in (-90, 90]."""
    return float(np.degrees(np.arctan(slope)))


def cross_coupling(family: str, slope: float) -> float:
    """The virtual-gate coefficient a slope gives.

    dot-1 lines: alpha_12 = L12 / L11 = -1 / slope   (how much P2 moves dot 1)
    dot-2 lines: alpha_21 = L21 / L22 = -slope       (how much P1 moves dot 2)
    """
    return float(-1.0 / slope) if family == "dot1" else float(-slope)


def families_present(charge_states: np.ndarray) -> Dict[str, bool]:
    """Whether a family has a line worth measuring in the window at all.

    A family is present when at least MIN_FAMILY_PX pixels sit on a change
    of that dot's occupation.  A dot whose line only clips a corner has
    nothing to measure by any method; those device-families are reported as
    absent and left out of the error statistics, never counted against one
    method.
    """
    out = {}
    for k, fam in enumerate(FAMILIES):
        n = charge_states[..., k]
        edge = np.zeros(n.shape, dtype=bool)
        edge[:-1, :] |= n[1:, :] != n[:-1, :]
        edge[:, :-1] |= n[:, 1:] != n[:, :-1]
        out[fam] = bool(edge.sum() >= MIN_FAMILY_PX)
    return out


# ── the fit ───────────────────────────────────────────────────────────────

def rasterise(xy: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """(column, row) points -> binary image, nearest pixel."""
    img = np.zeros(shape, dtype=bool)
    if len(xy):
        c = np.clip(np.rint(xy[:, 0]).astype(int), 0, shape[1] - 1)
        r = np.clip(np.rint(xy[:, 1]).astype(int), 0, shape[0] - 1)
        img[r, c] = True
    return img


def segments(xy: np.ndarray, shape: Tuple[int, int], threshold: int,
             line_length: int, line_gap: int, seed: int = 0
             ) -> List[Tuple[float, float, np.ndarray]]:
    """[(angle_deg, length_px, points)] of the straight segments in a set.

    Angles are of the line, from the +V1 axis, in (-90, 90].  Each Hough
    segment's angle is replaced by the total-least-squares direction of the
    points within REFINE_PX of it, so a segment drawn between integer pixel
    ends does not carry the lattice's rounding.  `points` are those points
    (or the two segment ends when fewer than two lie near it).
    """
    if len(xy) < 2:
        return []
    from skimage.transform import probabilistic_hough_line
    found = probabilistic_hough_line(rasterise(xy, shape),
                                     threshold=threshold,
                                     line_length=line_length,
                                     line_gap=line_gap, theta=THETA, rng=seed)
    out = []
    for (x0, y0), (x1, y1) in found:
        p0, p1 = np.array([x0, y0], float), np.array([x1, y1], float)
        d = p1 - p0
        length = float(np.hypot(*d))
        if length == 0:
            continue
        u = d / length
        rel = xy - p0
        along = rel @ u
        perp = np.abs(rel @ np.array([-u[1], u[0]]))
        near = xy[(perp <= REFINE_PX) & (along >= -REFINE_PX)
                  & (along <= length + REFINE_PX)]
        if len(near) >= 2:
            c = near - near.mean(axis=0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            u = vt[0]
        else:
            near = np.stack([p0, p1])
        ang = float(np.degrees(np.arctan2(u[1], u[0])))
        ang = (ang + 90.0) % 180.0 - 90.0           # -> [-90, 90)
        if ang == -90.0:
            ang = 90.0
        out.append((ang, length, near))
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def joint_angle(segs: Sequence[Tuple[float, float, np.ndarray]],
                centre: float) -> float:
    """One direction shared by every segment of a family.

    The family's lines are parallel, so its slope is fitted once from all of
    them: each segment's points are centred on their own mean (its own
    offset) and the stacked residuals give the common direction by total
    least squares.  Segments more than TRIM_DEG from the family's median
    angle are left out, so one stray segment does not tilt the rest.  Long
    segments carry more points and weigh more, as they should.
    """
    # segments given as (angle, length) alone carry no points to fit
    use = [s for s in segs if len(s) > 2 and abs(s[0] - centre) <= TRIM_DEG]
    if not use:
        return centre
    c = np.concatenate([s[2] - s[2].mean(axis=0) for s in use])
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ang = float(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
    # a line's angle is defined mod 180: take the copy nearest the median,
    # so a near-vertical family is not split between -90 and +90
    return min((ang - 180.0, ang, ang + 180.0), key=lambda a: abs(a - centre))


def assign_families(segs: Sequence[Tuple[float, float, np.ndarray]]
                    ) -> Dict[str, Optional[float]]:
    """{family: angle in degrees, or None} from a list of segments.

    Only segments sloping down are lead transitions.  Vertical segments
    (+90) are taken as -90, the steep end of the same family.  The split
    between steep and shallow is a length-weighted Otsu threshold on angle;
    when there is no second group (or the groups are closer than
    ONE_FAMILY_DEG) the single group goes to dot1 if it is steeper than the
    diagonal and to dot2 otherwise.
    """
    segs = [((-90.0 if s[0] >= 89.95 else s[0]),) + tuple(s[1:])
            for s in segs]
    segs = [s for s in segs if s[0] < 0]
    a = np.array([s[0] for s in segs])
    w = np.array([s[1] for s in segs])
    out: Dict[str, Optional[float]] = {f: None for f in FAMILIES}
    if not len(a):
        return out
    order = np.argsort(a)
    a, w = a[order], w[order]
    segs = [segs[i] for i in order]
    best_k, best_var = None, -1.0
    for k in range(1, len(a)):
        w0, w1 = w[:k].sum(), w[k:].sum()
        m0 = (a[:k] * w[:k]).sum() / w0
        m1 = (a[k:] * w[k:]).sum() / w1
        var = w0 * w1 * (m1 - m0) ** 2
        if var > best_var:
            best_k, best_var = k, var
    if best_k is not None:
        steep = _weighted_median(a[:best_k], w[:best_k])
        shallow = _weighted_median(a[best_k:], w[best_k:])
        if shallow - steep >= ONE_FAMILY_DEG:
            out["dot1"] = joint_angle(segs[:best_k], steep) if JOINT else steep
            out["dot2"] = (joint_angle(segs[best_k:], shallow) if JOINT
                           else shallow)
            return out
    one = _weighted_median(a, w)
    out["dot1" if one < -45.0 else "dot2"] = (joint_angle(segs, one) if JOINT
                                              else one)
    return out


def fit_slopes(xy: np.ndarray, shape: Tuple[int, int], hough: Dict,
               seed: int = 0) -> Dict[str, Optional[float]]:
    """{family: fitted line angle in degrees, or None} for one point set."""
    return assign_families(segments(np.asarray(xy, float).reshape(-1, 2),
                                    shape, seed=seed, **hough))


def mask_points(mask: np.ndarray) -> np.ndarray:
    """(N, 2) (column, row) points of a binary map — (V1, V2) in pixels."""
    rr, cc = np.nonzero(mask > 0.5)
    return np.stack([cc, rr], axis=1).astype(float)


# ── the two baselines ─────────────────────────────────────────────────────

def canny_points(Z: np.ndarray, sigma: float, low_q: float, high_q: float,
                 background_px: float = 0.0) -> np.ndarray:
    """Canny edges of the full dense scan, as (column, row) points.

    The raw sensor image sits on the steep, curved flank of the sensor's own
    Coulomb peak, and a charge transition is a small step on top of it; left
    in, that flank is the largest gradient in the image and Canny draws it.
    So the conventional preprocessing is offered as a knob: subtract a
    Gaussian-blurred copy (background_px wide, 0 = off).  The image is then
    scaled by its own min and max — legitimate here, this arm measured all of
    it — and thresholds are quantiles of the gradient magnitude, so one
    setting means the same on every device.  Edges within BORDER_PX of the
    frame are dropped: the background residual is largest at the steep
    corner, and without this Canny draws the frame itself (seen on the
    examples; it fitted a vertical "family" to the right edge).
    """
    from scipy.ndimage import gaussian_filter
    from skimage.feature import canny
    Z = np.asarray(Z, dtype=float)
    if background_px:
        Z = Z - gaussian_filter(Z, background_px, mode="nearest")
    lo, hi = float(Z.min()), float(Z.max())
    Zn = (Z - lo) / (hi - lo) if hi > lo else np.zeros_like(Z)
    edges = canny(Zn, sigma=sigma, low_threshold=low_q,
                  high_threshold=high_q, use_quantiles=True, mode="nearest")
    b = BORDER_PX
    edges[:b, :] = edges[-b:, :] = False
    edges[:, :b] = edges[:, -b:] = False
    return mask_points(edges)


BORDER_PX = 3       # Canny edges this close to the frame are artefacts
STEP_WINDOW = 2     # neighbours either side that define a ray's local slope


def step_residual(trace: np.ndarray) -> np.ndarray:
    """|Δ_i - b_i| / |b_i| along one ray, one value per gap.

    b_i, the background slope at gap i, is the median of the 2*STEP_WINDOW
    gaps around it (gap i itself left out), so a smooth curve — the sensor
    flank — cancels.  Dividing by |b_i| makes the criterion scale-free: the
    flank is many times steeper near the fan's origin than far from it, and
    an absolute threshold fires on its curvature there (measured: 30 % of
    such "steps" lie within 1.5 px of a line, against 14 % for a point picked
    at random on a ray; this rule reaches 48 % at k = 1).  The first and
    last STEP_WINDOW gaps are left at 0: with neighbours on one side only,
    curvature and a step cannot be told apart there.
    """
    d = np.diff(np.asarray(trace, dtype=float))
    n = len(d)
    out = np.zeros(n)
    for i in range(STEP_WINDOW, n - STEP_WINDOW):
        nb = np.r_[d[max(0, i - STEP_WINDOW):i], d[i + 1:i + 1 + STEP_WINDOW]]
        b = np.median(nb)
        out[i] = abs(d[i] - b) / (abs(b) + 1e-30)
    return out


def ray_step_points(sample_dir: str, n_rays: int, n_points: int, k: float
                    ) -> np.ndarray:
    """The steps found on each ray, as (column, row) points — no network.

    Same fan, same nearest-cell sampling as ray_peaks.measure.  On each trace
    a step is a peak of step_residual above k: a jump k times the local
    background slope.  Only the ray's own values enter, so the rule needs no
    normalisation at all.  Repeated samples of one pixel (near the
    fan's origin) are dropped first: they are one measurement, not two.  A
    step sits between two samples; its position is their midpoint.
    """
    ux, uy, Z = load_grid(sample_dir)
    pts: List[np.ndarray] = []
    for angle in fan_angles(n_rays):
        line = ray_polyline(angle, n_points, ux, uy)
        row, col = voltage_to_pixel(line[:, 0], line[:, 1], ux, uy)
        keep = np.r_[True, (np.diff(row) != 0) | (np.diff(col) != 0)]
        line, row, col = line[keep], row[keep], col[keep]
        trace = Z[row, col].astype(float)
        if len(trace) < 4:
            continue
        r = step_residual(trace)
        idx, _ = find_peaks(np.r_[0.0, r, 0.0], height=k)
        idx = idx - 1
        if len(idx):
            # positions in fractional pixels, so a step between two samples
            # is not snapped onto either of them
            fx = (line[:, 0] - ux[0]) / (ux[-1] - ux[0]) * (len(ux) - 1)
            fy = (line[:, 1] - uy[0]) / (uy[-1] - uy[0]) * (len(uy) - 1)
            pts.append(np.stack([0.5 * (fx[idx] + fx[idx + 1]),
                                 0.5 * (fy[idx] + fy[idx + 1])], axis=1))
    return np.concatenate(pts) if pts else np.empty((0, 2))


# ── scoring ───────────────────────────────────────────────────────────────

def device_truth(sample_dir: str) -> Dict:
    """True slopes, angles and which families are in the window."""
    with open(os.path.join(sample_dir, "device.json")) as f:
        rec = json.load(f)
    cap = rec["capacitance"]
    slopes = true_slopes(cap["Cdd"], cap["Cgd"])
    states = np.load(os.path.join(sample_dir, "numpy", "simulation",
                                  "charge_states.npy"))
    return {"sample": os.path.basename(sample_dir),
            "slope": slopes,
            "angle": {f: slope_to_angle(s) for f, s in slopes.items()},
            "present": families_present(states)}


def angle_error(fit: Optional[float], true: float) -> float:
    """|fit - true| in degrees; NaN when the method returned nothing."""
    return float("nan") if fit is None else abs(fit - true)


def summarise(errors: Sequence[float]) -> Dict:
    """Median, mean, 90th percentile and hit rates of a list of angle errors.

    NaN (no fit) counts as a miss in every hit rate and is left out of the
    location statistics; n_failed says how many there were.
    """
    e = np.asarray(errors, dtype=float)
    ok = e[~np.isnan(e)]
    out = {"n": int(len(e)), "n_failed": int(np.isnan(e).sum())}
    if len(ok):
        out.update(median_deg=float(np.median(ok)), mean_deg=float(ok.mean()),
                   p90_deg=float(np.percentile(ok, 90)))
    else:
        out.update(median_deg=float("nan"), mean_deg=float("nan"),
                   p90_deg=float("nan"))
    for t in (0.5, 1.0, 2.0, 5.0):
        out[f"within_{t:g}deg"] = float(np.mean(np.nan_to_num(e, nan=1e9) <= t)) \
            if len(e) else float("nan")
    return out


# ── the sensor's own truth ────────────────────────────────────────────────

def sensor_slopes(cap: Dict) -> Dict[str, float]:
    """The slopes the SENSOR image shows, with the sensor dot in the Maxwell
    matrix (qarray's ChargeSensedDotArray).

    The line maps the network learns are drawn by DotArray without the
    sensor; the charge-sensor image Canny and the ray steps read is
    simulated with it, and the extra capacitance moves each dot's slope by a
    fraction of a degree (median 0.07, max 0.95 degrees over the 550-device
    pool).  Both are reported, so no arm is scored against a truth it could
    not have seen.
    """
    from qarray.DotArrays._helper_functions import \
        _convert_to_maxwell_with_sensor
    _, inv, cgd = _convert_to_maxwell_with_sensor(
        np.array(cap["Cdd"], float), np.array(cap["Cgd"], float),
        np.array(cap["Cds"], float), np.array(cap["Cgs"], float))
    L = np.asarray(inv) @ np.asarray(cgd)
    return {"dot1": float(-L[0, 0] / L[0, 1]),
            "dot2": float(-L[1, 0] / L[1, 1])}


# ── selection on validation devices ───────────────────────────────────────

CAP_DEG = 10.0      # a failed or wild fit costs this much in the selection


def selection_score(errors: Sequence[float]) -> float:
    """Mean angle error with every error capped at CAP_DEG and a failed fit
    counted as CAP_DEG — low is good.  Capped so one wild device cannot
    decide a setting; failures counted so returning nothing is never safe."""
    e = np.nan_to_num(np.asarray(errors, float), nan=CAP_DEG)
    return float(np.minimum(e, CAP_DEG).mean()) if len(e) else float("inf")


def errors_for(point_sets: Sequence[np.ndarray], truths: Sequence[Dict],
               shape: Tuple[int, int], hough: Dict) -> List[float]:
    """Angle errors over every present device-family, one arm, one setting."""
    out = []
    for xy, t in zip(point_sets, truths):
        fit = fit_slopes(xy, shape, hough)
        out += [angle_error(fit[f], t["angle"][f])
                for f in FAMILIES if t["present"][f]]
    return out


def choose(candidates: Dict[str, Sequence[np.ndarray]],
           truths: Sequence[Dict], shape: Tuple[int, int]) -> Dict:
    """Best (point-set setting, Hough setting) on the validation devices.

    candidates maps a label of the arm's own knob (e.g. a Canny setting) to
    the point sets that knob gives on the validation devices.
    """
    best = None
    for label, sets in candidates.items():
        for hough in HOUGH_GRID:
            s = selection_score(errors_for(sets, truths, shape, hough))
            if best is None or s < best["score"]:
                best = {"knob": label, "hough": dict(hough), "score": s}
    return best


# ── what the slopes are for: virtual gates ────────────────────────────────

def alphas_from_angles(a1: Optional[float], a2: Optional[float]
                       ) -> Optional[Tuple[float, float]]:
    """(a12, a21) from the fitted dot-1 and dot-2 line angles, or None."""
    if a1 is None or a2 is None:
        return None
    return (cross_coupling("dot1", np.tan(np.radians(a1))),
            cross_coupling("dot2", np.tan(np.radians(a2))))


def residual_crosstalk(L2: np.ndarray, alpha: Optional[Tuple[float, float]]
                       ) -> Tuple[float, float]:
    """(r12, r21) still left after virtual gates built from `alpha`.

    U1 = P1 + a12 P2, U2 = a21 P1 + P2, i.e. U = A P.  The dots see the
    virtual gates through L A^-1; its off-diagonal over its diagonal is how
    much each gate still moves the other dot.  alpha = None is no virtual
    gates at all (A = 1): the residual is then the true a12, a21.
    """
    A = np.eye(2) if alpha is None else np.array([[1.0, alpha[0]],
                                                 [alpha[1], 1.0]])
    Lv = L2 @ np.linalg.inv(A)
    return float(Lv[0, 1] / Lv[0, 0]), float(Lv[1, 0] / Lv[1, 1])
