"""
The charge state (n1, n2) of every region of the gate plane, from a binary
line map alone.

    full CSD   label_states(Y)                     the ceiling of the method
    U-Net      label_states(prob > threshold)      from a few % of the grid

One function, called the same way on both maps:

    regions      connected non-line pixels (4-connectivity); crumbs < 5 px
                 count as line
    fill         every line pixel takes the nearest region
    borders      every touching pair of regions, with its contact pixels
    family       principal-axis angle of each border: positive slope ->
                 interdot; negative -> Otsu (length-weighted) into the
                 steeper dot-1 and the shallower dot-2 family
    step         which side gets the extra charge, from the border normal
    propagate    maximum spanning tree by border length, (0, 0) at the
                 top-right corner region
    consistency  the fraction of the borders left out of the tree whose
                 step agrees with the labels the tree gave

The carriers are holes: occupation grows towards negative gate voltage, so
(0, 0) sits towards the top right, n1 grows to lower V_P1 and n2 to lower
V_P2 (checked on charge_states.npy, see device_variation).
"""
import json
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .virtual_gates import _otsu_split, pixel_size

MIN_REGION_PX = 5        # smaller components are crumbs between close lines
MIN_BORDER_PX = 2        # a direction needs two contacts
FAMILIES = ("dot1", "dot2", "interdot")
STEP = {"dot1": (1, 0), "dot2": (0, 1), "interdot": (1, -1)}
LOW_STATES = ((0, 0), (1, 0), (0, 1), (1, 1))


# ── the truth ─────────────────────────────────────────────────────────────

def load_states(sample_dir: str) -> np.ndarray:
    """(H, W, 2) int (n1, n2); row = V_P2 with row 0 lowest, column = V_P1."""
    return np.load(os.path.join(sample_dir, "numpy", "simulation",
                                "charge_states.npy")).astype(int)


def state_code(cs: np.ndarray) -> np.ndarray:
    """(n1, n2) -> one integer per pixel, for counting and comparing."""
    return cs[..., 0] * 1000 + cs[..., 1]


def device_variation(cs: np.ndarray, device: Dict) -> Dict:
    """Step 0: how hard one device's window is, from its true states."""
    codes, counts = np.unique(state_code(cs), return_counts=True)
    cap = device["capacitance"]
    corner = cs[-1, -1]
    return {
        "n_states": int(len(codes)),
        "max_n1": int(cs[..., 0].max()),
        "max_n2": int(cs[..., 1].max()),
        "has_00": int(((cs[..., 0] == 0) & (cs[..., 1] == 0)).any()),
        "corner_n1": int(corner[0]), "corner_n2": int(corner[1]),
        "corner_is_00": int(corner[0] == 0 and corner[1] == 0),
        "median_region_px": float(np.median(counts)),
        "d1g1": float(cap["Cgd"][0][0]), "d2g2": float(cap["Cgd"][1][1]),
        "d1d2": float(cap["Cdd"][0][1]),
        # the sign convention, measured rather than assumed: occupation must
        # not grow towards positive V_P1 (columns) or V_P2 (rows)
        "n1_grows_to_neg_V1": int((np.diff(cs[..., 0], axis=1) <= 0).all()),
        "n2_grows_to_neg_V2": int((np.diff(cs[..., 1], axis=0) <= 0).all()),
    }


# ── the method ────────────────────────────────────────────────────────────

_FOUR = ndimage.generate_binary_structure(2, 1)


def close_frame(B: np.ndarray) -> np.ndarray:
    """
    Lines run onto the frame.

    The simulator's line maps stop one pixel short of the image edge, and
    two regions then join around the end of the line.  Copying the line
    pixels of the second row/column onto the outermost one closes that gap.
    """
    B = np.array(B, bool)
    B[0] |= B[1]
    B[-1] |= B[-2]
    B[:, 0] |= B[:, 1]
    B[:, -1] |= B[:, -2]
    return B


def regions(B: np.ndarray, min_size: int = MIN_REGION_PX):
    """(label image, n): 0 on lines and crumbs, 1..n one per region."""
    lab, n = ndimage.label(~close_frame(B), structure=_FOUR)
    if n == 0:
        return lab, 0
    size = np.bincount(lab.ravel())
    keep = size >= min_size
    keep[0] = False
    new = np.zeros(n + 1, int)
    new[keep] = np.arange(1, keep.sum() + 1)
    return new[lab], int(keep.sum())


def fill(lab: np.ndarray) -> np.ndarray:
    """Every line pixel takes the label of the nearest region."""
    if not (lab > 0).any():
        return lab
    _, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    return lab[idx[0], idx[1]]


def borders(F: np.ndarray, dx: float, dy: float) -> List[Dict]:
    """
    Every pair of touching regions of the filled map F.

    Each contact is two 4-neighbouring pixels in different regions; its
    midpoint is a point of the border and (pixel in a) - (pixel in b) is the
    border normal pointing into a.  Returns one dict per pair a < b with the
    border length (contacts), its angle in degrees [0, 180) in mV, and the
    mean normal into a in mV.
    """
    pairs = {}
    for axis in (0, 1):
        A = F[:-1, :] if axis == 0 else F[:, :-1]
        Bn = F[1:, :] if axis == 0 else F[:, 1:]
        r, c = np.nonzero(A != Bn)
        la, lb = A[r, c], Bn[r, c]
        # midpoint in pixels; the step goes from (la) to (lb) along +axis
        mr = r + (0.5 if axis == 0 else 0.0)
        mc = c + (0.5 if axis == 1 else 0.0)
        for k in range(len(r)):
            a, b, s = la[k], lb[k], 1.0
            if a > b:
                a, b, s = b, a, -1.0
            # normal into a: a lies at -axis of b when s = +1
            nrm = (-s * dy, 0.0) if axis == 0 else (0.0, -s * dx)
            pairs.setdefault((a, b), []).append((mr[k], mc[k], *nrm))
    out = []
    for (a, b), pts in pairs.items():
        p = np.array(pts)
        x = p[:, 1] * dx
        y = p[:, 0] * dy
        x, y = x - x.mean(), y - y.mean()
        th = 0.5 * np.degrees(np.arctan2(2 * (x * y).sum(),
                                         (x * x).sum() - (y * y).sum()))
        out.append({"a": int(a), "b": int(b), "length": len(p),
                    "theta": float(th % 180.0), "points": p[:, :2],
                    "normal": (float(p[:, 3].mean()), float(p[:, 2].mean()))})
    return out


def classify(bs: List[Dict]) -> None:
    """Family of every border, in place (None if it has no direction)."""
    for e in bs:
        e["family"] = None
        if e["length"] < MIN_BORDER_PX:
            continue
        if 0.0 < e["theta"] < 90.0:
            e["family"] = "interdot"
    neg = [e for e in bs if e["length"] >= MIN_BORDER_PX
           and (90.0 <= e["theta"] < 180.0 or e["theta"] == 0.0)]
    # exactly vertical (90) or horizontal (0) borders have no slope sign;
    # they are dot borders, at the steep and shallow ends of the range
    ang = np.array([180.0 if e["theta"] == 0.0 else e["theta"] for e in neg])
    if len(neg) >= 2 and np.ptp(ang) > 0:
        split = _otsu_split(ang, np.array([e["length"] for e in neg], float))
        for e, t in zip(neg, ang):
            e["family"] = "dot1" if t < split else "dot2"
    else:
        for e, t in zip(neg, ang):
            e["family"] = "dot1" if t < 135.0 else "dot2"


def step(e: Dict) -> np.ndarray:
    """(n1, n2) of region a minus region b across border e."""
    nx, ny = e["normal"]             # points from b into a, in mV
    f = e["family"]
    if f == "dot1" or f == "interdot":
        sign = 1 if nx < 0 else -1   # a on the lower-V_P1 side gets n1 + 1
    else:
        sign = 1 if ny < 0 else -1   # a on the lower-V_P2 side gets n2 + 1
    return sign * np.array(STEP[f])


def label_states(B: np.ndarray, dx: float, dy: float) -> Dict:
    """
    (n1, n2) of every pixel from a binary line map.

    B is (H, W), row = V_P2 with row 0 lowest, column = V_P1; dx, dy the
    pixel size in mV.  Returns the label image (H, W, 2) (-999 where a
    region was not reached), the filled region map, the borders with their
    family and step, the tree edges, and the consistency (nan when every
    border is in the tree).
    """
    lab, n = regions(B)
    H, W = np.shape(B)
    if n == 0:
        return {"labels": np.full((H, W, 2), -999), "regions": lab,
                "n_regions": 0, "borders": [], "tree": [],
                "consistency": float("nan"), "n_checked": 0,
                "unreached_px": H * W}
    F = fill(lab)
    bs = borders(F, dx, dy)
    classify(bs)
    use = [e for e in bs if e["family"] is not None]
    for e in use:
        e["step"] = step(e)

    root = int(F[-1, -1])
    lab_of = {root: np.zeros(2, int)}
    tree = set()
    if use:
        i = np.array([e["a"] for e in use]) - 1
        j = np.array([e["b"] for e in use]) - 1
        w = -np.array([e["length"] for e in use], float)   # max tree
        G = coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
        T = minimum_spanning_tree(G)
        T = T + T.T
        order, pred = breadth_first_order(T, root - 1, directed=False)
        by_pair = {(e["a"], e["b"]): e for e in use}
        for v in order[1:]:
            u = pred[v]
            a, b = v + 1, u + 1
            e = by_pair[(min(a, b), max(a, b))]
            d = e["step"] if e["a"] == a else -e["step"]
            lab_of[a] = lab_of[b] + d
            tree.add((e["a"], e["b"]))

    checked = agree = 0
    for e in use:
        if (e["a"], e["b"]) in tree:
            continue
        if e["a"] in lab_of and e["b"] in lab_of:
            checked += 1
            agree += int((lab_of[e["a"]] - lab_of[e["b"]] == e["step"]).all())

    table = np.full((n + 1, 2), -999)
    for k, v in lab_of.items():
        table[k] = v
    labels = table[F]
    return {"labels": labels, "regions": F, "n_regions": n, "borders": bs,
            "tree": sorted(tree),
            "consistency": agree / checked if checked else float("nan"),
            "n_checked": checked,
            "unreached_px": int((labels[..., 0] == -999).sum())}


# ── fitting straight lines to a line map ──────────────────────────────────

# (shortest piece kept in px, longest extension in px, keep the original map)
FIT_GRID = [(m, e, k) for m in (3, 5, 8) for e in (5, 10, 20, 40)
            for k in (0, 1)]


def _draw(B, r0, c0, r1, c1):
    """Set the pixels of the straight segment (r0,c0)-(r1,c1) in B."""
    n = int(np.ceil(max(abs(r1 - r0), abs(c1 - c0)))) + 1
    rr = np.rint(np.linspace(r0, r1, n)).astype(int)
    cc = np.rint(np.linspace(c0, c1, n)).astype(int)
    ok = (rr >= 0) & (rr < B.shape[0]) & (cc >= 0) & (cc < B.shape[1])
    B[rr[ok], cc[ok]] = True


def fit_lines(B: np.ndarray, min_len: int = 5, max_ext: int = 20,
              keep: int = 1) -> np.ndarray:
    """
    Straight-line repair of a binary line map.

    The map is thinned and cut at its junctions into pieces (the same
    pieces slopes_from_lines fits).  Every piece of at least min_len pixels
    is replaced by its least-squares straight segment, between the extreme
    projections of its pixels.  Each end of that segment is then extended
    along the segment's own direction, one pixel at a time, for at most
    max_ext pixels; the extension is drawn only if it reaches another line
    pixel or the frame, so a line is joined to what it was heading for and
    never left dangling.  With keep, the original map is kept underneath.
    """
    from skimage.morphology import skeletonize
    B = np.asarray(B, bool)
    H, W = B.shape
    skel = skeletonize(B)
    nb = ndimage.convolve(skel.astype(int), np.array([[1, 1, 1], [1, 0, 1],
                                                      [1, 1, 1]]),
                          mode="constant")
    lab, n = ndimage.label(skel & (nb < 3), structure=np.ones((3, 3)))
    out = B.copy() if keep else np.zeros_like(B)
    segs = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        piece = np.zeros_like(B)
        piece[sl] = lab[sl] == k
        r, c = np.nonzero(piece)
        if len(r) < min_len:
            continue
        mr, mc = r.mean(), c.mean()
        X = np.stack([r - mr, c - mc], 1)
        u = np.linalg.svd(X, full_matrices=False)[2][0]   # (dr, dc), unit
        t = X @ u
        p0 = (mr + t.min() * u[0], mc + t.min() * u[1])
        p1 = (mr + t.max() * u[0], mc + t.max() * u[1])
        _draw(out, *p0, *p1)
        # the piece's own thick line: an extension must leave it first
        own = ndimage.distance_transform_edt(~piece) <= 2.5
        ends = []
        for j, p, d in ((np.argmax(t), p1, u), (np.argmin(t), p0, -u)):
            er, ec = r[j], c[j]
            w = skel[max(er - 1, 0):er + 2, max(ec - 1, 0):ec + 2]
            o = piece[max(er - 1, 0):er + 2, max(ec - 1, 0):ec + 2]
            if not (w & ~o).any():        # a free end: nothing attached
                ends.append((p, d))
        segs.append((ends, own))
    for ends, own in segs:
        for start, d in ends:
            path = []
            for s in range(1, max_ext + 1):
                rr = int(round(start[0] + s * d[0]))
                cc = int(round(start[1] + s * d[1]))
                if not (0 <= rr < H and 0 <= cc < W):
                    break                                  # reached frame
                path.append((rr, cc))
                if B[rr, cc] and not own[rr, cc]:
                    break                                  # reached a line
            else:
                continue                                   # reached nothing
            for rr, cc in path:
                out[rr, cc] = True
    return out


def fit_knob(k) -> str:
    m, e, keep = k
    return (f"min piece {m}px, extend up to {e}px, "
            f"{'keep' if keep else 'replace'} original")


# ── the non-ML reference: lines from the dense sensor image ───────────────

# Two conventional detectors on the full dense charge-sensor scan, each with
# its knobs.  Every setting is scanned on the validation devices and the best
# one is applied, unchanged, to the test devices.
#   canny:    (sigma, low quantile, high quantile, background px, dilate px,
#              border px)
#   gradient: (sigma, quantile, background px, dilate px, border px)
DENSE_GRID = (
    [("canny", s, lo, hi, bg, dil, b) for s in (1.0, 2.0)
     for lo, hi in ((0.80, 0.90), (0.90, 0.95))
     for bg in (0.0, 3.0, 6.0) for dil in (0, 1) for b in (0, 3)]
    + [("gradient", s, q, bg, dil, b) for s in (0.7, 1.0, 1.5)
       for q in (0.80, 0.85, 0.90, 0.93, 0.96)
       for bg in (4.0, 8.0) for dil in (0, 1) for b in (0, 3)]
    + [("steps", w, rel, q, thin, dil) for w in (5, 7) for rel in (0, 1)
       for q in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
       for thin in (0, 1) for dil in (0, 1)])


def step_residual_map(Z: np.ndarray, w: int, rel: bool,
                      measured: np.ndarray = None) -> np.ndarray:
    """How much each pixel-to-pixel difference departs from its neighbours.

    Along each axis: d = first difference, b = its running median over w
    pixels (the smooth background slope of the sensor), residual |d - b|;
    with rel, divided by |b| + median |d|, so a step counts relative to the
    local slope.  A smooth background, however steep, leaves almost nothing;
    a charge transition is a jump.  The larger of the two axes is kept.

    With `measured`, the two global numbers (the min-max scaling and the
    median |d|) are taken over measured pixels only, so nothing from an
    unmeasured pixel enters the result.
    """
    from scipy.ndimage import median_filter
    Z = np.asarray(Z, dtype=float)
    ref = Z[measured] if measured is not None and measured.any() else Z
    Z = (Z - ref.min()) / (np.ptp(ref) or 1.0)
    out = np.zeros(Z.shape)
    for ax in (0, 1):
        d = np.diff(Z, axis=ax)
        b = median_filter(d, size=(w, 1) if ax == 0 else (1, w),
                          mode="nearest")
        r = np.abs(d - b)
        if rel:
            if measured is not None:
                both = (measured[1:] & measured[:-1]) if ax == 0 else \
                    (measured[:, 1:] & measured[:, :-1])
                scale = np.median(np.abs(d[both])) if both.any() else \
                    np.median(np.abs(d))
            else:
                scale = np.median(np.abs(d))
            r = r / (np.abs(b) + scale)
        out = np.maximum(out, np.pad(r, [(0, 1), (0, 0)] if ax == 0
                                     else [(0, 0), (0, 1)]))
    return out


def _reframe(E: np.ndarray, b: int) -> np.ndarray:
    """Drop edges within b px of the frame, then run the edges found at
    distance b straight out to the frame.

    The sensor's own background is steepest at the frame, and a detector
    draws the frame itself there; that line cuts a thin false region off
    the corner.  Dropping the band removes it; extending the surviving
    edges keeps the real lines closed onto the frame.
    """
    if not b:
        return E
    E = E.copy()
    E[:b, :] = E[b, :]
    E[-b:, :] = E[-b - 1, :]
    E[:, :b] = E[:, b][:, None]
    E[:, -b:] = E[:, -b - 1][:, None]
    return E


def dense_line_map(Z: np.ndarray, knob) -> np.ndarray:
    """Binary line map from the dense sensor image Z (row = V_P2).

    The route without a network, with every pixel measured.

    canny     subtract a Gaussian-blurred copy (the sensor's smooth
              background), scale by min and max, Canny with quantile
              thresholds.
    gradient  |grad Z| at scale sigma, minus its own Gaussian-blurred copy
              (removes the smooth flank of the sensor peak, keeps the steps),
              pixels above a quantile, thinned to one-pixel lines.

    Quantile thresholds make one setting mean the same on every device.
    Optional dilation closes one-pixel breaks.  Unlike slopes.canny_points,
    edges at the frame are kept: removing them opens the gaps that merge two
    regions.
    """
    from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
    from skimage.feature import canny
    from skimage.morphology import skeletonize
    Z = np.asarray(Z, dtype=float)
    Z = (Z - Z.min()) / (np.ptp(Z) or 1.0)
    if knob[0] == "canny":
        _, s, lo, hi, bg, dil, b = knob
        if bg:
            Z = Z - gaussian_filter(Z, bg, mode="nearest")
        E = canny(Z, sigma=s, low_threshold=lo, high_threshold=hi,
                  use_quantiles=True, mode="nearest")
    elif knob[0] == "gradient":
        _, s, q, bg, dil, b = knob
        G = gaussian_gradient_magnitude(Z, s, mode="nearest")
        G = G - gaussian_filter(G, bg, mode="nearest")
        E = skeletonize(G > np.quantile(G, q))
    else:
        _, w, rel, q, thin, dil = knob
        R = step_residual_map(Z, w, bool(rel))
        E = R > np.quantile(R, q)
        if thin:
            E = skeletonize(E)
        b = 0
    E = _reframe(E, b)
    if dil:
        E = ndimage.binary_dilation(E, iterations=dil)
    return E


def dense_knob(k) -> str:
    if k[0] == "canny":
        _, s, lo, hi, bg, dil, b = k
        return (f"Canny sigma={s:g} q=({lo:g},{hi:g}) background={bg:g}px "
                f"dilate={dil}px border={b}px")
    if k[0] == "gradient":
        _, s, q, bg, dil, b = k
        return (f"gradient sigma={s:g} q={q:g} background={bg:g}px "
                f"dilate={dil}px border={b}px")
    _, w, rel, q, thin, dil = k
    return (f"step residual window={w}px "
            f"{'relative' if rel else 'absolute'} q={q:g} "
            f"thin={'yes' if thin else 'no'} dilate={dil}px")


# ── the metrics ───────────────────────────────────────────────────────────

def keep_mask(Y: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Pixels farther than tol px from a true line: the ones scored."""
    return ndimage.distance_transform_edt(~np.asarray(Y, bool)) > tol


def best_shift(pred: np.ndarray, true: np.ndarray, m: np.ndarray):
    """The constant (dn1, dn2) that makes the most kept pixels right."""
    ok = m & (pred[..., 0] != -999)
    if not ok.any():
        return np.zeros(2, int)
    d = true[ok] - pred[ok]
    u, c = np.unique(d, axis=0, return_counts=True)
    return u[np.argmax(c)]


def _centroid(mask, dx, dy):
    r, c = np.nonzero(mask)
    return np.array([c.mean() * dx, r.mean() * dy])


def score(res: Dict, cs: np.ndarray, Y: np.ndarray, dx: float,
          dy: float) -> Dict:
    """Every per-device metric of one labelling against the truth."""
    m = keep_mask(Y)
    P = res["labels"]
    sh = best_shift(P, cs, m)
    Ps = np.where(P == -999, -999, P + sh)
    nk = m.sum()

    right_rel = (Ps == cs).all(-1) & m
    right_abs = (P == cs).all(-1) & m
    tp = np.where(P[..., 0] == -999, -10 ** 6, P.sum(-1))
    dtot = cs.sum(-1)[m] - tp[m]
    ok = tp[m] > -10 ** 5
    tsh = np.bincount(dtot[ok] - dtot[ok].min()).argmax() + dtot[ok].min() \
        if ok.any() else 0
    right_tot = (tp[m] + tsh == cs.sum(-1)[m]).mean()

    tcode, pcode = state_code(cs), state_code(Ps)
    states = np.unique(tcode[m])
    recall = np.mean([(pcode[m & (tcode == s)] == s).mean() > 0.5
                      for s in states]) if len(states) else np.nan
    prec = []
    for k in range(1, res["n_regions"] + 1):
        rk = m & (res["regions"] == k)
        if rk.any():
            prec.append((tcode[rk] == pcode[rk]).mean() > 0.5)
    out = {"acc_rel": float(right_rel.sum() / nk),
           "acc_abs": float(right_abs.sum() / nk),
           "acc_total": float(right_tot),
           "state_recall": float(recall),
           "state_precision": float(np.mean(prec)) if prec else np.nan,
           "shift_n1": int(sh[0]), "shift_n2": int(sh[1]),
           "consistency": res["consistency"], "n_checked": res["n_checked"],
           "n_regions": res["n_regions"],
           "unreached_%": 100.0 * res["unreached_px"] / P[..., 0].size}
    for s in LOW_STATES:
        c = s[0] * 1000 + s[1]
        t, p = m & (tcode == c), m & (pcode == c)
        key = f"{s[0]}{s[1]}"
        out[f"iou_{key}"] = (float((t & p).sum() / (t | p).sum())
                             if t.any() else np.nan)
        if key == "11":
            out["centre_err_11_mV"] = (
                float(np.linalg.norm(_centroid(t, dx, dy)
                                     - _centroid(p, dx, dy)))
                if t.any() and p.any() else np.nan)
    return out


def score_few(res: Dict, cs: np.ndarray, Y: np.ndarray,
              nmax: int = 3) -> Dict:
    """
    The few-electron region only: true n1 <= nmax and n2 <= nmax.

    Absolute labels (no shift), because the top-right corner is (0,0) on
    every device and an experiment needs the real occupation.

    acc_few       fraction of the region's scored pixels with the right
                  (n1, n2)
    bal_few       the same per true state, averaged over the states present
                  (each state counts once, however large)
    land_11       1 if the gates parked at the centre of the predicted (1,1)
                  region (its pixel farthest from the region's edge) are
                  really in (1,1); 0 if not or if no (1,1) was predicted;
                  nan if (1,1) is not in the window
    """
    m = keep_mask(Y)
    roi = (cs[..., 0] <= nmax) & (cs[..., 1] <= nmax)
    P = res["labels"]
    tcode, pcode = state_code(cs), state_code(P)
    sel = m & roi
    out = {"few_px_%": 100.0 * sel.mean()}
    if not sel.any():
        out.update(acc_few=np.nan, bal_few=np.nan, land_11=np.nan,
                   n_states_few=0)
        return out
    right = tcode == pcode
    out["acc_few"] = float(right[sel].mean())
    states = np.unique(tcode[sel])
    out["bal_few"] = float(np.mean([right[sel & (tcode == s)].mean()
                                    for s in states]))
    out["n_states_few"] = int(len(states))
    if not (tcode == 1001).any():
        out["land_11"] = np.nan
    else:
        p11 = pcode == 1001
        if not p11.any():
            out["land_11"] = 0.0
        else:
            dist = ndimage.distance_transform_edt(np.pad(p11, 1))[1:-1, 1:-1]
            r, c = np.unravel_index(np.argmax(dist), dist.shape)
            out["land_11"] = float(tcode[r, c] == 1001)
    return out


def wrong_steps(res: Dict, cs: np.ndarray, Y: np.ndarray,
                lim: int = 3) -> np.ndarray:
    """(2 lim + 1)^2 histogram of (dn1, dn2) = shifted label - truth over
    the wrong kept pixels; row = dn2, column = dn1, clipped to +-lim."""
    m = keep_mask(Y)
    P = res["labels"]
    sh = best_shift(P, cs, m)
    ok = m & (P[..., 0] != -999)
    d = (P + sh - cs)[ok]
    d = d[(d != 0).any(-1)]
    d = np.clip(d, -lim, lim) + lim
    h = np.zeros((2 * lim + 1, 2 * lim + 1), int)
    np.add.at(h, (d[:, 1], d[:, 0]), 1)
    return h


def pixel_mV(device: Dict, shape) -> tuple:
    return pixel_size(device["voltage_window"], shape)


def read_device(sample_dir: str) -> Dict:
    with open(os.path.join(sample_dir, "device.json")) as f:
        return json.load(f)


# ── statistics ────────────────────────────────────────────────────────────

def stat(values: Sequence[float]) -> Dict:
    """mean, std, median and n over the finite values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "n": int(len(v))}


# ── tables ────────────────────────────────────────────────────────────────

METRICS = ("acc_rel", "acc_abs", "acc_total", "state_recall",
           "state_precision", "iou_11", "centre_err_11_mV", "consistency")
PERCENT = {"acc_rel", "acc_abs", "acc_total", "state_recall",
           "state_precision"}
TABLE_COLS = [("pixel acc. rel. %", "acc_rel"),
              ("pixel acc. abs. %", "acc_abs"),
              ("total-charge acc. %", "acc_total"),
              ("state recall %", "state_recall"),
              ("state precision %", "state_precision"),
              ("(1,1) IoU", "iou_11"),
              ("(1,1) centre err. (mV)", "centre_err_11_mV"),
              ("consistency", "consistency")]


def summary_row(rows: Sequence[Dict]) -> Dict:
    """One table row from the per-device rows of one method.

    The absolute accuracy is taken only on devices whose true top-right
    corner is (0, 0).
    """
    out = {}
    for k in METRICS:
        rr = [r for r in rows if k != "acc_abs" or r["corner_is_00"]]
        s = stat([r[k] for r in rr])
        if k in PERCENT:
            s = {q: (100 * v if q != "n" else v) for q, v in s.items()}
        out[k] = s
    out["n_devices"] = len(rows)
    out["coverage_%"] = 100.0 * float(np.mean([r["coverage"] for r in rows]))
    return out


def fmt(s: Dict, digits: int = 1) -> str:
    """mean ± std [median]."""
    if not s["n"]:
        return "—"
    return (f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f} "
            f"[{s['median']:.{digits}f}]")


def _digits(k):
    return 1 if k in PERCENT else (3 if k == "centre_err_11_mV" else 2)


def table_lines(summaries: Sequence[Dict]) -> List[List[str]]:
    head = (["method", "rays × points", "coverage %"]
            + [c for c, _ in TABLE_COLS])
    body = [[s["method"], s["budget"], f"{s['coverage_%']:.2f}"]
            + [fmt(s[k], _digits(k)) for _, k in TABLE_COLS]
            for s in summaries]
    return [head] + body


TABLE_NOTE = (
    "Each cell: mean ± std over devices, median in brackets. Pixels within "
    "1 px of a true transition line are not scored. 'rel.': after the best "
    "constant shift (Δn1, Δn2) per device; 'abs.': no shift, top-right "
    "corner taken as (0,0), only devices whose true corner is (0,0). "
    "Total-charge accuracy uses its own best constant shift of n1 + n2. "
    "(1,1) IoU and centre error use the shifted labels; the centre error is "
    "over devices where both the true and the predicted (1,1) region exist. "
    "Consistency: fraction of non-tree borders whose step agrees with the "
    "propagated labels.")


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
        f.write("\n" + TABLE_NOTE + "\n")
    with open(out_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rays", "points", "coverage_%", "n_devices"]
                   + [f"{k}_{q}" for _, k in TABLE_COLS
                      for q in ("mean", "std", "median", "n")])
        for s in summaries:
            w.writerow([s["method"], s["rays"], s["points"],
                        f"{s['coverage_%']:.4f}", s["n_devices"]]
                       + [s[k][q] for _, k in TABLE_COLS
                          for q in ("mean", "std", "median", "n")])
    tex_head = ["method", r"rays $\times$ points", r"coverage (\%)",
                r"acc.\ rel.\ (\%)", r"acc.\ abs.\ (\%)",
                r"total (\%)", r"recall (\%)", r"precision (\%)",
                r"(1,1) IoU", r"(1,1) err.\ (mV)", "consistency"]
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


VARIATION_KEYS = [("charge states in the window", "n_states", 0),
                  ("largest n1", "max_n1", 0), ("largest n2", "max_n2", 0),
                  ("median pixels per state", "median_region_px", 0),
                  ("d1g1 (cell size, dot 1)", "d1g1", 2),
                  ("d2g2 (cell size, dot 2)", "d2g2", 2),
                  ("d1d2 (interdot coupling)", "d1d2", 2)]
VARIATION_COUNTS = [("window contains (0,0)", "has_00"),
                    ("top-right corner is (0,0)", "corner_is_00"),
                    ("n1 never grows towards +V_P1", "n1_grows_to_neg_V1"),
                    ("n2 never grows towards +V_P2", "n2_grows_to_neg_V2")]


def variation_table(rows: Sequence[Dict], out_base: str, caption: str):
    """Step 0: one line per quantity, median and range over the devices."""
    lines = ["| quantity | median | min | max |", "|---|---|---|---|"]
    for name, k, d in VARIATION_KEYS:
        v = np.array([r[k] for r in rows], float)
        lines.append(f"| {name} | {np.median(v):.{d}f} | {v.min():.{d}f} | "
                     f"{v.max():.{d}f} |")
    n = len(rows)
    for name, k in VARIATION_COUNTS:
        lines.append(f"| {name} | {sum(r[k] for r in rows)}/{n} devices "
                     f"| | |")
    with open(out_base + ".md", "w", encoding="utf-8") as f:
        f.write(caption + "\n\n" + "\n".join(lines) + "\n")
    return lines


# ── figures ───────────────────────────────────────────────────────────────

INK, MUTED = "#111111", "#555555"
FULL_COLOR, UNET_COLOR, DENSE_COLOR = "#111111", "#2a78d6", "#eb6834"
LABEL_MIN_PX = 40        # regions smaller than this get no (n1, n2) text
ERR_N1, ERR_N2, ERR_BOTH = (0.92, 0.45, 0.10), (0.16, 0.42, 0.84), (0.1,) * 3


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)


def _state_rgb(L: np.ndarray) -> np.ndarray:
    """A colour per (n1, n2); white where no label was reached."""
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab20")
    idx = (3 * L[..., 0] + 7 * L[..., 1]) % 20
    rgb = cm(idx)[..., :3]
    rgb[L[..., 0] == -999] = 1.0
    return rgb


def _write_labels(ax, L, ext, fs=4.5):
    """n1,n2 at the centre of every connected state region."""
    H, W = L.shape[:2]
    x0, x1, y0, y1 = ext
    code = state_code(L)
    for c in np.unique(code):
        lab, n = ndimage.label(code == c, structure=_FOUR)
        for k in range(1, n + 1):
            r, cc = np.nonzero(lab == k)
            if len(r) < LABEL_MIN_PX or L[r[0], cc[0], 0] == -999:
                continue
            # the region's pixel nearest its centroid, so the text sits
            # inside even a curved region; the text is the label itself,
            # which may be negative where propagation went wrong
            j = np.argmin((r - r.mean()) ** 2 + (cc - cc.mean()) ** 2)
            n1, n2 = L[r[j], cc[j]]
            ax.text(x0 + cc[j] * (x1 - x0) / (W - 1),
                    y0 + r[j] * (y1 - y0) / (H - 1),
                    f"{n1},{n2}", fontsize=fs, ha="center",
                    va="center", color=INK)


def _map_axes(a, first_col):
    a.set_aspect("equal")
    a.tick_params(labelsize=5, length=2, pad=1)
    a.set_xlabel("$V_\\mathrm{P1}$ (mV)", fontsize=5.5, labelpad=1)
    if first_col:
        a.set_ylabel("$V_\\mathrm{P2}$ (mV)", fontsize=5.5, labelpad=1)
    else:
        a.set_yticklabels([])


def fig_variation(out_dir, var_rows, examples):
    """Fig. 1: how much the evaluation devices differ, and six examples.

    examples: (name, charge_states, voltage_window, Y) of six devices.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.45,
                          left=0.07, right=0.99, top=0.92, bottom=0.08)
    top = gs[0].subgridspec(1, 4, wspace=0.45)
    n = len(var_rows)

    def col(k):
        return np.array([r[k] for r in var_rows])
    a = fig.add_subplot(top[0])
    a.hist(col("n_states"), bins=20, color=UNET_COLOR)
    a.set_xlabel("charge states in window", fontsize=7)
    a.set_ylabel("devices", fontsize=7)
    a = fig.add_subplot(top[1])
    a.hist(col("median_region_px"), bins=20, color=UNET_COLOR)
    a.set_xlabel("median px per state", fontsize=7)
    a = fig.add_subplot(top[2])
    m = int(max(col("max_n1").max(), col("max_n2").max()))
    b = np.arange(0.5, m + 1.5)
    a.hist([col("max_n1"), col("max_n2")], bins=b,
           color=[UNET_COLOR, "#eb6834"], label=["max $n_1$", "max $n_2$"])
    a.legend(fontsize=6, frameon=False)
    a.set_xlabel("largest occupation", fontsize=7)
    a = fig.add_subplot(top[3])
    a.scatter(col("d1d2"), col("n_states"), s=5, color=UNET_COLOR)
    a.set_xlabel("$d_1d_2$ (interdot)", fontsize=7)
    a.set_ylabel("states in window", fontsize=7)
    for ax in fig.axes:
        ax.tick_params(labelsize=6)
    fig.text(0.07, 0.985,
             f"{n} devices.  Top-right corner state is (0,0) in "
             f"{int(col('corner_is_00').sum())}/{n}; the window contains "
             f"(0,0) in {int(col('has_00').sum())}/{n}.",
             fontsize=7, va="top", color=MUTED)
    bot = gs[1].subgridspec(1, len(examples), wspace=0.12)
    for k, (name, cs, ext, Y) in enumerate(examples):
        a = fig.add_subplot(bot[k])
        a.imshow(_state_rgb(cs), origin="lower", extent=ext,
                 interpolation="nearest")
        _write_labels(a, cs, ext, fs=3.4)
        _map_axes(a, k == 0)
        a.set_title(f"{name}: {len(np.unique(state_code(cs)))} states",
                    fontsize=6)
    _save(fig, out_dir, "fig1_device_variation")
    plt.close(fig)


def fig_examples(out_dir, picks):
    """Fig. 2: good / typical / bad device, (a) truth (b) U-Net map
    (c) U-Net labels (d) error map.

    picks: dicts with name, label, cs, ext, Y, B (binary map), res, score.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axs = plt.subplots(len(picks), 4,
                            figsize=(7.2, 2.0 * len(picks) + 0.35),
                            gridspec_kw=dict(wspace=0.08, hspace=0.5,
                                             left=0.075, right=0.995,
                                             top=0.9, bottom=0.07))
    titles = ["(a) truth", "(b) U-Net line map", "(c) U-Net labels",
              "(d) errors"]
    for k, p in enumerate(picks):
        ext, cs, res = p["ext"], p["cs"], p["res"]
        m = keep_mask(p["Y"])
        L = res["labels"]
        sh = best_shift(L, cs, m)
        Ls = np.where(L == -999, -999, L + sh)
        ax = axs[k]
        kw = dict(origin="lower", extent=ext, interpolation="nearest")
        ax[0].imshow(_state_rgb(cs), **kw)
        _write_labels(ax[0], cs, ext)
        ax[1].imshow(p["B"], cmap=ListedColormap(["white", INK]), vmin=0,
                     vmax=1, **kw)
        ax[2].imshow(_state_rgb(Ls), **kw)
        _write_labels(ax[2], Ls, ext)
        err = np.full(cs.shape[:2] + (3,), 0.82)
        wrong = m & ~(Ls == cs).all(-1)
        d = Ls - cs
        # a wrong pixel's colour says which occupation is off
        e1, e2 = d[..., 0] != 0, d[..., 1] != 0
        err[wrong & e1 & ~e2] = ERR_N1
        err[wrong & ~e1 & e2] = ERR_N2
        err[wrong & e1 & e2] = ERR_BOTH
        err[~m] = 1.0
        ax[3].imshow(err, **kw)
        for c, a in enumerate(ax):
            _map_axes(a, c == 0)
            if k == 0:
                a.set_title(titles[c], fontsize=7, pad=12)
        s = p["score"]
        con = ("–" if not np.isfinite(s["consistency"])
               else f"{s['consistency']:.2f}")
        ax[0].text(0.0, 1.03,
                   f"{p['label']}, {p['name']}:  rel. accuracy "
                   f"{100 * s['acc_rel']:.1f} %,  state recall "
                   f"{100 * s['state_recall']:.0f} %,  consistency {con}",
                   transform=ax[0].transAxes, fontsize=6.3, ha="left",
                   va="bottom")
    fig.text(0.5, 0.008, "(d) grey: correct;  white: within 1 px of a true "
             "line (not scored);  wrong: orange = n1 only, blue = n2 only, "
             "black = both n1 and n2",
             ha="center", fontsize=6, color=MUTED)
    _save(fig, out_dir, "fig2_labelling_examples")
    plt.close(fig)


def fig_vs_coverage(out_dir, summaries):
    """Fig. 3: relative pixel accuracy and state recall vs coverage."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    un = [s for s in summaries if s["method"] == "U-Net"]
    full = [s for s in summaries if s["method"] == "full CSD"][0]
    dense = [s for s in summaries
             if s["method"] not in ("U-Net", "full CSD")]
    for a, k, lab in ((ax[0], "acc_rel", "pixel accuracy, relative (%)"),
                      (ax[1], "state_recall", "state recall (%)")):
        x = np.array([s["coverage_%"] for s in un])
        a.plot(x, [s[k]["median"] for s in un], "o", color=UNET_COLOR, ms=4,
               label="U-Net, median")
        a.plot(x, [s[k]["mean"] for s in un], "x", color=UNET_COLOR, ms=4,
               label="U-Net, mean")
        a.plot(100, full[k]["median"], "s", color=FULL_COLOR, ms=5,
               label="full CSD, median")
        a.plot(100, full[k]["mean"], "+", color=FULL_COLOR, ms=7,
               label="full CSD, mean")
        for d in dense:
            a.plot(100, d[k]["median"], "D", color=DENSE_COLOR, ms=4.5,
                   label=f"{d['method']}, median")
            a.plot(100, d[k]["mean"], "x", color=DENSE_COLOR, ms=6,
                   label=f"{d['method']}, mean")
        a.set_xscale("log")
        t = [1, 2, 5, 10, 20, 50, 100]
        a.set_xticks(t)
        a.set_xticklabels([str(v) for v in t])
        a.set_xlabel("coverage of the grid (%)", fontsize=8)
        a.set_ylabel(lab, fontsize=8)
        a.set_ylim(0, 102)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.6)
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    _save(fig, out_dir, "fig3_accuracy_vs_coverage")
    plt.close(fig)


def fig_failures(out_dir, hist, lim, rows_full, rows_unet, label,
                 rows_dense=()):
    """Fig. 4: (dn1, dn2) of wrong pixels; accuracy vs states and d1d2."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    h = hist.astype(float)
    h[h == 0] = np.nan
    im = ax[0].imshow(h, origin="lower", cmap="viridis", norm=LogNorm(),
                      extent=(-lim - 0.5, lim + 0.5, -lim - 0.5, lim + 0.5))
    cb = fig.colorbar(im, ax=ax[0], shrink=0.85)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("wrong pixels", fontsize=6)
    ax[0].set_xlabel("$\\Delta n_1$ (label − truth)", fontsize=7)
    ax[0].set_ylabel("$\\Delta n_2$", fontsize=7)
    ax[0].set_title(f"U-Net {label}, all test devices (clipped at ±{lim})",
                    fontsize=6.5)
    for a, k, xl in ((ax[1], "n_states", "charge states in window"),
                     (ax[2], "d1d2", "$d_1d_2$ (interdot)")):
        series = [(rows_full, FULL_COLOR, "s", "full CSD"),
                  (rows_unet, UNET_COLOR, "o", f"U-Net {label}")]
        if rows_dense:
            series.insert(1, (rows_dense, DENSE_COLOR, "D",
                              rows_dense[0]["method"]))
        for rr, c, mk, nm in series:
            a.plot([r[k] for r in rr], [100 * r["acc_rel"] for r in rr], mk,
                   ms=3, mfc="none", mec=c, mew=0.7, label=nm)
        a.set_xlabel(xl, fontsize=7)
        a.set_ylabel("pixel accuracy, rel. (%)", fontsize=7)
        a.set_ylim(0, 102)
        a.grid(alpha=0.25, lw=0.6)
    ax[1].legend(fontsize=6, frameon=False, loc="lower left")
    for a in ax:
        a.tick_params(labelsize=6)
    _save(fig, out_dir, "fig4_failures")
    plt.close(fig)
