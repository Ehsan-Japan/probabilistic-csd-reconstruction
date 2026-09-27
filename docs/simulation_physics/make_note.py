# -*- coding: utf-8 -*-
"""
make_note.py -- the numbers, checks and figures for simulation_physics.pdf.

    python docs/simulation_physics/make_note.py      # figures + values.tex
    latexmk -pdf simulation_physics.tex              # (run in this folder)

Everything the note says about how QArray turns capacitances and gate
voltages into transition lines is re-implemented here from the equations
in the note and compared, pixel for pixel, with what QArray (and this
repository's simulator) actually produced for one stored device.  If an
equation in the note were wrong, a check below would fail and the note
would say so -- nothing is typed in by hand.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from qarray import ChargeSensedDotArray, DotArray               # noqa: E402
from csdrecon.simulation.dqd_simulator import _charge_state_changes  # noqa

DEVICE = os.path.join(ROOT, "data", "_device_pools",
                      "devices_n550_res100_c1df7b6bf", "sample_10")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
NMAX = 6            # brute-force search: 0..NMAX-1 holes per dot is ample

rec = json.load(open(os.path.join(DEVICE, "device.json")))
cap = {k: np.array(v, dtype=float) for k, v in rec["capacitance"].items()}
x0, x1, y0, y1 = rec["voltage_window"]
RES = rec["resolution"]
GAMMA = rec["coulomb_peak_width"]
TEMP = rec["temperature"]


# ── 1. the capacitance matrices, as QArray builds them ────────────────────
def maxwell(cdd_nm, cgd_nm):
    """Eq. (2): diagonal = row sums of both non-Maxwell matrices."""
    off = cdd_nm.copy()
    np.fill_diagonal(off, 0.0)
    C = np.diag(cdd_nm.sum(1) + cgd_nm.sum(1)) - off
    return C, np.linalg.inv(C)


C, Cinv = maxwell(cap["Cdd"], cap["Cgd"])
Cg = cap["Cgd"]                                   # 2 x 3, positive
A = Cinv @ Cg                                     # 2 x 3 lever matrix

# full matrices with the sensor as a third island (sensor self term = 0)
cdd_full = np.zeros((3, 3))
cdd_full[:2, :2] = cap["Cdd"]
cdd_full[2, :2] = cap["Cds"][0]
cdd_full[:2, 2] = cap["Cds"][0]
cgd_full = np.vstack([cap["Cgd"], cap["Cgs"]])
Cf, Cfinv = maxwell(cdd_full, cgd_full)

# QArray's own objects, for the comparisons
dot = DotArray(Cdd=cap["Cdd"], Cgd=cap["Cgd"], charge_carrier="hole")
sen = ChargeSensedDotArray(Cdd=cap["Cdd"], Cgd=cap["Cgd"], Cds=cap["Cds"],
                           Cgs=cap["Cgs"], coulomb_peak_width=GAMMA, T=TEMP)
checks = {}
checks["C matches QArray"] = np.allclose(C, dot.cdd)
checks["C_full matches QArray"] = np.allclose(Cf, sen.cdd_full)
checks["hole sign: cgd = -C_g"] = np.allclose(dot.cgd, -Cg)


# ── 2. the energy and the ground state, re-implemented ────────────────────
def energy(n, V, Ci=Cinv, G=Cg):
    """Eq. (3) for holes: F = (n + C_g V)^T C^-1 (n + C_g V)."""
    q = n + np.einsum("ij,...j->...i", G, V)
    return np.einsum("...i,ij,...j->...", q, Ci, q)


CONFIGS = np.array([(a, b) for a in range(NMAX) for b in range(NMAX)],
                   dtype=float)


def ground_state(V):
    """Eq. (4): the integer configuration of lowest energy."""
    F = np.stack([energy(n, V) for n in CONFIGS], axis=-1)
    return CONFIGS[np.argmin(F, axis=-1)]


vx = np.linspace(x0, x1, RES)
vy = np.linspace(y0, y1, RES)
VX, VY = np.meshgrid(vx, vy)                     # rows follow V2
V = np.stack([VX, VY, np.zeros_like(VX)], axis=-1)

n_ours = ground_state(V)
n_qarray = np.rint(dot.do2d_open("P1", x0, x1, RES, "P2", y0, y1, RES))
stored = np.load(glob.glob(os.path.join(DEVICE, "**", "charge_states.npy"),
                           recursive=True)[0])
checks["ground state = QArray DotArray (all pixels)"] = \
    np.array_equal(n_ours, n_qarray)
checks["ground state = stored charge_states.npy"] = \
    np.array_equal(n_ours.astype(np.int8), stored)

# ── 3. the label: a pixel whose state differs from its right/up neighbour
lab_ours = _charge_state_changes(n_ours)          # (RES-1, RES-1)
gt = np.load(glob.glob(os.path.join(DEVICE, "**", "ground_truth_labels.npy"),
                       recursive=True)[0])
checks["label = stored ground truth (interior)"] = np.array_equal(
    lab_ours.astype(np.uint8), gt[:RES - 1, :RES - 1])
line_frac = gt.mean()

# ── 4. the sensor signal, re-implemented ──────────────────────────────────
NPEAK = sen.n_peak


def lorentz(x, g):
    return 1.0 / ((x / g) ** 2 + 1.0)


def sensor_signal(V, n_dots):
    """Eq. (6): Lorentzians in the sensor's energy steps."""
    Gf = cgd_full
    v_s = -(np.einsum("ij,...j->...i", -Gf, V))[..., 2:]   # C_gs V (holes)
    # QArray rounds the continuous sensor charge  cgd_full . V  (negative
    # for holes) and perturbs it by k = -NPEAK..NPEAK
    Ns = np.round(np.einsum("ij,...j->...i", -Gf, V)[..., 2:])
    Fk = []
    for k in range(-NPEAK, NPEAK + 1):
        n_full = np.concatenate([n_dots, Ns + k], axis=-1)
        q = n_full - np.einsum("ij,...j->...i", -Gf, V)
        Fk.append(np.einsum("...i,ij,...j->...", q, Cfinv, q))
    Fk = np.stack(Fk, axis=0)
    return lorentz(np.diff(Fk, axis=0), GAMMA).sum(axis=0), v_s


vg_q = sen.gate_voltage_composer.do2d("P1", x0, x1, RES, "P2", y0, y1, RES)
z_qarray, n_sen = sen.charge_sensor_open(vg_q)
z_ours, _ = sensor_signal(vg_q, np.asarray(n_sen))
checks["sensor signal = QArray (all pixels)"] = np.allclose(
    z_ours, z_qarray[..., 0], rtol=1e-9, atol=1e-15)
checks["QArray V grid = ours (V3 = 0)"] = np.allclose(vg_q, V)

# ── 5. worked example: one pixel, every configuration ─────────────────────
pix = (RES // 2, RES // 2)                       # centre of the window
Vp = V[pix]
ng = -(Cg @ Vp)                                  # induced charge, holes
F_tab = np.array([[energy(np.array([a, b], float), Vp) for b in range(4)]
                  for a in range(4)])
n_star = n_ours[pix]

# ── 6. the transition-line equations ──────────────────────────────────────
# F(n + d) - F(n) = d^T C^-1 d + 2 d^T C^-1 (n + C_g V) = 0   (Eq. 5)
# with V = (V1, V2, 0):  2 (d^T A)[0] V1 + 2 (d^T A)[1] V2 = -(d^T C^-1 d
#                                             + 2 d^T C^-1 n)
def slope(d):
    r = np.asarray(d, float) @ A
    return -r[0] / r[1]


s_dot1 = slope([1, 0])
s_dot2 = slope([0, 1])
s_inter = slope([1, -1])
# spacing between successive dot-1 lines along V1 at fixed n2, V2
dV1 = Cinv[0, 0] / A[0, 0]
dV2 = Cinv[1, 1] / A[1, 1]

# measured from the ground truth.  Each boundary between two SPECIFIC
# states (n, n + d) is one straight segment; the lines of one family are
# broken into such segments at the triple points, so a fit across a whole
# family would mix offsets.  Collect the midpoints between neighbouring
# pixels whose states differ by +-d, group them by the state pair, and fit
# the longest segment of each family.
def segments(d):
    groups = {}
    for (dy, dx) in ((0, 1), (1, 0)):
        a = n_ours[:RES - dy, :RES - dx]
        b = n_ours[dy:, dx:]
        diff = b - a
        for sgn in (1, -1):
            m = np.all(diff == sgn * np.array(d), -1)
            yy, xx = np.nonzero(m)
            for r, c in zip(yy, xx):
                lo = tuple(a[r, c].astype(int)) if sgn == 1 else                     tuple(b[r, c].astype(int))
                groups.setdefault(lo, []).append(
                    (vx[c] + dx * (vx[1] - vx[0]) / 2,
                     vy[r] + dy * (vy[1] - vy[0]) / 2))
    return groups


def fit_longest(d):
    g = segments(d)
    if not g:
        return np.nan, 0, None, []
    key = max(g, key=lambda k: len(g[k]))
    Q = np.array(g[key])
    k = np.polyfit(Q[:, 0], Q[:, 1], 1)[0]
    return k, len(Q), key, Q


m_dot1, k1, key1, Q1 = fit_longest((1, 0))
m_dot2, k2, key2, Q2 = fit_longest((0, 1))
m_inter, k3, key3, Q3 = fit_longest((1, -1))
seg = {"dot1": Q1, "dot2": Q2, "inter": Q3}

# ── figures ────────────────────────────────────────────────────────────────
ext = [x0, x1, y0, y1]
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9})

# Fig 1: ground-state map with the transition labels
states = sorted({tuple(int(v) for v in s) for s in n_ours.reshape(-1, 2)})
idx = {s: i for i, s in enumerate(states)}
imap = np.vectorize(lambda a, b: idx[(int(a), int(b))])(n_ours[..., 0],
                                                         n_ours[..., 1])
cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:len(states)]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
axs[0].imshow(imap, origin="lower", extent=ext, cmap=ListedColormap(cols),
              interpolation="nearest", aspect="auto")
for s, i in idx.items():
    yy, xx = np.nonzero(imap == i)
    if len(xx) > 40:
        axs[0].text(vx[int(np.median(xx))], vy[int(np.median(yy))],
                    "(%d,%d)" % s, ha="center", va="center", fontsize=6.5)
axs[0].set_title("(a) ground state $(n_1, n_2)$", fontsize=9)
axs[1].imshow(1 - gt, origin="lower", extent=ext, cmap="gray",
              interpolation="nearest", aspect="auto")
axs[1].set_title("(b) label: state changes", fontsize=9)
for ax in axs:
    ax.set_xlabel("$V_1$")
axs[0].set_ylabel("$V_2$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "states_and_label.pdf"))
plt.close(fig)

# Fig 2: energy along a cut, the lower envelope, and the sensor signal
t = np.linspace(0, 1, 600)
pA = np.array([x0, y0]) + 0.05 * np.array([x1 - x0, y1 - y0])
pB = np.array([x0, y0]) + 0.95 * np.array([x1 - x0, y1 - y0])
Vc = np.stack([pA[0] + t * (pB[0] - pA[0]), pA[1] + t * (pB[1] - pA[1]),
               np.zeros_like(t)], axis=-1)
Fc = np.stack([energy(n, Vc) for n in CONFIGS], axis=-1)
gs_c = np.argmin(Fc, axis=-1)
used = sorted(set(gs_c))
z_c, _ = sensor_signal(Vc, CONFIGS[gs_c])
fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True,
                        gridspec_kw={"height_ratios": [2.2, 1]})
Fmin = Fc.min(-1)
dF = Fc - Fmin[:, None]              # energy above the ground state
pal = plt.get_cmap("tab10")
for j in range(len(CONFIGS)):
    if dF[:, j].min() > 0.6:
        continue
    on = j in used
    axs[0].plot(t, dF[:, j], lw=1.4 if on else 0.6,
                alpha=1 if on else 0.35,
                color=pal(used.index(j) % 10) if on else "#aaaaaa",
                label="(%d,%d)" % tuple(CONFIGS[j].astype(int)) if on
                else None)
chg = np.nonzero(np.diff(gs_c))[0]
for c in chg:
    for ax in axs:
        ax.axvline(t[c], color="#999999", lw=0.6, ls=":")
axs[0].set_ylim(-0.02, 0.6)
axs[0].set_ylabel("$F(\\mathbf{n};\\mathbf{V}) - F_{\\min}(\\mathbf{V})$")
axs[0].legend(fontsize=6.5, ncol=4, frameon=False, loc="upper left")
axs[1].plot(t, z_c, color="#b03030", lw=1.2)
axs[1].set_ylabel("sensor $z$")
axs[1].set_xlabel("position along the cut (0 = lower left, 1 = upper right)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "energy_cut.pdf"))
plt.close(fig)

# Fig 3: predicted slopes over the ground truth
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.imshow(1 - gt, origin="lower", extent=ext, cmap="gray",
          interpolation="nearest", aspect="auto")
for name, s, col in (("dot 1", s_dot1, "#1f77b4"), ("dot 2", s_dot2,
                     "#d62728"), ("interdot", s_inter, "#2ca02c")):
    P = np.asarray(seg[{"dot 1": "dot1", "dot 2": "dot2",
                        "interdot": "inter"}[name]])
    if len(P) == 0:
        continue
    # drawn over the fitted segment only, a little longer than it
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    half = 0.5 * np.hypot(np.ptp(P[:, 0]), np.ptp(P[:, 1])) + 0.05
    dx = half / np.sqrt(1 + s * s)
    ax.plot([cx - dx, cx + dx], [cy - s * dx, cy + s * dx], color=col, lw=2,
            alpha=0.85, label="%s, Eq. (7)" % name)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.legend(fontsize=6.5, loc="upper right", frameon=True)
ax.set_xlabel("$V_1$")
ax.set_ylabel("$V_2$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "slopes.pdf"))
plt.close(fig)

# Fig 4: the sensor map from Eq. (6); QArray's agrees (see values.tex)
fig, ax = plt.subplots(figsize=(3.8, 3.2))
im = ax.imshow(z_ours, origin="lower", extent=ext, cmap="hot",
               aspect="auto", interpolation="nearest")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _p: "%.1f" % (v * 1e5)))
cb.set_label(r"sensor signal $z$ ($\times 10^{-5}$)")
ax.set_xlabel("$V_1$")
ax.set_ylabel("$V_2$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sensor.pdf"))
plt.close(fig)


# ── values.tex ─────────────────────────────────────────────────────────────
def mat(M, fmt="%.4f"):
    rows = [" & ".join(fmt % v for v in r) for r in np.atleast_2d(M)]
    return "\\begin{pmatrix}" + " \\\\ ".join(rows) + "\\end{pmatrix}"


L = []
def d(name, val):
    L.append("\\newcommand{\\%s}{%s}" % (name, val))


d("devname", "sample\\_10")
d("res", RES)
d("vxlo", "%.3f" % x0); d("vxhi", "%.3f" % x1)
d("vylo", "%.3f" % y0); d("vyhi", "%.3f" % y1)
d("gam", "%g" % GAMMA)
_e = int(np.floor(np.log10(TEMP)))
_m = TEMP / 10.0 ** _e
d("temp", ("10^{%d}" % _e) if abs(_m - 1) < 1e-9 else ("%g\times10^{%d}" % (_m, _e)))
d("npeak", NPEAK)
d("Cddnm", mat(cap["Cdd"]))
d("Cgdnm", mat(cap["Cgd"]))
d("Cdsnm", mat(cap["Cds"]))
d("Cgsnm", mat(cap["Cgs"]))
d("Cmax", mat(C))
d("Cinv", mat(Cinv))
d("Amat", mat(A))
d("Cfull", mat(Cf))
d("ConeOne", "%.4f" % C[0, 0])
d("dOneOne", "%.4f" % cap["Cdd"][0, 0])
d("dOneTwo", "%.4f" % cap["Cdd"][0, 1])
d("gOneSum", "%.4f" % cap["Cgd"][0].sum())
d("sSens", "%.4f" % (cap["Cds"].sum() + cap["Cgs"].sum()))
d("CssFull", "%.4f" % Cf[2, 2])
d("pixV", "(%.4f,\\ %.4f,\\ 0)" % (Vp[0], Vp[1]))
d("pixNg", "(%.4f,\\ %.4f)" % (ng[0], ng[1]))
d("pixN", "(%d,\\ %d)" % tuple(n_star.astype(int)))
tab = []
for a in range(4):
    cells = []
    for b in range(4):
        s = "%.4f" % F_tab[a, b]
        if (a, b) == tuple(n_star.astype(int)):
            s = "\\mathbf{%s}" % s
        cells.append("$%s$" % s)
    tab.append("$n_1=%d$ & " % a + " & ".join(cells) + " \\\\")
d("Ftable", "\n".join(tab))
d("slopeOne", "%.3f" % s_dot1)
d("slopeTwo", "%.3f" % s_dot2)
d("slopeInter", "%.3f" % s_inter)
d("mslopeOne", "%.3f" % m_dot1); d("nOne", k1)
d("pairOne", "(%d,%d)" % key1); d("pairTwo", "(%d,%d)" % key2); d("pairInter", "(%d,%d)" % key3)
d("mslopeTwo", "%.3f" % m_dot2); d("nTwo", k2)
d("mslopeInter", "%.3f" % m_inter); d("nInter", k3)
d("dVone", "%.4f" % dV1)
d("dVtwo", "%.4f" % dV2)
d("EcOne", "%.4f" % Cinv[0, 0])
d("EcTwo", "%.4f" % Cinv[1, 1])
d("Em", "%.4f" % Cinv[0, 1])
d("linefrac", "%.1f" % (100 * line_frac))
d("nstates", len(states))
d("statelist", ", ".join("(%d,%d)" % s for s in states))
_zd = np.abs(z_ours - z_qarray[..., 0]).max()
d("zmaxdiff", "exactly zero" if _zd == 0 else "$%.1e$" % _zd)
d("zlo", "%.2f" % (z_ours.min() * 1e5))
d("zhi", "%.2f" % (z_ours.max() * 1e5))
# how far the sensor sits from resonance: the smallest |F_{k+1} - F_k|
_Ns = np.round(np.einsum("ij,...j->...i", -cgd_full, vg_q)[..., 2:])
_step = []
for k in (-1, 0):
    nf0 = np.concatenate([np.asarray(n_sen), _Ns + k], -1)
    nf1 = np.concatenate([np.asarray(n_sen), _Ns + k + 1], -1)
    q0 = nf0 - np.einsum("ij,...j->...i", -cgd_full, vg_q)
    q1 = nf1 - np.einsum("ij,...j->...i", -cgd_full, vg_q)
    _step.append(np.abs(np.einsum("...i,ij,...j->...", q1, Cfinv, q1)
                        - np.einsum("...i,ij,...j->...", q0, Cfinv, q0)))
d("detmin", "%.3f" % np.min(_step))
d("detminovergam", "%.0f" % (np.min(_step) / GAMMA))
d("npix", RES * RES)
rows = []
for k, v in checks.items():
    rows.append("%s & %s \\\\" % (k.replace("_", "\\_"),
                                   "\\textbf{pass}" if v else
                                   "\\textbf{FAIL}"))
d("checktable", "\n".join(rows))
with open(os.path.join(HERE, "values.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

for k, v in checks.items():
    print("%-48s %s" % (k, "pass" if v else "FAIL"))
print("slopes  predicted %.3f %.3f %.3f   measured %.3f(%d) %.3f(%d) "
      "%.3f(%d)" % (s_dot1, s_dot2, s_inter, m_dot1, k1, m_dot2, k2,
                    m_inter, k3))
print("pixel", pix, "V", Vp, "n*", n_star)
