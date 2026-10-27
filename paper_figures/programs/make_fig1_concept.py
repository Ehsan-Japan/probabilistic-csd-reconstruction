"""
make_fig1_concept.py -- Fig. 1 of Manuscript_ver6: the concept of two-step
localization against generative reconstruction, as a vector schematic
(user's specification, 2026-10-27, second layout).

    python paper_figures/programs/make_fig1_concept.py

One source for every output: this script computes the geometry and writes
the SVG; the PDF and the PNG are that SVG printed by headless Chrome, and
the HTML preview embeds the same SVG. Drawn at its print size, 15.49 cm
(439 pt) wide. Every text is measured with the Arial metrics Chrome uses,
and build() refuses to write a figure in which two texts overlap, a text
runs into a square or a block, or leaves its box.

Banner: the shared goal. Each panel: offline training data (dashed box,
left) and the online pipeline (solid boxes, right).
(a) Generative reconstruction (Ref. 12): sparse grid -> generative model
    -> full estimated CSD -> classical edge/ridge detector -> extracted
    transition-line CSD (one line broken, one spurious fragment).
(b) Two-step localization: sparse 1D rays -> U-Net -> probabilistic
    guidance map, one false branch (Step 1) -> local sweeps (+-2 px) ->
    1D step detection -> extracted transition-line CSD, the false branch
    rejected (Step 2).
The two final squares draw the same honeycomb: the ground state of a
constant-interaction double dot computed here (make_concept_ai.states); no
image is a result of this work or of Ref. 12. Writes
method_docs_ver2/3_methods/figures/fig1_concept.{svg,pdf,png,html}.
"""
import base64
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_concept_ai as C                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures" / "method_docs_ver2" / "3_methods" / "figures"
# --parallel: a variant (user, 2026-10-27) in which the offline rows of
# (a) and (b) are the same four stages -- source (box), raw data (stack),
# labelling (box), labels (image) -- written to fig1_concept_parallel.*;
# the manuscript keeps using fig1_concept.*
PARALLEL = "--parallel" in sys.argv
NAME = "fig1_concept_parallel" if PARALLEL else "fig1_concept"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

W = 439.0                               # pt, the printed width
SLATE, BLUE, ORANGE = "#4A5568", "#1E40AF", "#B45309"
INK, MUTED = "#111827", "#4B5563"
LINE, RED, GREY, GREEN = "#1F2937", "#DC2626", "#9CA3AF", "#15803D"
TRANS = "#E11D48"                       # transition lines on a CSD
FONT = "Arial, Helvetica, sans-serif"
METRIC = {False: ImageFont.truetype("arial.ttf", 100),
          True: ImageFont.truetype("arialbd.ttf", 100)}

# two rows per panel (user, 2026-10-27): row 1 the offline training data,
# stage by stage, across the full width; row 2 the online pipeline
D = 50.0                                # every square of row 2
FLOW_L, FLOW_R = 4.0, 431.0             # the online box(es)
C0, B1, C2 = 44.0, 104.0, 180.0         # input, model block, model output
XI, DI = 24.0, 58.0                     # the input square: larger, further left
B1_W = 54.0                  # equal arrows either side (user)
STEP1_R, STEP2_L = 238.0, 260.0         # (b): 22 pt between the two steps
C3, C4 = 266.0, 354.0                   # sweeps, final
DET = (258.0, 326.0)                    # (a): the line-detection block
XT = (54.0, 156.0, 264.0, 372.0)        # row 1: centre of each stage
XT_W = (94.0, 104.0, 104.0, 104.0)        # row 1: caption widths
HT = 93.0                               # height of row 1
GAP = 10.5                              # between row 1 and row 2
IN, IN_TOP = 4.0, 14.0                  # (b): steps inside the online box
STEP2_TITLE = ["Step 2: Physical verification on device"]
BADGE_Y = 7.0                           # headers sit on the border
SQ = BADGE_Y + 44.0                     # squares: this far below box top
HC = SQ + D + 17 + 7.0 * 1.18 + 7       # height of every pipeline box


def width(s, size, bold=False):
    return METRIC[bold].getlength(s) * size / 100.0


# ---- the geometry, in unit coordinates (u right, v up) -----------------
def geometry(n1=None, n2=None, with_false=True):
    if n1 is None:
        n1, n2 = C.states()
    sig = C.signal(n1, n2)
    polys = C.lines_of(n1, n2)
    pts = np.vstack([np.asarray(p) for p in polys])

    def emptiest(us, vs):
        best, cell = -1.0, None
        for cu in np.linspace(*us, 30):
            for cv in np.linspace(*vs, 30):
                d = np.min(np.linalg.norm(pts - (cu, cv), axis=1))
                if d > best:
                    best, cell = d, np.array((cu, cv))
        return best, cell

    # the false branch of (b): a short segment through the emptiest point
    # of the lower-right cell, clear of every true line
    false = None
    if with_false:
        best, cell = emptiest((0.55, 0.9), (0.08, 0.35))
        half = 0.75 * best * np.array((0.86, 0.5))
        false = [tuple(cell - half), tuple(cell + half)]
        samp = np.linspace(cell - half, cell + half, 11)
        assert np.min(np.linalg.norm(samp[:, None] - pts[None],
                                     axis=2)) > 0.04
    # the spurious fragment in the labels of (a): a short arc in the
    # lower-left cell, away from the broken line
    best, cell = emptiest((0.1, 0.4), (0.12, 0.4))
    t = np.linspace(-0.6, 0.6, 9)
    r = 0.55 * best
    spur = [tuple(cell + r * np.array((np.sin(a), 0.35 * np.cos(a))))
            for a in t]
    # the broken line of (a): the gap in an interdot line (slope > 0, short,
    # inside the square), the middle half of it left out
    cand = []
    for i, p in enumerate(polys):
        p = np.asarray(p)
        for k in range(len(p) - 1):
            d = p[k + 1] - p[k]
            L = np.linalg.norm(d)
            mid = (p[k] + p[k + 1]) / 2
            if d[0] * d[1] > 0 and 0.06 < L < 0.25 and \
                    np.all((mid > 0.25) & (mid < 0.75)):
                cand.append((abs(mid[0] - 0.5) + abs(mid[1] - 0.5), i, k))
    assert cand, "no interdot segment found"
    _, gi, gk = min(cand)
    return sig, polys, false, spur, (gi, gk)


def missed_region(polys, target=(0.30, 0.66), r=0.065):
    """the centre (on a transition line, nearest target) and radius of the
    stretch the network does not predict"""
    pts = []
    for p in polys:
        p = np.asarray(p)
        for k in range(len(p) - 1):
            pts += list(np.linspace(p[k], p[k + 1], 20))
    pts = np.asarray(pts)
    c = pts[np.argmin(np.linalg.norm(pts - np.asarray(target), axis=1))]
    return c, r


def png_uri(a, cmap):
    buf = io.BytesIO()
    plt.imsave(buf, a, cmap=cmap, origin="lower", format="png",
               vmin=float(a.min()), vmax=float(a.max()))
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode()


def line_raster(polys, n=200):
    """the polylines drawn into an n x n array (1 on a line)"""
    a = np.zeros((n, n))
    for p in polys:
        p = np.asarray(p)
        for k in range(len(p) - 1):
            for f in np.linspace(0, 1, 60):
                u, v = p[k] + f * (p[k + 1] - p[k])
                if 0 <= u <= 1 and 0 <= v <= 1:
                    a[int(v * (n - 1)), int(u * (n - 1))] = 1.0
    return a


# ---- SVG ------------------------------------------------------------------
class Svg:
    def __init__(self):
        self.el, self.defs, self.nclip = [], [], 0
        self.texts, self.solid, self.inside = [], [], []

    def add(self, s):
        self.el.append(s)

    def text(self, x, y, s, size=8.0, fill=INK, weight="normal",
             anchor="start", rotate=None, style="normal", box=None):
        bold = weight == "bold"
        w = width(s, size, bold)
        x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        bb = (x0, y - 0.72 * size, x0 + w, y + 0.21 * size)
        if rotate:                                   # the V2 label only
            bb = (x - 0.72 * size, y - w / 2, x + 0.21 * size, y + w / 2)
        self.texts.append((bb, s))
        if box is not None:
            self.inside.append((bb, box, s))
        rot = ' transform="rotate(%g %.2f %.2f)"' % (rotate, x, y) \
            if rotate else ""
        self.add('<text x="%.2f" y="%.2f" font-size="%g" fill="%s" '
                 'font-weight="%s" font-style="%s" text-anchor="%s"%s>%s'
                 '</text>' % (x, y, size, fill, weight, style, anchor, rot,
                              s.replace("&", "&amp;")))

    def lines(self, x, y, rows, size=8.0, fill=INK, weight="normal",
              anchor="middle", lead=1.18, box=None):
        """rows: strings, or (string, weight[, fill]) tuples"""
        for k, r in enumerate(rows):
            s, wt, fl = (r, weight, fill) if isinstance(r, str) else \
                (r[0], r[1], r[2] if len(r) > 2 else fill)
            self.text(x, y + k * size * lead, s, size, fl, wt, anchor,
                      box=box)

    def rrect(self, x, y, w, h, stroke, sw=1.5, r=6, fill="none",
              dash=None):
        self.add('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="%g" '
                 'fill="%s" stroke="%s" stroke-width="%g"%s/>'
                 % (x, y, w, h, r, fill, stroke, sw,
                    ' stroke-dasharray="%s"' % dash if dash else ""))

    def path(self, pts, stroke, sw, opacity=1.0, dash=None, cap="round",
             clip=None):
        d = "M" + " L".join("%.2f %.2f" % p for p in pts)
        self.add('<path d="%s" fill="none" stroke="%s" stroke-width="%g" '
                 'stroke-opacity="%g" stroke-linecap="%s" '
                 'stroke-linejoin="round"%s%s/>'
                 % (d, stroke, sw, opacity, cap,
                    ' stroke-dasharray="%s"' % dash if dash else "",
                    ' clip-path="url(#%s)"' % clip if clip else ""))

    def head(self, x1, y1, ang, colour, s=5.5):
        bx, by = x1 - s * np.cos(ang), y1 - s * np.sin(ang)
        px, py = -np.sin(ang) * s * 0.45, np.cos(ang) * s * 0.45
        self.add('<path d="M%.2f %.2f L%.2f %.2f L%.2f %.2f Z" fill="%s"/>'
                 % (x1, y1, bx + px, by + py, bx - px, by - py, colour))
        return bx, by

    def arrow(self, x0, y0, x1, y1, colour, sw=1.4):
        bx, by = self.head(x1, y1, np.arctan2(y1 - y0, x1 - x0), colour)
        self.path([(x0, y0), (bx, by)], colour, sw, cap="butt")

    def clip_rect(self, x, y, w, h):
        self.nclip += 1
        cid = "c%d" % self.nclip
        self.defs.append('<clipPath id="%s"><rect x="%.2f" y="%.2f" '
                         'width="%.2f" height="%.2f"/></clipPath>'
                         % (cid, x, y, w, h))
        return cid

    def check(self):
        """no two texts overlap, no text in a square or a block, every
        boxed text inside its box"""
        def hit(a, b, pad=0.3):
            return a[0] < b[2] - pad and b[0] < a[2] - pad and \
                a[1] < b[3] - pad and b[1] < a[3] - pad
        bad = []
        for i, (a, s) in enumerate(self.texts):
            for b, t in self.texts[i + 1:]:
                if hit(a, b):
                    bad.append("text/text: %r / %r" % (s, t))
            for b, t in self.solid:
                if hit(a, b) and not any(s == u and b == bx for _, bx, u
                                         in self.inside):
                    bad.append("text/%s: %r" % (t, s))
        for a, b, s in self.inside:
            if a[0] < b[0] or a[2] > b[2] or a[1] < b[1] or a[3] > b[3]:
                bad.append("outside its box: %r" % s)
        assert not bad, "\n".join(bad)

    def render(self, h):
        return ('<svg xmlns="http://www.w3.org/2000/svg" '
                'xmlns:xlink="http://www.w3.org/1999/xlink" '
                'width="%gpt" height="%gpt" viewBox="0 0 %g %g" '
                'font-family="%s">\n<rect width="%g" height="%g" '
                'fill="#FFFFFF"/>\n<defs>%s</defs>\n%s\n</svg>\n'
                % (W, h, W, h, FONT, W, h, "".join(self.defs),
                   "\n".join(self.el)))


def build():
    sig, polys, false, spur, (gi, gk) = geometry()
    MISS = missed_region(polys)
    # the front measured device of (a): its own lines and detector errors
    _s, mpolys, _f, mspur, mgap = geometry(
        *C.states(160, 0.30, 0.20, 2.2, (0.1, 0.4)), with_false=False)
    MEAS_DEV = (mpolys, mspur, mgap)
    lo, hi = float(sig.min()), float(sig.max())
    n = sig.shape[0]
    S = Svg()

    def at(x, y, u, v, d=D):
        return x + d * u, y + d * (1 - v)

    def tile(poly, x, y, d=D):
        return [at(x, y, u, v, d) for u, v in poly]

    def grey(u, v):
        val = sig[min(n - 1, int(v * (n - 1))), min(n - 1, int(u * (n - 1)))]
        g = int(255 * (0.85 - 0.7 * (val - lo) / (hi - lo)))
        return "#%02X%02X%02X" % (g, g, g)

    def frame(x, y, d=D, sw=1.0):
        S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" fill="none" '
              'stroke="#000000" stroke-width="%g"/>' % (x, y, d, d, sw))

    def square(x, y, what, d=D):
        S.solid.append(((x, y, x + d, y + d), what))

    def image(x, y, d, uri):
        S.add('<image x="%.2f" y="%.2f" width="%g" height="%g" '
              'preserveAspectRatio="none" href="%s"/>' % (x, y, d, d, uri))

    def axes(x, y, d=D):
        for k in (0.2, 0.5, 0.8):
            S.path([(x + d * k, y + d), (x + d * k, y + d + 2.5)], INK, 0.6,
                   cap="butt")
            S.path([(x, y + d * k), (x - 2.5, y + d * k)], INK, 0.6,
                   cap="butt")
        S.text(x + d / 2, y + d + 9, "V\u2081", 7, MUTED, anchor="middle",
               style="italic")
        S.text(x - 5.5, y + d / 2, "V\u2082", 7, MUTED, anchor="middle",
               rotate=-90, style="italic")

    def caption(xc, y, rows, box=None):
        S.lines(xc, y + D + 19, rows, 7.5, INK, box=box)

    def block(x, y, w, rows, colour, sizes, h=D - 6):
        S.rrect(x, y + D / 2 - h / 2, w, h, colour, 1.5, 5, "#FFFFFF")
        bx = (x + 2, y + D / 2 - h / 2 + 1, x + w - 2, y + D / 2 + h / 2 - 1)
        S.solid.append((bx, "block"))
        hs = [sz * 1.18 for sz in sizes]
        yy = y + D / 2 - sum(hs) / 2 + 0.72 * sizes[0] + 0.7
        for (s, wt), sz, h in zip(rows, sizes, hs):
            S.text(x + w / 2, yy, s, sz, INK, wt, "middle", box=bx)
            yy += h
        return x, x + w

    def halo_arrow(pts, colour, crossings):
        """an elbow arrow; a white gap is cut where it crosses a border"""
        for (x, y0, y1) in crossings:
            S.add('<rect x="%.2f" y="%.2f" width="5" height="%.2f" '
                  'fill="#FFFFFF"/>' % (x - 2.5, y0, y1 - y0))
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        bx, by = S.head(xb, yb, np.arctan2(yb - ya, xb - xa), colour, 5)
        S.path(pts[:-1] + [(bx, by)], colour, 1.1, cap="butt")

    def stack(x, y, s, draw, n=3, dx=3.0, dy=3.0):
        """n diagrams, the front one (k = 0) bottom-left"""
        for k in range(n - 1, -1, -1):
            x0, y0 = x + dx * k, y + dy * (n - 1 - k)
            S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" '
                  'fill="#FFFFFF"/>' % (x0, y0, s, s))
            draw(k, x0, y0)
            S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" '
                  'fill="none" stroke="#6B7280" stroke-width="0.6"/>'
                  % (x0, y0, s, s))

    def honeycomb(x, y, d, colour, sw, pl=None):
        cid = S.clip_rect(x, y, d, d)
        for p in (polys if pl is None else pl):
            S.path(tile(p, x, y, d), colour, sw, clip=cid)

    def jagged(x, y, d, colour, sw, pl=None):
        """the honeycomb as a classical detector draws it from an image:
        each line resampled, shifted sideways at random (a ragged, saw-like
        edge) and drawn in short pieces of varying width"""
        rng = np.random.default_rng(11)
        cid = S.clip_rect(x, y, d, d)
        for p in (polys if pl is None else pl):
            p = np.asarray(p)
            L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0),
                                                  axis=1)])
            s = np.arange(0, L[-1], 0.018)
            q = np.c_[np.interp(s, L, p[:, 0]), np.interp(s, L, p[:, 1])]
            t = np.gradient(q, axis=0)
            nrm = np.c_[-t[:, 1], t[:, 0]]
            nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-12
            off = 0.0045 * rng.standard_normal(len(q)) \
                + 0.0035 * (-1.0) ** np.arange(len(q))
            q = q + off[:, None] * nrm
            for k in range(0, len(q) - 1, 3):
                S.path(tile(q[k:k + 4], x, y, d), colour,
                       sw * rng.uniform(0.5, 1.6), clip=cid)

    def fragments(x, y, d, colour, sw, pl=None):
        """the lines as a local detector returns them from an estimated
        image: broken into pieces of random length with gaps between them,
        each piece slightly shifted and tilted, of varying width"""
        rng = np.random.default_rng(5)
        cid = S.clip_rect(x, y, d, d)
        for p in (polys if pl is None else pl):
            p = np.asarray(p)
            L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0),
                                                  axis=1)])
            s0 = rng.uniform(0, 0.04)
            while s0 < L[-1]:
                s1 = min(L[-1], s0 + rng.uniform(0.04, 0.22))
                s = np.linspace(s0, s1, 6)
                q = np.c_[np.interp(s, L, p[:, 0]), np.interp(s, L, p[:, 1])]
                t = q[-1] - q[0]
                nrm = np.array((-t[1], t[0])) / (np.linalg.norm(t) + 1e-12)
                shift = rng.normal(0, 0.02)
                tilt = rng.normal(0, 0.18)
                f = np.linspace(-0.5, 0.5, len(q))[:, None]
                q = q + (shift + tilt * f * np.linalg.norm(t)) * nrm
                S.path(tile(q, x, y, d), colour, sw * rng.uniform(0.6, 1.7),
                       cap="butt", clip=cid)
                s0 = s1 + rng.uniform(0.015, 0.07)

    def errors(x, y, d, sw, colour=LINE, ring=RED, rough=False,
               dev=None):
        """the honeycomb as a line detector returns it: the interdot line
        (in both the n1 and the n2 contours) cut by a mask, a spurious arc
        added; both ringed in red"""
        pl, sp, (i0, k0) = dev if dev else (polys, spur, (gi, gk))
        a, b = np.asarray(pl[i0][k0]), np.asarray(pl[i0][k0 + 1])
        gx_, gy_ = at(x, y, *((a + b) / 2), d)
        gr = 0.25 * d * np.linalg.norm(b - a)
        S.defs.append('<mask id="gap%d" maskUnits="userSpaceOnUse"><rect '
                      'x="0" y="0" width="%g" height="2000" fill="#FFFFFF"/>'
                      '<circle cx="%.2f" cy="%.2f" r="%.2f" fill="#000000"/>'
                      '</mask>' % (S.nclip, W, gx_, gy_, gr))
        S.add('<g mask="url(#gap%d)">' % S.nclip)
        if rough == "fragments":
            fragments(x, y, d, colour, sw, pl)
        elif rough:
            jagged(x, y, d, colour, sw, pl)
        else:
            honeycomb(x, y, d, colour, sw, pl)
        S.add('</g>')
        cid = S.clip_rect(x, y, d, d)
        S.path(tile(sp, x, y, d), colour, sw, clip=cid)
        sx, sy = at(x, y, *np.mean(sp, axis=0), d)
        for cx, cy, r in ((gx_, gy_, 1.5 * gr), (sx, sy, 0.17 * d)):
            S.add('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="none" '
                  'stroke="%s" stroke-width="0.7" stroke-dasharray="1.4 1"/>'
                  % (cx, cy, r, ring))

    def tab(x, ytop, s, size, colour):
        w = width(s, size, True)
        S.add('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
              'fill="#FFFFFF"/>' % (x - 3, ytop - size * 0.6, w + 6,
                                    size * 1.2))
        S.text(x, ytop + 0.36 * size, s, size, colour, "bold")

    def wrap(s, size, bold, maxw):
        """greedy word wrap to maxw points; "\n" forces a break"""
        if "\n" in s:
            return [ln for part in s.split("\n")
                    for ln in wrap(part, size, bold, maxw)]
        out, cur = [], ""
        for wd in s.split(" "):
            t = (cur + " " + wd).strip()
            if cur and width(t, size, bold) > maxw:
                out.append(cur)
                cur = wd
            else:
                cur = t
        out.append(cur)
        return out

    def textrows(xc, y, rows, maxw, box, big=7.0, small=6.4):
        """rows: (text, weight[, colour]); the first row bold at `big`,
        the rest at `small`; each wrapped to maxw; returns the last
        baseline"""
        rows = [r for r in rows if not (len(r) > 2 and r[2] == MUTED)]
        yy, last = y, y - 9            # nothing left: a note starts at y
        for i, r in enumerate(rows):
            s, wt = r[0], r[1]
            fl = r[2] if len(r) > 2 else INK
            sz = big if wt == "bold" else small
            for ln in wrap(s, sz, wt == "bold", maxw):
                S.text(xc, yy, ln, sz, fl, wt, "middle", box=box)
                last = yy
                yy += sz * 1.18
        return last

    def badge(top, rows, colour, right=FLOW_R):
        xc = C4 + D / 2
        bw = 2 * (right - 4 - xc)
        y0, y1 = top + BADGE_Y, top + SQ - 7
        S.rrect(xc - bw / 2, y0, bw, y1 - y0, colour, 1.0, 4, "#FFFFFF")
        S.add('<path d="M%.2f %.2f L%.2f %.2f L%.2f %.2f Z" fill="%s"/>'
              % (xc - 3.5, y1, xc + 3.5, y1, xc, y1 + 4, colour))
        bx = (xc - bw / 2 + 1.5, y0 + 1, xc + bw / 2 - 1.5, y1 - 1)
        lines = [(ln, wt) for s, wt in rows
                 for ln in wrap(s, 6.8, wt == "bold", bw - 7)]
        lead = 6.8 * 1.18
        yy = (y0 + y1) / 2 - lead * (len(lines) - 1) / 2 + 2.5
        for ln, wt in lines:
            S.text(xc, yy, ln, 6.8, colour, wt, "middle", box=bx)
            yy += lead

    def note(xc, y, s, good, box, maxw):
        """one advantage (check) or cost (cross), centred on xc; returns
        the last baseline"""
        lines = wrap(s, 6.6, False, maxw - 9)
        w = max(width(ln, 6.6) for ln in lines)
        x = xc - (w + 9) / 2
        if good:
            S.path([(x, y - 3), (x + 2.2, y - 0.6), (x + 6, y - 6)], GREEN,
                   1.3)
        else:
            for sg in (1, -1):
                S.path([(x + 0.5, y - 3 - 2.6 * sg),
                        (x + 5.5, y - 3 + 2.6 * sg)], RED, 1.2)
        for k, ln in enumerate(lines):
            S.text(x + 9, y + 7.8 * k, ln, 6.6, INK, box=box)
        return y + 7.8 * (len(lines) - 1)

    def offline(top, colour, slots, good):
        """row 1: how the training data are made, stage by stage; the box
        grows with its captions"""
        bx = (6, top + 2, W - 6, top + 400)
        # a caption marked "above" stands over its image; the row moves
        # down to make room for the longest such caption
        above = [len([ln for r in c for ln in wrap(r[0], 7.0, True, mw)
                      if r[1] == "bold"])
                 for mw, (_k, _a, c, _n, *f) in zip(XT_W, slots) if f]
        lift = (max(above) * 7.0 * 1.18 + 3) if above else 0.0
        y, ext, low = top + 9 + lift, [], 0.0
        for xc, mw, (kind, arg, cap, nt, *flag) in zip(XT, XT_W, slots):
            if kind == "stack":                       # a few diagrams
                stack(xc - 22, y, 38, arg)
                ext.append((xc - 22, xc + 22))
            elif kind == "many":                      # as many as wanted
                n = len(SIMS)
                w = 34 + 2.6 * (n - 1)
                stack(xc - w / 2, y - 2, 34, arg, n, 2.6, 1.4)
                ext.append((xc - w / 2, xc + w / 2))
            elif kind == "image":
                arg(xc - 22, y, 44)
                ext.append((xc - 22, xc + 22))
            else:                                     # a process block
                rows, sizes = arg
                block(xc - 38, y + 22 - D / 2, 76, rows, colour, sizes,
                      h=38)
                ext.append((xc - 38, xc + 38))
            if flag:                      # caption above, note below
                textrows(xc, top + 9 + 7.0 * 0.85, cap, mw, bx)
                last = y + 53 - 9
            else:
                last = textrows(xc, y + 53, cap, mw, bx)
            low = max(low, note(xc, last + 9, nt, good, bx, mw))
        for (_, a), (b, _) in zip(ext[:-1], ext[1:]):
            S.arrow(a + 3, y + 22, b - 3, y + 22, colour, 1.2)
        h = low + 5 - top
        S.rrect(4, top, W - 8, h, colour, 1.1, 6, dash="4 2.5")
        tab(14, top, "Offline: training data preparation", 8.0, colour)
        return top + h

    def trains(bottom, top, colour):
        """from the training data (row 1) down into the model block"""
        xc = B1 + B1_W / 2
        S.add('<rect x="%.2f" y="%.2f" width="6" height="4" fill="#FFFFFF"/>'
              % (xc - 3, top - 2))
        y1 = top + SQ + 3
        bx, by = S.head(xc, y1, np.pi / 2, colour, 5)
        S.path([(xc, bottom), (bx, by)], colour, 1.3, cap="butt")
        S.text(xc + 4, bottom + 8.6, "trains", 6.5, colour)

    def sweeps(x, y):
        """one sweep every ~0.13 along each line; a line shared by the n1
        and n2 contours is swept once"""
        cid = S.clip_rect(x, y, D, D)
        kept = []
        for p in [np.array(false)] + polys:
            p = np.asarray(p)
            L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0),
                                                  axis=1)])
            step = 0.11 if p.shape[0] > 2 else L[-1] / 3.2
            for s in np.arange(step / 2.6, L[-1], step):
                k = int(max(0, min(np.searchsorted(L, s) - 1, len(p) - 2)))
                f = (s - L[k]) / max(L[k + 1] - L[k], 1e-9)
                q = p[k] + f * (p[k + 1] - p[k])
                if np.linalg.norm(q - MISS[0]) < MISS[1] + 0.04:
                    continue
                if 0.0 <= q[0] <= 1.0 and 0.03 <= q[1] <= 0.97 and \
                        all(np.linalg.norm(q - o) > 0.07 for o in kept):
                    kept.append(q)
        for u, v in kept:
            S.path(tile([(u - 0.08, v), (u + 0.08, v)], x, y), ORANGE, 1.4,
                   cap="butt", clip=cid)

    def labels_with_errors(x, y, d):
        image(x, y, d, MEAS[0])
        errors(x, y, d, 1.0, TRANS, "#FFFFFF", rough=True, dev=MEAS_DEV)
        S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" fill="none" '
              'stroke="#6B7280" stroke-width="0.6"/>' % (x, y, d, d))

    def binary(x, y, d):
        """the exact label: clean lines where the charge state changes"""
        S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" '
              'fill="#FFFFFF"/>' % (x, y, d, d))
        honeycomb(x, y, d, TRANS if PARALLEL else "#000000", 0.9)
        S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" fill="none" '
              'stroke="#6B7280" stroke-width="0.6"/>' % (x, y, d, d))

    def fit(x, y, s, size, colour, weight, maxw, anchor="start"):
        """one line, shrunk until it fits maxw"""
        while width(s, size, weight == "bold") > maxw:
            size -= 0.25
        S.text(x, y, s, size, colour, weight, anchor)

    # ------------------------------------------------------------ banner
    # (user, 2026-10-27) wording from "Figure 1 Exact Text Replacement
    # Reference.pdf", right-hand column
    S.rrect(4, 4, W - 8, 21, INK, 1.0, 4)
    lab = "Shared objective: "
    goal = ("reconstruct the charge transition-line CSD from sparse "
            "measurements, bypassing time-consuming 2D raster scans")
    sz = 8.5
    while width(lab, sz, True) + width(goal, sz) > W - 20:
        sz -= 0.25
    w0, w1 = width(lab, sz, True), width(goal, sz)
    x = W / 2 - (w0 + w1) / 2
    S.text(x, 17.8, lab, sz, INK, "bold", box=(6, 5, W - 6, 24))
    S.text(x + w0, 17.8, goal, sz, INK, box=(6, 5, W - 6, 24))

    # ---------------------------------------------------------------- (a)
    fit(6, 41, "(a) Conventional CSD reconstruction", 10.5, SLATE, "bold",
        W - 12)
    meas = lambda k, x0, y0: image(x0, y0, 38, MEAS[k])
    slots_a = [
        ("stack", meas, [("Experimental CSD dataset", "bold"),
                         ("acquired from fabricated devices", "normal",
                          MUTED)],
         "bounded by measurement time"),
        ("block", ([("Manual", "bold"), ("curation", "bold"),
                    ("(expert classification)", "normal")],
                   [7.5, 7.5, 6.2]),
         [("subset manually screened for quality", "normal", MUTED)],
         "requires manual expert curation"),
        ("block", ([("2D edge / ridge", "bold"), ("filters", "bold"),
                    ("(Canny, Frangi)", "normal")], [7.5, 7.5, 6.5]),
         [("heuristic extraction from measured CSDs", "normal", MUTED)],
         "no independent ground truth"),
        ("image", labels_with_errors,
         [("Detector-derived ground-truth labels", "bold"),
          ("filter artifacts ringed", "normal", MUTED)],
         "labels inherit detector errors")]
    if PARALLEL:
        slots_a = [
            ("block", ([("Experimental 2D", "bold"), ("raster scans", "bold"),
                        ("(fabricated devices)", "normal")],
                       [7.5, 7.5, 6.2]),
             [], "time-consuming acquisition"),
            ("stack", meas, [("Experimental CSD dataset", "bold")],
             "bounded by measurement time"),
            # (user, 2026-10-27) "annotation" as in (b)
            ("block", ([("Manual curation", "bold"),
                        ("and annotation", "bold"),
                        ("(classical filters:", "normal"),
                        ("Canny, Frangi)", "normal")],
                       [7.2, 7.2, 6.2, 6.2]),
             [], "lacks ground truth"),
            ("image", labels_with_errors,
             [("Detector-derived ground-truth labels", "bold")],
             "labels inherit detector errors")]
    bottom = offline(48.5, SLATE, slots_a, False)
    top = bottom + GAP
    S.rrect(FLOW_L, top, FLOW_R - FLOW_L, HC, SLATE)
    fb = (FLOW_L + 2, top + 2, FLOW_R - 2, top + HC - 2)
    tab(FLOW_L + 10, top, "Online reconstruction", 8.0, SLATE)
    trains(bottom, top, SLATE)
    y = top + SQ
    cy = y + D + 17                     # first baseline under a square
    yi = y + D / 2 - DI / 2             # the input square, centred
    # 1 sparse grid
    for i in range(9):
        for j in range(9):
            u, v = (i + 0.5) / 9, (j + 0.5) / 9
            S.add('<circle cx="%.2f" cy="%.2f" r="1.4" fill="%s"/>'
                  % (*at(XI, yi, u, v, DI), grey(u, v)))
    frame(XI, yi, DI)
    square(XI, yi, "square", DI)
    axes(XI, yi, DI)
    textrows(XI + DI / 2, yi + DI + 17, [("Sparse 2D measurements",
                                           "bold"),
                              ("uniform grid or 1D cuts on device",
                               "normal", MUTED)], 96, fb)
    # 2 the model
    block(B1, y, B1_W, [("Conditional", "bold"), ("generative", "bold"),
                        ("model", "bold")], SLATE, [7.2, 7.2, 7.2])
    # 3 full estimate
    image(C2, y, D, EST)
    frame(C2, y)
    square(C2, y, "square")
    textrows(C2 + D / 2, cy, [("Reconstructed full-plane CSD", "bold"),
                              ("estimates continuous signal at every pixel",
                               "normal", MUTED)], 118, fb)
    # 4 classical edge / ridge detection
    dx0, dx1 = block(DET[0], y, DET[1] - DET[0],
                     [("Classical 2D", "bold"), ("edge / ridge", "bold"),
                      ("detection", "bold"), ("(Canny & Frangi", "normal"),
                      ("filters)", "normal")],
                     SLATE, [7.0, 7.0, 7.0, 6.3, 6.3], h=D + 8)
    # 5 the same CSD as in (b), its lines in red: one broken, one
    # spurious, as a detector returns them from an estimate
    # (user, 2026-10-27) white background; predicted lines red, the ground
    # truth thin black on top, so the misses show
    S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" fill="#FFFFFF"/>'
          % (C4, y, D, D))
    errors(C4, y, D, 1.4, TRANS, INK, rough="fragments")
    honeycomb(C4, y, D, "#000000", 0.55)
    frame(C4, y)
    square(C4, y, "square")
    textrows(C4 + D / 2, cy, [("Estimated vs real\ntransition lines",
                               "bold"),
                              ("propagates synthesis & filter errors",
                               "normal", MUTED)], 96, fb)
    ym = y + D / 2
    for x0, x1 in ((XI + DI, B1), (B1 + B1_W, C2), (C2 + D, dx0),
                   (dx1, C4)):
        S.arrow(x0 + 2.5, ym, x1 - 2.5, ym, SLATE)
    badge(top, [("Transition lines inferred in AI-synthesized image",
                 "bold"),
                ("via classical 2D filters (error-prone)", "normal")],
          SLATE)

    # ---------------------------------------------------------------- (b)
    head = top + HC + 18
    fit(6, head, "(b) Simulation-guided two-step localization (this work)",
        10.5, BLUE, "bold", W - 12)
    sim = lambda k, x0, y0: image(x0, y0, 34, SIMS[k])
    bottom = offline(head + 7.5, BLUE, [
        ("block", ([("Fast QD", "bold"), ("simulator", "bold"),
                    ("(random capacitance", "normal"),
                    ("sampling)", "normal")], [7.2, 7.2, 6.2, 6.2]),
         [("constant-capacitance solver (QArray)", "normal", MUTED)],
         "negligible compute cost" if PARALLEL else
         "negligible compute cost (~ms/diagram)"),
        ("many", sim, [("Simulated CSD dataset", "bold"),
                       ("unbounded synthetic sample size", "normal",
                        MUTED)],
         "zero experimental acquisition"),
        ("block", ([("Exact charge-state", "bold"),
                    ("boundaries (n₁, n₂)", "bold")], [7.2, 7.2]),
         [("derived directly from integer occupation", "normal", MUTED)],
         "zero manual annotation"),
        # the reference PDF is cut off below this row's first line; the
        # second line and the note are kept from the previous version
        ("image", binary, [("Exact ground-truth labels", "bold"),
                           ("black = transition line", "normal", MUTED)],
         "labels are exact by construction" if PARALLEL else
         "exact by construction")], True)
    top0 = bottom + GAP
    S.rrect(FLOW_L, top0, FLOW_R - FLOW_L, HC + IN_TOP + IN, SLATE)
    tab(FLOW_L + 10, top0, "Online reconstruction", 8.0, SLATE)
    top = top0 + IN_TOP
    l1, r2 = FLOW_L + IN, FLOW_R - IN
    S.rrect(l1, top, STEP1_R - l1, HC, BLUE)
    S.rrect(STEP2_L, top, r2 - STEP2_L, HC, ORANGE)
    s1 = (l1 + 2, top + 2, STEP1_R - 2, top + HC - 2)
    s2 = (STEP2_L + 2, top + 2, r2 - 2, top + HC - 2)
    tab(l1 + 7, top, "Step 1: Measurement guidance", 7.3, BLUE)
    tab(STEP2_L + 8, top, STEP2_TITLE[0], 7.6, ORANGE)
    S.add('<rect x="%.2f" y="%.2f" width="6" height="4" fill="#FFFFFF"/>'
          % (B1 + B1_W / 2 - 3, top0 - 2))
    trains(bottom, top, BLUE)
    y = top + SQ
    cy = y + D + 17
    yi = y + D / 2 - DI / 2
    # 1 rays from the maximum-voltage corner, each to the V1 or V2 axis;
    # points start a little away from the corner and are evenly spaced,
    # so that the corner is not crowded
    for k in range(1, 8):
        th = np.radians(90 * k / 8)
        reach = min(1 / np.cos(th), 1 / np.sin(th)) * 0.985
        for r in np.arange(0.24, reach + 1e-9, 0.105):
            u, v = 1 - r * np.cos(th), 1 - r * np.sin(th)
            if 0 <= u <= 1 and 0 <= v <= 1:
                S.add('<circle cx="%.2f" cy="%.2f" r="1.3" fill="%s"/>'
                      % (*at(XI, yi, u, v, DI), grey(u, v)))
    frame(XI, yi, DI)
    square(XI, yi, "square", DI)
    axes(XI, yi, DI)
    textrows(XI + DI / 2, yi + DI + 17, [("Sparse 1D ray sweeps",
                                           "bold"),
                              ("radial voltage sweeps across (V₁, "
                               "V₂)", "normal", MUTED)], 80, s1)
    # 2 the model
    block(B1, y, B1_W, [("U-Net", "bold"), ("segmentation", "bold"),
                        ("model", "bold")], BLUE, [7.2, 7.2, 7.2])
    # 3 guidance map: blue probability bands, one of them false
    image(C2, y, D, GUIDE)
    frame(C2, y)
    square(C2, y, "square")
    # (user, 2026-10-27) centred under its square, no hyphen
    textrows(C2 + D / 2, cy, [("Transition line\nprobability map"
                               if PARALLEL else
                               "Transition line probability map", "bold"),
                         ("predicts candidate transition-line regions",
                          "normal", MUTED),
                         ("(guides where to measure)", "normal", MUTED)],
             88, s1)
    # 4 local sweeps across the candidates, the false one included
    image(C3, y, D, GUIDE_FAINT)
    sweeps(C3, y)
    frame(C3, y)
    square(C3, y, "square")
    textrows(C3 + D / 2, cy, [("Targeted local sweeps", "bold"),
                         ("short sweeps across candidate regions",
                          "normal", MUTED)], 66, s2)
    # 1D step detection, between columns 4 and 5
    ym = y + D / 2
    gx = C3 + D + 6
    S.path([(gx, ym - 8), (gx + 11, ym - 8.5), (gx + 12, ym - 17),
            (gx + 25, ym - 17.5)], ORANGE, 1.3)
    S.arrow(C3 + D + 2.5, ym, C4 - 2.5, ym, ORANGE)
    xm = (C3 + D + C4) / 2
    # (user, 2026-10-27) no symbols (w, r_k) in the figure
    S.lines(xm, ym + 9, ["1D step", "detection"], 6.5, ORANGE)
    # 5 the same CSD as in (a), its lines in red and exact; the false
    # branch, dashed in the candidate colour, rejected (black x)
    S.add('<rect x="%.2f" y="%.2f" width="%g" height="%g" fill="#FFFFFF"/>'
          % (C4, y, D, D))
    mx_, my_ = at(C4, y, *MISS[0])
    S.defs.append('<mask id="miss" maskUnits="userSpaceOnUse"><rect x="0" '
                  'y="0" width="%g" height="3000" fill="#FFFFFF"/><circle '
                  'cx="%.2f" cy="%.2f" r="%.2f" fill="#000000"/></mask>'
                  % (W, mx_, my_, MISS[1] * D))
    S.add('<g mask="url(#miss)">')
    honeycomb(C4, y, D, TRANS, 1.8)
    S.add('</g>')
    honeycomb(C4, y, D, "#000000", 0.55)
    cid = S.clip_rect(C4, y, D, D)
    S.path(tile(false, C4, y), "#2563EB", 1.1, dash="2 1.6", clip=cid)
    mx, my = at(C4, y, *np.mean(false, axis=0))
    for sgn in (1, -1):
        S.path([(mx - 3, my - 3 * sgn), (mx + 3, my + 3 * sgn)], INK, 1.5)
    frame(C4, y)
    square(C4, y, "square")
    textrows(C4 + D / 2, cy, [("Estimated vs real\ntransition lines",
                               "bold"),
                              ("physically verified on device",
                               "normal", MUTED),
                              ("false candidate × rejected;",
                               "normal", MUTED),
                              ("unpredicted parts unverified",
                               "normal", MUTED)], 88, s2)
    for x0, x1 in ((XI + DI, B1), (B1 + B1_W, C2)):
        S.arrow(x0 + 2.5, ym, x1 - 2.5, ym, BLUE)
    # from Step 1 to Step 2: border to border, touching neither
    S.arrow(STEP1_R + 2.5, ym, STEP2_L - 2.5, ym, INK, 1.6)
    badge(top, right=FLOW_R - IN, rows=[
        ("Transition lines verified directly by the device's own charge "
                 "sensor", "bold"), ("(physical ground truth)", "normal")],
          colour=ORANGE)

    S.check()
    return S.render(top + HC + IN + 4)


def chrome(svg_path, html_path, pdf_path, png_path, h):
    svg = svg_path.read_text(encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "@page{size:%gpt %gpt;margin:0}html,body{margin:0;padding:0;"
        "background:#fff}svg{display:block}</style></head><body>%s"
        "</body></html>" % (W, h, svg), encoding="utf-8")
    url = html_path.resolve().as_uri()
    subprocess.run([CHROME, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer", "--print-to-pdf=%s" % pdf_path,
                    url], check=True, capture_output=True)
    wpx, hpx = int(np.ceil(W * 96 / 72)), int(np.ceil(h * 96 / 72))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu",
                    "--hide-scrollbars", "--force-device-scale-factor=5",
                    "--window-size=%d,%d" % (wpx, hpx),
                    "--screenshot=%s" % png_path, url], check=True,
                   capture_output=True)


def main():
    global EST, MEAS, SIMS, GUIDE, GUIDE_FAINT
    OUT.mkdir(parents=True, exist_ok=True)
    sig, polys, false, _s, _g = geometry()
    EST = png_uri(gaussian_filter(sig, 7.0), "gray")
    # three devices: measured (noisy, drifting) for (a), simulated for (b)
    MEAS, SIMS = [], []
    rng = np.random.default_rng(7)
    for c, a, sh in ((0.30, 0.20, (0.1, 0.4)), (0.42, 0.25, (0.5, 0.1)),
                     (0.36, 0.18, (0.2, 0.2))):
        m1, m2 = C.states(160, c, a, 2.2, sh)
        clean = C.signal(m1, m2)
        span = clean.max() - clean.min()
        drift = np.cumsum(rng.standard_normal(clean.shape[0]))[:, None]
        noisy = clean + 0.08 * span * rng.standard_normal(clean.shape) \
            + 0.02 * span * drift / np.abs(drift).max()
        MEAS.append(png_uri(gaussian_filter(noisy, 0.8), "gray"))
    # ten simulated devices: one flat level per charge state, sharp edges,
    # no noise and no drift -- clearly computed, not measured
    for k in range(10):
        c, a = 0.30 + 0.014 * k, 0.17 + 0.009 * (k % 5)
        s1, s2 = C.states(120, c, a, 2.1 + 0.05 * (k % 3),
                          (0.1 + 0.05 * k, 0.45 - 0.035 * k))
        lvl = (0.37 * s1 + 0.61 * s2) % 1.0
        flat = LinearSegmentedColormap.from_list("fl", ["#5A5A5A", "#F2F2F2"])
        SIMS.append(png_uri(lvl, flat))
    SIMS.reverse()          # the front diagram (k = 0) is the last made
    # the guidance map: true lines and the false branch, blurred into bands
    (mu, mv), mr = missed_region(polys)
    raster = line_raster(polys)
    gv, gu = np.mgrid[0:raster.shape[0], 0:raster.shape[1]] / (
        raster.shape[0] - 1)
    raster[np.hypot(gu - mu, gv - mv) < mr] = 0.0
    true = gaussian_filter(raster, 4.5)
    fal = gaussian_filter(line_raster([false]), 4.5)
    g = np.clip(np.maximum(true, fal) / (0.5 * true.max()), 0, 1) ** 0.55
    blues = LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", "#2563EB"])
    GUIDE = png_uri(np.clip(g, 0, 1), blues)
    faint = LinearSegmentedColormap.from_list("w2f", ["#FFFFFF", "#A9C2F5"])
    GUIDE_FAINT = png_uri(np.clip(g, 0, 1), faint)
    global SIMCS, BINARY, FINAL_BG
    SIMCS = png_uri(sig, "gray")
    rng = np.random.default_rng(3)
    noisy = sig + 0.035 * (sig.max() - sig.min()) * rng.standard_normal(
        sig.shape)
    light = LinearSegmentedColormap.from_list("lg", ["#7A7A7A", "#F4F4F4"])
    FINAL_BG = png_uri(noisy, light)
    # the label: 1 where the charge state changes to the next pixel
    n1, n2 = C.states(40)
    edge = np.zeros(n1.shape, bool)
    for f in (n1, n2):
        edge[:-1] |= f[:-1] != f[1:]
        edge[:, :-1] |= f[:, :-1] != f[:, 1:]
    BINARY = png_uri(np.where(edge, 0.0, 1.0), "gray")
    svg_text = build()
    h = float(svg_text.split('height="')[1].split("pt")[0])
    svg = OUT / (NAME + ".svg")
    svg.write_text(svg_text, encoding="utf-8")
    chrome(svg, OUT / (NAME + ".html"), OUT / (NAME + ".pdf"),
           OUT / (NAME + ".png"), h)
    for ext in ("svg", "pdf", "png", "html"):
        f = OUT / ("%s.%s" % (NAME, ext))
        print("%-4s %s %s" % (ext, "ok" if f.exists() else "MISSING", f))


if __name__ == "__main__":
    main()
