"""
make_concept_ai.py -- Fig. 1 of Manuscript_ver6 as a schematic (user,
2026-10-27): the concept, not results. Illustrator figure, driven over COM
with an ExtendScript, as make_pipeline_comparison_ai.py.

    python paper_figures/programs/make_concept_ai.py

(a) Generative reconstruction: a model trained on measured, manually
    annotated diagrams turns a sparse measurement into an estimate of the
    charge-sensor signal at every pixel; a detector then finds the lines in
    that estimate -- the lines are decided in an estimated image.
(b) Two-step localization: a U-Net trained on simulated diagrams with exact
    labels turns a few rays into the places where lines lie (Step 1); local
    sweeps on the device across each candidate keep a line only where the
    charge-sensor signal steps (Step 2) -- the lines are decided by the
    device's own charge sensor. One candidate is false and is rejected.

No image is a result of this work or of Ref. 12: the honeycomb is a
two-dot constant-interaction ground state computed here, the "signal" a
smooth function of its charge state. The two rows use the same columns, so
that what each method predicts sits one above the other. Writes
method_docs_ver2/3_methods/figures/fig_concept_ai.{ai,pdf,png}.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_pixel_labelling_ai as base                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures" / "method_docs_ver2" / "3_methods" / "figures"
TILES = OUT / "concept_tiles"
NAME = "fig_concept_ai"

W, T, BW = 439.0, 48.0, 50.0          # page width (15.49 cm), tile, box
ROW_H = 158.0
INK, MUT = "#222222", "#666666"
GEN_C, GEN_BG = "#555555", "#F2F2F2"
PRED_C, PRED_BG = "#3F51B5", "#EEF0FA"
VER_C, VER_BG = "#B26A00", "#FBF3E6"
BAND_C, LINE_C, SWEEP_C = "#A9C9E8", "#0B2F6B", "#E08A00"
# columns, shared by both rows: input, network, its output, step 2 /
# detector, lines, verdict
XC = dict(inp=6, net=70, out=136, mid=208, fin=298, verdict=352)
rgb = base.rgb


# ---- the schematic honeycomb ------------------------------------------
def states(n=360, c=0.38, a=0.22, span=2.1, shift=(0.55, 0.45)):
    """Ground-state (n1, n2) of a constant-interaction double dot on the
    unit square, (u, v) -> (V1, V2) = span * (u, v) + shift."""
    u = np.linspace(0, 1, n)
    V1, V2 = np.meshgrid(span * u + shift[0], span * u + shift[1])
    best, n1, n2 = None, None, None
    for i in range(6):
        for j in range(6):
            E = 0.5 * (i * i + j * j) + c * i * j - i * (V1 + a * V2) \
                - j * (V2 + a * V1)
            if best is None:
                best, n1, n2 = E, np.full(E.shape, i), np.full(E.shape, j)
            else:
                m = E < best
                best = np.where(m, E, best)
                n1, n2 = np.where(m, i, n1), np.where(m, j, n2)
    return n1, n2


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    d = np.abs(np.cross(b - a, pts - a)) / (np.linalg.norm(b - a) + 1e-12)
    k = int(np.argmax(d))
    if d[k] > eps:
        return np.vstack([_rdp(pts[:k + 1], eps)[:-1], _rdp(pts[k:], eps)])
    return np.vstack([a, b])


def lines_of(n1, n2):
    """The transition lines as polylines in unit coordinates (u right,
    v up): the contours where n1 or n2 changes."""
    n = n1.shape[0]
    fig = plt.figure()
    polys = []
    for f in (n1, n2):
        cs = plt.contour(np.linspace(0, 1, n), np.linspace(0, 1, n), f,
                         levels=np.arange(0.5, 6, 1.0))
        for segs in cs.allsegs:
            for s in segs:
                if len(s) > 2:
                    polys.append(_rdp(np.asarray(s), 0.004))
    plt.close(fig)
    return polys


def signal(n1, n2):
    """A schematic charge-sensor signal: one level per charge state on a
    smooth background, blurred."""
    n = n1.shape[0]
    u = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, u)
    s = 0.55 * np.cos(1.3 * n1 + 0.8 * n2) + 0.9 * (U + V) / 2
    return gaussian_filter(s, 2.0)


def make_tiles():
    TILES.mkdir(parents=True, exist_ok=True)
    n1, n2 = states()
    sig = signal(n1, n2)
    est = gaussian_filter(sig, 6.0)          # an estimate: smoothed
    plt.imsave(TILES / "estimate.png", est, cmap="gray", origin="lower")
    for k, (c, a, sh) in enumerate(((0.30, 0.20, (0.1, 0.4)),
                                    (0.42, 0.25, (0.5, 0.1)),
                                    (0.36, 0.18, (0.2, 0.2)))):
        m1, m2 = states(220, c, a, 2.2, sh)
        plt.imsave(TILES / f"measured_{k}.png", signal(m1, m2), cmap="gray",
                   origin="lower")
    return n1, n2, sig


# ---- the ExtendScript ---------------------------------------------------
class Js:
    def __init__(self, h):
        self.js = ["var doc = app.documents.add(DocumentColorSpace.RGB, "
                   "%g, %g);" % (W, h), base.JSX_HELPERS, r"""
function place(path, x, y, w, h){
  var p = doc.placedItems.add(); p.file = new File(path);
  p.width = w; p.height = h; p.left = X(x); p.top = Y(y); p.embed();
}
function rbox(x, y, w, h, stroke, fill, sw){
  var r = doc.pathItems.roundedRectangle(Y(y), X(x), w, h, 5, 5);
  if (fill) { r.filled = true; r.fillColor = col(fill); } else r.filled = false;
  r.stroked = true; r.strokeColor = col(stroke); r.strokeWidth = sw; return r;
}
"""]

    def a(self, s):
        self.js.append(s)

    def text(self, x, y, lines, size, colour, just="l", font="R"):
        for k, ln in enumerate(lines):
            self.a('text(%g,%g,[["%s",%s]],%g,%s,"%s");'
                   % (x, y + k * size * 1.18, ln.replace('"', '\\"'), font,
                      size, rgb(colour), just))

    def poly(self, pts, colour, sw, dash=None, cap_round=True):
        q = ",".join("[%.2f,%.2f]" % (x, y) for x, y in pts)
        self.a("var p = line([%s],%s,%g%s);%s" % (
            q, rgb(colour), sw, ",[%s]" % ",".join(map(str, dash))
            if dash else "", " p.strokeCap = StrokeCap.ROUNDENDCAP;"
            " p.strokeJoin = StrokeJoin.ROUNDENDJOIN;" if cap_round else ""))

    def arrow(self, x0, x1, y, colour=INK):
        self.a("arrow(%g,%g,%g,%g,%s,1.0,6);" % (x0, y, x1, y, rgb(colour)))

    def frame(self, x, y, w=T, h=T, fill="#FFFFFF"):
        self.a("rect(%g,%g,%g,%g,%s,%s,0.6);" % (x, y, w, h, rgb(fill),
                                                 rgb("#888888")))


def clip(poly, lo, hi):
    """The pieces of a polyline inside the square [lo, hi]^2 (Liang-Barsky
    per segment), so that thick strokes stay inside their tile."""
    out, cur = [], []
    for a, b in zip(poly[:-1], poly[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d, t0, t1 = b - a, 0.0, 1.0
        ok = True
        for p_, q_ in ((-d[0], a[0] - lo), (d[0], hi - a[0]),
                       (-d[1], a[1] - lo), (d[1], hi - a[1])):
            if abs(p_) < 1e-12:
                if q_ < 0:
                    ok = False
                    break
                continue
            r = q_ / p_
            if p_ < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        if not ok or t0 > t1:
            if len(cur) > 1:
                out.append(cur)
            cur = []
            continue
        pa, pb = a + t0 * d, a + t1 * d
        if not cur:
            cur = [tuple(pa)]
        cur.append(tuple(pb))
        if t1 < 1.0:
            out.append(cur)
            cur = []
    if len(cur) > 1:
        out.append(cur)
    return out


def to_tile(poly, x, y, s=T):
    """unit coordinates (v up) -> page points inside the tile at (x, y)"""
    return [(x + s * u, y + s * (1 - v)) for u, v in poly]


def build(n1, n2, sig):
    H = 2 * ROW_H + 4
    J = Js(H)
    polys = lines_of(n1, n2)
    n = sig.shape[0]

    def sample(u, v):
        i = min(n - 1, int(v * (n - 1)))
        j = min(n - 1, int(u * (n - 1)))
        return sig[i, j]
    lo, hi = float(sig.min()), float(sig.max())
    hot = plt.get_cmap("hot")

    def hexcol(val, cmap, a=0.15, b=0.9):
        r, g, b_, _ = cmap(a + (b - a) * (val - lo) / (hi - lo))
        return "#%02X%02X%02X" % (int(255 * r), int(255 * g), int(255 * b_))

    def tile_lines(x, y, colour, sw, s=T):
        for p in polys:
            J.poly(to_tile(p, x, y, s), colour, sw)

    def model_box(x, y, title, sub, c, fill):
        J.a("rbox(%g,%g,%g,%g,%s,%s,1.0);" % (x, y, BW, T, rgb(c),
                                                rgb(fill)))
        J.text(x + BW / 2, y + 12, title, 7.5, INK, "c", "B")
        J.text(x + BW / 2, y + 14 + 9.5 * len(title), sub, 6, MUT, "c")

    # the false candidate of Step 1: a short segment with no line under it
    # placed where it is farthest from every true line, so that the
    # rejected candidate cannot be mistaken for a true line
    pts = np.vstack([np.asarray(p) for p in polys])
    best, false = -1.0, None
    for cu in np.linspace(0.2, 0.8, 25):
        for cv in np.linspace(0.2, 0.8, 25):
            seg = np.array([(cu - 0.09, cv - 0.06), (cu + 0.09, cv + 0.06)])
            samp = np.linspace(seg[0], seg[1], 9)
            dmin = np.min(np.linalg.norm(samp[:, None, :] - pts[None], axis=2))
            if dmin > best:
                best, false = dmin, [tuple(seg[0]), tuple(seg[1])]
    assert best > 0.05, best

    for r, row in enumerate(("a", "b")):
        y0 = r * (ROW_H + 4)
        yt = y0 + 80                          # top of the tiles and boxes
        yl = yt + T + 4                       # labels under them
        if row == "a":
            J.a("rbox(%g,%g,%g,%g,%s,%s,0.8);" % (2, y0 + 2, W - 4,
                                                    ROW_H - 4, rgb(GEN_C),
                                                    rgb(GEN_BG)))
            J.text(8, y0 + 7, ["(a) Generative reconstruction"], 8.5,
                   GEN_C, "l", "B")
            J.text(8, y0 + 19, ["estimates the signal at every pixel, "
                                "including the Coulomb-blockaded regions"],
                   6.5, MUT)
            # training: measured, annotated diagrams
            for k in range(3):
                xs, ys = XC["net"] + 10 + 4 * (2 - k), y0 + 26 + 4 * k
                J.a('place("%s",%g,%g,26,26);' % (
                    str(TILES / f"measured_{k}.png").replace("\\", "/"), xs,
                    ys))
                J.a("rect(%g,%g,26,26,null,%s,0.5);" % (xs, ys,
                                                        rgb("#888888")))
            J.text(XC["net"] + 48, y0 + 30, ["trained on measured,",
                                             "manually annotated",
                                             "diagrams"], 6, GEN_C)
            J.a("arrow(%g,%g,%g,%g,%s,0.8,5);" % (
                XC["net"] + BW / 2, y0 + 62, XC["net"] + BW / 2, yt - 1,
                rgb(GEN_C)))
            # sparse measurement: a grid of measured points
            J.frame(XC["inp"], yt, fill="#111111")
            for i in range(10):
                for j in range(10):
                    u, v = (i + 0.5) / 10, (j + 0.5) / 10
                    g = int(255 * (0.25 + 0.7 * (sample(u, v) - lo)
                                   / (hi - lo)))
                    J.a("dot(%g,%g,0.9,%s);" % (
                        XC["inp"] + T * u, yt + T * (1 - v),
                        rgb("#%02X%02X%02X" % (g, g, g))))
            J.text(XC["inp"] + T / 2, yl, ["sparse", "measurement"], 6.5,
                   INK, "c")
            model_box(XC["net"], yt, ["generative", "model"],
                      ["estimates", "the signal"], GEN_C, "#FFFFFF")
            J.a('place("%s",%g,%g,%g,%g);' % (
                str(TILES / "estimate.png").replace("\\", "/"), XC["out"],
                yt, T, T))
            J.a("rect(%g,%g,%g,%g,null,%s,0.6);" % (XC["out"], yt, T, T,
                                                     rgb("#888888")))
            J.text(XC["out"] + T / 2, yl, ["signal estimated", "at every pixel"],
                   6.5, INK, "c")
            model_box(XC["mid"], yt, ["edge / ridge", "detector"],
                      ["applied to", "the estimate"], GEN_C, "#FFFFFF")
            J.frame(XC["fin"], yt)
            tile_lines(XC["fin"], yt, "#777777", 0.9)
            J.text(XC["fin"] + T / 2, yl, ["lines found", "in the estimate"],
                   6.5, INK, "c")
            for x0, x1 in ((XC["inp"] + T, XC["net"]), (XC["net"] + BW,
                                                       XC["out"]),
                           (XC["out"] + T, XC["mid"]),
                           (XC["mid"] + BW, XC["fin"])):
                J.arrow(x0 + 3, x1 - 3, yt + T / 2, GEN_C)
            J.text(XC["verdict"] + 42, yt + 12, ["lines decided", "in an",
                                                 "estimated image"], 7.5,
                   GEN_C, "c")
        else:
            # Step 1 and Step 2 bands
            xs2 = XC["mid"] - 14
            J.a("rbox(%g,%g,%g,%g,%s,%s,0.8);" % (2, y0 + 2, xs2 - 6,
                                                    ROW_H - 4, rgb(PRED_C),
                                                    rgb(PRED_BG)))
            J.a("rbox(%g,%g,%g,%g,%s,%s,0.8);" % (xs2, y0 + 2,
                                                    XC["verdict"] - 4 - xs2,
                                                    ROW_H - 4, rgb(VER_C),
                                                    rgb(VER_BG)))
            J.text(8, y0 + 7, ["(b) Two-step localization (this work)"], 8.5,
                   PRED_C, "l", "B")
            J.text(8, y0 + 19, ["predicts candidate regions, then the "
                                   "device decides"], 6.5, MUT)
            J.text(8, y0 + 32, ["Step 1: Line prediction"], 7, PRED_C, "l",
                   "B")
            J.text(xs2 + 6, y0 + 32, ["Step 2: Line verification on the "
                                      "device"], 7, VER_C, "l", "B")
            # training: simulated diagrams with exact labels
            for k in range(3):
                xs, ys = XC["net"] + 10 + 4 * (2 - k), y0 + 36 + 4 * k
                J.a("rect(%g,%g,26,26,%s,%s,0.5);" % (
                    xs, ys, rgb("#FFFFFF"), rgb("#888888")))
            for p in polys:
                J.poly(to_tile(p, XC["net"] + 10, y0 + 44, 26), LINE_C, 0.5)
            J.text(XC["net"] + 48, y0 + 42, ["trained on simulated",
                                             "diagrams, exact labels"], 6,
                   PRED_C)
            J.a("arrow(%g,%g,%g,%g,%s,0.8,5);" % (
                XC["net"] + BW / 2, y0 + 73, XC["net"] + BW / 2, yt - 1,
                rgb(PRED_C)))
            # rays from the maximum-voltage corner
            J.frame(XC["inp"], yt, fill="#E6E6E6")
            for k in range(1, 9):
                th = np.radians(90 * k / 9)
                for m in range(1, 15):
                    u = 1 - m / 14 * np.cos(th) * 1.0
                    v = 1 - m / 14 * np.sin(th) * 1.0
                    if 0 <= u <= 1 and 0 <= v <= 1:
                        J.a("dot(%g,%g,0.85,%s);" % (
                            XC["inp"] + T * u, yt + T * (1 - v),
                            rgb(hexcol(sample(u, v), hot))))
            J.text(XC["inp"] + T / 2, yl, ["a few rays"], 6.5, INK, "c")
            model_box(XC["net"], yt, ["U-Net"],
                      ["predicts", "candidate regions"], PRED_C, "#FFFFFF")
            # where the lines lie: candidate bands, one of them false
            J.frame(XC["out"], yt)
            for p in polys + [np.array(false)]:
                for q in clip(p, 0.05, 0.95):
                    J.poly(to_tile(q, XC["out"], yt), BAND_C, 4.2)
            J.text(XC["out"] + T / 2, yl, ["candidate regions"], 6.5, INK,
                   "c")
            # local sweeps across each candidate
            J.frame(XC["mid"], yt)
            for p in polys + [np.array(false)]:
                for q in clip(p, 0.05, 0.95):
                    J.poly(to_tile(q, XC["mid"], yt), BAND_C, 4.2)
            for p in polys + [np.array(false)]:
                L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0),
                                                       axis=1)])
                for s in np.arange(0.04, L[-1], 0.13):
                    k = int(np.searchsorted(L, s) - 1)
                    k = max(0, min(k, len(p) - 2))
                    f = (s - L[k]) / max(L[k + 1] - L[k], 1e-9)
                    u, v = p[k] + f * (p[k + 1] - p[k])
                    if not (0.08 <= u <= 0.92 and 0.04 <= v <= 0.96):
                        continue
                    J.poly(to_tile([(u - 0.07, v), (u + 0.07, v)],
                                   XC["mid"], yt), SWEEP_C, 0.9)
            J.text(XC["mid"] + T / 2, yl, ["local sweeps", "across each"],
                   6.5, INK, "c")
            # step detection: a trace with a step
            gx0, gy = XC["mid"] + T + 8, yt + 8
            J.poly([(gx0, gy + 12), (gx0 + 10, gy + 11.5), (gx0 + 11, gy + 2),
                    (gx0 + 24, gy + 1)], VER_C, 1.0)
            J.text(XC["mid"] + T + (XC["fin"] - XC["mid"] - T) / 2, yt + 31,
                   ["step", "detection"], 6, VER_C, "c")
            # verified lines: the true ones kept, the false one rejected
            J.frame(XC["fin"], yt)
            tile_lines(XC["fin"], yt, LINE_C, 1.3)
            J.poly(to_tile(false, XC["fin"], yt), "#B0B0B0", 1.0, [2, 1.5])
            cx, cy = to_tile([np.mean(false, axis=0)], XC["fin"], yt)[0]
            J.poly([(cx - 2.6, cy - 2.6), (cx + 2.6, cy + 2.6)], "#C00000",
                   1.0)
            J.poly([(cx - 2.6, cy + 2.6), (cx + 2.6, cy - 2.6)], "#C00000",
                   1.0)
            J.text(XC["fin"] + T / 2, yl, ["verified lines"], 6.5, INK, "c")
            for x0, x1, c in ((XC["inp"] + T, XC["net"], PRED_C),
                              (XC["net"] + BW, XC["out"], PRED_C),
                              (XC["out"] + T, XC["mid"], VER_C),
                              (XC["mid"] + T, XC["fin"], VER_C)):
                J.arrow(x0 + 3, x1 - 3, yt + T / 2, c)
            J.text(XC["verdict"] + 42, yt + 12, ["lines decided", "by the device's",
                                                 "charge sensor"], 7.5,
                   VER_C, "c", "B")
            # a check mark under the verdict
            vx, vy = XC["verdict"] + 42, yt + 48
            J.poly([(vx - 6, vy - 1), (vx - 2, vy + 3), (vx + 6, vy - 6)],
                   VER_C, 1.6)
            J.text(vx, vy + 8, ["the device is", "the final judge"], 6.5,
                   VER_C, "c")
    out = str(OUT / NAME).replace("\\", "/")
    J.a(r"""
var o = new IllustratorSaveOptions(); o.pdfCompatible = true;
doc.saveAs(new File("%(o)s.ai"), o);
var pdf = new PDFSaveOptions(); pdf.preserveEditability = false;
doc.saveAs(new File("%(o)s.pdf"), pdf);
var png = new ExportOptionsPNG24(); png.horizontalScale = 600; png.verticalScale = 600;
png.antiAliasing = true; png.transparency = false; png.artBoardClipping = true;
doc.exportFile(new File("%(o)s.png"), ExportType.PNG24, png);
doc.close(SaveOptions.DONOTSAVECHANGES);
"done";
""" % dict(o=out))
    return "\n".join(J.js)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n1, n2, sig = make_tiles()
    jsx = OUT / (NAME + ".jsx")
    jsx.write_text(build(n1, n2, sig), encoding="utf-8")
    ps = ("$a = New-Object -ComObject Illustrator.Application; "
          "$a.UserInteractionLevel = -1; "
          "$a.DoJavaScriptFile('%s')" % str(jsx))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    for ext in ("ai", "pdf", "png"):
        f = OUT / ("%s.%s" % (NAME, ext))
        print("%-4s %s" % (ext, "ok" if f.exists() else "MISSING"), f)


if __name__ == "__main__":
    main()
