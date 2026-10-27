"""Illustrator figure: the true transition lines, and what a pixel grid makes of them.

One real device of the study (sample_1 of the device pool), simulated with
the same QArray model and capacitances as the dataset:

  (a) the continuous gate-voltage plane with its true transition lines, no
      grid.  The lines are the boundaries between charge configurations,
      traced on a grid ~15x finer than the paper's and simplified to
      straight segments; they are drawn as vectors.
  (b) a 16 x 16-pixel region at the paper's spacing (100 x 100 pixels over
      the window).  Each pixel is coloured by the configuration at its
      centre and labelled by the rule of Sec. 3.1.1.  These labels are the
      stored ground truth of this device -- checked below, pixel for pixel.
  (c) the same region on a grid twice as coarse.  Same device, same rule,
      different labels: the labelling is a property of the chosen spacing.

Output: paper_figures/deck_figures/pixel_quantisation/  (.ai .pdf .png .svg)
"""
import json
import subprocess
from pathlib import Path

import contourpy
import numpy as np
from qarray import DotArray

import make_pixel_labelling_ai as base

ROOT = Path(__file__).resolve().parents[2]
DEVICE = (ROOT / "data" / "_device_pools" / "devices_n550_res100_c1df7b6bf"
          / "sample_1")
OUT = ROOT / "paper_figures" / "deck_figures" / "pixel_quantisation"
NAME = "pixel_quantisation"

RES = 100                    # the paper's grid
Z0, ZN = (26, 28), 16        # zoom: first pixel (col, row), pixels per side
FINE = 1500                  # samples per side for the continuous lines

dev = json.loads((DEVICE / "device.json").read_text())
X0, X1, Y0, Y1 = dev["voltage_window"]
CAP = dev["capacitance"]
MODEL = DotArray(Cdd=CAP["Cdd"], Cgd=CAP["Cgd"], charge_carrier="hole")
DX = (X1 - X0) / (RES - 1)   # pixel spacing; centres are np.linspace points
DY = (Y1 - Y0) / (RES - 1)


def occupation(xa, xb, nx, ya, yb, ny):
    """Integer (n1, n2) on a do2d grid; axis 0 is V2, axis 1 is V1."""
    n = MODEL.do2d_open(x_gate="P1", x_min=xa, x_max=xb, x_res=nx,
                        y_gate="P2", y_min=ya, y_max=yb, y_res=ny)
    return np.rint(np.nan_to_num(n, nan=0)).astype(int)


def labels(n):
    """The dataset rule: differs from the +V1 or +V2 neighbour."""
    lab = np.zeros(n.shape[:2], bool)
    lab[:, :-1] |= (n[:, 1:] != n[:, :-1]).any(-1)
    lab[:-1, :] |= (n[1:, :] != n[:-1, :]).any(-1)
    return lab


def rdp(pts, eps):
    """Douglas-Peucker: the staircase of a traced boundary -> its segments."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.hypot(*ab)
    d = (np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
         if L > 0
         else np.hypot(*(pts - a).T))
    k = int(np.argmax(d))
    if d[k] > eps:
        return np.vstack([rdp(pts[:k + 1], eps)[:-1], rdp(pts[k:], eps)])
    return np.vstack([a, b])


def true_lines(xa, xb, ya, yb, n=FINE):
    """Polylines (in volts) of every charge-configuration boundary."""
    occ = occupation(xa, xb, n, ya, yb, n)
    xs, ys = np.linspace(xa, xb, n), np.linspace(ya, yb, n)
    out = []
    for k in (0, 1):
        z = occ[..., k].astype(float)
        gen = contourpy.contour_generator(xs, ys, z)
        for lev in np.arange(z.min() + 0.5, z.max(), 1.0):
            for seg in gen.lines(lev):
                out.append(rdp(np.asarray(seg), 0.6 * (xb - xa) / n))
    return out


# ── the data ─────────────────────────────────────────────────────────────
full = occupation(X0, X1, RES, Y0, Y1, RES)
stored = np.load(DEVICE / "numpy" / "simulation" / "charge_states.npy")
truth = np.load(DEVICE / "numpy" / "simulation" / "ground_truth_labels.npy")
assert np.array_equal(full, stored), "occupation differs from the stored run"
# the stored map leaves the last row and column unlabelled (they have no
# +V neighbour); everywhere else it is exactly the rule
assert np.array_equal(labels(full)[:-1, :-1], truth.astype(bool)[:-1, :-1]), \
    "rule != stored labels"
assert not truth[-1].any() and not truth[:, -1].any()

c0, r0 = Z0
# zoom box: from the edge of pixel (c0, r0) to the far edge of the last one
BX0, BX1 = X0 + (c0 - 0.5) * DX, X0 + (c0 + ZN - 0.5) * DX
BY0, BY1 = Y0 + (r0 - 0.5) * DY, Y0 + (r0 + ZN - 0.5) * DY

# (b) the paper's grid: the stored configuration and labels themselves
occ_b = full[r0:r0 + ZN, c0:c0 + ZN]
lab_b = truth.astype(bool)[r0:r0 + ZN, c0:c0 + ZN]
# (c) spacing 2*DX over the same box; one extra row and column so the
# pixels on the far edges still have their +V1 / +V2 neighbour
M = ZN // 2
occ_c_ext = occupation(BX0 + DX, BX0 + DX + M * 2 * DX, M + 1,
                       BY0 + DY, BY0 + DY + M * 2 * DY, M + 1)
occ_c = occ_c_ext[:M, :M]
lab_c = labels(occ_c_ext)[:M, :M]

lines_full = true_lines(X0, X1, Y0, Y1)
lines_zoom = true_lines(BX0, BX1, BY0, BY1, 600)

states = sorted({tuple(s) for s in occ_b.reshape(-1, 2)}
                | {tuple(s) for s in occ_c.reshape(-1, 2)})
PALE = ["#FBEAD6", "#E4EFD9", "#EFE3F1", "#FFF3C4", "#DDEFEC", "#F4DCDC",
        "#E9E9E9", "#F9E0C0"]
assert len(states) <= len(PALE), states
STATE = {s: PALE[k] for k, s in enumerate(states)}

LAB_FILL, LAB_EDGE = base.LAB_FILL, base.LAB_EDGE
LINE, GRID, INK, MUT = base.LINE, base.GRID, base.INK, base.MUT
rgb = base.rgb

# ── layout (points, y down) ──────────────────────────────────────────────
W, H = 640.0, 240.0
SIDE = 172.0
PA = (52.0, 12.0)            # panel tops-left
PB = (262.0, 12.0)
PC = (452.0, 12.0)


def js_str(s):
    # \u escapes are left for JavaScript to read
    return s.replace('"', '\\"')


def build_jsx():
    js = ["var doc = app.documents.add(DocumentColorSpace.RGB, %g, %g);"
          % (W, H), base.JSX_HELPERS]
    a = js.append

    def poly(pts, sw, colour, dash=None):
        a("line([%s],%s,%g%s);" % (
            ",".join("[%.2f,%.2f]" % tuple(p) for p in pts), rgb(colour), sw,
            "" if dash is None else ",%s" % dash))

    def t(x, y, segs, size, colour, just="l"):
        body = ",".join('["%s",%s%s]' % (js_str(s), f, ",1" if sub else "")
                        for s, f, sub in segs)
        a('text(%g,%g,[%s],%g,%s,"%s");' % (x, y, body, size, rgb(colour),
                                          just))

    V1 = [("V", "I", 0), ("1", "I", 1)]
    V2 = [("V", "I", 0), ("2", "I", 1)]

    # ── (a) continuous plane ─────────────────────────────────────────────
    ax, ay = PA
    sx = SIDE / (X1 - X0)
    sy = SIDE / (Y1 - Y0)

    def A(x, y):
        return ax + (x - X0) * sx, ay + (Y1 - y) * sy
    a("rect(%g,%g,%g,%g,[255,255,255],%s,0.6);" % (ax, ay, SIDE, SIDE,
                                                   rgb(MUT)))
    for seg in lines_full:
        poly([A(*p) for p in seg], 1.1, LINE)
    # configuration of the larger regions, at their centroid
    cells = {}
    for (r, c), s in np.ndenumerate(full[..., 0] * 10 + full[..., 1]):
        # keep the labels clear of the zoom box
        if r0 - 6 <= r < r0 + ZN + 6 and c0 - 6 <= c < c0 + ZN + 6:
            continue
        cells.setdefault(s, []).append((r, c))
    for s, rc in cells.items():
        if len(rc) < 260:
            continue
        rc = np.array(rc)
        r, c = np.median(rc, axis=0)
        x, y = A(X0 + c * DX, Y0 + r * DY)
        t(x, y - 4, [("(%d, %d)" % (s // 10, s % 10), "R", 0)], 6.5, MUT, "c")
    # zoom box and its connectors to (b)
    bx, by = A(BX0, BY1)
    bw, bh = (BX1 - BX0) * sx, (BY1 - BY0) * sy
    a("var zb = rect(%g,%g,%g,%g,null,%s,0.8); zb.strokeDashes=[2,1.5];"
      % (bx, by, bw, bh, rgb(INK)))
    poly([(bx + bw, by), PB], 0.4, MUT, "[2,2]")
    poly([(bx + bw, by + bh), (PB[0], PB[1] + SIDE)], 0.4, MUT, "[2,2]")
    # axes: ticks in mV at round values
    for v in np.arange(np.ceil(X0 * 2) / 2, X1 + 1e-9, 0.5):
        x, _ = A(v, Y0)
        poly([(x, ay + SIDE), (x, ay + SIDE + 3)], 0.6, INK)
        t(x, ay + SIDE + 5, [("%g" % v, "R", 0)], 7, INK, "c")
    for v in np.arange(np.ceil(Y0 * 2) / 2, Y1 + 1e-9, 0.5):
        _, y = A(X0, v)
        poly([(ax - 3, y), (ax, y)], 0.6, INK)
        t(ax - 5, y - 4.5, [("%g" % v, "R", 0)], 7, INK, "r")
    t(ax + SIDE / 2, ay + SIDE + 16, V1 + [(" (mV)", "R", 0)], 8, INK, "c")
    a('var yl = text(%g,%g,[["V",I],["2",I,1],[" (mV)",R]],8,%s,"c");'
      'yl.rotate(90); yl.left = X(%g); yl.top = Y(%g) + yl.height/2;'
      % (0, 0, rgb(INK), ax - 34, ay + SIDE / 2))

    # ── (b) and (c): the same box on two grids ──────────────────────────
    def gridded(P, occ, lab, n, title, sub):
        px0, py0 = P
        cell = SIDE / n
        for r in range(n):
            for c in range(n):
                x, y = px0 + c * cell, py0 + (n - 1 - r) * cell
                if lab[r, c]:
                    a("rect(%g,%g,%g,%g,%s,null,0);"
                      % (x, y, cell, cell, rgb(LAB_FILL)))
                else:
                    a("rect(%g,%g,%g,%g,%s,%s,0.35);"
                      % (x, y, cell, cell, rgb(STATE[tuple(occ[r, c])]),
                         rgb(GRID)))
        for r in range(n):
            for c in range(n):
                if lab[r, c]:
                    x, y = px0 + c * cell, py0 + (n - 1 - r) * cell
                    a("rect(%g,%g,%g,%g,null,%s,0.8);"
                      % (x, y, cell, cell, rgb(LAB_EDGE)))
        for r in range(n):
            for c in range(n):
                x, y = px0 + (c + 0.5) * cell, py0 + (n - 0.5 - r) * cell
                a("dot(%g,%g,%g,%s);" % (x, y, 0.9 if n > 10 else 1.3,
                                         rgb(INK)))
        zx = SIDE / (BX1 - BX0)
        zy = SIDE / (BY1 - BY0)
        for seg in lines_zoom:
            poly([(px0 + (x - BX0) * zx, py0 + (BY1 - y) * zy)
                  for x, y in seg], 1.4, LINE)
        a("rect(%g,%g,%g,%g,null,%s,0.6);" % (px0, py0, SIDE, SIDE, rgb(MUT)))
        # spacing bar under the panel: one pixel
        yb = py0 + SIDE + 6
        poly([(px0, yb), (px0 + cell, yb)], 0.8, INK)
        poly([(px0, yb - 2.5), (px0, yb + 2.5)], 0.8, INK)
        poly([(px0 + cell, yb - 2.5), (px0 + cell, yb + 2.5)], 0.8, INK)
        t(px0 + cell + 4, yb - 4.5, sub, 7.5, INK)

    def panel_title(P, letter):
        # a boxed letter in the top-left corner, as in the U-Net figure;
        # what each panel shows is said in the caption (your call,
        # 2026-09-30)
        x, y, b = P[0] + 4, P[1] + 4, 11
        a("rect(%g,%g,%g,%g,[255,255,255],%s,0.6);" % (x, y, b, b, rgb(INK)))
        t(x + b / 2, y + 1.3, [(letter, "B", 0)], 8, INK, "c")

    panel_title(PA, "a")
    gridded(PB, occ_b, lab_b, ZN, "b", [("\\u0394", "R", 0)])
    panel_title(PB, "b")
    gridded(PC, occ_c, lab_c, M, "c", [("2\\u0394", "R", 0)])
    panel_title(PC, "c")

    # ── legend ───────────────────────────────────────────────────────────
    ly = PA[1] + SIDE + 26
    lx = PB[0]
    poly([(lx, ly + 4), (lx + 18, ly + 4)], 1.4, LINE)
    t(lx + 24, ly, [("true transition line", "R", 0)], 7.5, INK)
    lx += 108
    a("rect(%g,%g,9,9,%s,%s,0.35);" % (lx, ly, rgb(PALE[0]), rgb(GRID)))
    a("dot(%g,%g,1.1,%s);" % (lx + 4.5, ly + 4.5, rgb(INK)))
    t(lx + 14, ly, [("configuration at the pixel centre", "R", 0)], 7.5, INK)
    lx += 138
    a("rect(%g,%g,9,9,%s,%s,0.8);" % (lx, ly, rgb(LAB_FILL), rgb(LAB_EDGE)))
    t(lx + 14, ly, [("labelled pixel", "R", 0)], 7.5, INK)

    out = str(OUT).replace("\\", "/")
    a(r"""
var base = "%(out)s/%(name)s";
var o = new IllustratorSaveOptions(); o.pdfCompatible = true;
doc.saveAs(new File(base + ".ai"), o);
var pdf = new PDFSaveOptions(); pdf.preserveEditability = false;
doc.saveAs(new File(base + ".pdf"), pdf);
var png = new ExportOptionsPNG24(); png.horizontalScale = 400; png.verticalScale = 400;
png.antiAliasing = true; png.transparency = false; png.artBoardClipping = true;
doc.exportFile(new File(base + ".png"), ExportType.PNG24, png);
var svg = new ExportOptionsSVG(); svg.embedRasterImages = true;
svg.fontType = SVGFontType.OUTLINEFONT;
doc.exportFile(new File(base + ".svg"), ExportType.SVG, svg);
doc.close(SaveOptions.DONOTSAVECHANGES);
"done";
""" % dict(out=out, name=NAME))
    return "\n".join(js)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jsx = OUT / (NAME + ".jsx")
    jsx.write_text(build_jsx(), encoding="utf-8")
    print("device %s: labels on the paper's grid match the stored ground "
          "truth; states in the zoom: %s" % (DEVICE.name, states))
    print("zoom box V1 %.4f..%.4f mV, V2 %.4f..%.4f mV; spacing %.2f uV"
          % (BX0, BX1, BY0, BY1, 1e3 * DX))
    ps = ("$a = New-Object -ComObject Illustrator.Application; "
          "$a.UserInteractionLevel = -1; "
          "$a.DoJavaScriptFile('%s')" % str(jsx))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    for ext in ("ai", "pdf", "png", "svg"):
        f = OUT / ("%s.%s" % (NAME, ext))
        print("%-6s %s" % (ext, "ok" if f.exists() else "MISSING"), f)


if __name__ == "__main__":
    main()
