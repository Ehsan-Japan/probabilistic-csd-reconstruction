"""Illustrator figure: how a zero-width transition line becomes labelled pixels.

Writes an ExtendScript (.jsx) and runs it in Adobe Illustrator over COM.
The labelled pixels are not drawn by hand: they are computed here with the
rule of Sec. 3.1.1 -- a pixel is labelled if its charge configuration
differs from that of its neighbour in the +V1 or +V2 direction -- applied to
a straight transition line on an 8 x 8 patch of the grid.

Output: paper_figures/deck_figures/pixel_labelling/  (.ai, .pdf, .png, .svg)
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deck_figures" / "pixel_labelling"
NAME = "pixel_labelling"

N = 8            # pixels per side of the patch shown
CELL = 34.0      # pt per pixel
GX, GY = 58.0, 26.0          # grid top-left on the artboard (y down)
W, H = 590.0, 350.0          # artboard

# the true boundary, in pixel units with y up: config 1 where y > C - S*x
S, C = 0.55, 5.8


def config(i, j):
    x, y = i + 0.5, j + 0.5
    return int(y > C - S * x)


def labelled(i, j):
    c = config(i, j)
    return ((i + 1 < N and config(i + 1, j) != c)
            or (j + 1 < N and config(i, j + 1) != c))


# colours
REG = {0: "#E6EDF5", 1: "#FBEAD6"}
LAB_FILL, LAB_EDGE = "#9ECAE1", "#08519C"
LINE = "#C0392B"
GRID, INK, MUT = "#B5B5B5", "#222222", "#666666"


# Drawing helpers shared with make_pixel_quantisation_ai.py.  Coordinates
# are artboard points, x right and y DOWN from the top-left corner.
JSX_HELPERS = r"""
var ab = doc.artboards[0].artboardRect;
function X(x){ return ab[0] + x; }
function Y(y){ return ab[1] - y; }
function col(c){ var k = new RGBColor(); k.red=c[0]; k.green=c[1]; k.blue=c[2]; return k; }
function rect(x, y, w, h, fill, stroke, sw){
  var r = doc.pathItems.rectangle(Y(y), X(x), w, h);
  if (fill) { r.filled = true; r.fillColor = col(fill); } else r.filled = false;
  if (stroke) { r.stroked = true; r.strokeColor = col(stroke); r.strokeWidth = sw; } else r.stroked = false;
  return r;
}
function line(pts, stroke, sw, dash){
  var p = doc.pathItems.add(), q = [];
  for (var k = 0; k < pts.length; k++) q.push([X(pts[k][0]), Y(pts[k][1])]);
  p.setEntirePath(q); p.filled = false; p.stroked = true;
  p.strokeColor = col(stroke); p.strokeWidth = sw;
  if (dash) p.strokeDashes = dash;
  return p;
}
function dot(x, y, r, fill){
  var e = doc.pathItems.ellipse(Y(y - r), X(x - r), 2*r, 2*r);
  e.filled = true; e.fillColor = col(fill); e.stroked = false; return e;
}
function head(x, y, dx, dy, s, fill){
  // filled arrowhead with its tip at (x, y), pointing along (dx, dy)
  var n = Math.sqrt(dx*dx + dy*dy); dx /= n; dy /= n;
  var bx = x - dx*s, by = y - dy*s, px_ = -dy*s*0.42, py_ = dx*s*0.42;
  var p = doc.pathItems.add();
  p.setEntirePath([[X(x), Y(y)], [X(bx+px_), Y(by+py_)], [X(bx-px_), Y(by-py_)]]);
  p.closed = true; p.filled = true; p.fillColor = col(fill); p.stroked = false;
}
function arrow(x0, y0, x1, y1, c, sw, s){
  line([[x0, y0], [x1 - (x1-x0)*0.0, y1]], c, sw);
  head(x1, y1, x1 - x0, y1 - y0, s, c);
}
// rich text: segments [[text, font, subscript?], ...]
function text(x, y, segs, size, c, just){
  var t = doc.textFrames.add();
  var s = ""; for (var k = 0; k < segs.length; k++) s += segs[k][0];
  t.contents = s;
  var ca = t.textRange.characterAttributes;
  ca.size = size; ca.fillColor = col(c);
  var at = 0;
  for (var k = 0; k < segs.length; k++) {
    var L = segs[k][0].length;
    for (var m = 0; m < L; m++) {
      var ch = t.textRange.characters[at + m].characterAttributes;
      ch.textFont = app.textFonts.getByName(segs[k][1]);
      if (segs[k][2]) ch.baselinePosition = FontBaselineOption.SUBSCRIPT;
    }
    at += L;
  }
  if (just == "c") t.textRange.paragraphAttributes.justification = Justification.CENTER;
  if (just == "r") t.textRange.paragraphAttributes.justification = Justification.RIGHT;
  t.left = X(x); t.top = Y(y);
  if (just == "c") t.left = X(x) - t.width/2;
  if (just == "r") t.left = X(x) - t.width;
  return t;
}
function boxed(t){
  var g = t.geometricBounds, pd = 2;
  var r = doc.pathItems.rectangle(g[1] + pd, g[0] - pd, g[2] - g[0] + 2*pd, g[1] - g[3] + 2*pd);
  r.filled = true; r.fillColor = col([255,255,255]); r.stroked = false; r.opacity = 85;
  t.zOrder(ZOrderMethod.BRINGTOFRONT);
}
var R = "ArialMT", I = "Arial-ItalicMT", B = "Arial-BoldMT";
"""


def rgb(h):
    h = h.lstrip("#")
    return "[%d,%d,%d]" % tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def px(i, j):
    """top-left (x, y-down) of pixel (i, j), j counted from the bottom"""
    return GX + i * CELL, GY + (N - 1 - j) * CELL


def build_jsx():
    js = []
    a = js.append
    a("var doc = app.documents.add(DocumentColorSpace.RGB, %g, %g);" % (W, H))
    a(JSX_HELPERS)

    # pixels, coloured by charge configuration; labelled ones on top
    for i in range(N):
        for j in range(N):
            x, y = px(i, j)
            if labelled(i, j):
                a("rect(%g,%g,%g,%g,%s,%s,1.1);"
                  % (x, y, CELL, CELL, rgb(LAB_FILL), rgb(LAB_EDGE)))
            else:
                a("rect(%g,%g,%g,%g,%s,%s,0.5);"
                  % (x, y, CELL, CELL, rgb(REG[config(i, j)]), rgb(GRID)))
    # redraw labelled edges so neighbouring grey edges do not cover them
    for i in range(N):
        for j in range(N):
            if labelled(i, j):
                x, y = px(i, j)
                a("rect(%g,%g,%g,%g,null,%s,1.1);"
                  % (x, y, CELL, CELL, rgb(LAB_EDGE)))

    # pixel centres: where the simulator evaluates
    for i in range(N):
        for j in range(N):
            x, y = px(i, j)
            a("dot(%g,%g,1.6,%s);" % (x + CELL / 2, y + CELL / 2, rgb(INK)))

    # the true transition line, clipped to the patch
    def to_art(xp, yp):
        return GX + xp * CELL, GY + (N - yp) * CELL
    x0, y0 = to_art(0, C)
    x1, y1 = to_art(N, C - S * N)
    a("line([[%g,%g],[%g,%g]],%s,1.8);" % (x0, y0, x1, y1, rgb(LINE)))

    # region labels
    a('boxed(text(%g,%g,[["(",R],["N",I],["1",I,1],[" + 1, ",R],["N",I],["2",I,1],[")",R]],9,%s,"r"));'
      % (GX + N * CELL - 6, GY + 5, rgb(INK)))
    a('boxed(text(%g,%g,[["(",R],["N",I],["1",I,1],[", ",R],["N",I],["2",I,1],[")",R]],9,%s,"l"));'
      % (GX + 5, GY + N * CELL - 16, rgb(INK)))

    # example pixel P: labelled, with both +V1 and +V2 neighbours across
    # the line; marked with a letter
    P = next((i, j) for i in range(2, N - 2) for j in range(N - 1)
             if labelled(i, j) and config(i + 1, j) != config(i, j)
             and config(i, j + 1) != config(i, j))
    x, y = px(*P)
    a('text(%g,%g,[["P",B]],9,%s,"c");' % (x + 7, y + CELL - 12, rgb(LAB_EDGE)))

    # axes
    ay = GY + N * CELL + 14
    a("arrow(%g,%g,%g,%g,%s,0.8,6);" % (GX, ay, GX + N * CELL + 14, ay, rgb(INK)))
    a('text(%g,%g,[["V",I],["1",I,1]],10,%s,"l");'
      % (GX + N * CELL + 18, ay - 7, rgb(INK)))
    ax_ = GX - 14
    a("arrow(%g,%g,%g,%g,%s,0.8,6);" % (ax_, GY + N * CELL, ax_, GY - 12, rgb(INK)))
    a('text(%g,%g,[["V",I],["2",I,1]],10,%s,"c");' % (ax_, GY - 26, rgb(INK)))

    # one-pixel scale bar under the first column
    sy = ay + 12
    a("line([[%g,%g],[%g,%g]],%s,0.8);" % (GX, sy, GX + CELL, sy, rgb(MUT)))
    a("line([[%g,%g],[%g,%g]],%s,0.8);" % (GX, sy - 3, GX, sy + 3, rgb(MUT)))
    a("line([[%g,%g],[%g,%g]],%s,0.8);"
      % (GX + CELL, sy - 3, GX + CELL, sy + 3, rgb(MUT)))
    a('text(%g,%g,[["1 pixel = unit of distance (incl. \\u03C4)",R]],8,%s,"l");'
      % (GX + CELL + 6, sy - 5, rgb(MUT)))

    # ---- legend -----------------------------------------------------------
    lx, ly = GX + N * CELL + 58, GY + 4
    s = 14
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (lx, ly, s, s, rgb(REG[0]), rgb(GRID)))
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (lx + s, ly, s, s, rgb(REG[1]), rgb(GRID)))
    a('text(%g,%g,[["pixel colour: charge configuration",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly + 1, rgb(INK)))
    ly += 24
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (lx + s / 2, ly, s, s, rgb(REG[0]), rgb(GRID)))
    a("dot(%g,%g,1.6,%s);" % (lx + s, ly + s / 2, rgb(INK)))
    a('text(%g,%g,[["pixel centre: the single voltage",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly - 4, rgb(INK)))
    a('text(%g,%g,[["pair the simulator evaluates",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly + 7, rgb(INK)))
    ly += 26
    a("line([[%g,%g],[%g,%g]],%s,1.8);" % (lx, ly + s / 2, lx + 2 * s, ly + s / 2, rgb(LINE)))
    a('text(%g,%g,[["true transition line (zero width)",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly + 1, rgb(INK)))
    ly += 24
    a("rect(%g,%g,%g,%g,%s,%s,1.1);"
      % (lx + s / 2, ly, s, s, rgb(LAB_FILL), rgb(LAB_EDGE)))
    a('text(%g,%g,[["labelled pixel: last one before the",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly - 4, rgb(INK)))
    a('text(%g,%g,[["configuration changes",R]],9,%s,"l");'
      % (lx + 2 * s + 8, ly + 7, rgb(INK)))

    # ---- the rule, on a 2 x 2 close-up ------------------------------------
    c2 = 30.0
    rx, ry = lx + 34, ly + 58          # top-left of the close-up
    # P bottom-left, +V1 neighbour right, +V2 neighbour above
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (rx, ry, c2, c2, rgb(REG[1]), rgb(GRID)))
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (rx + c2, ry + c2, c2, c2, rgb(REG[1]), rgb(GRID)))
    a("rect(%g,%g,%g,%g,%s,%s,0.5);" % (rx + c2, ry, c2, c2, rgb(REG[1]), rgb(GRID)))
    a("rect(%g,%g,%g,%g,%s,%s,1.1);" % (rx, ry + c2, c2, c2, rgb(LAB_FILL), rgb(LAB_EDGE)))
    a('text(%g,%g,[["P",B]],9,%s,"c");' % (rx + 7, ry + 2 * c2 - 12, rgb(LAB_EDGE)))
    # arrows from P's centre to its two neighbours' centres
    pcx, pcy = rx + c2 / 2, ry + 1.5 * c2
    a("dot(%g,%g,1.6,%s);" % (pcx, pcy, rgb(INK)))
    a("dot(%g,%g,1.6,%s);" % (pcx + c2, pcy, rgb(INK)))
    a("dot(%g,%g,1.6,%s);" % (pcx, pcy - c2, rgb(INK)))
    a("arrow(%g,%g,%g,%g,%s,0.8,5);" % (pcx + 3, pcy, pcx + c2 - 3, pcy, rgb(INK)))
    a("arrow(%g,%g,%g,%g,%s,0.8,5);" % (pcx, pcy - 3, pcx, pcy - c2 + 3, rgb(INK)))
    a('text(%g,%g,[["+",R],["V",I],["1",I,1]],8,%s,"l");'
      % (rx + 2 * c2 + 5, pcy - 6, rgb(INK)))
    a('text(%g,%g,[["+",R],["V",I],["2",I,1]],8,%s,"c");'
      % (pcx, ry - 13, rgb(INK)))
    # the boundary passing between P and both neighbours
    a("line([[%g,%g],[%g,%g]],%s,1.8);"
      % (rx - 4, ry + 19.8, rx + 62, ry + 66, rgb(LINE)))
    ty = ry + 2 * c2 + 12
    a('text(%g,%g,[["P is labelled if its configuration differs",R]],8.5,%s,"l");'
      % (lx, ty, rgb(INK)))
    a('text(%g,%g,[["from its +",R],["V",I],["1",I,1],[" or its +",R],["V",I],["2",I,1],[" neighbour",R]],8.5,%s,"l");'
      % (lx, ty + 11, rgb(INK)))

    # ---- save & export ----------------------------------------------------
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
    ps = ("$a = New-Object -ComObject Illustrator.Application; "
          "$a.UserInteractionLevel = -1; "
          "$a.DoJavaScriptFile('%s')" % str(jsx))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    for ext in ("ai", "pdf", "png", "svg"):
        f = OUT / ("%s.%s" % (NAME, ext))
        print("%-6s %s" % (ext, "ok" if f.exists() else "MISSING"), f)


if __name__ == "__main__":
    main()
