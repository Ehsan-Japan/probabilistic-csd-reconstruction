"""Illustrator figure: the pipeline of Hernandes et al. [12] beside ours.

  (a) Hernandes et al.: sparse data -> conditional diffusion model -> full
      charge-sensor image -> Canny edges + Frangi ridges -> lines.
  (b) This work: sparse data -> U-Net -> P(transition line) per pixel ->
      threshold -> lines.

Every statement about [12] was checked against the paper's PDF
(csd-materials/citations/V. Hernandes et al/paper/): 128 x 128 images; the
loss "is simply the mean squared error between the original and the
predicted image"; rNMSE, PSNR, SSIM and IoU, F1, Hausdorff in its Tables I
and II; transition lines extracted with the Canny algorithm and the Frangi
filter.  Ours: grid_train (weighted BCE + soft Dice on the line labels),
the stored threshold of the headline checkpoint, 100 x 100.

The image tiles in BOTH rows are one held-out device of this work (the one
of fig_inputs), for illustration only -- [12] works on measured diagrams.
The tiles of row (b) are the real input, output and thresholded output of
the trained network; the "extracted lines" tile of row (a) is the device's
ground truth, since the detectors of [12] were not run here.

Output: paper_figures/deck_figures/pipeline_comparison/

--ref12 (2026-09-30) writes pipeline_comparison_ref12 instead, whose row (a)
carries the images of [12] itself, cropped from its Fig. 3 -- first from the
author's Presentation1.pptx, now from the PDF of [12] at full resolution: the masked input,
the diffusion reconstruction and the ridge comparison, for CSD 496 at
reduce factor 5.  That factor was read off the crop (dot period ~5 pixels
of 128), and [12] gives it as "approximately ... 4%" of the data.  Its
colours, from the caption of that figure: white = predicted ridge matching
the original, blue = detected in the prediction only, red = in the
original only.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf                                       # noqa: E402
from csdrecon.ml import grid_train, ray_peaks                   # noqa: E402
import make_pixel_labelling_ai as base                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures" / "deck_figures" / "pipeline_comparison"
TILES = OUT / "tiles"
NAME = "pipeline_comparison"


# ── tiles from the run ───────────────────────────────────────────────────
def make_tiles():
    TILES.mkdir(parents=True, exist_ok=True)
    sample = mf._sensor_dir()
    _ux, _uy, Z = mf._raw_grid(sample)
    truth = ray_peaks.load_ground_truth(sample) > 0.5
    m = ray_peaks.measure(sample, mf.N_RAYS, mf.N_POINTS)
    _gx, _gy, Zn = ray_peaks.load_grid(sample)
    ch = ray_peaks.to_channels(m, Zn.shape)
    net, ck = mf.net()
    prob = grid_train.predict(net, ch[None, :ray_peaks.NET_CHANNELS])[0]
    thr = float(ck["threshold"])
    vis = ch[ray_peaks.CH_VISITED] > 0.5
    sig = np.where(vis, ch[ray_peaks.CH_SIGNAL], np.nan)
    hot = plt.get_cmap("hot").copy()
    hot.set_bad("#e6e6e6")

    def tile(name, data, cmap, **kw):
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(data, origin="lower", cmap=cmap, interpolation="nearest",
                  aspect="auto", **kw)
        ax.axis("off")
        fig.savefig(TILES / (name + ".png"), dpi=300, facecolor="white")
        plt.close(fig)

    tile("input", sig, hot, vmin=0, vmax=1)
    # [12]'s input: horizontal and vertical line cuts (one of its two mask
    # types), drawn here as every 8th row and column of the same device --
    # an illustration of the pattern, not one of its line-cut factors
    cut = np.zeros(Zn.shape, bool)
    cut[::8, :] = True
    cut[:, ::8] = True
    tile("input_cuts", np.where(cut, Zn, np.nan), hot, vmin=0, vmax=1)
    tile("sensor", Z, "hot")
    tile("lines_truth", np.where(truth, 0.0, 1.0), "gray", vmin=0, vmax=1)
    tile("probability", prob, mf.J_CMAP, vmin=0, vmax=1)
    tile("lines_pred", np.where(prob > thr, 0.0, 1.0), "gray", vmin=0, vmax=1)
    return dict(device=os.path.basename(sample), threshold=thr,
                n_rays=mf.N_RAYS, n_points=mf.N_POINTS,
                coverage=100 * vis.mean(), res=Z.shape[0])


# ── the images of [12], from its own PDF ─────────────────────────────────
# Fig. 3 of [12] is embedded in its PDF as one 4448 x 3392 image (xref 331,
# page 7): two blocks of 4 columns (reduce factors 3, 5, 7, 9) by 6 rows.
# The author's Presentation1.pptx held 300-px crops of it; these are the same
# panels at the PDF's resolution.  The column was identified by matching the
# pptx crops against all four (ridges: r = 0.944 for factor 5, <= 0.837 for
# the others).
REF12_PDF = (ROOT / "csd-materials" / "citations" / "V. Hernandes et al"
             / "Reconstructing Quantum Dot Charge Stability Diagrams with "
               "Diffusion Models.pdf")
REF12_XREF = 331
REF12_COL = (632, 1121)                      # CSD 496, reduce factor 5
REF12_ROWS = {"h_masked": (264, 753), "h_original": (786, 1275),
              "h_diffusion": (1309, 1798), "h_ridges": (2354, 2843)}


def ref12_tiles():
    """Crop the four panels from Fig. 3 of [12], inside their frame."""
    import pymupdf
    from PIL import Image
    TILES.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(REF12_PDF))
    pix = pymupdf.Pixmap(doc, REF12_XREF)
    full = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    assert full.size == (4448, 3392), full.size
    x0, x1 = REF12_COL
    for name, (y0, y1) in REF12_ROWS.items():
        full.crop((x0 + 6, y0 + 6, x1 - 6, y1 - 6)).save(TILES / (name + ".png"))
    return {k: (x1 - x0 - 12) for k in REF12_ROWS}


# ── the drawing ──────────────────────────────────────────────────────────
# Laid out for a journal page (2026-09-30): the figure is printed at the
# text width, so the text is set large here (10-13 pt) to come out at about
# 6-8 pt, and nothing sits to the right of the last tile.
M = 8.0                          # outer margin
T = 112.0                        # tile side
BOXW = 118.0
G = 24.0                         # gap holding an arrow
LH = 12.0                        # label line height
FS_HEAD, FS_BOXB, FS_BOX, FS_LAB = 13, 11.5, 10.5, 10
X_IN = M
X_M1 = X_IN + T + G
X_MID = X_M1 + BOXW + G
X_M2 = X_MID + T + G
X_OUT = X_M2 + BOXW + G
W = X_OUT + T + M
ROW_H = 196.0
H = 2 * ROW_H - 4
INK, MUT = base.INK, base.MUT
A_COL, B_COL = "#6E6E6E", "#08519C"     # row accents: theirs grey, ours blue
rgb = base.rgb


def build_jsx(info, ref12=False, name=None):
    js = ["var doc = app.documents.add(DocumentColorSpace.RGB, %g, %g);"
          % (W, H), base.JSX_HELPERS]
    a = js.append
    a(r"""
function place(path, x, y, s){
  var p = doc.placedItems.add(); p.file = new File(path);
  p.width = s; p.height = s; p.left = X(x); p.top = Y(y); p.embed();
}
function box(x, y, w, h, stroke, fill){
  var r = doc.pathItems.roundedRectangle(Y(y), X(x), w, h, 6, 6);
  r.filled = true; r.fillColor = col(fill); r.stroked = true;
  r.strokeColor = col(stroke); r.strokeWidth = 1.2; return r;
}
""")

    def t(x, y, segs, size, colour, just="l"):
        body = ",".join('["%s",%s%s]' % (s.replace('"', '\\"'), f,
                                        ",1" if sub else "")
                        for s, f, sub in segs)
        a('text(%g,%g,[%s],%g,%s,"%s");' % (x, y, body, size, rgb(colour),
                                          just))

    def img(name, x, y):
        p = str(TILES / (name + ".png")).replace("\\", "/")
        a('place("%s",%g,%g,%g);' % (p, x, y, T))
        a("rect(%g,%g,%g,%g,null,%s,0.6);" % (x, y, T, T, rgb(MUT)))

    def arrow(x0, x1, y, colour):
        a("arrow(%g,%g,%g,%g,%s,1.3,8);" % (x0, y, x1, y, rgb(colour)))

    def model(x, y, title, lines, colour, fill):
        a("box(%g,%g,%g,%g,%s,%s);" % (x, y, BOXW, T, rgb(colour), rgb(fill)))
        n = len(title) + len(lines)
        y0 = y + T / 2 - n * 6.6
        k = 0
        for w in title:
            t(x + BOXW / 2, y0 + 13.2 * k, [(w, "B", 0)], FS_BOXB, INK, "c")
            k += 1
        for w in lines:
            t(x + BOXW / 2, y0 + 13.2 * k, [(w, "R", 0)], FS_BOX, INK, "c")
            k += 1

    def label(x, y, words, colour=INK, width=T):
        for k, w in enumerate(words):
            t(x + width / 2, y + 4 + LH * k, [(w, "R", 0)], FS_LAB, colour,
              "c")

    x_sym = "\\u00D7"
    theirs = dict(
        who="Hernandes et al. [12]:", what=" reconstruct the image, then "
        "find the lines", colour=A_COL, fill="#F2F2F2",
        inp="input_cuts", inlab=["sparse data:", "line cuts"],
        m1=(["conditional", "diffusion model"],
            ["loss: MSE to the", "measured image"]),
        mid="sensor", midlab=["full charge-sensor", "image, 128 %s 128" % x_sym],
        m2=(["Canny edges", "+ Frangi ridges"],
            ["applied after", "the network"]),
        out="lines_truth", outlab=["extracted lines"],
        ev=["scored on the image", "(rNMSE, PSNR, SSIM)", "and the lines",
            "(IoU, F1, Hausdorff)"])
    ours = dict(
        who="This work:", what=" infer the lines directly", colour=B_COL,
        fill="#EAF2FA",
        inp="input", inlab=["sparse data:", "rays"],
        m1=(["U-Net"], ["loss: weighted BCE", "+ Dice to the", "line labels"]),
        mid="probability", midlab=["P(transition line)", "per pixel, %d %s %d"
                                   % (info["res"], x_sym, info["res"])],
        m2=(["threshold"], ["chosen on", "validation devices"]),
        out="lines_pred", outlab=["predicted lines"],
        ev=["scored on the lines", "(F1@\\u03C4)"])
    if ref12:
        # row (a) from [12] itself: its Fig. 3, CSD 496, reduce factor 5
        theirs.update(
            inp="h_masked", inlab=["sparse data: grid mask,",
                                   "\\u2248 4 % of the image"],
            mid="h_diffusion", midlab=["diffusion reconstruction,",
                                       "128 %s 128" % x_sym],
            out="h_ridges", outlab=["ridges vs measured image:",
                                    "white both, blue extra,", "red missed"])
        ours.update(inlab=["sparse data: %d rays," % info["n_rays"],
                           "%.1f %% of the plane" % info["coverage"]])

    for r, (letter, row) in enumerate((("a", theirs), ("b", ours))):
        y = r * ROW_H + 4
        col = row["colour"]
        a('rect(%g,%g,13,13,[255,255,255],%s,0.7);' % (X_IN, y, rgb(INK)))
        t(X_IN + 6.5, y + 0.8, [(letter, "B", 0)], 10, INK, "c")
        t(X_IN + 20, y - 0.5, [(row["who"], "B", 0), (row["what"], "R", 0)],
          FS_HEAD, col)
        yt = y + 24
        img(row["inp"], X_IN, yt)
        label(X_IN, yt + T, row["inlab"], MUT)
        arrow(X_IN + T + 4, X_M1 - 4, yt + T / 2, col)
        model(X_M1, yt, row["m1"][0], row["m1"][1], col, row["fill"])
        arrow(X_M1 + BOXW + 4, X_MID - 4, yt + T / 2, col)
        img(row["mid"], X_MID, yt)
        label(X_MID, yt + T, row["midlab"])
        n_mid = len(row["midlab"])
        t(X_MID + T / 2, yt + T + 6 + LH * n_mid, [("network output", "B", 0)],
          FS_LAB, col, "c")
        a("var hl = rect(%g,%g,%g,%g,null,%s,1.3); hl.strokeDashes=[3.5,2.5];"
          % (X_MID - 5, yt - 5, T + 10, T + 14 + LH * (n_mid + 1), rgb(col)))
        arrow(X_MID + T + 4, X_M2 - 4, yt + T / 2, col)
        model(X_M2, yt, row["m2"][0], row["m2"][1], col, row["fill"])
        # what the result is scored on: under the second box, where the
        # row has room
        label(X_M2, yt + T, row["ev"], MUT, BOXW)
        arrow(X_M2 + BOXW + 4, X_OUT - 4, yt + T / 2, col)
        img(row["out"], X_OUT, yt)
        label(X_OUT, yt + T, row["outlab"], MUT)

    out = str(OUT).replace("\\", "/")
    a(r"""
var base = "%(out)s/%(name)s";
var o = new IllustratorSaveOptions(); o.pdfCompatible = true;
doc.saveAs(new File(base + ".ai"), o);
var pdf = new PDFSaveOptions(); pdf.preserveEditability = false;
doc.saveAs(new File(base + ".pdf"), pdf);
var png = new ExportOptionsPNG24(); png.horizontalScale = 500; png.verticalScale = 500;
png.antiAliasing = true; png.transparency = false; png.artBoardClipping = true;
doc.exportFile(new File(base + ".png"), ExportType.PNG24, png);
var svg = new ExportOptionsSVG(); svg.embedRasterImages = true;
svg.fontType = SVGFontType.OUTLINEFONT;
doc.exportFile(new File(base + ".svg"), ExportType.SVG, svg);
doc.close(SaveOptions.DONOTSAVECHANGES);
"done";
""" % dict(out=out, name=name or NAME))
    return "\n".join(js)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = make_tiles()
    print("tiles from %(device)s, %(n_rays)d x %(n_points)d, coverage "
          "%(coverage).2f %%, threshold %(threshold)g" % info)
    ref12 = "--ref12" in sys.argv
    name = NAME + ("_ref12" if ref12 else "")
    if ref12:
        print("crops from Fig. 3 of %s: %s" % (REF12_PDF.name, ref12_tiles()))
    jsx = OUT / (name + ".jsx")
    jsx.write_text(build_jsx(info, ref12, name), encoding="utf-8")
    ps = ("$a = New-Object -ComObject Illustrator.Application; "
          "$a.UserInteractionLevel = -1; "
          "$a.DoJavaScriptFile('%s')" % str(jsx))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    for ext in ("ai", "pdf", "png", "svg"):
        f = OUT / ("%s.%s" % (name, ext))
        print("%-6s %s" % (ext, "ok" if f.exists() else "MISSING"), f)


if __name__ == "__main__":
    main()
