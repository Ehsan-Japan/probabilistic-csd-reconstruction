"""
make_detector_explained.py -- a short PDF answering two questions:
the U-Net compared with a raster scan of equal cost and with the full scan;
how the raster size is chosen; in which direction the step detector
compares pixels; and that only measured pixels enter the decision.

    python docs/charge_states/make_detector_explained.py

Writes UNet_vs_raster_vs_full_scan.docx/.pdf into the project root
(the PDF via Word).  Figures go to docs/charge_states/detector_explained/.
Numbers: results/<run>/two_step/table.csv and the recomputation below.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.colors import to_rgb, LogNorm            # noqa: E402
from matplotlib.patches import Rectangle, Patch          # noqa: E402
from scipy.ndimage import median_filter                  # noqa: E402
from skimage.morphology import skeletonize               # noqa: E402

import run_19_two_step as R                              # noqa: E402
from csdrecon.ml import grid_train                       # noqa: E402
from csdrecon.study import charge_states as C           # noqa: E402
from run_7_virtual_gates import model_dir               # noqa: E402

DEVICE, BAND, Q = "sample_223", 5, 0.8
FIG = os.path.join(HERE, "detector_explained")
DOCX = os.path.join(ROOT, "UNet_vs_raster_vs_full_scan.docx")
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def grid_points(n_side, shape=(100, 100)):
    """Where the k x k raster sits (as in run_18.coarse_grid)."""
    m = np.zeros(shape, bool)
    ri = np.rint(np.linspace(0, shape[0] - 1, n_side)).astype(int)
    ci = np.rint(np.linspace(0, shape[1] - 1, n_side)).astype(int)
    m[np.ix_(ri, ci)] = True
    return m


def per_axis(Z, measured):
    """step_residual_map split into its two directions."""
    Z = np.asarray(Z, dtype=float)
    ref = Z[measured]
    Zs = (Z - ref.min()) / np.ptp(ref)
    out = []
    for ax in (0, 1):
        d = np.diff(Zs, axis=ax)
        b = median_filter(d, size=(5, 1) if ax == 0 else (1, 5),
                          mode="nearest")
        both = (measured[1:] & measured[:-1]) if ax == 0 else \
            (measured[:, 1:] & measured[:, :-1])
        r = np.abs(d - b) / (np.abs(b) + np.median(np.abs(d[both])))
        out.append(np.pad(r, [(0, 1), (0, 0)] if ax == 0 else [(0, 0),
                                                                 (0, 1)]))
    return out                       # [along V2 (vertical), along V1]


def gridlines(a, n=100, c="#d0d0d0", lw=0.15):
    for k in range(n + 1):
        a.axvline(k - 0.5, color=c, lw=lw)
        a.axhline(k - 0.5, color=c, lw=lw)


def fig_cost(D, i, path, k_u, k_b):
    X = D["X"][i]
    rays = X[1] > 0.5
    centre = skeletonize(D["B"][i])
    strip = R.dilate(centre, BAND) | rays
    panels = [(rays, "(a) U-Net: 8 rays × 60 points"),
              (grid_points(k_u), f"(b) raster {k_u} × {k_u}"),
              (strip, "(c) U-Net + ±5 px strip"),
              (grid_points(k_b), f"(d) raster {k_b} × {k_b}")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15))
    for a, (m, t) in zip(axs, panels):
        img = np.ones(m.shape + (3,))
        img[m] = to_rgb("#1f4e8c")
        a.imshow(img, origin="lower", interpolation="nearest")
        a.set_title(f"{t}\n{100 * m.mean():.1f}% of the pixels",
                    fontsize=7.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_neighbours(path):
    n = 11
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for a, ax_dir in zip(axs, ("V1", "V2")):
        img = np.ones((n, n, 3))
        k = 5
        for s in range(-2, 4):
            if ax_dir == "V1":
                img[k, k + s] = to_rgb("#a9c9e8")
            else:
                img[k + s, k] = to_rgb("#a9c9e8")
        img[k, k] = to_rgb("#c00000")
        a.imshow(img, origin="lower", interpolation="nearest")
        gridlines(a, n, "#888888", 0.5)
        nb = (k, k + 1) if ax_dir == "V1" else (k + 1, k)
        if ax_dir == "V1":
            a.annotate("", xy=(k + 1, k), xytext=(k, k),
                       arrowprops=dict(arrowstyle="->", lw=1.2))
        else:
            a.annotate("", xy=(k, k + 1), xytext=(k, k),
                       arrowprops=dict(arrowstyle="->", lw=1.2))
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("V$_1$ →")
        a.set_ylabel("V$_2$ →")
        a.set_title(f"({'a' if ax_dir == 'V1' else 'b'}) comparison along "
                    f"{ax_dir}", fontsize=8)
        del nb
    fig.legend(handles=[Patch(color="#c00000", label="pixel k being judged"),
                        Patch(color="#a9c9e8",
                              label="pixels whose values are used "
                                    "(k−2 … k+3)")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_directions(D, i, path):
    Y, B, Z, X = D["Y"][i], D["B"][i], D["Z"][i], D["X"][i]
    centre = skeletonize(B)
    measured = R.dilate(centre, BAND) | (X[1] > 0.5)
    checked = R.dilate(centre, BAND - 2) & R.inside(measured)
    rv, rh = per_axis(Z, measured)
    full = C.step_residual_map(Z, 5, True, measured)
    assert np.allclose(np.maximum(rv, rh), full)
    thr = float(np.quantile(full[checked], Q))
    kept = checked & (full > thr)
    assert np.array_equal(kept, R.two_step(D, i, Q, BAND)[0])

    fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.0))
    axs = axs.ravel()
    Zs = np.where(measured, Z, np.nan)
    axs[0].imshow(Zs, origin="lower", cmap="viridis",
                  interpolation="nearest")
    axs[0].set_title("(a) measured signal", fontsize=7.5)
    norm = LogNorm(1e-3, 3)
    for a, m, t in ((axs[1], rh, "(b) step value along V$_1$\n"
                     "(horizontal neighbours)"),
                    (axs[2], rv, "(c) step value along V$_2$\n"
                     "(vertical neighbours)")):
        im = a.imshow(np.where(checked, np.maximum(m, 1e-3), np.nan),
                      origin="lower", cmap="magma_r", norm=norm,
                      interpolation="nearest")
        a.set_title(t, fontsize=7.5)
    img = np.ones(Y.shape + (3,))
    img[checked] = to_rgb("#e6f0f9")
    img[Y] = to_rgb("#9a9a9a")
    img[kept] = to_rgb("#c00000")
    axs[3].imshow(img, origin="lower", interpolation="nearest")
    axs[3].set_title("(d) larger of (b), (c) > threshold:\nkept (red), "
                     "true (grey)", fontsize=7.5)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("V$_1$ →", fontsize=7)
        a.set_ylabel("V$_2$ →", fontsize=7)
    cb = fig.colorbar(im, ax=list(axs), shrink=0.45, pad=0.03)
    cb.set_label("step value", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return thr


def leak_study(val, test):
    """The earlier version: every pixel within BAND-2 of the centre lines
    examined, whether or not its detector window was measured."""
    def earlier(D, i, q):
        centre = skeletonize(D["B"][i])
        measured = R.dilate(centre, BAND) | (D["X"][i, 1] > 0.5)
        checked = R.dilate(centre, BAND - 2)
        return R.steps(D["Z"][i], checked, q, measured), \
            float(measured.mean())
    frac = []
    for i in range(len(test["Y"])):
        c = skeletonize(test["B"][i])
        m = R.dilate(c, BAND) | (test["X"][i, 1] > 0.5)
        ch = R.dilate(c, BAND - 2)
        frac.append(1 - (ch & R.inside(m)).sum() / ch.sum())
    q = R.choose(val, earlier)
    return (100 * float(np.mean(frac)), 100 * float(np.max(frac)),
            R.run(test, earlier, q))


def table_rows():
    with open(os.path.join(R.OUT, "table.csv"), newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_doc(thr, leak, tab):
    import docx
    from docx.shared import Pt, Cm
    doc = docx.Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(10)
    st.paragraph_format.space_after = Pt(5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2.0)

    def H(t):
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.bold = True
        r.font.size = Pt(12)

    def P(t):
        doc.add_paragraph(t)

    def B(t):
        p = doc.add_paragraph(t, style="List Bullet")
        p.paragraph_format.space_after = Pt(2)

    def T(rows, widths):
        tb = doc.add_table(rows=len(rows), cols=len(rows[0]))
        tb.style = "Table Grid"
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                cell = tb.cell(r, c)
                cell.text = v
                cell.paragraphs[0].runs[0].font.size = Pt(9)
                cell.paragraphs[0].runs[0].bold = (r == 0)
                cell.width = Cm(widths[c])
        doc.add_paragraph()

    def F(name, w=16.5):
        doc.add_picture(os.path.join(FIG, name), width=Cm(w))

    def row(method, setting=""):
        return next(r for r in tab if r["method"] == method
                    and r["setting"].startswith(setting))

    title = doc.add_paragraph()
    rt = title.add_run("U-Net, raster scan and full scan: how the lines "
                       "are found, and how the comparison is made fair")
    rt.bold = True
    rt.font.size = Pt(15)

    def raster(matched):
        return next(r for r in tab if r["method"] == "coarse scan"
                    and r["matched"] == matched)

    def k_of(r):
        return int(r["setting"].split(" x ")[0])

    def f3(r, key="f1"):
        return f"{float(r[key]):.3f}"
    u, b5 = row("U-Net only"), row("U-Net, then local check", "band ±5")
    cu, cb5, fs = raster("U-Net only"), raster("band ±5 px"), row("full scan")

    # 1 ---------------------------------------------------------------
    H("1. The comparison")
    P("Three ways of finding the transition lines, all scored on the same "
      "50 test devices (F1 with a tolerance of 1 pixel):")
    B("U-Net: measure along 8 rays, let the U-Net predict the lines, then "
      "measure a strip around them and keep the lines where the signal "
      "steps.")
    B("Raster scan: the conventional measurement, a regular k × k grid of "
      "points, then the same line detector.")
    B("Full scan: every one of the 100 × 100 pixels, then the same line "
      "detector. This is the reference that the method has to match.")
    T([["Method", "Measured", "Precision", "Recall", "F1@1"],
       ["U-Net only (rays)", f"{float(u['measured']):.1f}%", f3(u,
        "precision"), f3(u, "recall"), f3(u)],
       [f"Raster {k_of(cu)} × {k_of(cu)} (same cost)",
        f"{float(cu['measured']):.1f}%", f3(cu, "precision"),
        f3(cu, "recall"), f3(cu)],
       ["U-Net + ±5 px strip", f"{float(b5['measured']):.1f}%",
        f3(b5, "precision"), f3(b5, "recall"), f3(b5)],
       [f"Raster {k_of(cb5)} × {k_of(cb5)} (same cost)",
        f"{float(cb5['measured']):.1f}%", f3(cb5, "precision"),
        f3(cb5, "recall"), f3(cb5)],
       ["Full scan", "100%", f3(fs, "precision"), f3(fs, "recall"),
        f3(fs)]],
      (6.6, 2.2, 2.2, 2.2, 2.0))

    H("2. Why a raster scan, and why these sizes")
    P("Without the U-Net, the natural way to save measurement time is to "
      "measure the diagram on a coarser grid: a raster scan. It is what an "
      "experimentalist would do anyway, so it is the fair competitor. The "
      "question is: with the same number of measured points, who finds the "
      "lines better, the raster or the U-Net?")
    P("The size is therefore not free. For each U-Net setting, the raster "
      "gets k × k points, with k the smallest integer for which k² is at "
      "least the number of pixels that the U-Net setting measures (on "
      "average over the test devices). The raster thus never gets fewer "
      "measurements than the U-Net; if anything, a few more.")
    T([["U-Net setting", "Pixels measured", "Raster", "F1@1 U-Net / raster"],
       ["rays only", f"{float(u['measured']) * 100:.0f}",
        f"√{float(u['measured']) * 100:.0f} → {k_of(cu)} × {k_of(cu)} = "
        f"{k_of(cu) ** 2}", f"{float(u['f1']):.2f} / {float(cu['f1']):.2f}"],
       ["rays + ±5 px strip", f"{float(b5['measured']) * 100:.0f}",
        f"√{float(b5['measured']) * 100:.0f} → {k_of(cb5)} × {k_of(cb5)} = "
        f"{k_of(cb5) ** 2}",
        f"{float(b5['f1']):.2f} / {float(cb5['f1']):.2f}"]],
      (4.0, 3.0, 5.0, 4.4))
    F("cost.png")
    P("Fig. 1. The measured pixels (blue) of one test device. (a) and (b) "
      "cost the same. The strip in (c) depends on the predicted lines, so "
      "its cost varies from device to device; the raster in (d) matches its "
      "average over the 50 test devices, not this one device. The raster "
      "spreads its points evenly; the U-Net spends them along rays, then "
      "near the predicted lines.")
    P("The ±3, ±4 and ±6 px strips get their own matched rasters in the "
      "same way (Table III and Fig. 16 of the manuscript). For each raster "
      "the lines were found either on the coarse grid itself or after "
      "upsampling it to 100 × 100, whichever was better on the validation "
      "devices, so the raster is given its best chance.")

    # 2 ---------------------------------------------------------------
    H("3. How the algorithm finds the lines")
    P("There is no sweep in the algorithm itself. First the pixels of the "
      "strip are measured (in this simulation study, each pixel's value is "
      "taken from the simulated diagram). Then, for every pixel of the "
      "checked region, the algorithm compares the pixel with its "
      "neighbours in BOTH directions: horizontally (along V1) and "
      "vertically (along V2).")
    P("For one direction, at pixel k:")
    B("d = signal(k+1) − signal(k): the change to the next pixel.")
    B("b = median of d over the five positions k−2 … k+2: the usual change "
      "nearby, i.e. the smooth slope of the sensor background.")
    B("step value = |d − b| / (|b| + s), where s is the typical |d| over "
      "all measured pixels. On a smooth background d ≈ b and the step "
      "value is close to 0; at a transition the signal jumps, d is far "
      "from b, and the step value is large.")
    P("This is done once along V1 and once along V2, and the larger of the "
      "two step values is kept. Both directions are needed because the "
      "lines have different orientations. A nearly horizontal line is "
      "crossed when moving vertically, so it appears in the V2 comparison; "
      "a nearly vertical line appears in the V1 comparison; a diagonal "
      "line appears in both.")
    F("neighbours.png", 13.0)
    P("Fig. 2. The pixels that enter the step value of pixel k (red): its "
      "difference to the next pixel (arrow) and the four neighbouring "
      "differences for the median. Six pixels in a row (a) or a column (b).")
    F("directions.png", 13.5)
    P(f"Fig. 3. The two directions on test device {DEVICE}. (a) The "
      "measured signal (strip and rays). (b) Step value from horizontal "
      "neighbours: the near-vertical lines light up. (c) Step value from "
      "vertical neighbours: the near-horizontal lines light up. (d) Pixels "
      f"whose larger step value exceeds the threshold ({thr:.3f}) are kept "
      "as line pixels (red), shown over the true lines (grey).")
    P("The threshold is not a fixed number. Among all the checked pixels "
      f"of one device, the strongest {100 * (1 - Q):.0f}% are kept "
      f"(the {100 * Q:.0f}th percentile). This fraction was chosen on the "
      "validation devices, never on the test devices.")
    P("In a real experiment, the strip would be measured by short voltage "
      "sweeps across each predicted line. The direction of these sweeps "
      "does not matter for the algorithm, because it only needs the strip "
      "pixels to be measured; the two comparisons are made afterwards. The "
      "cost is counted as the number of measured pixels.")

    # 4 ---------------------------------------------------------------
    doc.add_page_break()
    H("4. Only measured pixels enter the decision")
    pct, mx, old = leak
    P("Fig. 2 shows that the step value at pixel k uses the pixels "
      "k−2 … k+3, one more on the + side than on the − side. In an earlier "
      "version, every pixel within ±3 px of the U-Net centre lines was "
      "examined; a pixel at distance 3 can need a pixel at distance 6, "
      "outside the measured ±5 px strip. In the simulation that value was "
      f"available and was used. This concerned {pct:.1f}% of the examined "
      f"pixels on average (at most {mx:.1f}% on one device).")
    P("This is now corrected: a pixel is examined only if all six pixels "
      "its step value uses, in both directions, were measured. For the "
      "same reason a strip must extend at least ±3 px; with ±1 or ±2 px "
      "almost no pixel can be examined, so these strips were dropped.")
    T([["±5 px strip", "Measured", "Precision", "Recall", "F1@1"],
       ["earlier version (some unmeasured pixels used)",
        f"{old['measured']:.1f}%", f"{old['precision']:.3f}",
        f"{old['recall']:.3f}", f"{old['f1']:.3f}"],
       ["corrected (in the manuscript now)",
        f"{float(b5['measured']):.1f}%", f3(b5, "precision"),
        f3(b5, "recall"), f3(b5)]],
      (8.0, 2.0, 1.8, 1.8, 1.8))
    doc.save(DOCX)


def main():
    os.makedirs(FIG, exist_ok=True)
    net, ck = grid_train.load(os.path.join(R.RUN_DIR, R.CELL,
                                           model_dir(R.RUN_DIR, R.CELL),
                                           "unet.pt"))
    thr_u = float(ck["threshold"])
    val, test = R.load(net, thr_u, "val"), R.load(net, thr_u, "test")
    i = test["names"].index(DEVICE)
    tab = table_rows()
    ks = {r["matched"]: int(r["setting"].split(" x ")[0]) for r in tab
          if r["method"] == "coarse scan"}
    fig_cost(test, i, os.path.join(FIG, "cost.png"), ks["U-Net only"],
             ks[f"band ±{BAND} px"])
    fig_neighbours(os.path.join(FIG, "neighbours.png"))
    thr = fig_directions(test, i, os.path.join(FIG, "directions.png"))
    leak = leak_study(val, test)
    print(leak)
    build_doc(thr, leak, tab)
    print("wrote", DOCX)


if __name__ == "__main__":
    main()
