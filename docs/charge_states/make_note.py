# -*- coding: utf-8 -*-
"""
make_note.py -- the figures and numbers for charge_states.pdf.

    python scripts/run_16_charge_states.py          # the study (tables, figs)
    python docs/charge_states/make_note.py          # step figures + values.tex
    latexmk -pdf charge_states.tex                  # (run in this folder)

The note walks one test device through label_states, step by step, twice:
on the full CSD's line map Y and on the binary map of one U-Net.  Every
panel is drawn from what the library functions return for that device, and
every number in the text is written into values.tex here or read from the
study's summary files -- nothing is typed in by hand.
"""
import csv
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from csdrecon.ml import grid_train                     # noqa: E402
from csdrecon.study import charge_states as C          # noqa: E402
from csdrecon.study import virtual_gates as V          # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_7_virtual_gates import model_dir              # noqa: E402

RUN = os.path.join(ROOT, "results",
                   "4-5-6-7-8_rays_40-50-60_points_500_samples")
CELL = "8_rays_60_points_500_samples"
RAYS, POINTS = 8, 60
DEVICE = "sample_223"            # the 'typical' device of the study's fig. 2
STUDY = os.path.join(RUN, "charge_states")
# the note is built inside the run it describes
OUT = os.path.join(STUDY, "method_note")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, MUTED = "#111111", "#555555"
FAM_COLOR = {"dot1": "#7b3fbf", "dot2": "#eb6834", "interdot": "#1baf7a",
             None: "#bbbbbb"}
FAM_LABEL = {"dot1": "dot 1", "dot2": "dot 2", "interdot": "interdot",
             None: "no direction"}
AGREE, DISAGREE = "#1baf7a", "#d62728"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 6.5, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5})

# ── the device and the two binary maps ──────────────────────────────────────
z = np.load(os.path.join(RUN, CELL, "test.npz"), allow_pickle=True)
names = [os.path.basename(str(s).replace("\\", "/")) for s in z["samples"]]
i = names.index(DEVICE)
sdir = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf", DEVICE)
dev = C.read_device(sdir)
cs = C.load_states(sdir)
Y = z["Y"][i] > 0.5
X = z["X"][i:i + 1]
net, ck = grid_train.load(os.path.join(RUN, CELL, model_dir(RUN, CELL), "unet.pt"))
THR = float(ck["threshold"])
prob = grid_train.predict(net, X)[0]
B_unet = prob > THR
dx, dy = C.pixel_mV(dev, Y.shape)
ext = dev["voltage_window"]
H, W = Y.shape
COV = float(X[0, 1].mean())

MAPS = {"full": ("full CSD", Y), "unet": (f"U-Net {RAYS}×{POINTS}", B_unet)}
RES = {k: C.label_states(B, dx, dy) for k, (_, B) in MAPS.items()}
SC = {k: C.score(RES[k], cs, Y, dx, dy) for k in MAPS}


def to_mV(r, c):
    return ext[0] + np.asarray(c) * dx, ext[2] + np.asarray(r) * dy


def style(ax, first=True, xlabel=True):
    ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.tick_params(length=2, pad=1)
    if xlabel:
        ax.set_xlabel("$V_\\mathrm{P1}$ (mV)", labelpad=1)
    else:
        ax.set_xticklabels([])
    if first:
        ax.set_ylabel("$V_\\mathrm{P2}$ (mV)", labelpad=1)
    else:
        ax.set_yticklabels([])


KW = dict(origin="lower", extent=ext, interpolation="nearest")
BW = ListedColormap(["white", INK])


def region_rgb(F, n, seed=3):
    rng = np.random.default_rng(seed)
    pal = plt.get_cmap("tab20")(rng.permutation(20) % 20)[:, :3]
    rgb = pal[F % 20]
    rgb[F == 0] = 1.0
    return rgb


def centroids(F, n):
    out = {}
    for k in range(1, n + 1):
        r, c = np.nonzero(F == k)
        out[k] = to_mV(r.mean(), c.mean())
    return out


# ── fig. A: from the probability map to regions ─────────────────────────────
def fig_regions():
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.95),
                            gridspec_kw=dict(wspace=0.08, hspace=0.32,
                                             left=0.07, right=0.995,
                                             top=0.93, bottom=0.09))
    for row, key in enumerate(("full", "unet")):
        name, B = MAPS[key]
        ax = axs[row]
        if key == "full":
            ax[0].imshow(Y, cmap=BW, vmin=0, vmax=1, **KW)
            ax[0].set_title("(a) full CSD line map $Y$")
        else:
            im = ax[0].imshow(prob, cmap="viridis", vmin=0, vmax=1, **KW)
            ax[0].contour(prob, [THR], colors="white", linewidths=0.5,
                          origin="lower", extent=ext)
            ax[0].set_title(f"(e) U-Net $P$, contour at {THR:g}")
        Bc = C.close_frame(B)
        show = np.zeros(B.shape, int)
        show[B] = 1
        show[Bc & ~np.asarray(B, bool)] = 2
        ax[1].imshow(show, cmap=ListedColormap(["white", INK, "#d62728"]),
                     vmin=0, vmax=2, **KW)
        ax[1].set_title(("(b)" if row == 0 else "(f)")
                        + " $B$, frame closed")
        lab0, n0 = C.ndimage.label(~Bc, structure=C._FOUR)
        lab, n = C.regions(B)
        crumbs = (lab0 > 0) & (lab == 0) & ~Bc
        rgb = region_rgb(lab, n)
        rgb[Bc] = 0.15
        rgb[crumbs] = (0.84, 0.15, 0.16)
        ax[2].imshow(rgb, **KW)
        ax[2].set_title(("(c)" if row == 0 else "(g)")
                        + f" regions: {n} (crumbs red)")
        F = RES[key]["regions"]
        ax[3].imshow(region_rgb(F, n), **KW)
        ax[3].set_title(("(d)" if row == 0 else "(h)")
                        + " lines filled")
        for c, a in enumerate(ax):
            style(a, first=c == 0, xlabel=row == 1)
    fig.text(0.5, 0.005, f"{DEVICE}.  Red in (b),(f): pixels added on the "
             "frame by close_frame.  Red in (c),(g): components under "
             f"{C.MIN_REGION_PX} px, treated as line.", ha="center",
             fontsize=6, color=MUTED)
    fig.savefig(os.path.join(FIG, "step1_regions.pdf"))
    fig.savefig(os.path.join(FIG, "step1_regions.png"), dpi=200)
    plt.close(fig)


# ── fig. B: borders, their direction and family ─────────────────────────────
def otsu_cut(bs):
    neg = [e for e in bs if e["family"] in ("dot1", "dot2")]
    ang = np.array([180.0 if e["theta"] == 0.0 else e["theta"] for e in neg])
    w = np.array([e["length"] for e in neg], float)
    if len(neg) >= 2 and np.ptp(ang) > 0:
        return ang, w, V._otsu_split(ang, w)
    return ang, w, 135.0


def fig_borders():
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=0.28,
                          hspace=0.4, left=0.07, right=0.985, top=0.92,
                          bottom=0.1)
    tv = V.true_gates(dev["capacitance"])["angles"]
    for row, key in enumerate(("full", "unet")):
        res = RES[key]
        F = res["regions"]
        a0 = fig.add_subplot(gs[row, 0])
        a0.imshow(region_rgb(F, res["n_regions"]) * 0.35 + 0.65, **KW)
        for e in res["borders"]:
            x, y = to_mV(e["points"][:, 0], e["points"][:, 1])
            a0.plot(x, y, "s", ms=0.9, mec="none",
                    color=FAM_COLOR[e["family"]])
        a0.set_title(("(a) " if row == 0 else "(d) ") + MAPS[key][0]
                     + ": borders by family")
        style(a0, first=True, xlabel=row == 1)

        # the step across every border: an arrow into the region that gains
        a1 = fig.add_subplot(gs[row, 1])
        a1.imshow(region_rgb(F, res["n_regions"]) * 0.35 + 0.65, **KW)
        for e in res["borders"]:
            if e["family"] is None:
                continue
            p = e["points"]
            r, c = p[:, 0].mean(), p[:, 1].mean()
            x, y = to_mV(r, c)
            nx, ny = e["normal"]
            if np.hypot(nx, ny) == 0:      # contacts cancel: no direction
                continue
            u = np.array([nx, ny]) / np.hypot(nx, ny)
            # the normal points into a; a gains when the step's sum ...
            s = e["step"]
            into_a = s[0] > 0 or (s[0] == 0 and s[1] > 0)
            if not into_a:
                u = -u
            L = 0.07 * (ext[1] - ext[0])
            a1.annotate("", xy=(x + u[0] * L, y + u[1] * L), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                        color=FAM_COLOR[e["family"]],
                                        mutation_scale=5))
        a1.set_title(("(b)" if row == 0 else "(e)")
                     + " step: arrow into the side that gains")
        style(a1, first=False, xlabel=row == 1)

        a2 = fig.add_subplot(gs[row, 2])
        ang, w, cut = otsu_cut(res["borders"])
        pos = [e for e in res["borders"] if e["family"] == "interdot"]
        bins = np.arange(0, 181, 5)
        if len(ang):
            a2.hist(ang, bins=bins, weights=w, color="#999999",
                    label="negative-slope borders")
        if pos:
            a2.hist([e["theta"] for e in pos], bins=bins,
                    weights=[e["length"] for e in pos],
                    color=FAM_COLOR["interdot"], label="interdot (slope > 0)")
        a2.axvline(cut, color=INK, lw=0.8, ls="--")
        a2.text(cut, a2.get_ylim()[1] * 0.97, " Otsu cut", fontsize=5.5,
                va="top")
        for f in ("dot1", "dot2", "interdot"):
            a2.axvline(tv[f], color=FAM_COLOR[f], lw=0.8)
        a2.set_xlim(0, 180)
        a2.set_xlabel("border angle θ (deg)")
        a2.set_ylabel("border length (contacts)")
        a2.set_title(("(c)" if row == 0 else "(f)")
                     + " angle histogram, length-weighted")
        if row == 0:
            a2.legend(fontsize=5, frameon=False, loc="upper left")
    h = [Line2D([], [], marker="s", ls="none", ms=3, color=FAM_COLOR[f],
                label=FAM_LABEL[f]) for f in ("dot1", "dot2", "interdot",
                                              None)]
    h.append(Line2D([], [], color=MUTED, lw=0.8,
                    label="true angle (capacitances), in (c),(f)"))
    fig.legend(handles=h, loc="upper center", ncol=5, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIG, "step2_borders.pdf"))
    fig.savefig(os.path.join(FIG, "step2_borders.png"), dpi=200)
    plt.close(fig)


# ── fig. C: the tree, the labels, the consistency ───────────────────────────
def fig_tree():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.1),
                            gridspec_kw=dict(wspace=0.08, hspace=0.32,
                                             left=0.07, right=0.995,
                                             top=0.9, bottom=0.09))
    for row, key in enumerate(("full", "unet")):
        res = RES[key]
        F, n = res["regions"], res["n_regions"]
        cen = centroids(F, n)
        root = int(F[-1, -1])
        ax = axs[row]
        ax[0].imshow(region_rgb(F, n) * 0.35 + 0.65, **KW)
        tree = set(res["tree"])
        L = {}
        for e in res["borders"]:
            if e["family"] is None:
                continue
            (xa, ya), (xb, yb) = cen[e["a"]], cen[e["b"]]
            lw = 0.4 + 2.2 * e["length"] / max(b["length"]
                                               for b in res["borders"])
            if (e["a"], e["b"]) in tree:
                ax[0].plot([xa, xb], [ya, yb], "-", color=INK, lw=lw)
        for e in res["borders"]:
            if e["family"] is None or (e["a"], e["b"]) in tree:
                continue
            (xa, ya), (xb, yb) = cen[e["a"]], cen[e["b"]]
            la = res["labels"][F == e["a"]][0]
            lb = res["labels"][F == e["b"]][0]
            ok = (la - lb == e["step"]).all()
            ax[0].plot([xa, xb], [ya, yb], "--", lw=0.8,
                       color=AGREE if ok else DISAGREE)
        xs = [cen[k][0] for k in cen]
        ys = [cen[k][1] for k in cen]
        ax[0].plot(xs, ys, "o", ms=2.2, color=INK)
        ax[0].plot(*cen[root], "*", ms=8, color="#f2c200", mec=INK, mew=0.4)
        ax[0].set_title(("(a) " if row == 0 else "(d) ") + MAPS[key][0]
                        + ": maximum spanning tree")
        Ls = res["labels"]
        ax[1].imshow(C._state_rgb(Ls), **KW)
        C._write_labels(ax[1], Ls, ext, fs=5)
        ax[1].set_title(("(b)" if row == 0 else "(e)")
                        + " labels, (0,0) at the top-right corner")
        ax[2].imshow(C._state_rgb(cs), **KW)
        C._write_labels(ax[2], cs, ext, fs=5)
        s = SC[key]
        con = ("–" if not np.isfinite(s["consistency"])
               else f"{s['consistency']:.2f}")
        ax[2].set_title(("(c)" if row == 0 else "(f)")
                        + f" truth;  acc. {100 * s['acc_rel']:.1f} %, "
                        f"consistency {con}")
        for c, a in enumerate(ax):
            style(a, first=c == 0, xlabel=row == 1)
    h = [Line2D([], [], color=INK, lw=1.5, label="tree edge (width ∝ "
                "border length)"),
         Line2D([], [], color=AGREE, ls="--", lw=0.8,
                label="non-tree border, step agrees"),
         Line2D([], [], color=DISAGREE, ls="--", lw=0.8,
                label="non-tree border, step disagrees"),
         Line2D([], [], marker="*", ls="none", ms=7, color="#f2c200",
                mec=INK, mew=0.4, label="root = top-right region, (0,0)")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIG, "step3_tree_labels.pdf"))
    fig.savefig(os.path.join(FIG, "step3_tree_labels.png"), dpi=200)
    plt.close(fig)


# ── fig. D: scoring ─────────────────────────────────────────────────────────
def fig_scoring():
    fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.15),
                            gridspec_kw=dict(wspace=0.08, left=0.09,
                                             right=0.995, top=0.88,
                                             bottom=0.17))
    m = C.keep_mask(Y)
    axs[0].imshow(~m, cmap=ListedColormap(["#d9d9d9", "white"]), **KW)
    axs[0].set_title(f"(a) scored pixels: {100 * m.mean():.1f} %")
    res = RES["unet"]
    Ls = res["labels"]
    sh = C.best_shift(Ls, cs, m)
    Lsh = np.where(Ls == -999, -999, Ls + sh)
    right = (Lsh == cs).all(-1)
    img = np.ones(Y.shape + (3,))
    img[m & right] = 0.82
    d = Lsh - cs
    e1, e2 = d[..., 0] != 0, d[..., 1] != 0
    img[m & e1 & ~e2] = C.ERR_N1
    img[m & ~e1 & e2] = C.ERR_N2
    img[m & e1 & e2] = C.ERR_BOTH
    axs[1].imshow(img, **KW)
    axs[1].set_title(f"(b) U-Net errors, shift ({sh[0]},{sh[1]})")
    base = np.zeros(Y.shape + (3,)) + 1
    zz = (cs[..., 0] == 0) & (cs[..., 1] == 0)
    base[m & zz] = 0.82
    base[m & ~zz] = C.ERR_BOTH
    axs[2].imshow(base, **KW)
    axs[2].set_title(f"(c) 'all (0,0)': {100 * zz[m].mean():.1f} % right")
    for c, a in enumerate(axs):
        style(a, first=c == 0)
    fig.savefig(os.path.join(FIG, "step4_scoring.pdf"))
    fig.savefig(os.path.join(FIG, "step4_scoring.png"), dpi=200)
    plt.close(fig)


# ── the numbers ─────────────────────────────────────────────────────────────
def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def baseline(split="test"):
    """Accuracy of labelling every scored pixel (0,0), per device."""
    rows = [r for r in read_summary(os.path.join(STUDY, "per_device.csv"))
            if r["split"] == split and r["method"] == "full CSD"]
    v = []
    for r in rows:
        d = os.path.join(ROOT, "data", "_device_pools",
                         "devices_n550_res100_c1df7b6bf", r["device"])
        s = C.load_states(d)
        y = np.load(os.path.join(d, "numpy", "simulation",
                                 "ground_truth_labels.npy")) > 0.5
        m = C.keep_mask(y)
        v.append(((s[..., 0] == 0) & (s[..., 1] == 0))[m].mean())
    return np.array(v)


def values():
    S = {(r["method"], r["rays"], r["points"]): r
         for r in read_summary(os.path.join(STUDY, "summary.csv"))}
    full = S[("full CSD", "", "")]
    un = sorted([r for k, r in S.items() if k[0] == "U-Net"],
                key=lambda r: float(r["coverage_%"]))
    mid = S[("U-Net", str(RAYS), str(POINTS))]
    best = un[-1]
    b = baseline()
    var = {}
    with open(os.path.join(STUDY, "per_device.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and r["method"] == "full CSD":
                var[r["device"]] = r
    nst = np.array([float(r["n_states"]) for r in var.values()])
    reg = np.array([float(r["median_region_px"]) for r in var.values()])
    corner = sum(int(float(r["corner_is_00"])) for r in var.values())
    sv = SC["unet"]
    sf = SC["full"]

    def f1(x, k="mean"):
        return f"{100 * float(x[f'{k}']):.1f}" if False else f"{float(x):.1f}"
    v = {
        "Device": DEVICE.replace("_", r"\_"),
        "Threshold": f"{THR:g}",
        "Coverage": f"{100 * COV:.2f}",
        "MinRegion": str(C.MIN_REGION_PX),
        "MinBorder": str(C.MIN_BORDER_PX),
        "NRegFull": str(RES["full"]["n_regions"]),
        "NRegUnet": str(RES["unet"]["n_regions"]),
        "NStatesDev": str(len(np.unique(C.state_code(cs)))),
        "AccDevFull": f"{100 * sf['acc_rel']:.1f}",
        "AccDevUnet": f"{100 * sv['acc_rel']:.1f}",
        "RecDevUnet": f"{100 * sv['state_recall']:.0f}",
        "ConDevUnet": ("--" if not np.isfinite(sv["consistency"])
                       else f"{sv['consistency']:.2f}"),
        "NTest": full["n_devices"],
        "NStMed": f"{np.median(nst):.0f}", "NStMin": f"{nst.min():.0f}",
        "NStMax": f"{nst.max():.0f}",
        "RegMin": f"{reg.min():.0f}", "RegMax": f"{reg.max():.0f}",
        "CornerOO": str(corner),
        "FullAcc": f"{float(full['acc_rel_mean']):.1f}",
        "FullRec": f"{float(full['state_recall_mean']):.1f}",
        "MidAcc": f"{float(mid['acc_rel_mean']):.1f}",
        "MidAccMed": f"{float(mid['acc_rel_median']):.1f}",
        "MidRec": f"{float(mid['state_recall_mean']):.1f}",
        "MidIoU": f"{float(mid['iou_11_mean']):.2f}",
        "MidIoUMed": f"{float(mid['iou_11_median']):.2f}",
        "MidCon": f"{float(mid['consistency_mean']):.2f}",
        "MidCov": f"{float(mid['coverage_%']):.2f}",
        "LoAcc": f"{float(un[0]['acc_rel_mean']):.1f}",
        "LoRec": f"{float(un[0]['state_recall_mean']):.1f}",
        "LoCov": f"{float(un[0]['coverage_%']):.2f}",
        "HiAcc": f"{float(best['acc_rel_mean']):.1f}",
        "HiAccMed": f"{float(best['acc_rel_median']):.1f}",
        "HiRec": f"{float(best['state_recall_mean']):.1f}",
        "HiCov": f"{float(best['coverage_%']):.2f}",
        "HiBudget": f"{best['rays']} $\\times$ {best['points']}",
        "BaseDev": f"{100 * ((cs[..., 0] == 0) & (cs[..., 1] == 0))[C.keep_mask(Y)].mean():.1f}",
        "ScoredDev": f"{100 * C.keep_mask(Y).mean():.1f}",
        "BaseMean": f"{100 * b.mean():.1f}",
        "BaseStd": f"{100 * b.std():.1f}",
    }
    with open(os.path.join(OUT, "values.tex"), "w", encoding="utf-8") as f:
        f.write("% written by make_note.py -- do not edit\n")
        for k, x in v.items():
            f.write(f"\\newcommand{{\\v{k}}}{{{x}}}\n")
    return v


def copy_study():
    """The study's tables and figures, so the note compiles on its own."""
    for n in ("fig1_device_variation", "fig2_labelling_examples",
              "fig3_accuracy_vs_coverage", "fig4_failures"):
        shutil.copy2(os.path.join(STUDY, "figures", n + ".pdf"),
                     os.path.join(FIG, n + ".pdf"))
    for n in ("summary.tex", "summary_validation.tex"):
        shutil.copy2(os.path.join(STUDY, n), os.path.join(OUT, n))


if __name__ == "__main__":
    fig_regions()
    fig_borders()
    fig_tree()
    fig_scoring()
    v = values()
    copy_study()
    print("written:", FIG)
    for k in ("AccDevFull", "AccDevUnet", "NRegFull", "NRegUnet"):
        print(" ", k, v[k])
