# -*- coding: utf-8 -*-
"""
fig_fit_steps.py -- fit_lines, one step per panel, on one test device.

    python docs/charge_states/fig_fit_steps.py

The steps re-run the same operations, in the same order and with the same
setting, as charge_states.fit_lines (the setting run_17 chose on the
validation devices); the last panel is checked to equal fit_lines' output.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from csdrecon.ml import grid_train                     # noqa: E402
from csdrecon.study import charge_states as C          # noqa: E402
from run_7_virtual_gates import model_dir              # noqa: E402

RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
CELL = "8_rays_60_points_500_samples"
DEVICE = "sample_223"
RUN_DIR = os.path.join(ROOT, "results", RUN)
OUT = os.path.join(RUN_DIR, "few_electron", "figures")
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")
INK = "#111111"
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5})


def main():
    with open(os.path.join(RUN_DIR, "few_electron", "fit_scan.csv")) as f:
        txt = next(r["setting"] for r in csv.DictReader(f)
                   if r["chosen"] == "1")
    min_len, max_ext, keep = next(k for k in C.FIT_GRID
                                  if C.fit_knob(k) == txt)
    z = np.load(os.path.join(RUN_DIR, CELL, "test.npz"), allow_pickle=True)
    names = [os.path.basename(str(s).replace("\\", "/"))
             for s in z["samples"]]
    i = names.index(DEVICE)
    net, ck = grid_train.load(os.path.join(RUN_DIR, CELL,
                                           model_dir(RUN_DIR, CELL),
                                           "unet.pt"))
    B = grid_train.predict(net, z["X"][i:i + 1])[0] > float(ck["threshold"])
    dev = C.read_device(os.path.join(POOL, DEVICE))
    cs = C.load_states(os.path.join(POOL, DEVICE))
    Y = z["Y"][i] > 0.5
    dx, dy = C.pixel_mV(dev, Y.shape)
    H, W = B.shape

    # the same operations as fit_lines, kept step by step for drawing
    skel = skeletonize(B)
    nb = ndimage.convolve(skel.astype(int), np.array([[1, 1, 1], [1, 0, 1],
                                                      [1, 1, 1]]),
                          mode="constant")
    lab, n = ndimage.label(skel & (nb < 3), structure=np.ones((3, 3)))
    segs, ext_drawn, ext_dropped = [], [], []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        piece = np.zeros_like(B)
        piece[sl] = lab[sl] == k
        r, c = np.nonzero(piece)
        if len(r) < min_len:
            continue
        mr, mc = r.mean(), c.mean()
        X = np.stack([r - mr, c - mc], 1)
        u = np.linalg.svd(X, full_matrices=False)[2][0]
        t = X @ u
        p0 = (mr + t.min() * u[0], mc + t.min() * u[1])
        p1 = (mr + t.max() * u[0], mc + t.max() * u[1])
        own = ndimage.distance_transform_edt(~piece) <= 2.5
        ends = []
        for j, p, d in ((np.argmax(t), p1, u), (np.argmin(t), p0, -u)):
            er, ec = r[j], c[j]
            w = skel[max(er - 1, 0):er + 2, max(ec - 1, 0):ec + 2]
            o = piece[max(er - 1, 0):er + 2, max(ec - 1, 0):ec + 2]
            if not (w & ~o).any():
                ends.append((p, d))
        segs.append((p0, p1, ends, own))
    for p0, p1, ends, own in segs:
        for start, d in ends:
            path, hit = [], False
            for s in range(1, max_ext + 1):
                rr = int(round(start[0] + s * d[0]))
                cc = int(round(start[1] + s * d[1]))
                if not (0 <= rr < H and 0 <= cc < W):
                    hit = True
                    break
                path.append((rr, cc))
                if B[rr, cc] and not own[rr, cc]:
                    hit = True
                    break
            (ext_drawn if hit else ext_dropped).append((start, path))
    F = C.fit_lines(B, min_len, max_ext, keep)

    ext = dev["voltage_window"]
    kw = dict(origin="lower", extent=ext, interpolation="nearest")
    bw = ListedColormap(["white", INK])
    to = lambda r, c: (ext[0] + np.asarray(c) * dx,          # noqa: E731
                       ext[2] + np.asarray(r) * dy)
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 4.4),
                           gridspec_kw=dict(wspace=0.1, hspace=0.42,
                                            left=0.07, right=0.995,
                                            top=0.93, bottom=0.08))
    ax = ax.ravel()
    ax[0].imshow(B, cmap=bw, vmin=0, vmax=1, **kw)
    ax[0].set_title("1. U-Net line map $B$\n(thick, with gaps)")
    ax[1].imshow(skel, cmap=bw, vmin=0, vmax=1, **kw)
    ax[1].set_title("2. thinned to 1 px\n(skeleton)")
    rng = np.random.default_rng(1)
    col = rng.random((n + 1, 3)) * 0.8
    img = np.ones((H, W, 3))
    img[lab > 0] = col[lab[lab > 0]]
    ax[2].imshow(img, **kw)
    ax[2].set_title("3. cut at crossings\n(one colour per piece)")
    ax[3].imshow(B, cmap=ListedColormap(["white", "#cccccc"]), vmin=0,
                 vmax=1, **kw)
    for p0, p1, _, _ in segs:
        x, y = to([p0[0], p1[0]], [p0[1], p1[1]])
        ax[3].plot(x, y, "-", color="#2a78d6", lw=1.0)
    ax[3].set_title("4. straight line fitted\nto each piece (blue)")
    ax[4].imshow(B, cmap=ListedColormap(["white", "#cccccc"]), vmin=0,
                 vmax=1, **kw)
    for p0, p1, _, _ in segs:
        x, y = to([p0[0], p1[0]], [p0[1], p1[1]])
        ax[4].plot(x, y, "-", color="#2a78d6", lw=0.8)
    for start, path in ext_dropped:
        if path:
            pr = [start[0]] + [p[0] for p in path]
            pc = [start[1]] + [p[1] for p in path]
            ax[4].plot(*to(pr, pc), ":", color="#999999", lw=0.8)
    for start, path in ext_drawn:
        pr = [start[0]] + [p[0] for p in path]
        pc = [start[1]] + [p[1] for p in path]
        ax[4].plot(*to(pr, pc), "-", color="#d62728", lw=1.4)
    ax[4].set_title("5. loose ends extended\n(red kept, grey dropped)")
    ax[5].imshow(F, cmap=bw, vmin=0, vmax=1, **kw)
    ax[5].set_title("6. result: $B$ + fitted lines\n+ kept extensions")
    r0 = C.label_states(B, dx, dy)
    r1 = C.label_states(F, dx, dy)
    for a, res, lab_ in ((ax[6], r0, "7. labels from $B$"),
                         (ax[7], r1, "8. labels from the result")):
        a.imshow(C._state_rgb(res["labels"]), **kw)
        C._write_labels(a, res["labels"], ext, fs=4)
        f = C.score_few(res, cs, Y)
        a.set_title(f"{lab_}\nbalanced acc. {100 * f['bal_few']:.0f} %")
    for c, a in enumerate(ax):
        C._map_axes(a, c % 4 == 0)
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
    fig.savefig(os.path.join(OUT, "fit_steps.png"), dpi=250)
    fig.savefig(os.path.join(OUT, "fit_steps.pdf"))
    plt.close(fig)
    print("extensions kept", len(ext_drawn), "dropped",
          sum(1 for _, p in ext_dropped if p), "pieces", len(segs),
          "setting", txt)


if __name__ == "__main__":
    main()
