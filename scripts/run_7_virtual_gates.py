"""
Virtual-gate matrices from the full CSD and from every U-Net.

    python scripts/run_7_virtual_gates.py

For each device of the split, three matrices:

    G_true   from the capacitances in device.json
    G_full   slopes_from_lines(Y)                    (the full CSD's line map)
    G_unet   slopes_from_lines(prob > threshold)     (one per U-Net)

slopes_from_lines is the same function, called the same way, for both maps.
The thresholds are the ones stored in each unet.pt (chosen on the validation
split during training); nothing is re-tuned here and nothing is trained.

The test split (test.npz) is the main table.  The same table is made for the
75 validation devices (VAL_FRACTION of train.npz, the slice grid_train.train
carved out with SEED) and labelled "validation": they chose each threshold,
so they are not held out.

Writes results/<run>/virtual_gates/:
    per_device.csv              every number per (split, method, device)
    summary.md/.csv/.tex        the test table
    summary_validation.*        the validation table
    figures/                    fig1-fig4, 300 dpi PNG + PDF
and copies the PDFs and tables to COPY_TO.
"""
import csv
import json
import os
import shutil

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
from csdrecon.ml import grid_train
from csdrecon.study import virtual_gates as V

# ══ settings ══════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["4-6-8-12-16-24_rays_40-60-100_points_500_samples",
        "4-5-6-7-8_rays_40-50-60_points_500_samples"]
MODEL_DIR = "model"                  # the seed-0 checkpoint of every cell
FIGURE_UNET = {                      # the U-Net shown in figs 1 and 2
    "4-6-8-12-16-24_rays_40-60-100_points_500_samples": "8_rays_40_points_500_samples",
    "4-5-6-7-8_rays_40-50-60_points_500_samples": "8_rays_60_points_500_samples"}
COPY_TO = os.path.join(ROOT, "paper_figures", "virtual_gate")
# ══════════════════════════════════════════════════════════════════════════


def cells(run_dir):
    """(name, rays, points) of every U-Net folder, by measured points."""
    out = []
    for n in os.listdir(run_dir):
        ck = os.path.join(run_dir, n, MODEL_DIR, "unet.pt")
        if n[0].isdigit() and os.path.isfile(ck):
            with open(os.path.join(run_dir, n, "config.json")) as f:
                c = json.load(f)
            out.append((n, c["n_rays"], c["n_points"]))
    return sorted(out, key=lambda c: (c[1] * c[2], c[1]))


def split_arrays(cell_dir, split):
    """(X, Y, sample folders) of the test devices or the validation slice."""
    if split == "test":
        z = np.load(os.path.join(cell_dir, "test.npz"))
        return z["X"], z["Y"], list(z["samples"])
    z = np.load(os.path.join(cell_dir, "train.npz"))
    n = len(z["Y"])
    vi = np.random.default_rng(grid_train.SEED).permutation(n)[
        :max(1, int(grid_train.VAL_FRACTION * n))]
    return z["X"][vi], z["Y"][vi], [z["samples"][i] for i in vi]


def device_info(sample_dir):
    with open(os.path.join(sample_dir, "device.json")) as f:
        d = json.load(f)
    return d


def run_split(run_dir, split, out_dir):
    cl = cells(run_dir)
    X0, Y, samples = split_arrays(os.path.join(run_dir, cl[0][0]), split)
    info = [device_info(s) for s in samples]
    truth = [V.true_gates(d["capacitance"]) for d in info]
    px = [V.pixel_size(d["voltage_window"], Y[i].shape)
          for i, d in enumerate(info)]
    rows, maps = [], {"Y": Y.astype(bool), "samples": np.array(samples)}

    def add(method, rays, points, binmap, i, coverage):
        fit = V.slopes_from_lines(binmap, *px[i])
        r = {"split": split, "method": method, "rays": rays,
             "points": points, "device": os.path.basename(samples[i]),
             "coverage": coverage, **V.device_row(fit, truth[i])}
        rows.append(r)

    for i in range(len(Y)):
        add("full CSD", "", "", Y[i], i, 1.0)
    for name, rays, points in cl:
        X, Yc, sc = split_arrays(os.path.join(run_dir, name), split)
        assert [os.path.basename(s) for s in sc] == \
            [os.path.basename(s) for s in samples], name
        net, ck = grid_train.load(os.path.join(run_dir, name, MODEL_DIR,
                                               "unet.pt"))
        pred = grid_train.predict(net, X) > float(ck["threshold"])
        maps[name] = pred
        for i in range(len(Y)):
            add("U-Net", rays, points, pred[i], i, float(X[i, 1].mean()))
        print(f"  {split:10s} {name}: threshold {ck['threshold']}")
    np.savez_compressed(os.path.join(out_dir, f"_maps_{split}.npz"), **maps)
    return rows


def summaries(rows):
    out = []
    keys = []
    for r in rows:
        k = (r["method"], r["rays"], r["points"])
        if k not in keys:
            keys.append(k)
    for m, rays, points in keys:
        rs = [r for r in rows if (r["method"], r["rays"], r["points"])
              == (m, rays, points)]
        s = V.summary_row(rs)
        s.update(method=m, rays=rays, points=points,
                 budget="—" if m == "full CSD" else f"{rays} × {points}")
        out.append(s)
    return out


def main():
    for run in RUNS:
        run_dir = os.path.join(ROOT, "results", run)
        out = os.path.join(run_dir, "virtual_gates")
        os.makedirs(out, exist_ok=True)
        print(run)
        rows = run_split(run_dir, "test", out) + \
            run_split(run_dir, "validation", out)
        with open(os.path.join(out, "per_device.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for split, base in (("test", "summary"),
                            ("validation", "summary_validation")):
            rs = [r for r in rows if r["split"] == split]
            n = len({r["device"] for r in rs})
            cap = (f"Virtual-gate matrix, {split} split ({n} devices), "
                   f"{run}." + (" Validation devices chose each U-Net's "
                                "threshold: not held out."
                                if split == "validation" else ""))
            V.write_tables(summaries(rs), os.path.join(out, base), cap)
        V.make_figures(run_dir, out, FIGURE_UNET[run])
        dest = os.path.join(COPY_TO, run)
        os.makedirs(dest, exist_ok=True)
        for sub in ("", "figures"):
            for n in os.listdir(os.path.join(out, sub)):
                if n.endswith((".pdf", ".md", ".csv", ".tex")):
                    shutil.copy2(os.path.join(out, sub, n), dest)


if __name__ == "__main__":
    main()
