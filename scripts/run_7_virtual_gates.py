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
# Each cell's checkpoint: seed 0 (model/), except a cell whose seed 0
# collapsed and was retrained -- then the seed in retrain_collapsed.json
# (model_seed<k>/), as run_11_merge_retrained.py takes it for the paper.
FIGURE_UNET = {                      # the U-Net shown in figs 1 and 2
    "4-6-8-12-16-24_rays_40-60-100_points_500_samples": "8_rays_40_points_500_samples",
    "4-5-6-7-8_rays_40-50-60_points_500_samples": "8_rays_60_points_500_samples"}
COPY_TO = os.path.join(ROOT, "paper_figures", "virtual_gate")
# ══════════════════════════════════════════════════════════════════════════


def model_dir(run_dir, name):
    f = os.path.join(run_dir, "retrain_collapsed.json")
    if os.path.isfile(f):
        with open(f) as fh:
            c = json.load(fh)["cells"].get(name)
        if c:
            return f"model_seed{c['chosen_seed']}"
    return "model"


def cells(run_dir):
    """(name, rays, points) of every U-Net folder, by measured points."""
    out = []
    for n in os.listdir(run_dir):
        ck = os.path.join(run_dir, n, model_dir(run_dir, n), "unet.pt")
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
        md = model_dir(run_dir, name)
        net, ck = grid_train.load(os.path.join(run_dir, name, md, "unet.pt"))
        pred = grid_train.predict(net, X) > float(ck["threshold"])
        maps[name] = pred
        for i in range(len(Y)):
            add("U-Net", rays, points, pred[i], i, float(X[i, 1].mean()))
        print(f"  {split:10s} {name}: {md}, threshold {ck['threshold']}")
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


def paragraph(rows, summ, cell):
    """The results paragraph, every number read from rows / summ."""
    full = summ[0]
    un = sorted(summ[1:], key=lambda s: s["coverage_%"])
    lo, hi = un[0], un[-1]
    r, p = int(cell.split("_")[0]), int(cell.split("_")[2])
    mid = next(s for s in un if (s["rays"], s["points"]) == (r, p))
    fr = [x for x in rows if x["method"] == "full CSD"]
    ok = {m: sum(abs(x[f"d{m}"]) < 0.005 for x in fr) for m in ("M12", "M21")}
    ur = [x for x in rows if x["method"] == "U-Net"
          and (x["rays"], x["points"]) == (r, p)]

    def med_rel(m, big):
        v = [x[f"rel_{m}"] for x in ur if (x[f"{m}_true"] > 0.4) == big
             and np.isfinite(x[f"rel_{m}"])]
        return float(np.median(v)), len(v)
    b12, n12 = med_rel("M12", True)
    s12, _ = med_rel("M12", False)
    b21, n21 = med_rel("M21", True)
    s21, _ = med_rel("M21", False)
    found = [s["interdot_found"] for s in un]
    dint = [s["dtheta_int"]["med_abs"] for s in un]
    n = full["n_devices"]
    return (
        f"The U-Nets were given {lo['coverage_%']:.2f}–{hi['coverage_%']:.2f} % "
        f"of the grid ({lo['budget']} to {hi['budget']} rays × points). "
        f"On the full CSD the same line fit recovers M12 and M21 with a median "
        f"relative error of {full['rel_M12']['med_abs']:.2f} % and "
        f"{full['rel_M21']['med_abs']:.2f} % ({ok['M12']}/{n} and "
        f"{ok['M21']}/{n} devices within 0.005 of the true value): the floor "
        f"of the fit itself on 100 × 100 px maps. From the U-Net "
        f"maps the median relative error of M12 falls from "
        f"{lo['rel_M12']['med_abs']:.1f} % at {lo['coverage_%']:.2f} % coverage "
        f"to {hi['rel_M12']['med_abs']:.1f} % at {hi['coverage_%']:.2f} %, and "
        f"that of M21 from {lo['rel_M21']['med_abs']:.1f} % to "
        f"{hi['rel_M21']['med_abs']:.1f} %; at {mid['budget']} "
        f"({mid['coverage_%']:.2f} %) it is {mid['rel_M12']['med_abs']:.1f} % "
        f"and {mid['rel_M21']['med_abs']:.1f} %. The U-Net error is not "
        f"uniform: at {mid['budget']} the devices with a true coupling above "
        f"0.4 are underestimated (median {b12:+.0f} % for M12, n = {n12}; "
        f"{b21:+.0f} % for M21, n = {n21}), while those below 0.4 are nearly "
        f"unbiased ({s12:+.1f} % and {s21:+.1f} %). The interdot slope, and "
        f"so rho, is not recovered: interdot segments are a few pixels long, "
        f"the full CSD yields an interdot family on only "
        f"{full['interdot_found']}/{n} devices (median |dθ_int| "
        f"{full['dtheta_int']['med_abs']:.1f}°, median |rho error| "
        f"{full['rel_rho']['med_abs']:.0f} %), and the {min(found)}–"
        f"{max(found)} devices per U-Net on which an interdot family is found "
        f"have median |dθ_int| of {min(dint):.0f}–{max(dint):.0f}°, i.e. "
        f"spurious pieces. rho is therefore reported but not claimed. A dot "
        f"family was missing on {full['dot_failed']}/{n} devices for the full "
        f"CSD and on at most {max(s['dot_failed'] for s in un)}/{n} for any "
        f"U-Net.")


def report_pdf(out, run, summ, summ_val, text):
    """report.pdf: test table, validation table, the paragraph, the figures."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(os.path.join(out, "report.pdf")) as pdf:
        for title, s in (("Test split", summ),
                         ("Validation split (chose the thresholds; "
                          "not held out)", summ_val)):
            lines = V.table_lines(s)
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(0.03, 0.96, f"Virtual-gate matrix — {title}\n{run}",
                     fontsize=10, va="top")
            ax = fig.add_axes([0.02, 0.30, 0.96, 0.58])
            ax.axis("off")
            widths = [0.06, 0.065, 0.06] + [0.125] * 6 + [0.055, 0.045]
            t = ax.table(cellText=lines[1:], colLabels=lines[0],
                         loc="upper center", cellLoc="center",
                         colWidths=widths)
            t.auto_set_font_size(False)
            t.set_fontsize(6.2)
            t.scale(1, 1.35)
            note = ("Each error cell: mean ± std over devices where the family "
                    "was found, median |error| in brackets. 'failed': devices "
                    "with no dot-1 or no dot-2 piece. rho only where an "
                    "interdot family was found.")
            body = note if s is summ_val else note + "\n\n" + text
            fig.text(0.03, 0.27, "\n".join(textwrap.wrap(body, 190)),
                     fontsize=7.5, va="top")
            pdf.savefig(fig)
            plt.close(fig)
        for name in ("fig1_examples", "fig2_parity", "fig3_angle_errors",
                     "fig4_error_vs_coverage"):
            img = plt.imread(os.path.join(out, "figures", name + ".png"))
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(8.3, 8.3 * h / w))
            fig.add_axes([0, 0, 1, 1]).imshow(img)
            fig.axes[0].axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)


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
        summ = {}
        for split, base in (("test", "summary"),
                            ("validation", "summary_validation")):
            rs = [r for r in rows if r["split"] == split]
            n = len({r["device"] for r in rs})
            cap = (f"Virtual-gate matrix, {split} split ({n} devices), "
                   f"{run}." + (" Validation devices chose each U-Net's "
                                "threshold: not held out."
                                if split == "validation" else ""))
            summ[split] = summaries(rs)
            V.write_tables(summ[split], os.path.join(out, base), cap)
        picks = V.make_figures(run_dir, out, FIGURE_UNET[run])
        text = paragraph([r for r in rows if r["split"] == "test"],
                         summ["test"], FIGURE_UNET[run])
        with open(os.path.join(out, "paragraph.md"), "w",
                  encoding="utf-8") as f:
            f.write(text + "\n\nFig. 1 devices (U-Net "
                    f"{FIGURE_UNET[run]}, mean |rel. error| of M12, M21): "
                    + ", ".join(f"{a} {b} ({c:.1f} %)" for a, b, c in
                                picks["examples"]) + "\n")
        report_pdf(out, run, summ["test"], summ["validation"], text)
        dest = os.path.join(COPY_TO, run)
        os.makedirs(dest, exist_ok=True)
        for sub in ("", "figures"):
            for n in os.listdir(os.path.join(out, sub)):
                if n.endswith((".pdf", ".md", ".csv", ".tex")):
                    shutil.copy2(os.path.join(out, sub, n), dest)


if __name__ == "__main__":
    main()
