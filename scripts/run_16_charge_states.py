"""
Charge states (n1, n2) from the full CSD and from every U-Net line map.

    python scripts/run_16_charge_states.py

For each device of the split, the same function labels two inputs:

    full CSD   charge_states.label_states(Y)
    U-Net      charge_states.label_states(prob > threshold)   one per U-Net

and both are scored against charge_states.npy.  The thresholds are the ones
stored in each unet.pt; nothing is re-tuned and nothing is trained.

The test split (test.npz) is the main table.  The same table is made for the
75 validation devices (the slice grid_train.train carved out of train.npz)
and labelled "validation": they chose each threshold, so they are not held
out.

Writes results/<run>/charge_states/:
    variation_<split>.md        Step 0: how much the devices differ
    per_device.csv              every metric per (split, method, device),
                                with the Step 0 quantities
    summary.md/.csv/.tex        the test table
    summary_validation.*        the validation table
    wrong_steps.npz             (dn1, dn2) histograms of the wrong pixels
    paragraph.md                the results paragraph, every number read here
    figures/                    fig1-fig4, 300 dpi PNG + PDF
"""
import csv
import os

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
from csdrecon.config import paths
from csdrecon.ml import grid_train
from csdrecon.ml.ray_peaks import load_grid
from csdrecon.study import charge_states as C
from run_7_virtual_gates import cells, model_dir, split_arrays

# ══ settings ══════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["4-5-6-7-8_rays_40-50-60_points_500_samples",
        "4-6-8-12-16-24_rays_40-60-100_points_500_samples"]
DENSE = "dense, no ML"               # the non-network reference row
FIGURE_UNET = {                      # the U-Net shown in figs 2 and 4
    "4-6-8-12-16-24_rays_40-60-100_points_500_samples": "8_rays_40_points_500_samples",
    "4-5-6-7-8_rays_40-50-60_points_500_samples": "8_rays_60_points_500_samples"}
N_VARIATION_EXAMPLES = 6
WRONG_LIM = 3                        # fig. 4 histogram clipped at +-3
# ══════════════════════════════════════════════════════════════════════════


def local(sample):
    """The device folder in this project's data/, whatever path the split
    was saved with (test.npz stores the absolute path of the machine that
    cut it)."""
    s = str(sample).replace("\\", "/").rstrip("/").split("/")
    return paths.data_path("_device_pools", s[-2], s[-1])


def choose_dense(dirs, Y, states, px, out):
    """The dense detector's setting: best mean state recall on these
    (validation) devices.  Every setting's scores go to dense_scan.csv."""
    Z = [load_grid(d)[2] for d in dirs]
    scan = []
    for k in C.DENSE_GRID:
        sc = [C.score(C.label_states(C.dense_line_map(Z[i], k), *px[i]),
                      states[i], Y[i], *px[i]) for i in range(len(Y))]
        scan.append({"setting": C.dense_knob(k),
                     "state_recall": float(np.mean([s["state_recall"]
                                                    for s in sc])),
                     "acc_rel": float(np.mean([s["acc_rel"] for s in sc])),
                     "knob": k})
    best = max(scan, key=lambda r: (r["state_recall"], r["acc_rel"]))
    with open(os.path.join(out, "dense_scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "state_recall_mean", "acc_rel_mean",
                    "chosen"])
        for r in sorted(scan, key=lambda r: -r["state_recall"]):
            w.writerow([r["setting"], f"{r['state_recall']:.4f}",
                        f"{r['acc_rel']:.4f}", int(r is best)])
    print(f"  dense setting chosen on validation: {best['setting']} "
          f"(recall {100 * best['state_recall']:.1f} %)")
    return best["knob"]


def run_split(run_dir, split, dense_knob=None, out=None):
    cl = cells(run_dir)
    _, Y, samples = split_arrays(os.path.join(run_dir, cl[0][0]), split)
    Y = Y.astype(bool)
    dirs = [local(s) for s in samples]
    names = [os.path.basename(d) for d in dirs]
    info = [C.read_device(d) for d in dirs]
    states = [C.load_states(d) for d in dirs]
    px = [C.pixel_mV(d, Y[i].shape) for i, d in enumerate(info)]
    var = [C.device_variation(states[i], info[i]) for i in range(len(Y))]
    rows, hists, maps = [], {}, {"Y": Y, "samples": np.array(names)}

    def add(method, rays, points, B, i, coverage, key):
        res = C.label_states(B, *px[i])
        sc = C.score(res, states[i], Y[i], *px[i])
        rows.append({"split": split, "method": method, "rays": rays,
                     "points": points, "device": names[i],
                     "coverage": coverage, **sc, **var[i]})
        h = C.wrong_steps(res, states[i], Y[i], WRONG_LIM)
        hists[key] = hists.get(key, 0) + h

    for i in range(len(Y)):
        add("full CSD", "", "", Y[i], i, 1.0, "full")
    if dense_knob is None:
        dense_knob = choose_dense(dirs, Y, states, px, out)
    maps[DENSE] = np.array([C.dense_line_map(load_grid(d)[2], dense_knob)
                            for d in dirs])
    for i in range(len(Y)):
        add(DENSE, "", "", maps[DENSE][i], i, 1.0, DENSE)
    for name, rays, points in cl:
        X, _, sc = split_arrays(os.path.join(run_dir, name), split)
        assert [os.path.basename(local(s)) for s in sc] == names, name
        md = model_dir(run_dir, name)
        net, ck = grid_train.load(os.path.join(run_dir, name, md, "unet.pt"))
        pred = grid_train.predict(net, X) > float(ck["threshold"])
        maps[name] = pred
        for i in range(len(Y)):
            add("U-Net", rays, points, pred[i], i, float(X[i, 1].mean()),
                name)
        print(f"  {split:10s} {name}: {md}, threshold {ck['threshold']}")
    ctx = {"dirs": dirs, "names": names, "info": info, "states": states,
           "px": px, "var": var, "maps": maps}
    ctx["dense_knob"] = dense_knob
    return rows, hists, ctx


def summaries(rows):
    keys = []
    for r in rows:
        k = (r["method"], r["rays"], r["points"])
        if k not in keys:
            keys.append(k)
    out = []
    for m, rays, points in keys:
        rs = [r for r in rows if (r["method"], r["rays"], r["points"])
              == (m, rays, points)]
        s = C.summary_row(rs)
        s.update(method=m, rays=rays, points=points,
                 budget="—" if m != "U-Net" else f"{rays} × {points}")
        out.append(s)
    return out


def figures(out, summ, rows, hists, ctx, cell):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    var, names, states = ctx["var"], ctx["names"], ctx["states"]
    Y = ctx["maps"]["Y"]

    # fig 1: six devices spread over the number of states
    o = np.argsort([v["n_states"] for v in var])
    pick = o[np.linspace(0, len(o) - 1, N_VARIATION_EXAMPLES).astype(int)]
    C.fig_variation(fig_dir, var,
                    [(names[i], states[i], ctx["info"][i]["voltage_window"],
                      Y[i]) for i in pick])

    # fig 2: good / typical / bad by the U-Net's relative accuracy
    parts = cell.split("_")
    r_, p_ = int(parts[0]), int(parts[2])
    ur = [r for r in rows if r["method"] == "U-Net"
          and (r["rays"], r["points"]) == (r_, p_)]
    acc = np.array([r["acc_rel"] for r in ur])
    o = np.argsort(-acc)
    picks = []
    for lab, j in (("good", o[0]), ("typical", o[len(o) // 2]),
                   ("bad", o[-1])):
        i = names.index(ur[j]["device"])
        B = ctx["maps"][cell][i]
        res = C.label_states(B, *ctx["px"][i])
        picks.append({"name": names[i], "label": lab, "cs": states[i],
                      "ext": ctx["info"][i]["voltage_window"], "Y": Y[i],
                      "B": B, "res": res,
                      "score": C.score(res, states[i], Y[i], *ctx["px"][i])})
    C.fig_examples(fig_dir, picks)
    C.fig_vs_coverage(fig_dir, summ)
    full = [r for r in rows if r["method"] == "full CSD"]
    dense = [r for r in rows if r["method"] == DENSE]
    C.fig_failures(fig_dir, hists[cell], WRONG_LIM, full, ur,
                   f"{r_}×{p_}", dense)
    return [(p["label"], p["name"], p["score"]["acc_rel"]) for p in picks]


def paragraph(summ, rows, cell):
    """The results paragraph, every number read from summ / rows."""
    full = summ[0]
    dn = next(s for s in summ if s["method"] == DENSE)
    un = sorted([s for s in summ if s["method"] == "U-Net"],
                key=lambda s: s["coverage_%"])
    lo, hi = un[0], un[-1]
    r_, p_ = int(cell.split("_")[0]), int(cell.split("_")[2])
    mid = next(s for s in un if (s["rays"], s["points"]) == (r_, p_))
    best = max(un, key=lambda s: s["acc_rel"]["mean"])
    n = full["n_devices"]

    def a(s, k="acc_rel", q="mean"):
        return s[k][q]
    return (
        f"The U-Nets were given {lo['coverage_%']:.2f}–{hi['coverage_%']:.2f} "
        f"% of the grid ({lo['budget']} to {hi['budget']} rays × points). "
        f"Labelling the full CSD's line map gives the ceiling of the method: "
        f"{a(full):.1f} % of the scored plane gets the right (n1, n2) after "
        f"one constant shift per device (median {a(full, q='median'):.1f} %), "
        f"{a(full, 'acc_abs'):.1f} % with no shift and the top-right corner "
        f"taken as (0,0), and {a(full, 'state_recall'):.1f} % of the charge "
        f"states are recovered ({n} test devices; pixels within 1 px of a "
        f"true line are not scored). From the U-Net line maps the relative "
        f"pixel accuracy is {a(lo):.1f} % at {lo['coverage_%']:.2f} % "
        f"coverage and {a(hi):.1f} % at {hi['coverage_%']:.2f} %; at "
        f"{mid['budget']} ({mid['coverage_%']:.2f} %) it is {a(mid):.1f} % "
        f"(median {a(mid, q='median'):.1f} %), the absolute accuracy "
        f"{a(mid, 'acc_abs'):.1f} %, and {a(mid, 'state_recall'):.1f} % of "
        f"the states are recovered against {a(full, 'state_recall'):.1f} % "
        f"for the full CSD. The best U-Net is {best['budget']} "
        f"({best['coverage_%']:.2f} %, {a(best):.1f} %). The (1,1) region, "
        f"the one used for a qubit, has IoU {a(mid, 'iou_11'):.2f} from the "
        f"{mid['budget']} U-Net against {a(full, 'iou_11'):.2f} from the "
        f"full CSD, with a centre error of "
        f"{mid['centre_err_11_mV']['median']:.3f} mV (median). The line "
        f"maps' self-consistency is {a(full, 'consistency'):.2f} for the "
        f"full CSD and {a(mid, 'consistency'):.2f} for {mid['budget']}. "
        f"Without a network, lines taken from the dense sensor image (every "
        f"pixel measured, no noise) give {a(dn):.1f} % pixel accuracy, "
        f"{a(dn, 'state_recall'):.1f} % state recall and (1,1) IoU "
        f"{a(dn, 'iou_11'):.2f}.")


def main():
    for run in RUNS:
        run_dir = os.path.join(ROOT, "results", run)
        print(run)
        if not os.path.isdir(run_dir):
            print(f"  {run_dir} is missing -- skipped")
            continue
        out = os.path.join(run_dir, "charge_states")
        os.makedirs(out, exist_ok=True)
        rows, summ, ctx_test, hist_test = [], {}, None, None
        knob = None
        # validation first: it chooses the dense detector's setting
        for split, base in (("validation", "summary_validation"),
                            ("test", "summary")):
            rs, hs, ctx = run_split(run_dir, split, knob, out)
            knob = ctx["dense_knob"]
            rows += rs
            n = len(ctx["names"])
            C.variation_table(
                ctx["var"], os.path.join(out, f"variation_{split}"),
                f"Device variation, {split} split ({n} devices), {run}.")
            cap = (f"Charge-state labelling, {split} split ({n} devices), "
                   f"{run}." + (" Validation devices chose each U-Net's "
                                "threshold: not held out."
                                if split == "validation" else ""))
            summ[split] = summaries(rs)
            C.write_tables(summ[split], os.path.join(out, base), cap)
            np.savez_compressed(os.path.join(out, f"wrong_steps_{split}.npz"),
                                **hs)
            if split == "test":
                ctx_test, hist_test = ctx, hs
        with open(os.path.join(out, "per_device.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        test_rows = [r for r in rows if r["split"] == "test"]
        picks = figures(out, summ["test"], test_rows, hist_test, ctx_test,
                        FIGURE_UNET[run])
        text = paragraph(summ["test"], test_rows, FIGURE_UNET[run])
        with open(os.path.join(out, "paragraph.md"), "w",
                  encoding="utf-8") as f:
            f.write(text + "\n\nFig. 2 devices (U-Net "
                    f"{FIGURE_UNET[run]}, relative pixel accuracy): "
                    + ", ".join(f"{a} {b} ({100 * c:.1f} %)"
                                for a, b, c in picks) + "\n"
                    + "\nDense detector setting (chosen on the validation "
                    f"devices by mean state recall): {C.dense_knob(knob)}\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
