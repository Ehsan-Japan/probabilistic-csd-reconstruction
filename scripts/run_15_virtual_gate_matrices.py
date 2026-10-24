"""
The virtual-gate matrix itself, per method, against the true one.

    python scripts/run_15_virtual_gate_matrices.py

Reads only what run_14_budget_slopes.py cached (<run>/budget_slopes/_cells/)
and trains nothing.  For every held-out test device it compares

    A = [[1, a12], [a21, 1]]        U = A P

built from the line fit of each method with the TRUE A, whose rows are the
lever-arm matrix normalised by its diagonal (a12 = L12/L11, a21 = L21/L22):

    dense scan     the line fit on Canny edges of the full dense scan
    rays alone     the line fit on the steps detected on the rays, no network
    network        the line fit on the network's line map from the rays
    true map       the line fit on the true line map (the fit's own floor)

The network's seed is the one with the best VALIDATION F1@1, as in run_14.
Only budgets whose ray-step arm and all seeds are cached are used.

Writes <run>/budget_slopes/vg_matrices/: matrices.csv (every device, method
and budget: measured and true a12, a21, |dA|, residual crosstalk),
summary.csv, and vg_matrices.pdf.
"""
import csv
import json
import os
import re

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from csdrecon.study import slopes as S
from run_14_budget_slopes import OUT as DATA, SEEDS, cell_config

OUT = os.path.join(DATA, "vg_matrices")
CACHE = os.path.join(DATA, "_cells")
COLOR = {"network": "#2a78d6", "dense scan": "#eb6834",
         "true map": "#1baf7a", "rays alone": "#eda100"}
INK, MUTED = "#111111", "#555555"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})


def _load(name):
    with open(os.path.join(CACHE, name)) as f:
        return json.load(f)


def budgets():
    """(rays, points) with ray steps and every seed cached, by coverage."""
    found = set()
    for n in os.listdir(CACHE):
        m = re.match(r"(\d+)x(\d+)_ray_steps\.json$", n)
        if m:
            r, p = map(int, m.groups())
            if all(os.path.isfile(os.path.join(
                    CACHE, f"{r}x{p}_network_seed{s}.json")) for s in SEEDS):
                found.add((r, p))
    with open(os.path.join(DATA, "cells.csv"), newline="") as f:
        cov = {(int(c["rays"]), int(c["points"])): float(c["coverage_%"])
               for c in csv.DictReader(f)}
    return sorted(((r, p, cov[(r, p)]) for r, p in found),
                  key=lambda b: b[2])


def true_alphas(samples):
    """True (a12, a21) and the 2x2 lever arms of each test device."""
    pool = cell_config(4, 40).pool_dir()
    out = {}
    for s in samples:
        with open(os.path.join(pool, s, "device.json")) as f:
            cap = json.load(f)["capacitance"]
        L = S.lever_arms(cap["Cdd"], cap["Cgd"])[:, :2]
        a = (L[0, 1] / L[0, 0], L[1, 0] / L[1, 1])
        # the true matrix removes the crosstalk exactly
        assert max(map(abs, S.residual_crosstalk(L, a))) < 1e-9, s
        out[s] = a
    return out


def rows_for(method, budget, devices, truth):
    out = []
    for d in devices:
        t12, t21 = truth[d["sample"]]
        ok = not (np.isnan(d["a12"]) or np.isnan(d["a21"]))
        out.append({
            "method": method, "budget": budget, "sample": d["sample"],
            "a12_true": t12, "a21_true": t21,
            "a12": d["a12"], "a21": d["a21"],
            "da12": d["a12"] - t12 if ok else np.nan,
            "da21": d["a21"] - t21 if ok else np.nan,
            "dA": float(np.hypot(d["a12"] - t12, d["a21"] - t21))
            if ok else np.nan,
            "crosstalk": d["crosstalk"]})
    return out


def summary(rows, method, budget, coverage, extra=None):
    g = [r for r in rows if r["method"] == method and r["budget"] == budget]
    dA = np.array([r["dA"] for r in g], float)
    xt = np.array([r["crosstalk"] for r in g], float)
    have = ~np.isnan(dA)
    return {"method": method, "budget": budget, "coverage_%": coverage,
            "n_devices": len(g), "matrices_built": int(have.sum()),
            "median_|da12|": float(np.nanmedian(np.abs(
                [r["da12"] for r in g]))) if have.any() else np.nan,
            "median_|da21|": float(np.nanmedian(np.abs(
                [r["da21"] for r in g]))) if have.any() else np.nan,
            "median_|dA|": float(np.nanmedian(dA)) if have.any() else np.nan,
            "p90_|dA|": float(np.nanpercentile(dA, 90)) if have.any()
            else np.nan,
            "crosstalk_median_%": 100 * float(np.nanmedian(xt))
            if have.any() else np.nan,
            # a device with no matrix counts as a miss, not as excluded
            "devices_crosstalk_le_5%": float(np.mean(
                np.nan_to_num(xt, nan=np.inf) <= 0.05)),
            **(extra or {})}


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{d}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    refs = _load("references.json")
    samples = [d["sample"] for d in refs["canny"]["devices"]]
    truth = true_alphas(samples)
    cells = budgets()

    rows, summ = [], []
    for method, arm in (("dense scan", "canny"), ("true map", "truth")):
        rows += rows_for(method, "-", refs[arm]["devices"], truth)
        summ.append(summary(rows, method, "-", 100.0 if arm == "canny"
                            else np.nan))
    picks = {}
    for r, p, cov in cells:
        b = f"{r}x{p}"
        st = _load(f"{b}_ray_steps.json")
        assert [d["sample"] for d in st["devices"]] == samples
        rows += rows_for("rays alone", b, st["devices"], truth)
        summ.append(summary(rows, "rays alone", b, cov))
        nets = [_load(f"{b}_network_seed{s}.json") for s in SEEDS]
        k = int(np.argmax([n["best_val_f1"] for n in nets]))
        picks[b] = SEEDS[k]
        assert [d["sample"] for d in nets[k]["devices"]] == samples
        rows += rows_for("network", b, nets[k]["devices"], truth)
        # every seed's median |dA|, to show the spread the pick sits in
        spread = []
        for s, n in zip(SEEDS, nets):
            rr = rows_for("network", b, n["devices"], truth)
            spread.append(float(np.nanmedian([x["dA"] for x in rr])))
        summ.append(summary(rows, "network", b, cov, {
            "seed": SEEDS[k], "median_|dA|_all_seeds": " / ".join(
                f"{v:.3f}" for v in spread)}))
    write_csv(os.path.join(OUT, "matrices.csv"), rows)
    write_csv(os.path.join(OUT, "summary.csv"), summ)
    pdf(rows, summ, cells, picks, truth, samples)
    print("wrote", OUT)


# ── the PDF ───────────────────────────────────────────────────────────────

def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pdf(rows, summ, cells, picks, truth, samples):
    get = {(s["method"], s["budget"]): s for s in summ}
    dense, tmap = get[("dense scan", "-")], get[("true map", "-")]
    big = f"{cells[-1][0]}x{cells[-1][1]}"
    path = os.path.join(OUT, "vg_matrices.pdf")
    with PdfPages(path) as pp:
        # ── page 1: the table ─────────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Virtual-gate matrix: dense-scan fit vs fit on "
                 "rays alone (and the network)", fontsize=13, weight="bold",
                 color=INK)
        notes = [
            "A = [[1, a12], [a21, 1]], U = A P.  True A from each device's "
            "capacitances; |dA| = sqrt(da12² + da21²).",
            "Same 50 held-out test devices for every method and budget.  "
            "Every knob chosen on validation devices.",
            "Network seed = best validation F1@1 (all-seed medians in the "
            "last column).  A device with no matrix counts as a miss in "
            "'≤ 5 %'.",
            "Budgets: the %d cells of run_14 cached so far (run stopped "
            "on low memory; the 60- and 100-point rows are incomplete)."
            % len(cells)]
        for i, t in enumerate(notes):
            fig.text(0.07, 0.915 - 0.022 * i, "• " + t, fontsize=8,
                     color=MUTED)
        head = ["method", "budget", "plane %", "matrices", "med |Δa12|",
                "med |Δa21|", "med |ΔA|", "p90 |ΔA|", "xt med %",
                "≤ 5 %", "|ΔA| all seeds"]
        body = [["no virtual gates", "–", "–", "–", "–", "–", "–", "–",
                 fmt(100 * float(np.median(_load("references.json")
                                           ["before"])), 1), "0 %", ""]]
        order = [("true map", "-"), ("dense scan", "-")] + [
            (m, f"{r}x{p}") for r, p, _ in cells
            for m in ("rays alone", "network")]
        for m, b in order:
            s = get[(m, b)]
            body.append([m, b.replace("x", " × ") if b != "-" else "–",
                         fmt(s["coverage_%"], 2) if b != "-" else
                         ("100" if m == "dense scan" else "–"),
                         f"{s['matrices_built']}/{s['n_devices']}",
                         fmt(s["median_|da12|"]), fmt(s["median_|da21|"]),
                         fmt(s["median_|dA|"]), fmt(s["p90_|dA|"]),
                         fmt(s["crosstalk_median_%"], 1),
                         "%d %%" % round(100 * s["devices_crosstalk_le_5%"]),
                         s.get("median_|dA|_all_seeds", "")])
        ax = fig.add_axes([0.04, 0.08, 0.92, 0.74])
        ax.axis("off")
        tb = ax.table(cellText=body, colLabels=head, loc="upper center",
                      cellLoc="center",
                      colWidths=[.13, .07, .07, .07, .08, .08, .08, .08,
                                 .07, .06, .17])
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5)
        tb.scale(1, 1.35)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("#cccccc")
            if i == 0:
                c.set_text_props(weight="bold")
            elif j == 0:
                c.set_text_props(color=COLOR.get(body[i - 1][0], INK))
        pp.savefig(fig)
        plt.close(fig)

        # ── page 2: against budget ────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        for ax, key, lab in ((axes[0], "median_|dA|",
                              "median |A − A_true| over devices"),
                             (axes[1], "matrices_built",
                              "devices with a matrix (of 50)")):
            for m, mk in (("rays alone", "s"), ("network", "o")):
                for pts, ls in ((40, "-"), (60, "--")):
                    cs = [(cov, get[(m, f"{r}x{p}")][key])
                          for r, p, cov in cells if p == pts]
                    if cs:
                        ax.plot(*zip(*cs), ls=ls, marker=mk, ms=5,
                                color=COLOR[m], mfc="none" if pts == 60
                                else COLOR[m],
                                label=f"{m}, {pts} points/ray")
            for m, s in (("dense scan", dense), ("true map", tmap)):
                ax.axhline(s[key], color=COLOR[m], lw=1.2,
                           ls="-" if m == "dense scan" else "-.",
                           label=m + (" (100 % of plane)"
                                      if m == "dense scan" else
                                      " (fit floor)"))
            ax.set_xscale("log")
            ticks = []
            for _, _, c in cells:          # 2.33 and 2.35 % share a tick
                if not ticks or c / ticks[-1] > 1.05:
                    ticks.append(c)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{c:.1f}" for c in ticks])
            ax.minorticks_off()
            ax.set_xlabel("fraction of the plane measured (%)")
            ax.set_ylabel(lab)
            _style(ax)
        axes[0].set_yscale("log")
        axes[1].set_ylim(0, 52)
        axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="lower center",
                       bbox_to_anchor=(0.5, 1.02))
        for ax, t in zip(axes, "ab"):
            ax.text(-0.08, 1.12 if t == "a" else 1.02, f"({t})", transform=ax.transAxes,
                    fontsize=10, weight="bold")
        fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.97))
        pp.savefig(fig)
        plt.close(fig)

        # ── page 3: a12, a21 measured vs true at the largest budget ───────
        fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
        for i, (m, b) in enumerate((("dense scan", "-"),
                                    ("rays alone", big),
                                    ("network", big))):
            g = [r for r in rows if r["method"] == m and r["budget"] == b]
            n_ok = sum(not np.isnan(r["dA"]) for r in g)
            for j, k in enumerate(("a12", "a21")):
                ax = axes[i, j]
                hi = max(r[f"{k}_true"] for r in g) * 1.1
                ax.plot([0, hi], [0, hi], color=MUTED, lw=0.8)
                ax.plot([r[f"{k}_true"] for r in g], [r[k] for r in g], "o",
                        ms=4, mfc="none", mew=1.1, color=COLOR[m])
                ax.set_xlim(0, hi)
                ax.set_ylim(-0.1, hi + 0.2)
                ax.set_xlabel(f"true {k}")
                ax.set_ylabel(f"measured {k}")
                ax.set_title(f"{m}" + (f", {b.replace('x', ' × ')}"
                                       if b != "-" else "")
                             + f"  ({n_ok}/50 matrices)", color=COLOR[m])
                _style(ax)
        fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.97))
        pp.savefig(fig)
        plt.close(fig)

        # ── page 4: the matrices of a few devices ─────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, f"Virtual-gate matrices of individual test "
                 f"devices (budget {big.replace('x', ' × ')}, network seed "
                 f"{picks[big]})", fontsize=11, weight="bold", color=INK)
        fig.text(0.07, 0.925, "First six test devices in file order (not "
                 "selected); r = residual crosstalk max(|r12|, |r21|).",
                 fontsize=8, color=MUTED)
        by = {(r["method"], r["budget"], r["sample"]): r for r in rows}
        y = 0.88
        for s in samples[:6]:
            t12, t21 = truth[s]
            fig.text(0.07, y, s, fontsize=9, weight="bold")
            entries = [("true", INK, t12, t21, None)]
            for m, b in (("dense scan", "-"), ("rays alone", big),
                         ("network", big)):
                r = by[(m, b, s)]
                entries.append((m, COLOR[m], r["a12"], r["a21"],
                                r["crosstalk"]))
            for k, (lab, col, a, c, xt) in enumerate(entries):
                x = 0.07 + 0.22 * k
                fig.text(x, y - 0.025, lab, fontsize=8, color=col)
                if np.isnan(a) or np.isnan(c):
                    fig.text(x, y - 0.06, "no matrix\n(a line family "
                             "not found)", fontsize=8, color=col,
                             family="monospace")
                    continue
                fig.text(x, y - 0.075, "⎡ 1.000  %6.3f ⎤\n⎣%6.3f   1.000 ⎦"
                         % (a, c), fontsize=8, family="monospace",
                         color=col)
                if xt is not None:
                    fig.text(x, y - 0.098, "r = %.1f %%" % (100 * abs(xt)),
                             fontsize=7.5, color=MUTED)
            y -= 0.14
        pp.savefig(fig)
        plt.close(fig)
    return path


if __name__ == "__main__":
    main()
