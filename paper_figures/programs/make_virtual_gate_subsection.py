# -*- coding: utf-8 -*-
"""
make_virtual_gate_subsection.py -- Sect. 4.2, virtual gates from sparse rays.

    python scripts/run_14_budget_slopes.py                       # the data
    python paper_figures/programs/make_virtual_gate_subsection.py

One JJAP document, built by make_method_docs.build from the journal's
template, so the fonts, styles, captions, cross-reference format and figure
and table numbering are the manuscript's own.  It is numbered to follow the
Results part (Fig. 13, Table II): its figures are 14 on and its table III.

EVERY NUMBER IN THE TEXT IS READ FROM run_14's OUTPUT (budget_slopes/
cells.csv, references.csv, per_device.csv) -- nothing is typed in.  The
figures are written into method_docs/4_Results_and_Discussion/figures/ and
copied beside the data in results/<run>/budget_slopes/figures/.

A copy of the document edited in Word is never overwritten; the new build
goes beside it as *_rebuilt.docx (make_method_docs' guard).
"""
import csv
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

import _figures                                                # noqa: E402
import make_method_docs as M                                   # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
RUN = os.path.join(REPO, "results",
                   "4-6-8-12-16-24_rays_40-60-100_points_500_samples")
DATA = os.path.join(RUN, "budget_slopes")
TARGETS = (0.03, 0.05, 0.10)          # "a few percent", three readings
HEADLINE = 0.05
FIG_OFFSET, TBL_OFFSET = 13, 2        # follows Fig. 13 and Table II
NUMBER, SLUG = "4_2", "Virtual_gate_extraction"
TITLE = "Virtual-gate extraction from sparse rays"
# ══════════════════════════════════════════════════════════════════════════

FIG_DIR = os.path.join(_figures.DOCS_ROOT, _figures.P5, "figures")
OUT_DOC = os.path.join(_figures.DOCS_ROOT, _figures.P5,
                       "%s_%s.docx" % (NUMBER, SLUG))
INK, MUTED = "#111111", "#555555"
# validated categorical slots 1-4 (network, Canny, true map, ray steps), as
# in run_12/run_13, and a ramp for points-per-ray
COLOR = {"network": "#2a78d6", "canny": "#eb6834", "truth": "#1baf7a",
         "ray_steps": "#eda100"}
POINT_SHADE = {40: "#86b6ef", 60: "#2a78d6", 100: "#184f95"}
MARK = {40: "o", 60: "s", 100: "D"}


# ── reading run_14 ────────────────────────────────────────────────────────

def _read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    cells = _read("cells.csv")
    refs = {r["arm"]: r for r in _read("references.csv")}
    net = [r for r in cells if r["arm"] == "network"]
    steps = {(int(r["rays"]), int(r["points"])): r for r in cells
             if r["arm"] == "ray_steps"}
    budgets = sorted({(int(r["rays"]), int(r["points"])) for r in net},
                     key=lambda b: (b[0] * b[1], b))
    out = []
    for b in budgets:
        rows = [r for r in net if (int(r["rays"]), int(r["points"])) == b]
        # the seed an experimenter would keep: best VALIDATION F1@1
        pick = max(rows, key=lambda r: _f(r["best_val_f1"]))
        xt = [_f(r["crosstalk_median_%"]) for r in rows]
        out.append({"rays": b[0], "points": b[1],
                    "n_points": b[0] * b[1],
                    "coverage": _f(pick["coverage_%"]),
                    "seed": int(pick["seed"]), "n_seeds": len(rows),
                    "f1": _f(pick["test_f1@1"]),
                    "xt": _f(pick["crosstalk_median_%"]),
                    "xt_p90": _f(pick["crosstalk_p90_%"]),
                    "xt_lo": min(xt), "xt_hi": max(xt),
                    "le": {t: _f(pick["crosstalk_le_%d%%" % round(100 * t)])
                           for t in TARGETS},
                    "rel": _f(pick["slope_rel_err_median_%"]),
                    "ang": _f(pick["angle_err_median_deg"]),
                    "fail": int(_f(pick["no_virtual_gates"])),
                    "steps_xt": _f(steps[b]["crosstalk_median_%"]),
                    "steps_fail": int(_f(steps[b]["no_virtual_gates"])),
                    "seeds_f1": sorted(_f(r["test_f1@1"]) for r in rows)})
    return out, refs


def smallest(cells, target):
    """The cheapest budget (fewest measured points) whose median residual
    crosstalk is at or below `target`, or None."""
    ok = [c for c in cells if c["xt"] <= 100 * target]
    return min(ok, key=lambda c: (c["n_points"], c["coverage"])) if ok \
        else None


# ── figures ───────────────────────────────────────────────────────────────

def _axes_style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


def fig_budget(cells, refs, path):
    """(a) median residual crosstalk, (b) devices within 5 %, vs coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for p in sorted({c["points"] for c in cells}):
        cs = sorted([c for c in cells if c["points"] == p],
                    key=lambda c: c["coverage"])
        x = [c["coverage"] for c in cs]
        y = [c["xt"] for c in cs]
        err = [[c["xt"] - c["xt_lo"] for c in cs],
               [c["xt_hi"] - c["xt"] for c in cs]]
        ax.errorbar(x, y, yerr=err, color=POINT_SHADE[p], marker=MARK[p],
                    ms=5, lw=1.6, capsize=2, label="network, %d points/ray" % p)
        ax.plot(x, [c["steps_xt"] for c in cs], ls=":", marker=MARK[p],
                mfc="none", color=COLOR["ray_steps"], ms=5, lw=1.2,
                label="ray steps, no network" if p == 40 else None)
    for arm, lab, ls in (("before (no virtual gates)", "no virtual gates",
                          "--"),
                         ("canny", "Canny, dense scan (100 %)", "-"),
                         ("truth", "fit to the true map", "-.")):
        c = {"before (no virtual gates)": "#999999", "canny": COLOR["canny"],
             "truth": COLOR["truth"]}[arm]
        ax.axhline(_f(refs[arm]["crosstalk_median_%"]), color=c, ls=ls,
                   lw=1.3, label=lab)
    ax.axhline(100 * HEADLINE, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fraction of the plane measured (%)")
    ax.set_ylabel("median residual crosstalk (%)")
    ax.text(-0.14, 1.02, "(a)", transform=ax.transAxes, fontsize=11,
            weight="bold")
    _axes_style(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)

    ax = axes[1]
    for p in sorted({c["points"] for c in cells}):
        cs = sorted([c for c in cells if c["points"] == p],
                    key=lambda c: c["coverage"])
        ax.plot([c["coverage"] for c in cs],
                [100 * c["le"][HEADLINE] for c in cs], color=POINT_SHADE[p],
                marker=MARK[p], ms=5, lw=1.6, label="%d points/ray" % p)
    for arm, c, ls in (("canny", COLOR["canny"], "-"),
                       ("truth", COLOR["truth"], "-.")):
        ax.axhline(100 * _f(refs[arm]["crosstalk_le_5%"]), color=c, ls=ls,
                   lw=1.3)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("fraction of the plane measured (%)")
    ax.set_ylabel("devices with crosstalk ≤ %d %% (%%)"
                  % round(100 * HEADLINE))
    ax.text(-0.14, 1.02, "(b)", transform=ax.transAxes, fontsize=11,
            weight="bold")
    _axes_style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".svg")
    plt.close(fig)


def fig_alpha(cell, path):
    """Measured against true a12, a21 at one budget: network vs Canny."""
    rows = _read("per_device.csv")
    net = [r for r in rows if r["arm"] == "network"
           and r["rays"] == str(cell["rays"])
           and r["points"] == str(cell["points"])
           and r["seed"] == str(cell["seed"])]
    canny = [r for r in rows if r["arm"] == "canny"]
    # the true coefficients, from the capacitances of each test device
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from csdrecon.study import slopes as S
    from run_14_budget_slopes import cell_config
    pool = cell_config(cell["rays"], cell["points"]).pool_dir()
    truth_a = {}
    for r in net:
        with open(os.path.join(pool, r["sample"], "device.json")) as f:
            cap = json.load(f)["capacitance"]
        L = S.lever_arms(cap["Cdd"], cap["Cgd"])
        truth_a[r["sample"]] = (L[0, 1] / L[0, 0], L[1, 0] / L[1, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, k, j, lab in ((axes[0], "a12", 0, r"$\alpha_{12}$"),
                          (axes[1], "a21", 1, r"$\alpha_{21}$")):
        hi = max(v[j] for v in truth_a.values()) * 1.12
        ax.plot([0, hi], [0, hi], color=MUTED, lw=0.8)
        for arm, rs, mk in (("canny", canny, "s"), ("network", net, "o")):
            x = [truth_a[r["sample"]][j] for r in rs]
            y = [_f(r[k]) for r in rs]
            ax.plot(x, y, mk, ms=5, mfc="none", mew=1.3, color=COLOR[arm],
                    label={"canny": "Canny, dense scan",
                           "network": "network, %d × %d rays"
                           % (cell["rays"], cell["points"])}[arm])
        ax.set_xlim(0, hi)
        ax.set_ylim(-0.05, hi + 0.15)
        ax.set_xlabel("true " + lab)
        ax.set_ylabel("measured " + lab)
        _axes_style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax, t in zip(axes, "ab"):
        ax.text(-0.14, 1.02, "(%s)" % t, transform=ax.transAxes,
                fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".svg")
    plt.close(fig)


# ── the text ──────────────────────────────────────────────────────────────

def _pct(x, d=1):
    return ("%%.%df" % d) % x


def _budget(c):
    return "%d × %d" % (c["rays"], c["points"])


def sections(cells, refs, answer, overview_name):
    before = _f(refs["before (no virtual gates)"]["crosstalk_median_%"])
    canny = refs["canny"]
    truth = refs["truth"]
    c_xt, t_xt = _f(canny["crosstalk_median_%"]), _f(truth["crosstalk_median_%"])
    n_dev = int(_f(canny["n_devices"]))
    lo = min(cells, key=lambda c: c["n_points"])
    hi = max(cells, key=lambda c: c["n_points"])
    best = min(cells, key=lambda c: c["xt"])
    found = {t: smallest(cells, t) for t in TARGETS}

    def reach(t):
        c = found[t]
        if c is None:
            return ("no budget in the sweep reaches a median of %d %%"
                    % round(100 * t))
        return ("%s rays × points (%s %% of the plane) is the smallest "
                "budget with a median of %d %% or less (%s %%)"
                % (_budget(c), _pct(c["coverage"], 2), round(100 * t),
                   _pct(c["xt"])))

    p1 = (
        "The quantity an experimenter extracts from a stability diagram is "
        "rarely the full set of transition lines. To compensate the "
        "capacitive crosstalk between plunger gates, it is sufficient to know "
        "the slopes of the two families of loading lines, from which the "
        "virtual gates U_{1} = P_{1} + α_{12}P_{2} and "
        "U_{2} = α_{21}P_{1} + P_{2} are constructed, with "
        "α_{12} = −1/s_{1} and α_{21} = −s_{2}, where "
        "s_{1} and s_{2} are the slopes dV_{2}/dV_{1} of the dot-1 and dot-2 "
        "lines. In the simulation the exact coefficients follow from the "
        "capacitance matrices of each device, α = (C_{dd}^{−1}C_{dg})"
        " normalised by its diagonal, so the accuracy of any extraction can "
        "be judged against the true value. We quantify it by the residual "
        "crosstalk, the off-diagonal of the lever-arm matrix seen through "
        "the virtual gates divided by its diagonal: the fraction by which "
        "one virtual gate still moves the other dot. Without virtual gates "
        "the residual is α itself; with the exact coefficients it is "
        "zero. For each device the larger of the two residuals is reported.")
    p2 = (
        "The same line-fitting procedure is applied to four point sets: the "
        "lines predicted by the network from the rays, Canny edges of the "
        "full dense scan (100 %% of the plane measured), the true line map, "
        "and the steps detected directly on the rays without the network. "
        "Straight segments are found by a probabilistic Hough transform, "
        "refined by total least squares, and grouped into the steep (dot-1) "
        "and shallow (dot-2) families; each family's slope is a single "
        "direction fitted jointly to all its segments. Every setting of the "
        "fit, of the Canny detector and of the step detector, and the "
        "training seed of the network, is chosen on the validation devices; "
        "the %d held-out test devices are used only for scoring. The fit to "
        "the true map measures the error contributed by the fitting itself."
        % n_dev)
    p3 = (
        "Figure 1(a) shows the median residual crosstalk on the test devices "
        "as a function of the fraction of the plane measured, for every "
        "measurement budget, and Table I lists the values. Without virtual "
        "gates the median crosstalk is %s %%. Virtual gates obtained by "
        "Canny from the dense scan reduce it to %s %%, and the fit to the "
        "true line map to %s %%, which is the floor set by the fitting "
        "procedure. With the network, the median residual decreases from %s "
        "%% at the smallest budget (%s, %s %% of the plane) to %s %% at the "
        "best (%s, %s %%). Without the network, the steps detected on the "
        "rays do not yield usable virtual gates at any budget: the median "
        "residual of this route stays between %s %% and %s %%, comparable to "
        "or larger than having no virtual gates at all."
        % (_pct(before), _pct(c_xt), _pct(t_xt),
           _pct(lo["xt"]), _budget(lo), _pct(lo["coverage"], 2),
           _pct(best["xt"]), _budget(best), _pct(best["coverage"], 2),
           _pct(min(c["steps_xt"] for c in cells)),
           _pct(max(c["steps_xt"] for c in cells))))
    p4 = (
        "The answer to the question posed at the outset therefore depends on "
        "what is meant by a few percent. For a median residual crosstalk of "
        "3 %%, %s; for 5 %%, %s; for 10 %%, %s. Figure 1(b) shows the "
        "complementary view, the fraction of devices whose residual is at "
        "most 5 %%: %s %% of devices for the dense Canny scan and %s %% for "
        "the fit to the true map, against %s %% for the network at %s."
        % (reach(0.03), reach(0.05), reach(0.10),
           _pct(100 * _f(canny["crosstalk_le_5%"]), 0),
           _pct(100 * _f(truth["crosstalk_le_5%"]), 0),
           _pct(100 * answer["le"][HEADLINE], 0), _budget(answer)))
    p5 = (
        "Figure 2 shows the true line maps of held-out devices replotted in "
        "the virtual gates obtained by each route at the %s budget. With the "
        "exact coefficients the dot-1 lines become vertical and the dot-2 "
        "lines horizontal; the residual tilt of the lines measures the "
        "error of each extraction. Figure 3 compares the extracted "
        "coefficients with the true values device by device."
        % _budget(answer))
    p6 = (
        "Three limitations qualify these numbers. First, the simulation is "
        "noise-free, which favours edge detection on the dense scan; the "
        "comparison with measured, noisy diagrams remains to be made. "
        "Second, the training of the network at a fixed budget depends on "
        "the random seed: at the budget of Fig. 2 the test F1@1 of the %d "
        "seeds ranges from %s to %s, and the error bars in Fig. 1(a) give "
        "the corresponding range of the median crosstalk. Third, part of "
        "every residual comes from the line fit itself, as the %s %% of the "
        "fit to the true map shows; a better fit would lower all curves."
        % (answer["n_seeds"], _pct(answer["seeds_f1"][0], 2),
           _pct(answer["seeds_f1"][-1], 2), _pct(t_xt)))

    head = ["Rays × points", "Plane measured (%)", "F1@1",
            "Slope error (%)", "Crosstalk, median (%)",
            "Crosstalk, seed range (%)", "Devices ≤ 5 % (%)"]
    rows = [[_budget(c), _pct(c["coverage"], 2), _pct(c["f1"], 3),
             _pct(c["rel"]), _pct(c["xt"]),
             "%s–%s" % (_pct(c["xt_lo"]), _pct(c["xt_hi"])),
             _pct(100 * c["le"][HEADLINE], 0)] for c in cells]
    rows += [["Canny, dense scan", "100", "–",
              _pct(_f(canny["slope_rel_err_median_%"])), _pct(c_xt), "–",
              _pct(100 * _f(canny["crosstalk_le_5%"]), 0)],
             ["Fit to true map", "–", "–",
              _pct(_f(truth["slope_rel_err_median_%"])), _pct(t_xt), "–",
              _pct(100 * _f(truth["crosstalk_le_5%"]), 0)],
             ["No virtual gates", "–", "–", "–", _pct(before),
              "–", "0"]]
    table = ("Table I. Virtual-gate extraction on the %d held-out devices for "
             "every measurement budget (network, seed chosen on validation), "
             "and the two references. Slope error is the median relative "
             "error |Δs/s| of the two line families; crosstalk is the "
             "larger residual of each device after virtualisation." % n_dev,
             head, rows, None, (len(cells),))
    return [(None, [p1, p2, p3, table, p4, p5, p6])]


def main():
    cells, refs = load()
    os.makedirs(FIG_DIR, exist_ok=True)
    answer = smallest(cells, HEADLINE) or min(cells, key=lambda c: c["xt"])
    f1 = os.path.join(FIG_DIR, "vg_budget.png")
    f3 = os.path.join(FIG_DIR, "vg_alpha.png")
    fig_budget(cells, refs, f1)
    fig_alpha(answer, f3)
    f2 = os.path.join(FIG_DIR, "vg_overview.png")
    src = os.path.join(RUN, "%d_rays_%d_points_500_samples" %
                       (answer["rays"], answer["points"]), "virtual_gates",
                       "overview.png")
    shutil.copyfile(src, f2)
    figures = [
        ("vg_budget.png",
         "(a) Median residual crosstalk after virtualisation against the "
         "fraction of the plane measured, for the network at 40, 60 and 100 "
         "points per ray (error bars: range over training seeds) and for the "
         "steps on the rays without the network (open symbols, dotted). "
         "Horizontal lines: no virtual gates, Canny on the dense scan, and "
         "the fit to the true map. (b) Fraction of held-out devices with a "
         "residual crosstalk of at most 5 %."),
        ("vg_overview.png",
         "True line maps of held-out devices in the physical gates (left) "
         "and in the virtual gates built from the exact coefficients and "
         "from each extraction route, at %s rays × points. Vertical "
         "and horizontal lines indicate correct virtual gates; the residual "
         "tilt of each family is given in each panel." % _budget(answer)),
        ("vg_alpha.png",
         "Extracted against true virtual-gate coefficients (a) "
         "α_{12} and (b) α_{21} for every held-out device: "
         "network at %s rays × points and Canny on the dense scan. The "
         "diagonal is exact extraction." % _budget(answer)),
    ]
    secs = sections(cells, refs, answer, f2)
    path, edited = M.build(NUMBER, SLUG, TITLE, None, secs, figures,
                           fig_offset=FIG_OFFSET, tbl_offset=TBL_OFFSET,
                           out=OUT_DOC)
    keep = os.path.join(DATA, "figures")
    os.makedirs(keep, exist_ok=True)
    for name, _ in figures:
        shutil.copyfile(os.path.join(FIG_DIR, name),
                        os.path.join(keep, name))
    print("wrote", path, "(your edited copy was kept)" if edited else "")
    return path


if __name__ == "__main__":
    main()
