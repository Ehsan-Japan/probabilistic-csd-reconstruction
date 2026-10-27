# -*- coding: utf-8 -*-
"""
make_method_docs.py -- the method, in five numbered JJAP documents.

Each part is one Word document built FROM THE JJAP REGULAR-PAPER TEMPLATE,
so the page size, margins, fonts and paragraph styles are the journal's and
not this script's:

    0_Cover_letter                    (the letter template, not the JJAP one)
    1_Abstract                        title page, abstract, keywords
    2_Introduction
    3_1_Device_simulation_and_dataset     3_2_Ray_based_measurement          |  the four are one section,
    3_3_Network_and_training           |  the METHOD, in the order the
    3_4_Threshold_and_tolerance       /   data flows through it
    4_Results

Figures go at the END of each document, one per page block, numbered within
the part and captioned -- the JJAP submission order (text, then figures).

EVERY NUMBER IN THE TEXT IS READ FROM THE RUN, not typed in: the hyper-
parameters come from csdrecon.ml.grid_train and the configuration's
config.json, the scores from comparison.csv.  A number that cannot be found
raises rather than being quietly omitted, because a method section with a
wrong constant in it is worse than one that failed to build.

    python paper_figures/programs/make_method_docs.py   # the documents
    python paper_figures/programs/make_pdfs.py          # the PDFs beside them

A DOCUMENT YOU EDITED BY HAND IS NEVER OVERWRITTEN.  Each one is
fingerprinted as it is written, so a rebuild can see that Word has been at
it since; that copy is left alone and the new version is written beside it
as <stem>_rebuilt.docx.  Documents that are replaced have their previous
copy kept in <part>/backups/.  See the guard below.

Feedback goes in <part>/<stem>.txt, one comment sheet per part, written by
make_method_notes.py and never overwritten either: you write in its YOU
WRITE half, the change and where it was made go back into its THE REPLY
half.
"""
import glob
import json
import math
import os
import re
import shutil
import sys

import copy

import docx
import docx.text.paragraph
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))   # paper_figures/programs
PKG = os.path.dirname(HERE)                         # paper_figures
sys.path.insert(0, HERE)

import _figures                                                # noqa: E402
import _run                                                    # noqa: E402
from csdrecon.config import capacitance_config as cc           # noqa: E402
from csdrecon.ml import grid_model, grid_train                 # noqa: E402

OUT_DIR = os.path.join(PKG, "method_docs")
TEMPLATE = os.path.join(
    "F:", os.sep, "Seminar Fujita", "Conferences&Journals",
    "Japanese Journal of Applied Physics_2026", "Drafts",
    "AO and TF Feedbacks", "20240730_template-RP.docx")
# The cover letter is a LETTER, not a manuscript part: it is built from the
# letter actually sent with the 2025 JJAP paper, so its page setup, face and
# spacing are the ones that journal has already accepted.
COVER_TEMPLATE = os.path.join(PKG, "templates", "cover_letter-RP.docx")
# Likewise the title page: it is the title page of the 2025 JJAP paper, and
# everything on it -- the Arial 14 bold title, the Times New Roman 12
# authors, the italic affiliations, the single-spaced abstract and the
# three custom styles they sit in -- is the template's, not this script's.
TITLE_TEMPLATE = os.path.join(PKG, "templates", "title_page-RP.docx")
# The template's own style names, as it defines them.
T_TITLE = "スタイル AUTHORS + 右 :  4.5 字1"
T_AFFIL = "スタイル AFFILIATIONS + 右 :  4.5 字"
T_EMAIL = "スタイル Email + 右 :  4.5 字"

# The template's own style names.  Japanese names are the template's, not a
# mistake: it was made in a Japanese Word and the styles must be referred to
# exactly as it defines them or Word silently falls back to Normal.
S_TITLE = "表題1"
S_ABSTRACT = "Normal"
S_SECTION = "SectionTitle"
S_SUBSECTION = "SubSectionTitle"
S_BODY = "MainText"

# The full author block.  It appears ONCE, on the title page: these are
# seven parts of one manuscript, not seven manuscripts.
# {..} marks a superscript; _para_marked turns it into one.
# The paper's title, written once.  The title page shows it as the title;
# the cover letter quotes it; the manuscript-details block repeats it as
# submission metadata.
PAPER_TITLE = ("Probabilistic reconstruction of charge stability diagrams "
               "from sparse ray measurements")

AUTHORS = ("Ehsan Alizadeh Kashtiban{1,2,*}, Takafumi Fujita{2,3,4,5,6}, "
           "and Akira Oiwa{2,4,5,6}")
AFFILIATIONS = (
    "{1}Department of Physics, Graduate School of Science, The University of Osaka, "
    "Toyonaka, Osaka, 560-0043, Japan",
    "{2}SANKEN, The University of Osaka, Ibaraki, Osaka 567-0047, Japan",
    "{3}Artificial Intelligence Research Center, The Institute of Scientific "
    "and Industrial Research, The University of Osaka, 8-1 Mihogaoka, Ibaraki, "
    "Osaka 567-0047, Japan",
    "{4}Center for Quantum Information and Quantum Biology, The University of "
    "Osaka, 1-2 Machikaneyama, Toyonaka, Osaka 560-0043, Japan",
    "{5}Center for Spintronics Research Network, Graduate School of "
    "Engineering Science, The University of Osaka, 1-3 Machikaneyama, Toyonaka, "
    "Osaka 560-8531, Japan",
    "{6}Spintronics Research Network Division, Institute for Open "
    "Transdisciplinary Research (OTRI), The University of Osaka, Osaka 565-0871, "
    "Japan",
)
EMAIL = "*E-mail: alizadeh21@sanken.osaka-u.ac.jp"

TEXT_WIDTH_IN = 6.10          # 8.27 page - 1.06 left - 1.10 right margin

# ── the JJAP body format, from the journal's own Page Setup and
# Paragraph dialogs (margin_info.png / paragraph_info.png) ───────────
FONT = "Times New Roman"
FONT_PT = 12
# Headings are the one thing NOT in the body face.
HEAD_FONT = "Arial"
HEAD_PT = 14
# SubSectionTitle, the second level the template defines, is 11 pt.
SUBHEAD_PT = 11
# The title page's title, from the title-page template: Arial 14 bold.
TITLE_FONT = "Arial"
TITLE_PT = 14
LINE_SPACING = 1.5            # "1.5 lines"
# Word's "1.5 ch" first-line indent is 1.5 character widths, and one
# character width is the font size, so at 12 pt it is 18 pt.
INDENT_PT = 1.5 * FONT_PT
MARGIN_IN = {"top": 0.91, "bottom": 0.91, "left": 1.06, "right": 1.10}


# ── the numbers, read from the run ────────────────────────────────────────
def _separation():
    """The train/test separation evidence stored by stage 1."""
    path = os.path.join(_run.CFG_DIR, "dataset_summary.json")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    sep = d["separation"]
    assert sep.get("available"), "no separation evidence in " + path
    assert d["intervals"]["applicable"] is False, (
        "this run DID split the parameter intervals -- Table I below "
        "claims train and test share one distribution and would be wrong")
    return sep


def _equal_coverage():
    """Budgets that cost the same but are shaped differently.

    Many rays with few points each against few rays with many points, at
    the same fraction of the plane: the comparison that says which of the
    two is worth the measurement budget.  Both must have converged, or the
    comparison is between a result and a failure.  Every such pair within
    0.1 % of the plane is counted, because quoting only the pair with the
    biggest lead would be choosing the answer first."""
    ok = _run.converged()
    rows = [r for r in _run.comparison_rows() if ok.get(r["configuration"])]
    pairs = []
    for a in rows:
        for b in rows:
            if int(a["n_rays"]) <= int(b["n_rays"]):
                continue
            gap = abs(float(a["coverage"]) - float(b["coverage"]))
            if gap > 0.001:
                continue
            pairs.append({
                "gap": gap, "coverage": 100 * float(a["coverage"]),
                "many_rays": (int(a["n_rays"]), int(a["n_points"]),
                              float(a["f1@1"])),
                "few_rays": (int(b["n_rays"]), int(b["n_points"]),
                             float(b["f1@1"])),
                "more_rays_wins": float(a["f1@1"]) > float(b["f1@1"])})
    pairs.sort(key=lambda d: d["gap"])
    return {"pairs": pairs, "n": len(pairs),
            "wins": sum(1 for d in pairs if d["more_rays_wins"]),
            "closest": pairs[0] if pairs else None}


def facts():
    cfg = _run.cfg_json()
    row = next(r for r in _run.comparison_rows()
               if r["configuration"] == _run.CONFIG)
    ok = _run.converged()
    lo, hi = cfg["voltage_window"][0], cfg["voltage_window"][1]
    n_val = int(grid_train.VAL_FRACTION * int(cfg["n_train"]))
    infl = []
    for r in _run.comparison_rows():
        base, wide = float(r["coverage"]), float(r["coverage@1"])
        if base > 0:
            infl.append(wide / base)
    return {
        "n_train": int(cfg["n_train"]), "n_test": int(cfg["n_test"]),
        "n_total": int(cfg["n_train"]) + int(cfg["n_test"]),
        "res": int(cfg["resolution"]), "split_seed": cfg["split_seed"],
        "window_mv": hi - lo, "offset_scale": cfg["offset_scale"],
        "peak_width": cfg["coulomb_peak_width"],
        "temperature": cfg["temperature"],
        "epochs": int(cfg["epochs"]), "train_seed": int(cfg["train_seed"]),
        "n_val": n_val, "n_fit": int(cfg["n_train"]) - n_val,
        "val_pct": int(round(100 * grid_train.VAL_FRACTION)),
        "lr": grid_train.LEARNING_RATE, "batch": grid_train.BATCH_SIZE,
        "max_pos_weight": grid_train.MAX_POS_WEIGHT,
        "thresholds": grid_train.THRESHOLDS,
        "widths": [grid_model.WIDTH * 2 ** i for i in range(4)],
        # counted off the network itself: 1 949 409 for WIDTH = 32
        "n_params": grid_model.RayToLinesNet().n_params,
        "n_rays": _run.N_RAYS, "n_points": _run.N_POINTS,
        "threshold": _run.THRESHOLD,
        # the too-high cut of the threshold figure; the same default and
        # override as make_figures.LADDER_HIGH
        "ladder_high": float(os.environ.get("CSD_LADDER_HIGH", 0.9)),
        "coverage": 100 * float(row["coverage"]),
        "coverage1": 100 * float(row["coverage@1"]),
        "f1": {t: float(row["f1@%d" % t]) for t in (0, 1, 2, 3)},
        "prec1": float(row["precision@1"]), "rec1": float(row["recall@1"]),
        "iou": float(row["iou"]),
        "n_budgets": len(ok), "n_failed": sum(1 for v in ok.values() if not v),
        # pixel accuracy of the CONVERGED budgets: its range at tau = 0 and
        # its lowest value from tau = 1 on (the Results text quotes both)
        "acc0": [float(r["accuracy@0"]) for r in _run.comparison_rows()
                 if ok.get(r["configuration"])],
        "acc1_min": min(float(r["accuracy@%d" % t])
                        for r in _run.comparison_rows()
                        if ok.get(r["configuration"]) for t in (1, 2, 3)),
        "rays_swept": sorted({int(r["n_rays"]) for r in _run.comparison_rows()}),
        "points_swept": sorted({int(r["n_points"])
                                for r in _run.comparison_rows()}),
        "equal_cov": _equal_coverage(),
        "infl_lo": min(infl), "infl_hi": max(infl),
        "line_frac": 100 * float(row["true_line_fraction"]),
        "intervals": cc.DEFAULT_INTERVALS,
        "separation": _separation(),
        "device": os.path.basename(_run.DEVICE),
        "device_f1": _run.DEVICE_F1, "test_mean": _run.TEST_MEAN_F1,
    }


F = facts()

# The text calls the quoted configuration "the best-performing configuration
# (8 × 60)" (your wording, 2026-09-28).  That is a claim about the run, so it
# is checked: no other budget may have a higher F1@1.
_BEST = max(_run.comparison_rows(), key=lambda r: float(r["f1@1"]))
if (int(_BEST["n_rays"]), int(_BEST["n_points"])) != (F["n_rays"],
                                                        F["n_points"]):
    raise SystemExit(
        "the text calls %d x %d the best-performing configuration, but "
        "%s x %s has the higher F1@1 (%.3f)"
        % (F["n_rays"], F["n_points"], _BEST["n_rays"], _BEST["n_points"],
           float(_BEST["f1@1"])))
BEST_CFG = "the best-performing configuration (%d × %d)" % (F["n_rays"],
                                                             F["n_points"])

_WORDS = ("zero one two three four five six seven eight nine ten eleven "
          "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
          "nineteen twenty").split()


def _word(n):
    """A count spelled out, as the text writes small numbers."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def _retrain_sentence():
    """What run_10 did, read from retrain_collapsed.json and the seed-0
    training summaries -- so the text cannot outlive a change to either."""
    path = os.path.join(_run.RUN_DIR, "retrain_collapsed.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    cells = rec["cells"]
    names = sorted(cells, key=lambda n: (int(n.split("_")[0]),
                                         int(n.split("_rays_")[1].split("_")[0])))
    old = []
    for n in names:
        with open(os.path.join(_run.RUN_DIR, n, "model",
                               "training_summary.json")) as fh:
            old.append(float(json.load(fh)["best_val_f1"]))
    budgets = ["%s × %s" % (n.split("_")[0],
                                   n.split("_rays_")[1].split("_")[0])
               for n in names]
    listed = ", ".join(budgets[:-1]) + " and " + budgets[-1]
    # the seed-0 best validation F1@1 of every budget that converged
    others = []
    for path in glob.glob(os.path.join(_run.RUN_DIR, "*_rays_*_points_*",
                                       "model", "training_summary.json")):
        if os.path.basename(os.path.dirname(os.path.dirname(path))) \
                not in cells:
            with open(path) as fh:
                others.append(float(json.load(fh)["best_val_f1"]))
    # "converged normally": every chosen seed passed run_10's rule
    assert all(c.get("chosen_seed") for c in cells.values()), cells
    # YOUR TEXT, word for word, 2026-09-29, in your order.  Checked that
    # day: 0.4155-0.4380 for the four, 0.6596-0.8126 for the other eleven
    # (4 x 40 is 0.6596, hence a range and not "at least 0.66").  Every
    # number here is read from the files, not typed in.
    return ("For %s of the %s measurement budgets (%s), the first training run did "
            "not converge: the best validation F1@1 over all %d epochs was "
            "only %.2f–%.2f, far below the %.2f–%.2f reached by the other "
            "measurement budgets. These %s measurement budgets were retrained with a different "
            "random seed, with the training data, architecture and "
            "training procedure unchanged, and all %s then converged "
            "normally."
            % (_word(len(names)), _word(F["n_budgets"]), listed,
               F["epochs"], min(old), max(old), min(others), max(others),
               _word(len(names)), _word(len(names))))


def _monotonic_sentence():
    """Whether F1@1 rises with the ray count at each point count, as the
    table says -- with the exception named, not smoothed over."""
    rows = _run.comparison_rows()
    good, bad = [], []
    for p in F["points_swept"]:
        s = sorted((int(r["n_rays"]), float(r["f1@1"]))
                   for r in rows if int(r["n_points"]) == p)
        drops = [(a, b) for a, b in zip(s, s[1:]) if b[1] <= a[1]]
        (bad if drops else good).append((p, drops))
    if not bad:
        return ("For a fixed number of points per ray, the score increases "
                "monotonically with the number of rays.")
    g = " and ".join(str(p) for p, _ in good)
    out = ("For %s points per ray, the score increases monotonically with "
           "the number of rays." % g) if good else ""
    for p, drops in bad:
        for a, b in drops:
            out += (" At %d points per ray, %d rays (F1@1 = %.3f) scores "
                    "below %d rays (%.3f)." % (p, b[0], b[1], a[0], a[1]))
    return out.strip()


def _equal_cov_short():
    ec = F["equal_cov"]
    return ("At equal coverage, allocating the measurement budget to additional rays "
            "yields a higher score than allocating it to additional points "
            "per ray in %s of the %s comparable pairs."
            % (_word(ec["wins"]), _word(ec["n"])))


def _equal_coverage_summary():
    """The same finding as _equal_coverage_sentence(), without the figure.

    Written for the Conclusions, where there is no Figure 1 to point at."""
    ec = F["equal_cov"]
    out = ("At equal coverage, additional rays are generally worth more "
           "than additional points per ray: of the %d converged pairs "
           "lying within 0.1 %% of the plane of each other, %d favour the "
           "measurement budget with more rays." % (ec["n"], ec["wins"]))
    for d in [x for x in ec["pairs"] if not x["more_rays_wins"]]:
        (ra, pa, fa) = d["many_rays"]
        (rb, pb, fb) = d["few_rays"]
        out += (" The finding is not universal: at %.1f %% of the plane, "
                "%d rays x %d points reaches %.3f against %.3f for %d "
                "rays x %d points."
                % (d["coverage"], ra, pa, fa, fb, rb, pb))
    return out


def _equal_coverage_sentence():
    """The equal-coverage claim, stated with the pairs that contradict it.

    The tally and the exceptions are read from the run rather than typed,
    so the paragraph cannot quietly become a claim about the winning pairs
    only.  Runs that did not converge are already excluded upstream, in
    _equal_coverage()."""
    ec = F["equal_cov"]
    out = ("At equal coverage, allocating the measurement budget to "
           "additional rays generally outperforms allocating it to "
           "additional points per ray, although the trend is not uniform: "
           "of the %d converged pairs lying within 0.1 %% of the plane of "
           "each other, %d favour the measurement budget with more rays."
           % (ec["n"], ec["wins"]))
    losses = [d for d in ec["pairs"] if not d["more_rays_wins"]]
    for d in losses:
        (ra, pa, fa) = d["many_rays"]
        (rb, pb, fb) = d["few_rays"]
        out += (" The exception is %d rays x %d points, which reaches F1 "
                "at tau = 1 of %.3f against %.3f for %d rays x %d points "
                "at the same %.1f %% of the plane."
                % (ra, pa, fa, fb, rb, pb, d["coverage"]))
    out += (" Figure 1 presents the comparison directly, including the "
            "pair that does not follow the trend."
            if losses else
            " Figure 1 presents the comparison directly.")
    return out





# What each capacitance entry does, for Table I.  Keyed by entry name so a
# renamed or added interval shows up as a missing key rather than silently
# being described as something it is not.
_CONTROLS = {
    "d1d1": "dot 1 self capacitance (charging energy)",
    "d2d2": "dot 2 self capacitance (charging energy)",
    "d1d2": "interdot capacitance: the anticrossing",
    "d1g1": "dot 1 primary gate: honeycomb period",
    "d2g2": "dot 2 primary gate: honeycomb period",
    "d1g2": "dot 1 cross gate: line slope",
    "d2g1": "dot 2 cross gate: line slope",
    "d1g3": "dot 1 residual coupling to gate 3",
    "d2g3": "dot 2 residual coupling to gate 3",
    "s1d1": "dot 1 to sensor coupling",
    "s1d2": "dot 2 to sensor coupling",
    "s1g1": "gate 1 to sensor cross-talk",
    "s1g2": "gate 2 to sensor cross-talk",
    "s1g3": "gate 3 to sensor cross-talk",
}


# THE CODE'S KEYS ARE QARRAY'S; THE PAPER'S SYMBOLS ARE THE FIGURE'S.
# QArray names a matrix after the pair it couples (Cgd, "gates and dots")
# but shapes it (n_dot, n_gate), so its entries read the other way round:
# c_d1g1 sits in Cgd.  The figure resolves that by naming the matrix in the
# order its own entries use -- C_dg with entries c_{d_i g_j} -- and this
# table follows the figure.  Its LaTeX source is
# illustrator/capacitance_matrices.tex.  Kept as an explicit map so that if
# the code ever renames a matrix, this raises instead of printing a name
# the figure does not carry.
_MATRIX_LABEL = {"Cdd": "C_{dd}", "Cgd": "C_{dg}", "Cds": "C_{sd}",
                 "Cgs": "C_{sg}"}


def _table_i_rows():
    """[rows, indices of the last row of each matrix block]."""
    rows, ends = [], []
    for matrix, entries in F["intervals"].items():
        if matrix not in _MATRIX_LABEL:
            raise SystemExit("no figure label for matrix %r" % matrix)
        for name, spec in entries.items():
            bands = cc.as_bands(spec)
            interval = " U ".join("%.2f - %.2f" % (lo, hi) for lo, hi in bands)
            rows.append([_MATRIX_LABEL[matrix], name, interval])
        ends.append(len(rows))          # 1-based: row 0 is the header
    return rows, ends[:-1]              # the last block ends at the table rule


def _table_ii_rows():
    f, sep = F, F["separation"]
    return [
        ["Devices generated", "%d" % f["n_total"]],
        ["Training devices", "%d" % f["n_train"]],
        ["  of which fit the weights", "%d" % f["n_fit"]],
        ["  of which validation", "%d" % f["n_val"]],
        ["Held-out (test) devices", "%d" % f["n_test"]],
        ["Split performed on", "device identity (not image)"],
        ["Split seed", "%s" % f["split_seed"]],
        ["Parameter intervals", "identical for both sets"],
        ["Parameter-space dimensions", "%d" % sep["dimensions"]],
        ["Min. train-to-test distance", "%.3f" % sep["min_distance"]],
        ["Mean nearest train-to-test distance",
         "%.3f" % sep["mean_nearest_distance"]],
        ["Min. train-to-train distance (yardstick)",
         "%.3f" % sep["min_within_train"]],
        ["Devices shared between sets", "0 (checked by ID and by hash)"],
    ]


# The abstract.  It is the title page's own abstract -- printed under the
# author block with no heading of its own, which is how the JJAP template
# lays a title page out -- so it lives here rather than in a section.
# The abstract, as the author wrote it.  It is NOT assembled from the run
# like the rest of the text: it is his words, kept verbatim.  What the run
# still decides is whether it is TRUE -- _check_abstract() below refuses to
# build if a number in it has drifted from comparison.csv.
# The abstract, as the author wrote it.  It is NOT assembled from the run
# like the rest of the text: it is his words, kept verbatim.  What the run
# still decides is whether it is TRUE -- _check_abstract() below refuses to
# build if a number in it has drifted from comparison.csv.
# The abstract, as the author wrote it.  It is NOT assembled from the run
# like the rest of the text: it is his words, kept verbatim.  What the run
# still decides is whether it is TRUE -- _check_abstract() below refuses to
# build if a number in it has drifted from comparison.csv.
# YOUR ABSTRACT, word for word, 2026-09-26.
ABSTRACT = (
    # YOUR TEXT, word for word, 2026-09-28.  4.7 % and 0.82 are the 8 x 60
    # headline; _check_abstract() verifies both against the run.
    "Operating spin qubits in gate-defined quantum dots requires knowing "
    "each dot's charge occupation, which is read from the charge stability "
    "diagram (CSD). Dense CSD scans are slow, and recent methods "
    "reconstruct the charge-sensor image from a sparse scan and extract "
    "transition lines with classical detectors, whose errors also "
    "contaminate the training labels. Simulators, in contrast, provide the "
    "transition lines exactly. We therefore train on simulated data and "
    "reconstruct the lines directly. Because a sparse scan leaves most of "
    "the voltage plane unobserved, the reconstruction is probabilistic. A "
    "U-Net trained on a constant-capacitance simulator takes a few "
    "one-dimensional voltage sweeps (rays) and returns a per-pixel "
    "probability of a transition line. The map is binarised with a "
    "validation-selected threshold and scored with a tolerance-based F1. "
    "With eight rays sampling 4.7 % of the voltage plane, the lines of "
    "held-out simulated devices are recovered with F1 = 0.82.")


def _free_parameters():
    """How many numbers the four matrices print, and how many are free.

    Cdd is built by MatrixGenerator.generate_symmetric, so its off-diagonal
    pair is one draw, not two.  Both counts are derived from the shapes the
    generator uses and from DEFAULT_INTERVALS, so neither can drift from the
    code: writing either one by hand is how the text came to say 13.
    """
    shapes = {"Cdd": (2, 2), "Cgd": (2, 3), "Cds": (1, 2), "Cgs": (1, 3)}
    symmetric = {"Cdd"}
    printed = free = 0
    per = []
    for name, (rows, cols) in shapes.items():
        cells = rows * cols
        n = len(cc.DEFAULT_INTERVALS[name])
        if name in symmetric:
            assert n == rows * (rows + 1) // 2, (name, n)
        else:
            assert n == cells, (name, n, cells)
        printed += cells
        free += n
        per.append((_MATRIX_LABEL[name], n))   # the paper's symbol, not QArray's key
    return printed, free, per


def _check_abstract():
    """Refuse to build a title page whose abstract has gone stale.

    The abstract is written by hand, so a re-run of the sweep could leave a
    number in it that the results no longer support.  Each claim is checked
    against the run; a mismatch names the number rather than being quietly
    published."""
    # Checked at the precision the abstract states them (2026-09-26):
    # F1 to two decimals, coverage to one.  The swept ranges are no longer
    # in the abstract, so they are no longer checked here.
    _words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    claims = [
        ("%d rays" % F["n_rays"],
         "%s rays" % _words.get(F["n_rays"], str(F["n_rays"]))),
        ("%d points" % F["n_points"], "%d points" % F["n_points"]),
        ("coverage %.1f %%" % F["coverage"], "%.1f %%" % F["coverage"]),
        ("F1@1 %.2f" % F["f1"][1], "F1 = %.2f" % F["f1"][1]),
        ("%d held-out devices" % F["n_test"], "%d held-out" % F["n_test"]),
    ]
    # A number is checked only if the abstract states that kind of claim
    # at all (2026-09-27: the abstract carries no result numbers now).
    stated = {"rays": " rays ", "points": " points", "coverage": "%",
              "F1@1": "F1 =", "held-out": "held-out"}
    claims = [(n, t) for n, t in claims
              if any(k in n and v in ABSTRACT for k, v in stated.items())]
    # "held-out" alone is not a count (2026-09-28: the abstract says "the
    # lines of held-out simulated devices"); check it only when a number
    # stands before it.
    if not re.search(r"\d+ held-out", ABSTRACT):
        claims = [(n, t) for n, t in claims if "held-out" not in n]
    missing = [name for name, text in claims if text not in ABSTRACT]
    if missing:
        raise SystemExit(
            "the abstract no longer matches the run: %s. "
            "Edit ABSTRACT in make_method_docs.py, or re-check the run."
            % ", ".join(missing))


def _fan_angles(n_rays):
    """The pipeline's own angles, so the text cannot disagree with it.

    ray_peaks.fan_angles is np.linspace(0, 90, n + 2)[1:-1]; it is spelled
    out here rather than imported because this file builds documents and
    must not need the measurement stack to do it."""
    return [90.0 * k / (n_rays + 1) for k in range(1, n_rays + 1)]


def _angle_set(angles):
    """`{18, 36, 54, 72}` written out, or elided when it is long."""
    deg = [("%g\u00b0" % a) for a in angles]
    if len(deg) <= 5:
        return ", ".join(deg)
    return ", ".join(deg[:3]) + ", \u2026, " + deg[-1]


def _ray_fan_counts():
    """What fig_ray_fan counted while drawing itself.

    The caption quotes how many measured points land on a transition line
    and how many peaks the detector reports.  Those are properties of the
    measurement, not of this file, so they are read from the sidecar the
    figure writes rather than typed here.  A missing sidecar raises: a
    caption with a made-up number in it is worse than a build that
    stopped."""
    path = _figures.out_path("ray_fan_counts.json")
    if not os.path.exists(path):
        raise SystemExit(
            "%s is missing -- run make_figures.fig_ray_fan first, which "
            "writes the counts its caption quotes" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_FREE = _free_parameters()
# The two self-capacitances couple a dot to ground, so they are not
# branches in the drawing; the text counts the branches with this.
_DRAWN = _FREE[1] - 2
TABLE_I_ROWS, TABLE_I_RULES = _table_i_rows()
TABLE_II_ROWS = _table_ii_rows()


def _table_results_rows():
    """Every budget of the sweep, read from comparison.csv: one row each,
    grouped by points per ray, with a rule after each group.  A budget
    that was retrained (retrain_collapsed.json) carries an asterisk."""
    rows = sorted(_run.comparison_rows(),
                  key=lambda r: (int(r["n_points"]), int(r["n_rays"])))
    out, ends = [], []
    for i, r in enumerate(rows):
        star = "*" if r["configuration"] in _run.SEEDS else ""
        out.append(["%s × %s%s" % (r["n_rays"], r["n_points"], star),
                    "%.2f" % (100 * float(r["coverage"])),
                    "%.1f" % float(r["threshold"]),
                    "%.3f" % float(r["precision@1"]),
                    "%.3f" % float(r["recall@1"]),
                    "%.3f" % float(r["f1@0"]),
                    "%.3f" % float(r["f1@1"])])
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is not None and nxt["n_points"] != r["n_points"]:
            ends.append(len(out))
    return out, ends


TABLE_R_ROWS, TABLE_R_RULES = _table_results_rows()


def _rounded(value, places=1):
    """`17.85` -> `17.9`, not `17.8`.

    The stored coverage is exactly 17.85, and a binary float of it sits a
    hair BELOW that, so "%.1f" rounds it down.  Half-up is what a reader
    checking the number against the CSV expects."""
    import decimal
    q = decimal.Decimal(1).scaleb(-places)
    return str(decimal.Decimal(repr(round(value, 6)))
               .quantize(q, rounding=decimal.ROUND_HALF_UP))


_SPELLED = ("zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "ten", "eleven", "twelve")


def _spelled(n):
    """`8` -> `eight`.

    A sentence does not open a count in digits, and the count is read from
    the run rather than typed, so it has to be spelled out here."""
    return _SPELLED[n] if n < len(_SPELLED) else str(n)


# ── the five parts ────────────────────────────────────────────────────────
# (number, slug, title, abstract, [(heading, [paragraph, ...]), ...],
#  [(figure file, caption), ...])
def parts():
    f = F
    thr_list = ", ".join("%g" % t for t in f["thresholds"])
    return [
        ("0", "cover_letter",
         "Cover letter",
         None,
         [
             ("To the Editor", [
                 "Dear Editor,",
                 "We wish to submit our manuscript, “%s”, for "
                 "consideration as a Regular Paper in the Japanese Journal "
                 "of Applied Physics." % PAPER_TITLE,
                 "Tuning a semiconductor quantum dot array requires the "
                 "charge stability diagram, and acquiring that diagram by a "
                 "dense two-dimensional voltage scan is the step that does "
                 "not scale. In this work we show that the "
                 "charge-transition lines of a double quantum dot can be "
                 "recovered from a small fraction of the gate-voltage plane "
                 "measured along one-dimensional rays. A fan of %d rays "
                 "sampled at %d points touches %.1f %% of the plane; a "
                 "U-Net is given only those measurements, as two image "
                 "channels, and returns one probability per pixel that a "
                 "transition line passes through it. On %d held-out devices "
                 "the recovered lines reach F1 = %.3f at a tolerance of one "
                 "pixel."
                 % (F["n_rays"], F["n_points"], F["coverage"], F["n_test"],
                    F["f1"][1]),
                 "We believe the work suits this journal for two reasons. "
                 "It continues a line we published here previously on "
                 "ray-based tuning of a GaAs quadruple-dot array [Jpn. J. "
                 "Appl. Phys. 64, 02SP22 (2025)], moving from classifying a "
                 "regime to reconstructing the diagram itself. And its "
                 "concern is the measurement cost of device tuning, which "
                 "is a practical obstacle to scaling quantum dot arrays "
                 "rather than a purely algorithmic question.",
                 "We have tried to report the result in a form that cannot "
                 "be over-read. The measurement budget is quoted as the "
                 "fraction of the plane actually measured. The threshold "
                 "that turns the probability map into lines is fitted on a "
                 "validation dataset carved out of the training devices and "
                 "never on the held-out set. Because transition lines are "
                 "one pixel wide, the score is a tolerant F1 and is always "
                 "given with its tolerance and with the area that tolerance "
                 "covers. We also state plainly that the devices are "
                 "simulated and the simulation is noise-free, so the claim "
                 "established here concerns recoverability of the geometry "
                 "and not yet validation against measured data.",
                 "The manuscript is original, has not been published "
                 "elsewhere, and is not under consideration by another "
                 "journal. All authors have read and approved the "
                 "submission and declare no conflict of interest. The code "
                 "and the analysis that produced every figure and number in "
                 "the manuscript are available in a public repository, "
                 "whose address is given in the paper.",
                 "Thank you for considering our work.",
             ]),
         ],
         [],
         {"letter": True}),

        ("1", "abstract",
         PAPER_TITLE,
         ABSTRACT,
         # The title page carries what the template carries: the title, the
         # authors, the affiliations, the e-mail and the abstract.  The
         # template has no heading on any of them and no block after the
         # abstract, so neither has this.
         [],
         [],
         {"title_page": True}),

        ("2", "introduction",
         "Introduction",
         # No part abstract: the text below IS the introduction, set word
         # for word, and a summary above it would be a second opening
         # paragraph saying the same thing.
         None,
         [
             # ONE continuous section.  The introduction is an argument --
             # the cost of a dense scan, the prior work, what this paper
             # adds -- and it reads as one; breaking it into numbered
             # subsections labels the argument instead of making it.  A
             # heading of None prints the paragraphs with no heading above
             # them; the part's own title carries the number.
             #
             # The wording is the author's, set word for word.  The NUMBERS
             # in it are not typed: every one is read from the run (see
             # facts()), so the text cannot outlive a re-run that changes
             # them.
             (None, [
                 # YOUR TEXT, word for word, 2026-09-27: the first three
                 # paragraphs of the Introduction, with references 4 and 5 new.
                 "Spin qubits in gate-defined semiconductor quantum dots are"
                 " a promising platform for quantum computing, owing to "
                 "their potential for scaling and their compatibility with "
                 "established semiconductor fabrication technology [1–3]. In"
                 " these devices, quantum information is encoded in the "
                 "spins of electrons confined by electrostatic potentials, "
                 "which are defined by the voltages applied to an array of "
                 "gate electrodes. Because the charging energy of a quantum "
                 "dot greatly exceeds the thermal energy at cryogenic "
                 "temperatures, electron transport through the dot is "
                 "governed by Coulomb blockade, and the number of electrons "
                 "it holds is quantized and set by the gate voltages [4]. "
                 "Operating a device therefore requires that these voltages "
                 "be tuned such that each dot holds a precisely known number"
                 " of electrons. As devices scale to larger arrays, the "
                 "number of gate voltages to be set grows accordingly, and "
                 "determining the electron occupation efficiently becomes a "
                 "central experimental challenge.",

                 "The elementary building block of such arrays is the double"
                 " quantum dot (DQD). A DQD consists of two dots coupled to "
                 "each other through a tunnel barrier and to separate "
                 "electron reservoirs, with each dot controlled primarily by"
                 " its own plunger gate [5]. The electron occupation (N₁, "
                 "N₂) of a DQD is determined from its charge stability "
                 "diagram (CSD) [5,6]. A CSD is acquired by sweeping the two"
                 " plunger-gate voltages V₁ and V₂ while recording the "
                 "signal of a capacitively coupled charge sensor [7]. Each "
                 "change in the occupation of either dot shifts the "
                 "electrostatic potential at the charge sensor and produces an "
                 "abrupt change in its signal. The resulting two-dimensional"
                 " map consists of regions of fixed charge configuration "
                 "separated by sharp charge transition lines. Owing to the "
                 "capacitive coupling between the dots, these lines form a "
                 "characteristic honeycomb pattern. Two families of lines "
                 "correspond to the loading of an electron from a reservoir "
                 "into either dot, and short interdot lines correspond to "
                 "the transfer of an electron between the dots. Because each"
                 " dot couples more strongly to its own plunger gate than to"
                 " the other, the loading lines of the two dots have "
                 "markedly different slopes in the (V₁, V₂) plane. "
                 "Identifying these lines, and counting them from the fully "
                 "depleted (0,0) configuration, establishes the charge "
                 "configuration throughout the diagram.",

                 "Most of the CSD, however, carries no information about the"
                 " charge configuration. Within each cell of the honeycomb, "
                 "the DQD is in Coulomb blockade: the charging energy "
                 "suppresses changes in the electron number, so the charge "
                 "configuration (N₁, N₂) remains fixed and the charge-sensor"
                 " response varies only smoothly with the gate voltages. The"
                 " charge configuration is therefore determined entirely by "
                 "the positions of the charge transition lines, which "
                 "constitute one-dimensional features occupying a small "
                 "fraction of the two-dimensional (V₁, V₂) plane. "
                 "Conventional CSD acquisition nevertheless employs dense "
                 "raster scanning on a uniform grid, in which V₁ is swept "
                 "over its full range for each value of V₂, so that every "
                 "pixel is measured irrespective of its information content."
                 " For an N × N grid, the number of measurements scales as "
                 # ", whereas the number of pixels intersected by
                 # transition lines scales only as O(N)" deleted (your
                 # call, 2026-09-27)
                 "O(N²). Consequently, the "
                 "overwhelming majority of acquisition time is spent "
                 "sampling Coulomb-blockaded regions whose charge "
                 "configuration is already fixed by the surrounding "
                 "transition lines. "
                 # added 2026-09-28 with references 8-11, as you gave it
                 "Reducing this cost is a central goal of automated device "
                 "tuning [8]. Machine-learning strategies that choose only "
                 "the most informative measurements [9] or replace "
                 "two-dimensional scans by one-dimensional rays [10,11] have "
                 "been demonstrated.",

                 "Hernandes et al. have recently addressed the cost of this "
                 "acquisition with a generative approach [12]. They trained "
                 "a conditional diffusion model on approximately 9000 "
                 "measured CSDs to reconstruct full diagrams from sparse "
                 "measurements. Two sampling strategies were evaluated: a "
                 "uniform grid of points, from which the transition lines "
                 "were preserved with as little as 4% of the data measured, "
                 "and horizontal and vertical line cuts, evaluated at "
                 "23\u201344% of the data. The model reconstructs the "
                 "charge-sensor signal over the entire plane, including the "
                 "featureless Coulomb-blockaded regions, and its output is "
                 "an image. The positions of the transition lines, which "
                 "are the quantity that determines the electron occupation, "
                 "are extracted from the reconstructed image only "
                 "afterwards, using classical edge- and ridge-detection "
                 "filters. The same filters, applied to the measured "
                 "diagrams, provide the reference lines against which the "
                 "reconstructions are evaluated [12]. Both the training data "
                 "and the reference transition lines therefore derive from "
                 # your text, word for word, 2026-09-27; checked against
                 # the Hernandes et al. PDF: Canny edges and Frangi ridges,
                 # 4000 diagrams labelled clean/noisy to train a classifier
                 "measurement. The approach is, however, inherently "
                 "constrained by its reliance on experimental training "
                 "data. The size of the training set is bounded by the "
                 "considerable cost of acquiring each diagram, and its "
                 "construction required expert annotation, in which a "
                 "subset of diagrams was manually classified by quality to "
                 "train a filtering classifier. Furthermore, the "
                 "reconstruction is evaluated against transition lines "
                 "extracted by conventional edge and ridge detectors (Canny "
                 "[13] and Frangi [14] filters), so the reference itself is subject "
                 "to the errors of these algorithms and does not constitute "
                 "an independent ground truth.",

                 "Recent advances in the simulation of quantum-dot "
                 "devices have made training data of this kind inexpensive "
                 "to generate. Fast solvers of the constant-capacitance "
                 "model are now available: QArray computes a 100 \u00d7 100 "
                 "pixel charge stability diagram of a 16-dot array in less "
                 "than a second, and smaller arrays in milliseconds, faster "
                 "than they could be measured experimentally [15]. Related "
                 # [17] moved to right after "QDarts" (your call, 2026-09-27)
                 "packages include QDsim [16] and QDarts [17], which finds "
                 "charge transitions in the presence of finite tunnel "
                 "couplings, non-constant charging energies, and charge-sensor "
                 "dots. Most "
                 "recently, QArray+ has extended this framework into a "
                 "physics-informed simulator [18]. It includes the coherent "
                 "tunnel coupling between neighbouring dots, whose "
                 "strength depends on the gate voltages. Near an interdot "
                 "transition, where two charge configurations are nearly "
                 "degenerate, the ground state becomes a superposition of "
                 "both. The simulated diagrams therefore exhibit the "
                 "avoided crossings and fractional charge occupations "
                 "observed in tunnel-coupled devices [18].",

                 # "Figure 1 sets out the method." deleted with the
                 # overview figure (your call, 2026-09-27).
                 "The simulator stands in "
                 "for the device while the network is trained, supplying "
                 "both the stability diagrams and the transition lines "
                 "that label them; the network is given only what a fan "
                 "of rays measured, and returns a probability map for a "
                 "device it was not trained on.",

                 "We therefore propose to train a neural network entirely "
                 "on simulated diagrams, as has been done for charge-state "
                 "recognition [19], and to employ it in experiments as "
                 "a probabilistic model of the transition-line positions. "
                 "Given only a small number of one-dimensional rays "
                 "measured on a device, the network returns, for every "
                 "pixel of the voltage plane, the probability that a "
                 "charge-transition line passes through it. It does so "
                 "without reconstructing the charge-sensor signal. "
                 # your wording, 2026-09-28
                 "The rays are two-gate sweeps radiating from one corner of "
                 "the gate-voltage window, in which both gate voltages "
                 "change simultaneously [10,20]. Unlike the "
                 "ray-based classifier of Ref. 10, which labels the charge "
                 "state near one point, the present network reconstructs "
                 "the transition lines over the whole window. "
                 "A horizontal line cut "
                 "runs nearly parallel to the loading lines of one dot and "
                 "a vertical cut to those of the other, so that each "
                 "axis-aligned cut predominantly samples a single family of "
                 # [12] added at your request, 2026-09-30
                 "lines [12]. Each oblique ray, in contrast, intersects both "
                 "families. The network is a U-Net [21] that receives the ray "
                 "data as two image channels. The first contains the charge-sensor "
                 "signal at the sampled pixels. The second is a binary mask "
                 "of the sampled pixels, which allows the network to "
                 "distinguish a sampled pixel within a Coulomb-blockaded "
                 "region from a pixel that was not sampled. In the present "
                 "work, the training devices are simulated with QArray.",

                 "Three methodological choices are stated explicitly, since "
                 "the reported results depend upon them. First, the "
                 "measurement budget is quantified as the fraction of the "
                 "voltage plane actually sampled by the rays, rather than "
                 "as a number of measurement points. Second, the threshold "
                 "that converts the probability map into binary transition "
                 "lines is selected on a validation subset drawn from the "
                 "training devices, never on the held-out test devices, and "
                 "it is reported with each result. "
                 # your text, word for word, 2026-09-29
                 "Third, the transition lines are marked by chains only one "
                 "pixel thick (Sec. 3.1.1). In a pixel-exact comparison, a "
                 "predicted line displaced by a single pixel therefore "
                 "shares few or none of its pixels with the truth, and the "
                 "score falls sharply, although the line has been located "
                 "to within one pixel. We therefore adopt a "
                 "tolerance-based F1 score, F1@\u03c4, and report it "
                 "throughout together with the tolerance \u03c4 and with "
                 "the fraction of the plane lying within \u03c4 pixels of a "
                 # "The strict intersection-over-union is reported alongside
                 # it." deleted at your request, 2026-09-29
                 "sampled pixel.",

                 "With %s rays of %d points each, sampling %s%% of the "
                 "voltage plane, the predicted transition lines reach "
                 "F1@1 = %.2f on %d held-out devices."
                 # "For reference, x% of the plane lies within one pixel of
                 # a sampled pixel." deleted at your request, 2026-09-30
                 % (_spelled(f["n_rays"]), f["n_points"],
                    _rounded(f["coverage"]), f["f1"][1], f["n_test"]),

                 "In this study, both training and evaluation are performed "
                 "on devices simulated within the constant-capacitance "
                 "model, with capacitances drawn independently at random "
                 "for each device, and the simulations are free of noise. "
                 "The result established here is therefore the first step "
                 "of the proposed approach: the charge-transition lines of "
                 "a double quantum dot can be located from rays sampling a "
                 "few percent of the voltage plane. Deployment of the "
                 "trained model on measured devices, whose diagrams are "
                 "subject to charge-sensor noise, charge switching, and deviations "
                 "from the constant-interaction picture, remains the "
                 "subject of future work. The code for the simulation, "
                 "training and evaluation is openly available [22].",
             ]),
         ],
         # The overview figure (method_overview.png) was deleted, your
         # call, 2026-09-27; the introduction has no figure now.
         [],
         # Numbered in order of first citation (JJAP).  Renumbered
         # 2026-09-28 when references 7-11, 13, 14, 19, 21, 23-25, 27 and 30
         # were added from your list: old 7->12, 8->15, 9->16, 10->17,
         # 11->18, 12->20, 13->22, 14->26, 15->28, 16->29.  Entries 7, 13,
         # 14, 21 and 25 were NOT checked against the publisher (the list
         # said so); check their pages through the DOI before submission.
         {"references": (
             "D. Loss and D. P. DiVincenzo, Phys. Rev. A 57, 120 (1998).",
             "R. Hanson, L. P. Kouwenhoven, J. R. Petta, S. Tarucha, and "
             "L. M. K. Vandersypen, Rev. Mod. Phys. 79, 1217 (2007).",
             "F. A. Zwanenburg, A. S. Dzurak, A. Morello, M. Y. Simmons, "
             "L. C. L. Hollenberg, G. Klimeck, S. Rogge, S. N. Coppersmith, "
             "and M. A. Eriksson, Rev. Mod. Phys. 85, 961 (2013).",
             # 4 and 5 new with your Introduction, 2026-09-27, as you gave them
             "L. P. Kouwenhoven, C. M. Marcus, P. L. McEuen, S. Tarucha, "
             "R. M. Westervelt, and N. S. Wingreen, in Mesoscopic Electron "
             "Transport, ed. L. L. Sohn, L. P. Kouwenhoven, and G. Schön "
             "(Kluwer, Dordrecht, 1997) NATO ASI Series E, Vol. 345, p. 105.",
             "W. G. van der Wiel, S. De Franceschi, J. M. Elzerman, "
             "T. Fujisawa, S. Tarucha, and L. P. Kouwenhoven, Rev. Mod. Phys. "
             "75, 1 (2002).",
             # 6 (was 4)
             "J. M. Elzerman, R. Hanson, J. S. Greidanus, L. H. Willems van "
             "Beveren, S. De Franceschi, L. M. K. Vandersypen, S. Tarucha, "
             "and L. P. Kouwenhoven, Phys. Rev. B 67, 161308(R) (2003).",
             # 7-11 new, 2026-09-28
             "M. Field, C. G. Smith, M. Pepper, D. A. Ritchie, J. E. F. Frost, "
             "G. A. C. Jones, and D. G. Hasko, Phys. Rev. Lett. 70, 1311 (1993).",
             "J. P. Zwolak and J. M. Taylor, Rev. Mod. Phys. 95, 011006 (2023).",
             "D. T. Lennon, H. Moon, L. C. Camenzind, L. Yu, D. M. Zumbühl, "
             "G. A. D. Briggs, M. A. Osborne, E. A. Laird, and N. Ares, "
             "npj Quantum Inf. 5, 79 (2019).",
             "J. P. Zwolak, T. McJunkin, S. S. Kalantre, S. F. Neyens, "
             "E. R. MacQuarrie, M. A. Eriksson, and J. M. Taylor, PRX Quantum 2, "
             "020335 (2021).",
             "J. Ziegler, F. Luthi, M. Ramsey, F. Borjans, G. Zheng, and "
             "J. P. Zwolak, Phys. Rev. Appl. 20, 034067 (2023).",
             # 12 (was 7)
             "V. Hernandes, J. Rogers, R. Koch, T. Spriggs, B. Undseth, "
             # the published version, as you gave it, 2026-09-28: JJAP
             # style, no title (DOI 10.1088/2632-2153/ae8070)
             "A. Chatterjee, L. M. K. Vandersypen, and E. Greplova, "
             "Mach. Learn.: Sci. Technol. 7, 045009 (2026).",
             # 13, 14 new: Canny and Frangi
             "J. Canny, IEEE Trans. Pattern Anal. Mach. Intell. PAMI-8, 679 "
             "(1986).",
             "A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever, "
             "Proc. MICCAI’98, Lecture Notes in Computer Science 1496 "
             "(Springer, Berlin, 1998) p. 130.",
             # 15-18 (were 8-11)
             # 6 from csd-materials/citations/Qarray.txt, 2026-09-26
             "B. van Straaten, J. Hickie, L. Schorling, J. Schuff, "
             "F. Fedele, and N. Ares, arXiv:2404.04994.",
             # 7 and 8 from csd-materials/citations/QDSim.txt and
             # QDarts.txt, 2026-09-26; preprints are cited as the previous
             # JJAP paper cites them: authors, arXiv number, no title.
             "V. Gualtieri, C. Renshaw-Whitman, V. Hernandes, and "
             "E. Greplova, arXiv:2404.02712.",
             "J. A. Krzywda, W. Liu, E. van Nieuwenburg, and O. Krause, "
             "arXiv:2404.02064.",
             # 9 from csd-materials/citations/QArray+.txt, 2026-09-26
             "P. Vaidhyanathan, B. van Straaten, A. Petrillo, R. Marchand, "
             "E. De Nicolo, M. Veldhorst, B. Khailany, T. L. Patti, and "
             "N. Ares, arXiv:2609.02736.",
             # 19 new
             "S. S. Kalantre, J. P. Zwolak, S. Ragole, X. Wu, N. M. Zimmerman, "
             "M. D. Stewart, and J. M. Taylor, npj Quantum Inf. 5, 6 (2019).",
             # 20 (was 12)
             # 10 your SSDM 2026 talk, "Ray-Based Sampling for Efficient
             # Extraction of Transition Lines in Double Quantum Dot
             # Stability Diagrams" -- in the format you gave, 2026-09-27
             "E. Alizadeh Kashtiban, T. Fujita, and A. Oiwa, presented at "
             "SSDM 2026, Int. Conf. Solid State Devices and Materials, "
             "2026.",
             # 21 new: the U-Net
             "O. Ronneberger, P. Fischer, and T. Brox, Proc. MICCAI 2015, "
             "Lecture Notes in Computer Science 9351 (Springer, Cham, 2015) "
             "p. 234.",
             # 22 (was 13)
             # 11 this repository, cited at the end of the Introduction
             # (your request, 2026-09-27); JJAP style for a web resource
             # the link only, no authors (your call, 2026-09-27)
             "https://github.com/Ehsan-Japan/"
             "probabilistic-csd-reconstruction",
             # 23-25 new: Dice loss, boundary F1, tunnel-coupled DQD
             "F. Milletari, N. Navab, and S.-A. Ahmadi, Proc. 4th Int. Conf. 3D "
             "Vision (3DV), 2016, p. 565.",
             "D. R. Martin, C. C. Fowlkes, and J. Malik, IEEE Trans. Pattern "
             "Anal. Mach. Intell. 26, 530 (2004).",
             "L. DiCarlo, H. J. Lynch, A. C. Johnson, L. I. Childress, "
             "K. Crockett, C. M. Marcus, M. P. Hanson, and A. C. Gossard, "
             "Phys. Rev. Lett. 92, 226801 (2004).",
             # 26 (was 14). Author lists of 26, 28 and 29 are taken from each
             # PDF's title page, not the Google Scholar exports.
             # 12 TRACS
             "R. Marchand, L. Schorling, C. Carlsson, J. Schuff, "
             "B. van Straaten, T. L. Patti, F. Fedele, J. Ziegler, "
             "P. Girdhar, P. Vaidhyanathan, and N. Ares, arXiv:2508.15710.",
             # 27 new: RF readout review
             "F. Vigneau, F. Fedele, A. Chatterjee, D. Reilly, F. Kuemmeth, "
             "M. F. Gonzalez-Zalba, E. Laird, and N. Ares, Appl. Phys. Rev. 10, "
             "021305 (2023).",
             # 28 (was 15)
             "C. Carlsson, J. Saez-Mollejo, F. Fedele, S. Calcaterra, "
             "D. Chrastina, G. Isella, G. Katsaros, and N. Ares, "
             "arXiv:2506.10834.",
             # 29 (was 16)
             # 14 published per the .txt (4 Mar 2026); the volume and page
             # are in neither the .txt nor the PDF (the arXiv version,
             # 2509.19537), so they are NOT given here.
             "M.-A. Roux, J. Rivard, V. Yon, A. Morel, D. Leclerc, "
             "C. Rohrbacher, E. B. Ndiaye, F. F. Tafuri, B. Bono, "
             "S. Kubicek, R. Loo, Y. Shimura, J. Jussot, C. Godfrin, "
             "D. Wan, K. De Greve, M.-A. Tétrault, D. Drouin, "
             "C. Lupien, M. Pioro-Ladrière, and E. Dupont-Ferrier, "
             "IEEE Trans. Quantum Eng. (2026).",
             # 30 new: variability after thermal cycling
             "H. Moon, D. T. Lennon, J. Kirkpatrick, N. M. van Esbroeck, "
             "L. C. Camenzind, L. Yu, F. Vigneau, D. M. Zumbühl, "
             "G. A. D. Briggs, M. A. Osborne, D. Sejdinovic, E. A. Laird, and "
             "N. Ares, Nat. Commun. 11, 4161 (2020).",
         )}),

        ("3_1", "device_simulation_and_dataset",
         "Device simulation and dataset construction",
         "We describe the simulated double-quantum-dot devices from which "
         "every charge stability diagram in this work is drawn, and the "
         "train/test split applied to them. %d devices are generated from a "
         "constant-capacitance model with randomly drawn capacitances, each "
         "rendered as a %d x %d pixel charge-sensor map over a %.0f x %.0f mV "
         "gate-voltage window."
         % (f["n_total"], f["res"], f["res"], f["window_mv"], f["window_mv"]),
         [
             ("1. The constant-capacitance model", [
                 "Each simulated device consists of a double quantum "
                 "dot (DQD) capacitively coupled to a single charge "
                 "sensor. It is modelled with the QArray simulator within "
                 "the constant-capacitance approximation, and the "
                 "equivalent circuit is shown in Figure 1(a). In this "
                 "approximation the device is treated as a network of "
                 "fixed capacitors. For a given gate-voltage vector "
                 "V_{g}, the ground-state charge configuration "
                 "n = (n_{1}, n_{2}) is the integer occupation that "
                 "minimises the electrostatic energy of the network. "
                 # YOUR TEXT, word for word, 2026-09-29: holes stated (the
                 # simulator runs DotArray(charge_carrier="hole")), and the
                 # transition lines defined here once
                 # Figure 3(a) cited here at your request, 2026-09-30,
                 # although Figure 2 is first cited later (your choice)
                 "The devices are simulated with holes as charge carriers, "
                 "so the occupation increases towards lower gate voltages "
                 "(Figure 3(a)). "
                 "Within the constant-capacitance model this only reverses "
                 "the sign of the gate voltage at which a carrier is added, "
                 "and the geometry of the stability diagram is the same as "
                 "for electrons. The transition lines of the stability "
                 "diagram are the lines in gate-voltage space at which two "
                 "charge configurations have equal energy. Crossing such a "
                 "line changes the number of carriers on one or both dots, "
                 "and the occupation jumps from one integer configuration "
                 "to another.",

                 # "The parameters of the model can be counted off the
                 # circuit." deleted (your call, 2026-09-26).
                 "Each of the three conducting islands (two "
                 "dots and the charge sensor) couples to each of the three "
                 "gates, giving %s capacitive branches, and the islands "
                 "couple to one another through c_{d1d2}, c_{s1d1} and "
                 "c_{s1d2}, giving %s more, so %s branches are drawn in "
                 "total in Figure 1(a). The two remaining parameters, "
                 "c_{d1d1} and c_{d2d2}, are self-capacitances and are "
                 "not shown in the drawing, for simplicity, so a device "
                 "is specified by %d capacitances in all."
                 % (_spelled(9), _spelled(3), _spelled(_DRAWN),
                    _FREE[1]),

                 "The same capacitances are collected into the four "
                 "matrices of Figure 1(b). The dot\u2013dot matrix "
                 "C_{dd} \u2208 \u211d^{2\u00d72} carries the dot "
                 "self-capacitances c_{d1d1} and c_{d2d2} on its "
                 "diagonal and the interdot mutual capacitance c_{d1d2} "
                 "off it; the dot\u2013gate matrix C_{dg} \u2208 "
                 "\u211d^{2\u00d73} has rows indexed by dot and columns "
                 "by gate, with entries c_{digj} (i = 1, 2; "
                 "j = 1, 2, 3); the charge-sensor\u2013dot matrix C_{sd} "
                 "\u2208 \u211d^{1\u00d72} has entries (c_{s1d1}, "
                 "c_{s1d2}); and the charge-sensor\u2013gate matrix C_{sg} "
                 "\u2208 \u211d^{1\u00d73} has entries (c_{s1g1}, "
                 "c_{s1g2}, c_{s1g3}). Because mutual capacitance is "
                 "reciprocal, c_{d1d2} = c_{d2d1} and C_{dd} is "
                 "symmetric, so the %d entries displayed hold the same "
                 "%d independent parameters: %s in C_{dd}, %s in C_{dg}, "
                 "%s in C_{sd} and %s in C_{sg}."
                 % ((_FREE[0], _FREE[1])
                    + tuple(_spelled(n) for _name, n in _FREE[2])),

                 # YOUR TEXT, word for word, 2026-09-28: one sentence in
                 # place of the two paragraphs before it.  Checked
                 # 2026-09-27: all 550 device.json records have
                 # attempts = 1, accepted = True.
                 "Each device realisation is a single draw of the %d "
                 "capacitance entries, each sampled independently and "
                 "uniformly from the intervals listed in Table I, so that "
                 "every device corresponds to a point in a %d-dimensional "
                 # "No draw is rejected, so the acceptance rate of the
                 # device pool is 1.00 by construction." deleted (your
                 # call, 2026-09-27); still true of the pool.
                 "parameter space."
                 % (f["separation"]["dimensions"],
                    f["separation"]["dimensions"]),
                 # Figure numbers here are the part's own: Fig. 1(b) is
                 # printed as Fig. 2(b) in 3_Methods and the manuscript.
                 ("Table I. Sampling intervals for the capacitance matrices "
                  "of Fig. 1(b), in QArray's dimensionless capacitance "
                  "units. For each device, every capacitance entry is drawn "
                  "independently from a uniform distribution over its "
                  "interval. No sampled device is filtered or rejected. "
                  "Training and test devices are drawn from the same "
                  "intervals and are separated by device identity only.",
                  ["Matrix", "Entry", "Interval"],
                  TABLE_I_ROWS, [0.95, 0.95, 1.25], TABLE_I_RULES),
                 "The charge-sensor response is computed with the noise model set "
                 "to NoNoise and nothing is added afterwards, so the maps "
                 "are noise-free. The Coulomb peak width is %.3g and the "
                 "electron temperature %.0e in simulation units."
                 % (f["peak_width"], f["temperature"]),
                 # The random-offset sentence and the "dominated by a large,
                 # smooth background" sentence were deleted (your call,
                 # 2026-09-27).  The window IS still offset per device
                 # (f["offset_scale"]); the text no longer says so.
                 # The offset sentence is back, reworded (your call,
                 # 2026-09-27); the scale is read from the run config.
                 "Each device is rendered over a %.0f x %.0f mV window at "
                 "%d x %d pixels. The window position is shifted at random "
                 "for each device, by up to %.0f %% of its width along each "
                 "axis, so that the transition lines do not appear at fixed "
                 "positions in the frame."
                 % (f["window_mv"], f["window_mv"], f["res"], f["res"],
                    100 * f["offset_scale"]),
                 "Figure 2 shows the charge-sensor signal (a) and the "
                 "transition-line map (b) that the simulator returns for one "
                 "device.",
                 # YOUR TEXT, word for word, 2026-09-29.  Checked against
                 # dqd_simulator._charge_state_changes: a pixel is 1 where
                 # its occupation differs from the next pixel along either
                 # gate axis, so the chain lies on the lower-voltage side.
                 # one paragraph now, the definition of a transition line
                 # left to the first paragraph (your text, 2026-09-29)
                 "The window is divided into a regular grid of "
                 "%d × %d pixels. "
                 # added at your request, 2026-09-30: the pixel centres
                 # are np.linspace points, both window edges included
                 # "and include both edges of the window" omitted at your
                 # request, 2026-09-30 (still true: np.linspace)
                 "The pixel centres are equally spaced, "
                 "so the %.0f mV width is divided into "
                 "%d intervals and the pixel spacing is Δ = %.0f mV/%d ≈ "
                 "%.1f µV. "
                 "Each pixel is a small square in the "
                 "(V₁, V₂) plane, represented by the voltage pair at its "
                 # rewritten at your request, 2026-09-30.  Checked against
                 # dqd_simulator._run_double_dot: the occupations come from
                 # DotArray(Cdd, Cgd) alone, the sensor from
                 # ChargeSensedDotArray with all four matrices.
                 "centre. At this single point the simulator determines the "
                 "charge configuration: from the capacitances of the network "
                 "in Figure 1(a), collected in the dot–dot and dot–gate "
                 "matrices C_{dd} and C_{dg} of Figure 1(b), it finds the "
                 "integer occupation (n_{1}, n_{2}) that minimises the "
                 "electrostatic energy of the network at that pair of gate "
                 "voltages. The charge-sensor signal is evaluated at the same "
                 "point, using in addition the charge-sensor matrices C_{sd} and "
                 "C_{sg}. "
                 # "Because the charge configuration is determined only at
                 # the pixel centres, ... depends on its resolution (Figure
                 # 3)." removed 2026-09-30: your paragraph below says it
                 "All distances "
                 "in this work, including the tolerance τ (Sec. 3.4.2), are "
                 "measured in units of the pixel spacing. "
                 # YOUR TEXT, word for word, 2026-09-30, in place of the
                 # zero-width / labelled-pixel sentences and "The labelled
                 # chain always lies within one pixel spacing of the true
                 # line."
                 "In the continuous (V₁, V₂) plane, a charge transition "
                 "line is a boundary of zero width separating two regions "
                 "of distinct charge configuration. When the plane is "
                 "sampled on a discrete grid with spacing Δ, such a "
                 # ΔV₁, ΔV₂ -> Δ and localizes -> localises, 2026-09-30
                 "boundary generically passes between adjacent "
                 "grid points and cannot be represented exactly. We "
                 "therefore define transition pixels as follows: a pixel is "
                 "labelled as a transition pixel if its charge "
                 "configuration differs from that of its nearest neighbour "
                 "in the positive V₁ or V₂ direction. Under this "
                 "definition, each labelled pixel is the last pixel "
                 "preceding a change in charge configuration, and each "
                 "transition line is represented by a connected chain of "
                 "pixels of single-pixel width, located on the "
                 "lower-voltage side of the true boundary. The discrete "
                 "representation consequently localises the boundary only "
                 "to within one grid spacing, and the resulting set of "
                 "labelled pixels depends on the resolution and alignment "
                 "of the sampling grid as well as on the physical "
                 "properties of the device. "
                 "Figure 3 shows the true "
                 "transition lines of one simulated device in the "
                 "continuous voltage plane, together with the labels "
                 "obtained from them on the grid used in this work and on "
                 "a grid twice as coarse."
                 % (f["res"], f["res"], f["window_mv"], f["res"] - 1,
                    f["window_mv"], f["res"] - 1,
                    1e3 * f["window_mv"] / (f["res"] - 1)),
             ]),
             # 3.1.2 "Ground truth" removed, and this subsection replaced
             # by your text, 2026-09-26.  Every count is read from the run;
             # "Figure 4" is the LOCAL number (3 until the pixel figure
             # went in before it, 2026-09-30).  The
             # near-duplicate check (separation) is no longer in the text.
             ("2. Train/test split", [
                 "A total of %d devices are generated by sampling the "
                 "capacitance intervals of Table I. The dataset is split by "
                 "device: %d devices are used for training and %d for "
                 "testing. Before training, %d %% of the training devices "
                 "(%d of %d) are set aside as a validation set, and the "
                 "network weights are fitted on the remaining %d devices. "
                 "The validation set is used to select the best-performing "
                 "epoch and the binarisation threshold (Section 3.4). The "
                 "test devices are not used for any training or "
                 "model-selection decision. Figure 4 illustrates this "
                 "partitioning."
                 % (f["n_total"], f["n_train"], f["n_test"], f["val_pct"],
                    f["n_val"], f["n_train"], f["n_fit"]),
             ]),
         ],
         [("dqd_model_full_bigb_v2.png",
           "The constant-capacitance model of the simulated device. (a) The "
           "capacitor network: two quantum dots QD1 and QD2, a charge-sensor dot "
           "QDs, and three gates. (b) The four capacitance matrices "
           "that define a device. A device is one uniform draw of the %d "
           "free parameters of Table I."
           % _FREE[1]),
          # 2026-09-26: the charge-sensor figure, the ray-fan figure and
          # the input channels of the network figure merged into one.
          # Cited from 3.2 by name, {fig:inputs}.
          ("fig_inputs.png",
           "One held-out device for the best-performing configuration "
           "(%d × %d). (a) The charge-sensor signal over the gate-voltage "
           "window. (b) The ground-truth transition lines computed by the "
           "simulator: the output the U-Net is trained to reproduce. (c) "
           "The normalised charge-sensor signal at the pixels sampled by the "
           "rays, grey elsewhere: the first input channel. (d) The binary "
           "sampling mask: the second input channel."
           % (f["n_rays"], f["n_points"])),
          # The schematic pixel_labelling.png (make_pixel_labelling_ai.py)
          # was here; deleted at your request, 2026-09-30.
          # make_pixel_quantisation_ai.py, 2026-09-30.  sample_1 of the
          # device pool; the script asserts that (b) is its stored ground
          # truth.  The spacing is (window)/(res - 1), as np.linspace.
          ("pixel_quantisation.png",
           # shortened at your request, 2026-09-30; the panel titles were
           # then taken out of the figure and said here instead
           "Transition lines of one simulated device. (a) The continuous "
           "voltage plane with the true transition lines and the charge "
           "configuration (n₁, n₂) of each region; the dashed square is "
           "enlarged in (b) and (c). (b) The grid of this work, spacing "
           "Δ = %.1f µV. (c) A grid twice as coarse, spacing 2Δ."
           % (1e3 * f["window_mv"] / (f["res"] - 1))),
          # Shortened at your request, 2026-09-29: the text of 3.1.2
          # already explains the split.  The counts are read from the run.
          ("fig_data_split.png",
           # "training" is the 500, as the figure labels it; 75 of them
           # are the validation devices
           "Partitioning of the %d simulated devices into %d training and "
           "%d test devices, with %d of the training devices reserved for "
           "validation."
           % (f["n_total"], f["n_train"], f["n_test"], f["n_val"]))]),

        ("3_2", "ray_based_measurement",
         "Ray-based measurement and network input",
         # YOUR TEXT, word for word, 2026-09-25, less the reference-budget
         # sentence (8 x 50, 3.88 %) you cut the same day.
         "In this section, we describe the sparse acquisition scheme and "
         "the tensor representation through which we present the acquired "
         "data to the network. We launch a set of rays from a single corner "
         "of the gate-voltage window and sample each ray at a fixed number "
         "of equally spaced points. We encode the resulting measurement as "
         "two image-shaped input channels: the first contains the "
         "charge-sensor signal at the pixels the rays visit, and the "
         "second is a binary occupancy mask that marks which pixels we "
         "measured.",
         [
             ("1. The ray fan", [
                 "A measurement budget is a pair "
                 "(n_{rays}, n_{points}). The rays emanate from the "
                 "maximum-voltage corner (V_{1}^{max}, V_{2}^{max}) of the "
                 "window, the k-th ray propagating along the inward unit "
                 "vector d_{k} = (\u2212cos \u03b8_{k}, "
                 "\u2212sin \u03b8_{k}) with",

                 EQ + "\u03b8_{k} = 90\u00b0 k / (n_{rays} + 1),"
                 "\u2003k = 1, \u2026, n_{rays}.",

                 "The angles are equispaced on the open interval "
                 "(0\u00b0, 90\u00b0); the endpoints are excluded because "
                 "an axis-aligned ray runs along the window boundary and "
                 "intersects few transition lines. For n_{rays} = %d this "
                 "gives \u03b8 \u2208 {%s} (Figure {fig:inputs}(c))."
                 % (f["n_rays"], _angle_set(_fan_angles(f["n_rays"]))),

                 # YOUR TEXT, word for word, 2026-09-25.
                 "Each ray corresponds to a single one-dimensional voltage "
                 "sweep. We chose this ray-based pattern primarily for its "
                 "simplicity. By distributing the rays across the voltage "
                 "window, we aimed to ensure that each ray intersects at "
                 "least one charge transition line, so that every sweep "
                 "carries information about the underlying stability "
                 "diagram. We restrict all ray angles to the open interval "
                 "(0\u00b0, 90\u00b0) so that the rays spread across the "
                 "interior of the window rather than running along its "
                 "edges. For simplicity, we space the rays at equal angles "
                 "and use the same number of points on every ray. We also "
                 "use the same set of rays for every device, rather than "
                 "adapting it to what we observe during the measurement. "
                 "We did not optimise "
                 "any of these choices. To examine how the measurement "
                 "budget affects performance, we evaluate %s rays, each "
                 "sampled at %s points per ray, giving %d configurations in "
                 "total. For each configuration, we use %d samples, of "
                 "which %d serve for training and %d for testing."
                 % (", ".join(str(n) for n in f["rays_swept"][:-1])
                    + ", and %d" % f["rays_swept"][-1],
                    ", ".join(str(n) for n in f["points_swept"][:-1])
                    + ", or %d" % f["points_swept"][-1],
                    len(f["rays_swept"]) * len(f["points_swept"]),
                    f["n_total"], f["n_train"], f["n_test"]),

                 # "Figure 1 shows the fan over one device, and the traces
                 # the rays return." -- that figure is gone (2026-09-26).

                 # YOUR TEXT, word for word, 2026-09-26.  Every number is
                 # read from the run: 388 distinct pixels was checked on all
                 # 50 test devices (the ray geometry is the same for every
                 # device, so the count is too).
                 "The measurement budget is defined as the total number of "
                 "single-point measurements. Each measured voltage pair is "
                 "assigned to the pixel whose centre lies closest to it. "
                 "Because all rays share a common origin, the first point "
                 "of every ray falls in the same pixel, and adjacent rays "
                 "may also be assigned to a common pixel in the following "
                 "step. The number of distinct pixels sampled is therefore "
                 "slightly smaller than the number of measurements. For the "
                 "configuration of %s rays of %d points, the %d measurements "
                 "would cover %d/(%d × %d) = %.2f %% of the grid if "
                 "every pixel were distinct, but they occupy %d distinct "
                 "pixels, or %.2f %%. Since a repeated measurement of the "
                 "same pixel provides no additional information to the "
                 "network, coverage is quantified throughout this work as "
                 "the fraction of distinct pixels sampled."
                 % (_spelled(f["n_rays"]), f["n_points"],
                    f["n_rays"] * f["n_points"],
                    f["n_rays"] * f["n_points"], f["res"], f["res"],
                    100.0 * f["n_rays"] * f["n_points"] / f["res"] ** 2,
                    int(round(f["coverage"] / 100.0 * f["res"] ** 2)),
                    f["coverage"]),
                 # The local-maxima paragraph was deleted (your call,
                 # 2026-09-26).  For the record, on the 50 test devices at
                 # 8 x 50, 66.6 % of the local maxima lie within 1 px of a
                 # true line (chance 13.5 %) and 85.9 % of the ray-line
                 # crossings have a maximum within one sample.
             ]),
             ("2. Input representation", [
                 # YOUR TEXT, word for word, 2026-09-28, replacing the whole
                 # subsection.  Checked against the code that day: the
                 # signal is min-max normalised over the full simulated
                 # diagram (ray_peaks.load_grid); every budget trained at
                 # train_seed 0 (torch.manual_seed: initial weights and
                 # batch order); the four retrained budgets and their seeds
                 # are in retrain_collapsed.json, described in 3.3.3; 6 x 50
                 # scores 0.661 and 5 x 50 0.736 at tau = 1.  The counts and
                 # the grid size are still read from the run.  "Fig." in the
                 # text is printed "Figure" (your rule, 2026-09-28).
                 # one paragraph, no colon and no list; this opening is your
                 # rewrite of 2026-09-28 (the closing full stop added)
                 "The rays measure only a few percent of the pixels, but a "
                 "convolutional network requires an input image with a value"
                 " at every pixel. The measured values are therefore placed "
                 "on the full %d × %d grid, and the network receives two "
                 "input channels [Figure {fig:inputs}(c) and (d)]. In the "
                 "first, the signal channel, each pixel crossed by a ray "
                 "holds the charge-sensor signal measured there, and every other "
                 "pixel is set to zero. The signal is normalised to [0, 1] "
                 "using the minimum and maximum of the full simulated "
                 "diagram. In the second, the mask channel, each pixel "
                 "crossed by a ray is set to one and every other pixel to "
                 "zero." % (f["res"], f["res"]),
                 "The mask is needed because the signal channel alone is "
                 "ambiguous. After normalisation, a measured pixel with low "
                 "charge-sensor signal also has a value at or near zero. A zero in "
                 "the signal channel could therefore mean either “measured,"
                 " and the signal here is low” or “not measured”. "
                 "These two cases carry different information. The first is "
                 "real evidence about the diagram at that pixel. The second "
                 "is no evidence at all, and the network must infer that "
                 "region from the surrounding rays. The mask tells the "
                 "network which case applies at each pixel.",
                 "Because both channels always cover the full grid, the size "
                 "of the input does not depend on the measurement budget. "
                 "Changing n_rays or n_points changes only how many pixels "
                 "are filled, not the size of the channels. "
                 # "budget" -> "measurement configuration", your request,
                 # 2026-09-30
                 "Every measurement configuration "
                 "therefore uses the same architecture, with the same number "
                 "of trainable parameters.",
                 "A separate network is trained for each of the %s measurement budgets. "
                 "The comparison between measurement budgets is meant to work like a "
                 "controlled experiment in which only the measurement "
                 "changes. Every network is therefore trained with the same "
                 "loss function, optimiser, number of epochs and training "
                 "devices, and its threshold is selected in the same way. If "
                 "the networks differed in size, a higher score at a larger "
                 "measurement budget could simply come from a larger model. With "
                 "identical networks, a difference in score reflects how "
                 "much information the rays provide."
                 % _word(f["n_budgets"]),
                 "One source of difference nevertheless remains. The initial "
                 "weights of the network, and the order in which the "
                 "training devices are presented, are drawn from a "
                 "pseudo-random number generator. In this study the "
                 "generator was started from the same seed for every "
                 "measurement budget, so all networks began from identical weights and "
                 "saw the training devices in the same order. The choice of "
                 "seed is, however, arbitrary. With a different seed, "
                 "training follows a different path and ends at a somewhat "
                 "different score. We refer to this spread as run-to-run "
                 "variation. %s measurement budgets show that it matters: their first "
                 "training runs did not converge, and they converged only "
                 "after being retrained with a different seed (Sec. 3.3.3). "
                 "Each measurement budget was trained to convergence only once, so the "
                 "size of this variation was not measured. Large "
                 "differences between measurement budgets can therefore be attributed "
                 "to the measurement. Small differences between neighbouring "
                 "measurement budgets, such as 6 × 50 scoring below 5 × 50, should not "
                 "be over-interpreted."
                 % _word(len(_run.SEEDS)).capitalize(),
             ]),
         ],
         # The ray-fan figure is gone (2026-09-26): the fan is shown by
         # the merged input figure of 3.1.
         []),

        ("3_3", "network_and_training",
         "Network architecture and training",
         # YOUR TEXT, word for word, 2026-09-23.  The widths and the
         # line fraction are still read from the run, not typed in.
         "The mapping from the two measured channels to a per-pixel "
         "transition-line probability is realized by a fully "
         "convolutional encoder\u2013decoder network of the U-Net type. "
         "The network has depth 3, with feature widths of %d, %d, and %d "
         "in the encoder and %d in the bottleneck. It is trained with a "
         "composite objective that combines a class-weighted binary "
         "cross-entropy and a soft Dice term. The Dice term is included "
         "because of the strong class imbalance: transition-line pixels "
         "make up only %.1f%% of the stability diagram."
         % (tuple(f["widths"]) + (f["line_frac"],)),
         [
             # YOUR TEXT, 2026-09-26, checked against grid_model.py: two
             # 3x3 conv + BatchNorm + GELU per stage, nearest-neighbour
             # upsampling to the skip size, 1x1 head, sigmoid at inference.
             # The resolution, widths, pooling chain and parameter count
             # are read from the code and the run.  "Figure 1" is the
             # LOCAL number, which renumber() prints as Figure 7.
             ("1. Architecture", [
                 "Figure 1 shows the architecture of the fully "
                 "convolutional U-Net. The input is a two-channel "
                 "%d × %d tensor (charge-sensor signal and sampling mask). "
                 "Each encoder stage (%d, %d and %d channels) consists of "
                 "two 3 × 3 convolutions, each followed by batch "
                 "normalisation and a GELU activation, and is followed by "
                 # " (100 -> 50 -> 25 -> 12)" deleted (your call,
                 # 2026-09-27)
                 "2 × 2 max pooling. The bottleneck has %d channels. In the "
                 "decoder, feature maps are upsampled by nearest-neighbour "
                 "interpolation to the size of the corresponding encoder "
                 "output and concatenated with it through skip "
                 "connections. A final 1 × 1 convolution with a "
                 "sigmoid produces a per-pixel transition-line probability "
                 # The parameter count (1 949 409) is no longer stated
                 # (your call, 2026-09-26); f["n_params"] still holds it.
                 "map of size %d × %d. Extracting "
                 "lines from this map is a separate decision step, "
                 "described in Sec. 3.4."
                 % ((f["res"], f["res"]) + tuple(f["widths"][:3])
                    + (f["widths"][3], f["res"], f["res"])),
             ]),
             # YOUR TEXT, word for word, 2026-09-26.  Checked against
             # grid_train.py: BCEWithLogitsLoss with pos_weight
             # min((1 - p)/p, MAX_POS_WEIGHT = 8), plus a soft Dice term.
             ("2. Loss function", [
                 "Transition-line pixels make up only a small fraction of "
                 "each diagram. A network that predicts no transition "
                 "anywhere would therefore still be correct for most pixels "
                 "while finding no lines at all. Two terms in the training "
                 "loss prevent this. First, the per-pixel binary "
                 "cross-entropy is weighted so that a missed "
                 "transition-line pixel costs more than an error on a pixel "
                 "without a line. Second, a Dice term [23] compares the "
                 "predicted and true line maps as a whole and penalises "
                 "predictions that place lines where there are none.",
             ]),
             ("3. Model selection and evaluation metric", [
                 # YOUR TEXT, word for word, 2026-09-26 (a full stop added
                 # at the end).  Checked against grid_train.train: after
                 # each epoch the validation F1@1 is computed and the best
                 # epoch's weights are kept.  The optimiser settings that
                 # were here (Adam, lr, batch, epochs, seed) are no longer
                 # in the text; f["lr"] etc. still hold them.
                 "After each training epoch, the network was evaluated on "
                 "the validation set, and the weights from the epoch with "
                 "the highest validation score were retained as the final "
                 "model rather than those from the last epoch. Performance "
                 "is measured with the F1 score, which combines precision "
                 "(the fraction of predicted line pixels that are correct) "
                 "and recall (the fraction of true line pixels that are "
                 "recovered) into a single number between 0 and 1. Because "
                 "the transition lines are only one pixel wide, a predicted "
                 "pixel is counted as correct if it lies within a small "
                 "distance, the tolerance τ, of a true line pixel. Model "
                 "selection uses a tolerance of one pixel (F1@1). The "
                 "motivation for this tolerance, and its effect on the "
                 "reported scores, are discussed in Sec. 3.4.",
             ] + ([_retrain_sentence()] if _retrain_sentence() else [])),
         ],
         # Back to the input | U-Net | output figure (your call,
         # 2026-09-26), with your caption of the same day.
         [("fig_model_flow.png",
           "Network input and output for one held-out device (%d rays "
           "× %d points). Left: normalised charge-sensor signal along the "
           "rays and the binary sampling mask. Centre: U-Net. Right: "
           "predicted %d × %d transition-line probability map."
           % (f["n_rays"], f["n_points"], f["res"], f["res"]))]),

        ("3_4", "threshold_and_tolerance",
         "From probability map to transition lines",
         "The network outputs a continuous probability map, while transition "
         "lines are binary, so two decisions stand between the output and a "
         "score. Where to cut the map is the threshold, chosen on the "
         "validation devices and stored in the checkpoint; for the "
         "best-performing configuration (%d × %d) it is %g. How strictly to compare the result with the truth "
         "is the tolerance tau, a distance in pixels. We report F1 at "
         "tau = 0 to 3 and quote tau = 1 as the headline, always beside the "
         "area that tolerance covers."
         % (f["n_rays"], f["n_points"], f["threshold"]),
         [
             # YOUR TEXT, word for word, 2026-09-26: the opening of 3.4,
             # before 3.4.1, unheaded.  The definitions match
             # grid_metrics.tolerant_f1 at tau = 0.
             (None, [
                 "The network assigns each pixel a probability that a "
                 "charge-transition line passes through it. To obtain the "
                 "transition lines themselves, this probability map is "
                 "binarised by applying a threshold. Pixels whose "
                 "probability exceeds the threshold are classified as lying "
                 "on a transition line, and all others as lying off it. The "
                 "reconstruction of the transition lines thereby becomes a "
                 "binary classification problem at the pixel level, and its "
                 "performance can be quantified with the standard metrics of "
                 "binary classification. A pixel predicted to lie on a "
                 "transition line that does lie on one is a true positive "
                 "(TP). A pixel predicted to lie on a transition line where "
                 "none exists is a false positive (FP). A pixel on a true "
                 "transition line that the prediction misses is a false "
                 "negative (FN). From these we define the precision "
                 "P = TP/(TP + FP), the fraction of predicted transition "
                 "pixels that are correct; the recall R = TP/(TP + FN), the "
                 "fraction of true transition pixels that are recovered; and "
                 "their harmonic mean, the F1 score, F1 = 2PR/(P + R). These "
                 "quantities are used throughout the remainder of this work.",
                 # your wording, 2026-09-27, replacing the trade-off
                 # sentence of the day before
                 "The value of the threshold determines how the errors are "
                 "distributed between the two kinds. If the threshold is set "
                 "low, even pixels with a small predicted probability are "
                 "classified as lying on a transition line. Few true "
                 "transitions are missed, so the recall is high, but many "
                 "pixels are wrongly marked as transitions, so the precision "
                 "is low. If the threshold is set high, only pixels "
                 "predicted with high confidence are retained. The marked "
                 "pixels are then mostly correct, so the precision is high, "
                 "but faint transitions are discarded, so the recall is low. "
                 "A suitable threshold must therefore be chosen to balance "
                 "these two effects.",
             ]),
             # YOUR TEXT, word for word, 2026-09-26, except the figure
             # references: Figs. 7-10 were merged into one figure the same
             # day, so "(Fig. 7)" is "(Fig. 7(b))" and "Figures 8-10" is
             # "Figures 7(a), (c) and (d)" (panels (a)/(b) swapped later the
             # same day).  Written here with the LOCAL number 1.
             ("1. Binarising the output: the threshold", [
                 "To obtain transition lines from the probability map, each "
                 "pixel must be classified as lying on a transition line or "
                 "not. This is done by applying a threshold to the "
                 # "The threshold balances missed transitions against
                 # spurious detections: ..." deleted (your call, 2026-09-27);
                 # the 3.4 opening now explains the trade-off.
                 "predicted probabilities. It is therefore a free parameter of the "
                 "method and must be fixed without reference to the test "
                 "data. We select it on a validation set of %d devices held "
                 "back from the training set. A range of candidate "
                 "thresholds is evaluated on these devices, and the value "
                 "giving the highest score is stored with the trained model. "
                 "This procedure does not use the held-out test devices at "
                 "any stage. The validation devices come from the training "
                 "distribution and are disjoint from the test set, so the "
                 "threshold is determined in the same way as any other "
                 "hyperparameter of the network. The procedure also matches "
                 "the intended application. On a real device the true "
                 "transition lines are unknown, so the threshold must be "
                 "fixed before the measurement using simulated data alone, "
                 "which is exactly what is done here. The validation curve "
                 "(Figure 1(b)) is nearly flat over a broad range of "
                 "thresholds and decreases only at very high values, so the "
                 "result is insensitive to the exact choice. Figures "
                 "1(a), (c) and (d) show a representative probability map "
                 "binarised at the selected threshold and at a deliberately "
                 "excessive one." % f["n_val"],
             ]),
             ("2. Scoring the output: the tolerance", [
                 "Transition lines are one pixel wide. Comparing them with "
                 "the ground truth pixel by pixel is unduly strict: a "
                 "perfectly recovered line displaced by a single pixel shares "
                 "no pixel with the truth and scores zero, although the "
                 "reconstruction is physically correct. Figure 2 shows this "
                 "case, constructed and scored with the same metric "
                 "code used throughout: strict F1 = 0.00, and F1 at "
                 "tau = 1 equal to 1.00.",
                 "We therefore report F1 at a tolerance tau, as is standard "
                 "in boundary-detection benchmarks [24]: a predicted "
                 "pixel counts as correct if a true line pixel lies within "
                 "tau pixels of it, and a true pixel counts as found if a "
                 "predicted one lies within tau. Distances are Euclidean. "
                 "Figure 3 shows what tau admits around a single pixel: "
                 "tau = 0 is the same pixel only, tau = 1 adds the four "
                 "side neighbours, and a diagonal neighbour is 1.41 pixels "
                 "away so it counts only from tau = 2.",
                 # 2026-09-26: what Fig. 10 shows is explained HERE; its
                 # caption was cut to one line.
                 "Figure 4 shows the effect of the tolerance on a single "
                 "prediction. It is a magnified region of one held-out "
                 "device, with the ground-truth lines in black, the network "
                 "output binarised at the selected threshold in red, and the "
                 "pixels within tau of the ground truth in grey. The "
                 "prediction and the ground truth are identical in every "
                 "panel; only the tolerance band grows from tau = 0 to 3, "
                 "and the precision, recall and F1 given above each panel "
                 "rise with it. The panel for tau = 1, the tolerance used "
                 "for model selection and for the headline results, is "
                 "highlighted.",
                 # 3.4.3 "Reporting" deleted (your call, 2026-09-27).  Its
                 # sentence citing the tolerance-curve figure moved here so
                 # that figure stays cited; the pixel-accuracy remark went
                 # with it (Results and Conclusions still make it).
                 "Figure 5 shows F1 against tau together with the "
                 "precision and recall behind it.",
             ]),
         ],
         # Figs. 7-10 (validation curve, probability map, two cuts)
         # merged into one figure, 2026-09-26.
         [("fig_threshold_panels.png",
           # (a) and (b) swapped 2026-09-26: the continuous map first.
           "Choosing and applying the threshold. (a) The U-Net output for "
           "one held-out device: the probability per pixel that a "
           "transition line passes through it. (b) Mean F1 at tau = 1 over "
           "the %d validation devices for every candidate threshold, with "
           "the stored value ringed. (c) The same map as (a) binarised at the "
           "selected threshold, P > %g, and (d) at a deliberately excessive "
           "one, P > %g, at which lines break up and are lost. In (c) and "
           "(d) the prediction is red and the ground truth black."
           % (f["n_val"], f["threshold"], f["ladder_high"])),
          ("tau_example.png",
           # Your wording, 2026-09-26.
           "Illustration of the tolerance τ used in the evaluation "
           "metric. Left to right: the ground-truth line (black), a "
           "prediction displaced by one pixel (red), the two superimposed, "
           # the F1 sentence is dropped (2026-09-26): the 3.4.2 text
           # already gives strict F1 = 0.00 and F1 at tau = 1 = 1.00
           "and the same pair with the τ = 1 tolerance band (grey)."),
          ("tau_neighbourhood.png",
           # Your wording, 2026-09-26.
           "Tolerance region around a single predicted pixel (red) for "
           "τ = 0, 1, 2 and 3 (left to right). Grey pixels lie within "
           "a Euclidean distance τ of the predicted pixel. Diagonal "
           "neighbours (distance √2 ≈ 1.41 px) are therefore "
           "included only for τ ≥ 2."),
          # The three full-map panels at tau = 0, 1 and 3 (p2l_4/5/6)
          # were dropped 2026-09-26: the zoom below shows the same thing.
          ("p2l_9_tau_zoom.png",
           # Your wording, 2026-09-26; the threshold is read from the run.
           # Cut to the essentials, 2026-09-26; the full description is
           # in the 3.4.2 text that cites this figure.
           "One prediction scored at τ = 0–3. Black: ground "
           "truth; red: output at P > %g; grey: pixels within τ of "
           "the ground truth. τ = 1 is highlighted." % f["threshold"]),
          ("p2l_7_tolerance_curve.png",
           # Your wording, 2026-09-26.  Budget and device count are read
           # from the run; the scores are per-device means
           # (grid_metrics.evaluate), hence "averaged".
           "Precision, recall and F1 score as functions of the tolerance "
           "τ for the %d × %d measurement budget, averaged over "
           "the %d test devices. The largest gain occurs between τ = 0 "
           "and τ = 1. For larger τ the scores increase only "
           "slightly." % (f["n_rays"], f["n_points"], f["n_test"]))]),

        ("4", "results_and_Discussion",
         "Results and Discussion",
         "We report the recovery of charge-transition lines as a function of "
         "the measurement budget, on %d held-out devices. The "
         "best-performing configuration (%d × %d) reaches F1 at tau = 1 of "
         "%.3f from "
         "%.2f %% of the gate-voltage plane, at a threshold of %g chosen on "
         "the validation dataset."
         % (f["n_test"], f["n_rays"], f["n_points"], f["f1"][1],
            f["coverage"], f["threshold"]),
         [
             # YOUR TEXT, word for word, 2026-09-26: section 4 as one
             # continuous text, no subsections.  "Figure 11" and "Fig. 12"
             # are written with the LOCAL numbers 1 and 2.  Every number is
             # read from the run; checked the same day: monotonic in rays
             # per point count (converged runs), the tau curve, coverage
             # 3.88 -> 17.85 %, pixel accuracy 0.851-0.908 at tau = 0 and
             # > 0.92 from tau = 1 (converged budgets), trivial 0.929.
             # One typo fixed: "budgetFigure 11" -> "budget. Figure 11".
             (None, [
                 "The recovery of charge-transition lines was evaluated on "
                 "the %d held-out devices as a function of the measurement "
                 "budget. Figure 1 shows F1@1 as a function of the fraction "
                 # 2026-09-27: the monotonic and equal-coverage sentences
                 # are now computed -- with the retrained budgets, 6 x 50
                 # scores below 5 x 50 and 2 of the 5 comparable pairs go
                 # the other way, which the typed text no longer matched.
                 "of the voltage plane sampled. %s The best-performing "
                 "configuration (%d × %d) reaches F1@1 = %.3f (precision %.3f, recall "
                 "%.3f) while sampling %.2f %% of the plane, at a "
                 "binarisation threshold of %g selected on the validation "
                 "dataset. %s"
                 % (f["n_test"], _monotonic_sentence(), f["n_rays"],
                    f["n_points"], f["f1"][1], f["prec1"], f["rec1"],
                    f["coverage"], f["threshold"], _equal_cov_short())
                 # 2026-09-27, your request: the final scores of every
                 # budget as a table, read from comparison.csv
                 + " The scores of all %s measurement budgets are listed in Table I."
                 % _word(f["n_budgets"]),
                 ("Table I. Scores of every measurement budget (rays × points "
                  "per ray) on the %d "
                  "held-out devices: fraction of the voltage plane sampled, "
                  "binarisation threshold selected on the validation dataset, "
                  "precision, recall and F1 at tau = 1, and strict F1 at "
                  "tau = 0. Measurement budgets "
                  "marked * were retrained from a different random "
                  "initialisation (Sec. 3.3). The best-performing "
                  "configuration (%d × %d) has the highest F1@1."
                  % (f["n_test"], f["n_rays"], f["n_points"]),
                  ["Measurement budget", "Coverage (%)", "Threshold", "Precision",
                   "Recall", "F1@0", "F1@1"],
                  TABLE_R_ROWS,
                  [1.3, 0.95, 0.9, 0.85, 0.7, 0.65, 0.65],
                  TABLE_R_RULES),
                 "The dependence of the scores on the tolerance τ is shown "
                 # YOUR TEXT, word for word, 2026-09-28
                 "in Figure 2 for three measurement budgets. For %s, F1 "
                 "rises from %.3f at τ = 0 to %.3f at τ = 1, and "
                 "then increases more slowly, reaching %.3f at τ = 2 and "
                 "%.3f at "
                 # "The dominant remaining error ... 18.91 % at tau = 1."
                 # deleted at your request, 2026-09-27
                 "τ = 3. Since the "
                 "improvement between τ = 0 and τ = 1 is obtained at "
                 "the expense of a several-fold increase in this area, "
                 "τ = 1, rather than a larger value, is adopted as the "
                 "headline tolerance. Pixel accuracy is included in "
                 "Figure 2(a) solely to illustrate why it is unsuitable as a "
                 # simplified at your request, 2026-09-27: no accuracy
                 # ranges any more, only why accuracy is not used
                 # no number (your call, 2026-09-27)
                 "metric. Transition lines make up only a few percent of "
                 "the diagram, so a prediction containing no lines at all "
                 "would already reach a high pixel accuracy. Accuracy is "
                 "therefore not used; the results are assessed with "
                 "precision, recall and their harmonic mean, the F1 score."
                 % (BEST_CFG, f["f1"][0], f["f1"][1], f["f1"][2],
                    f["f1"][3]),
                 # Rewritten at your request, 2026-09-30, around the
                 # pipeline figure (local Figure 3).  Every statement about
                 # Ref. 12 checked against its PDF: ~9000 measured diagrams,
                 # 128 x 128, MSE to the original image, Canny + Frangi run
                 # on "both ground truth and predicted CSDs", ~4 % grid,
                 # line cuts 23-44 %.  The separate "central step"
                 # paragraph (2026-09-27) is folded into the second one
                 # here, so it is not said twice.
                 # YOUR TEXT, 2026-09-30, with: the "closest work" opening
                 # you asked for folded into its first sentence; "sensor"
                 # -> "charge-sensor" (your rule); "summarized" ->
                 # "summarised"; Ref. xx -> 12; the figure cited; and the
                 # fairness sentence at the end.  Checked against Ref. 12:
                 # its U-Net takes "the sparse measurement y, the binary
                 # mask M" as channels, beside the noisy image and time.
                 # YOUR TEXT, word for word, 2026-09-30, in place of the
                 # paragraph and the two displayed pipelines; "Fig. 13" is
                 # written as the part's own "Figure 3" (printed as Figure
                 # 13), "sensor image" -> "charge-sensor image" (your rule),
                 # and the final full stop added.
                 "The work most closely related to the present study is that "
                 "of Hernandes et al.,^{12)} which differs from the present "
                 "method primarily in the quantity that is inferred. They "
                 "employ a conditional diffusion model with a similar input "
                 "encoding, in which the sparse charge-sensor signal and a "
                 "binary sampling mask are supplied as separate channels. "
                 "Their network, however, is trained to reconstruct the "
                 "complete charge-sensor image, and the charge transition "
                 "lines are obtained only in a subsequent step, by applying "
                 "edge and ridge detectors to the reconstructed image. The "
                 "two approaches are compared in Figure 3. In the approach "
                 "of Ref. 12, the sparse measurement is first used to "
                 "reconstruct the full charge-sensor image, from which the "
                 "transition lines are then extracted by a separate "
                 "detection stage. In the present work, the network infers "
                 "the transition lines directly from the sparse measurement, "
                 "producing for each pixel the probability that it lies on a "
                 "charge transition line, without any intermediate image "
                 "reconstruction or detection step.",
                 "In the image-reconstruction approach, most of the "
                 "network's output describes Coulomb-blockaded regions, "
                 "where the signal carries no information about the charge "
                 "configuration, and the transition lines depend on a "
                 "subsequent detection step whose threshold and scale "
                 "parameters introduce additional sources of error. In the "
                 "present approach, the transition lines are the direct "
                 "output of the network, so no intermediate detection stage "
                 "is required, and the result is a probabilistic estimate of "
                 "the quantity that determines the charge configuration. "
                 "Moreover, because the training labels are defined by the "
                 "charge configuration itself rather than extracted from a "
                 "measured image, the accuracy of the inferred lines can be "
                 "evaluated against an exact reference. Image reconstruction "
                 "retains the full charge-sensor response, which may be "
                 "useful for other characterisation tasks; for the purpose "
                 "of determining the charge configuration, however, the "
                 "direct inference of transition lines is sufficient.",

                 "The numerical results of the two studies are nevertheless "
                 "not directly comparable. Reference 12 reports that the "
                 "transition lines were preserved with about 4 %% of the "
                 "data measured on a uniform grid, and evaluates line cuts "
                 "at 23–44 %%; the present network reaches F1@1 = %.2f at "
                 "%s %% coverage. The data of Ref. 12 are measured and "
                 "noisy, whereas ours are simulated and noise-free; its "
                 "reference lines are the output of detectors, whereas ours "
                 "are exact; and the two studies score the lines with "
                 "different metrics. The present results therefore "
                 "establish the formulation on simulated devices, and a "
                 "comparison on the same measured diagrams, against the "
                 "same reference, remains to be made."
                 % (f["f1"][1], _rounded(f["coverage"])),
                 "The principal open question is whether a model trained on "
                 "noise-free constant-capacitance diagrams transfers to "
                 "fabricated devices. Measured diagrams contain charge-sensor "
                 "noise, charge switching, drift, and curved or broadened "
                 "lines arising from finite tunnel coupling [25] and "
                 "non-constant "
                 "charging energies, none of which are present in the "
                 "training data. Several strategies may reduce this gap, "
                 "including the introduction of realistic noise models "
                 "during training, the use of simulators that incorporate "
                 "tunnel coupling, such as QArray+,^{18)} and fine-tuning on a "
                 "small number of measured diagrams. The per-device min–max "
                 "normalisation of the input may also require revision, "
                 "since a single outlier caused by charge switching would "
                 "compress the dynamic range of the remaining signal.",
                 # Added 2026-09-27 at your request, verbatim (8 paragraphs).  The
                 # TRACS, Carlsson and Roux are references 26, 28 and 29
                 # (renumbered 2026-09-28), from csd-materials/citations.
                 "A further consideration concerns which transition lines "
                 "are needed in practice. The present metric weights every "
                 "line pixel equally, so the reported F1 score measures the "
                 "recovery of the diagram as a whole. Many tuning tasks, "
                 "however, do not require the complete diagram. Setting a "
                 "double dot to a target charge configuration, or locating "
                 "the operating point for spin readout, depends mainly on "
                 "the interdot transition lines and the triple points at "
                 "their ends. These features are short and therefore "
                 "contribute only a small fraction of the line pixels in "
                 "each diagram, so a high overall F1 score does not "
                 "guarantee that they are recovered accurately. Evaluating "
                 "the reconstruction specifically on interdot transitions "
                 "and triple points, and training the network with a loss "
                 "that emphasises them, is a natural extension of this work.",
                 "The choice of window is itself a limitation of the present"
                 " study. Every diagram is simulated over a voltage window "
                 "of fixed size, 2 × 2 mV, at a fixed resolution of 100 × "
                 # ", and the window is placed at a random position that
                 # generally does not contain the (0,0) configuration"
                 # deleted at your request, 2026-09-27
                 "100 pixels. "
                 "The variation in the capacitances changes the number of "
                 "honeycomb cells that fall inside this window, but the "
                 "network is never presented with windows of a different "
                 "voltage span or pixel density. In experiments, the scan "
                 "range and resolution are chosen by the experimenter and "
                 "differ between devices and tuning stages, and whether the "
                 "present network remains accurate under such changes has "
                 "not been tested.",
                 "A related approach that addresses several of these points "
                 "is the transformer-based model TRACS of Marchand et "
                 "al.^{26)} Rather than classifying individual pixels, TRACS "
                 "detects the triple points of a diagram together with their"
                 " connectivity, and thus directly returns the features "
                 "required for gate virtualisation and charge-state "
                 "initialisation. Input diagrams of arbitrary resolution are"
                 " rescaled to a common size. Trained entirely on QArray "
                 "simulations that include thermal broadening, white noise, "
                 "telegraph noise and latching, it was shown to generalise "
                 "to measured diagrams from three different device "
                 "architectures without retraining. However, TRACS requires "
                 "a fully measured charge stability diagram as input, and "
                 "therefore does not reduce the acquisition time, which is "
                 "the aim of the present work. Combining its feature-level "
                 "output with the sparse ray acquisition proposed here, so "
                 "that triple points and their connectivity are inferred "
                 "directly from a few voltage sweeps, is a promising "
                 "direction for future work.",
                 "Reducing the acquisition time is also the point at which "
                 "the present method connects to recent developments in "
                 "measurement hardware. The time required to characterise a "
                 "device consists of two parts: the time spent acquiring the"
                 " measurements, and the time spent processing them to "
                 "decide on the next step. The ray-based approach reduces "
                 "only the first, and its benefit therefore depends on the "
                 "second remaining small. Fast readout hardware, in "
                 "particular radio-frequency (RF) reflectometry [27], has "
                 "recently been shown to shorten the first part "
                 "considerably. "
                 # "(RF)" moved up to the first mention, 2026-09-28
                 "Carlsson et al. used RF charge sensing "
                 "to tune spin qubits automatically, with a median tuning "
                 "time of about 15 minutes, locating interdot transitions "
                 "with a neural network trained on QArray simulations.^{28)} "
                 "Roux et al. controlled the stability-diagram measurements "
                 "with an FPGA, which reduced the communication latency and "
                 "allowed shorter integration times and faster voltage ramps"
                 " to be exploited. Combining these FPGA-accelerated "
                 "measurements with a neural network for charge-transition "
                 "detection, they automatically tuned a SiGe quantum dot "
                 "into the single-electron regime with a success rate of 90 "
                 "%.^{29)} Once the measurements were accelerated, however, "
                 "the decision-making between measurements, performed on the"
                 " host computer, became the limiting step. They noted "
                 "accordingly that sparse measurement strategies are "
                 "advantageous only when the measurement time greatly "
                 "exceeds the time required to decide the next measurement.",
                 "With fast RF readout, a complete set of rays could be "
                 "acquired in a time far shorter than a conventional two-"
                 "dimensional raster scan, and the time needed for network "
                 "inference could then dominate. In the present method, the "
                 "network is evaluated only once, after all rays have been "
                 "acquired, so this cost is incurred a single time. The "
                 "network is also small and fully convolutional, which makes"
                 " its deployment on the host computer, or on the FPGA "
                 "itself as suggested by Roux et al.,^{29)} feasible. A short "
                 "measurement–prediction loop would further allow the "
                 "acquisition to become adaptive: the probability map "
                 "indicates where the model is uncertain, and this "
                 "information could be used to select the direction of the "
                 "next ray rather than fixing the ray pattern in advance. In"
                 " that case, the network would be evaluated after every "
                 "ray, and low-latency inference, ideally on the FPGA, would"
                 " become essential.",
                 "The placement of the rays could also exploit prior "
                 "knowledge of the device. In many experimental situations, "
                 "the approximate positions of the transition lines are "
                 "already known from earlier measurements of the same "
                 "device. These positions nevertheless change over time and "
                 "between cooldowns; thermal cycling, for example, can alter"
                 " the diagram as a whole [30]. In such cases, the task is not to"
                 " locate the transitions from scratch but to update their "
                 "positions quickly, and a few rays are better suited to "
                 "this than a repeated full raster scan. Prior knowledge "
                 "could be incorporated into the present method either by "
                 "placing the rays near the expected transitions, in "
                 "particular the interdot transitions of interest, or by "
                 "supplying a previously measured diagram to the network as "
                 "an additional input channel. The network would then need "
                 "to infer only how transition lines have moved rather "
                 "than where "
                 "they are. The setting studied here, in which no prior "
                 "information is used, therefore represents the most "
                 "demanding case, and fewer rays may suffice when such "
                 "information is available.",
             ]),
         ],
         [("results_budget_ladder.png",
           # three ladders, 40/50/60 points per ray (2026-09-27)
           "F1 at tau = 1 against the fraction of the plane measured, for "
           "%s points per ray; the number beside each point is the ray "
           "count. The best-performing configuration (%d × %d) is ringed."
           % (", ".join(str(n) for n in f["points_swept"][:-1])
              + " and %d" % f["points_swept"][-1],
              f["n_rays"], f["n_points"])),
          ("fig_tau_metrics.png",
           "(a) Pixel accuracy, (b) precision, (c) recall and (d) F1 "
           "against the tolerance tau, for three measurement budgets, on %d held-out "
           # 2026-09-26: the pixel-accuracy remark is gone; the reader is
           # pointed at the tau 0 -> 1 jump instead.  Checked: it is the
           # largest step for all three budgets drawn (+0.33 to +0.35;
           # the later steps are at most +0.15).
           "devices. Note the jump in F1 in (d) from tau = 0 to tau = 1, "
           "the largest step for all three measurement budgets." % f["n_test"]),
          # 2026-09-30.  Row (a) of the figure is Fig. 3 of Ref. 12, from
          # the author's Presentation1.pptx; check its licence before
          # submission.
          ("pipeline_comparison_ref12.png",
           "Pipelines of (a) Hernandes et al.^{12)} and (b) this work. "
           "Images in (a) are reproduced from Fig. 3 of Ref. 12 (grid "
           "mask, reduce factor 5): the masked measurement, the diffusion "
           "reconstruction, and its ridges compared with those of the "
           "measured image (white, both; blue, reconstruction only; red, "
           "measured image only). Images in (b): one held-out device of "
           "this work.")]),

        ("5", "conclusions",
         "Conclusions",
         "We summarise what this study establishes, the limitations under "
         "which it was established, and what remains to be done.",
         [
             # YOUR TEXT, word for word, 2026-09-26: the Conclusions as two
             # paragraphs, no subsections.  Numbers read from the run; "one
             # of four comparable pairs" from _equal_coverage() (3 of the 4
             # converged pairs within 0.1 % favour more rays).
             (None, [
                 # YOUR TEXT, word for word, 2026-09-27: the Conclusions as four
                 # paragraphs.  Its numbers are typed, so _check_conclusions()
                 # refuses to build if the run stops supporting them.
                 # your text, word for word, 2026-09-30
                 "We have formulated the reconstruction of a charge "
                 "stability diagram from sparse measurements as a "
                 "probabilistic inference of the positions of its charge "
                 "transition lines, the features that fully determine the "
                 "charge configuration of the double quantum dot. "
                 "Given a small number of "
                 "one-dimensional voltage sweeps, a U-Net returns, for every"
                 " pixel of the gate-voltage plane, the probability that a "
                 "transition line passes through it. The quantity that "
                 "determines the charge occupation is therefore obtained in "
                 "a single step, without an intermediate image or a separate"
                 " line detector. Because the network is trained entirely on"
                 " simulated devices, its training labels are exact and the "
                 "amount of training data is not limited by experimental "
                 "effort.",
                 "On 50 simulated test devices, the transition lines were "
                 "recovered with F1@1 between 0.66 and 0.82 while sampling "
                 "only 1.6–4.7 % of the voltage plane. The best-performing "
                 "configuration (8 × 60) reached F1@1 = 0.82 at a coverage of "
                 "4.7 %, and the score increased with the number of rays for"
                 " nearly all measurement budgets.",
                 "These results were obtained under idealised conditions. "
                 "The devices were simulated with a noise-free constant-"
                 "capacitance model, over a voltage window of fixed size, "
                 "and training and test devices were drawn from the same "
                 "capacitance intervals. The present work therefore "
                 "establishes that the geometry of a double-dot stability "
                 "diagram can in principle be recovered from rays covering a"
                 " few percent of the plane; it does not yet demonstrate "
                 "this on measured devices. In addition, the network outcome"
                 " depends on training, as shown by the four measurement budgets that "
                 "had to be retrained, so small differences between "
                 "individual measurement budgets should not be over-interpreted.",
                 "Several directions follow from this work. Training on "
                 "simulations that include noise and tunnel coupling, and "
                 "testing on measured diagrams, is the essential next step. "
                 "Beyond that, the method could be focused on the interdot "
                 "transitions and triple points that tuning actually "
                 "requires, informed by previously measured diagrams of the "
                 "same device, and combined with fast RF readout, for which "
                 "a small set of continuous voltage sweeps is naturally "
                 "suited. Together, these steps would turn sparse ray "
                 "measurements into a practical tool for the rapid "
                 "characterisation and re-tuning of quantum-dot devices.",
             ]),
         ],
         []),
    ]


def _check_conclusions():
    """The Conclusions are typed by hand; refuse to build if a number in
    them no longer matches the run, at the precision they state it."""
    rows = _run.comparison_rows()
    f1 = [float(r["f1@1"]) for r in rows]
    cov = [100 * float(r["coverage"]) for r in rows]
    text = " ".join(p for spec in parts() if spec[0] == "5"
                    for _h, paras in spec[4] for p in paras
                    if isinstance(p, str))
    claims = [
        "On %d simulated test devices" % F["n_test"],
        "F1@1 between %.2f and %.2f" % (min(f1), max(f1)),
        "sampling only %.1f–%.1f %%" % (min(cov), max(cov)),
        "The best-performing configuration (%d × %d)" % (F["n_rays"],
                                                         F["n_points"]),
        "reached F1@1 = %.2f at a coverage of %.1f %%" % (F["f1"][1],
                                                         F["coverage"]),
        "the %s measurement budgets that had to be retrained" % _word(len(_run.SEEDS)),
    ]
    missing = [c for c in claims if c not in text]
    if missing:
        raise SystemExit("the Conclusions no longer match the run: %s. "
                         "Edit them in make_method_docs.py."
                         % "; ".join(missing))


def _keep_full_resolution(doc):
    """Stop Word compressing the pictures.

    Word's default is to downsample every picture to its "default
    resolution" when it saves or exports a PDF; Fig. 13 came out of
    make_pdfs at 1219 px (200 dpi) from a 3420-px PNG.  The document
    setting doNotAutoCompressPictures ("Do not compress images in file")
    turns that off, for the .docx and for the PDF exported from it.
    Added 2026-09-30."""
    settings = doc.settings.element
    if settings.find(qn("w:doNotAutoCompressPictures")) is None:
        settings.append(OxmlElement("w:doNotAutoCompressPictures"))


# ── building one document ─────────────────────────────────────────────────
def _clear_body_after(doc, keep):
    """Delete every block after the first `keep` paragraphs."""
    body = doc.element.body
    paras = doc.paragraphs
    stop = paras[keep]._p
    seen = False
    for child in list(body):
        if child is stop:
            seen = True
            continue
        if seen and not child.tag.endswith("sectPr"):
            body.remove(child)


def _font(run, size=FONT_PT, bold=None, name=FONT):
    """Times New Roman at `size`, including the East-Asian slot.

    The template was made in a Japanese Word, so every run carries an
    eastAsia font as well as an ascii one.  Setting only run.font.name
    leaves the eastAsia slot pointing at the old font, and Word then
    renders some characters in it -- so both slots are set here."""
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    for slot in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(slot), name)


def _para_marked(doc, text, style, size=FONT_PT,
                 align=WD_ALIGN_PARAGRAPH.LEFT, name=FONT):
    """A paragraph in which {...} is set as a superscript.

    Affiliation markers are superscripts, not literal braces; writing
    them inline keeps the author block readable in the source."""
    p = doc.add_paragraph(style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = LINE_SPACING
    pf.first_line_indent = Pt(0)
    for chunk in re.split(r"({[^}]*})", text):
        if not chunk:
            continue
        mark = chunk.startswith("{")
        run = p.add_run(chunk[1:-1] if mark else chunk)
        _font(run, size, None, name)
        if mark:
            run.font.superscript = True
    return p


# ── how the journal writes a cross-reference ──────────────────────────────
# Read off the JJAP template and the previous paper (method_docs/
# *_template.docx, Manuscript_final_template.docx), which do two different
# things and do them consistently:
#
#   in the text      Figure 1, Figure 4(a), Table I     bold, word in full
#   in the caption   Fig. 1.   Table I.                 bold label, text not
#
# and the space inside a reference is a NARROW NO-BREAK SPACE (U+202F), so
# "Figure" never sits at the end of a line with its number on the next one.
# The journal's own template says it in Table I: "Scheme" cannot be used in
# APEX/JJAP, use "Figure".
NBSP = u"\u202f"   # narrow no-break space
# The panel letter is part of the reference and is bold with it, as in the
# previous paper's "Figure 4(a)" -- hence the lookahead rather than a
# trailing \b, which would refuse to end the match on the ")".
# The plural is matched too -- "Figures 2 to 4", "Tables I and II" -- with
# the second number in groups 3 and 4.  Until 2026-09-26 it was not, and a
# plural reference was neither renumbered nor set bold.
_NUM = r"[0-9]+(?:\([a-z]\))?|[IVXL]+"
REF = re.compile(
    r"\b(Figures?|Fig\.|Tables?)\s+(" + _NUM + r")"
    r"(?:(\s+(?:to|and)\s+|\s*[-–]\s*)(" + _NUM + r"))?(?![\w(])"
    # "Fig. 3 of Ref. 12" is another paper's figure: not ours to renumber
    # or link (2026-09-30)
    r"(?!\s+of\s+Ref\.)")
# The caption label: "Fig. 1." or "Table I." at the very start of a caption.
LABEL = re.compile(r"^(Fig\.|Table)\s+([0-9]+(?:\([a-z]\))?|[IVXL]+)\.")


# ── one numbering across the whole manuscript ─────────────────────────
# Every part is WRITTEN with its own figures numbered from 1 and its own
# tables from I, because that is what a part says when it is read on its
# own.  The parts are also pieces of ONE paper, and there the numbers have
# to run on: the first figure of 3.2 is Figure 4, because 3.1 used three.
#
# So the number in the source text is a LOCAL one, and the number that
# reaches the page is local + an offset counted off the parts before it.
# `renumber` is the only place that conversion happens, and every string on
# its way into a document goes through it -- body text, table caption, cell,
# figure caption -- so a reference and the thing it refers to cannot drift
# apart.  Panel letters ride along untouched: Figure 1(b) -> Figure 4(b).
_ROMAN = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
          (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
          (5, "V"), (4, "IV"), (1, "I"))


def roman(n):
    out = ""
    for value, sign in _ROMAN:
        while n >= value:
            out += sign
            n -= value
    return out


def unroman(text):
    value = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, seen = 0, 0
    for ch in reversed(text):
        v = value[ch]
        total += v if v >= seen else -v
        seen = max(seen, v)
    return total


def renumber(text, fig_offset=0, tbl_offset=0):
    """`text` with its figure and table numbers moved by the offsets.

    It moves what the journal actually writes and nothing else: "Figure 2",
    "Figure 4(a)" and "Table II" in the prose, and a caption's own
    "Fig. 2." or "Table I." at the very front of it."""
    if not text or not (fig_offset or tbl_offset):
        return text

    def shift(kind, number):
        if kind.startswith("Table"):
            return roman(unroman(number) + tbl_offset)
        m = re.match(r"([0-9]+)(\([a-z]\))?$", number)
        if not m:
            return number
        return "%d%s" % (int(m.group(1)) + fig_offset, m.group(2) or "")

    head = ""
    m = LABEL.match(text)
    if m:
        head = "%s %s." % (m.group(1), shift(m.group(1), m.group(2)))
        text = text[m.end():]
    def moved(m):
        out = "%s %s" % (m.group(1), shift(m.group(1), m.group(2)))
        if m.group(4):
            out += m.group(3) + shift(m.group(1), m.group(4))
        return out
    return head + REF.sub(moved, text)


def renumber_table(spec, fig_offset=0, tbl_offset=0):
    """A table spec (caption, headers, rows, ...) with its numbers moved."""
    def r(v):
        return renumber(v, fig_offset, tbl_offset) if isinstance(v, str) else v
    caption, headers, rows = spec[0], spec[1], spec[2]
    return ((r(caption), [r(h) for h in headers],
             [[r(v) for v in row] for row in rows]) + tuple(spec[3:]))


# A capacitance is written C_{dd} and its entries c_{d1g1}; a space of
# matrices is R^{2x3}.  The braces are markup, not text: they never reach
# the page.  One level only, which is all Word draws properly, and it is
# what the symbols in this paper need.
# A paragraph that opens with this marker is a displayed equation: it is
# set centred and unindented rather than as running text.  The marker is
# never printed.
EQ = "$$"

SCRIPT = re.compile(r"([_^])\{([^{}]*)\}")


def _scripted(chunk):
    """`C_{dd}` -> [("C", None), ("dd", "sub")]."""
    out, last = [], 0
    for m in SCRIPT.finditer(chunk):
        if m.start() > last:
            out.append((chunk[last:m.start()], None))
        out.append((m.group(2), "sub" if m.group(1) == "_" else "sup"))
        last = m.end()
    out.append((chunk[last:], None))
    return [(t, v) for t, v in out if t]


# ── citations, the JJAP way ───────────────────────────────────────────────
# The source writes a citation as "[4]" or "[1–3]" after a word.  JJAP sets
# it as a superscript with a closing parenthesis, on the shoulder of the
# word and AFTER any punctuation that follows -- as its own template does:
# "…your paper's existence.^{1-4)} It must state…".  The whole "1-4)" is
# superscript.  Only a bracket that follows a word (or ")") and one space
# counts, and a reference number never starts at 0, so an interval such
# as "normalised to [0, 1]" is never taken for one (it was, 2026-09-26).
CITE = re.compile(r"(?<=[A-Za-z0-9)])\s\[([1-9]\d*(?:\s*[–,-]\s*\d+)*)\]"
                  r"([.,;:]?)")


# ── named figure references ───────────────────────────────────────────────
# A part numbers its figures from 1 and renumber() adds the offset of the
# parts before it, which cannot express a reference BACK to an earlier
# part's figure.  "{fig:inputs}" names the figure instead; it is replaced by
# the figure's number in the paper when the text is written into Word,
# after every renumbering.  Added 2026-09-26.
FIGREF = {"inputs": "fig_inputs.png", "network": "fig_model_flow.png"}
_FIGNUM = None


def _paper_figure_numbers():
    global _FIGNUM
    if _FIGNUM is None:
        _FIGNUM, n = {}, 0
        for spec in parts():
            for fname, _cap in spec[5]:
                n += 1
                _FIGNUM[fname] = n
    return _FIGNUM


def _named_refs(text):
    if "{fig:" not in text:
        return text
    nums = _paper_figure_numbers()
    return re.sub(r"\{fig:(\w+)\}",
                  lambda m: str(nums[FIGREF[m.group(1)]]), text)


def _cite(text):
    """`technology [1–3]. In` -> `technology.^{1–3)} In`."""
    return CITE.sub(lambda m: "%s^{%s)}" % (m.group(2),
                                            re.sub(r"\s+", "", m.group(1))),
                    text)


# A citation once it is set: the superscript "4)", "1–3)" or "5,9)".
CITE_SUP = re.compile(r"^([1-9]\d*)(?:\s*[–,-]\s*\d+)*\)$")
# ...and as the source may already write it: "et al.,^{7)}".
CITE_SET = re.compile(r"\^\{([1-9]\d*(?:\s*[–,-]\s*\d+)*)\)\}")


def paper_references():
    """The paper's reference list -- the part that carries it."""
    for spec in parts():
        if len(spec) > 6 and "references" in spec[6]:
            return spec[6]["references"]
    return ()


def cited_references(sections):
    """[(number, entry)] for every reference `sections` cites, in order.

    A part printed on its own lists these at its end, under the paper's
    numbers, so each citation still jumps to its entry (your request,
    2026-09-28).  The paper's own list is untouched."""
    nums = set()

    def scan(text):
        for rx in (CITE, CITE_SET):
            for m in rx.finditer(text):
                for item in m.group(1).split(","):
                    ends = [int(x) for x in re.split(r"[–-]", item)]
                    nums.update(range(ends[0], ends[-1] + 1))

    for _heading, paragraphs in sections:
        for text in paragraphs:
            if isinstance(text, str):
                scan(text)
            elif isinstance(text, list):
                for item in text:
                    scan(item)
    refs = paper_references()
    return [(n, refs[n - 1]) for n in sorted(nums) if n <= len(refs)]


PANELS = re.compile(r"(?:\s*(?:,\s*and|,|and|to|–|-)\s*\([a-z]\))+")


def _anchor(kind, number):
    """The bookmark a figure or table caption carries: fig_8, tab_I.

    A panel letter is dropped -- "Figure 3(b)" jumps to Fig. 3."""
    number = re.sub(r"\(.*$", "", number)
    return ("tab_" if kind.startswith("Table") else "fig_") + number


def _bookmark(p, run, name):
    """Wrap `run` in a Word bookmark called `name`."""
    global _BM_ID
    _BM_ID += 1
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(_BM_ID))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(_BM_ID))
    run._r.addprevious(start)
    run._r.addnext(end)


_BM_ID = 0


def _link(p, runs, anchor):
    """Move `runs` into an internal hyperlink to bookmark `anchor`.

    Word keeps these as live links when it exports the PDF, so a
    reference in the text jumps to its figure or table."""
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("w:anchor"), anchor)
    hl.set(qn("w:history"), "1")
    runs[0]._r.addprevious(hl)
    for r in runs:
        hl.append(r._r)


def _runs(p, text, size, bold, name, label=False):
    """Fill `p` with runs, setting cross-references the way JJAP does.

    A reference in the text ("Figure 2", "Table I") is bold and carries a
    narrow no-break space; with `label`, the caption's own "Fig. 2." at the
    front is bold and the caption that follows it is not.

    Every reference is also a link to its caption, and every caption label
    is the bookmark the link points at (2026-09-26)."""
    text = _cite(_named_refs(text))
    pieces = []                                   # (text, bold, link, mark)
    if label:
        m = LABEL.match(text)
        if m:
            pieces.append((m.group(0).replace(" ", NBSP), True, None,
                           _anchor(m.group(1), m.group(2))))
            text = text[m.end():]
    last = 0
    for m in REF.finditer(text):
        if m.start() > last:
            pieces.append((text[last:m.start()], None, None, None))
        kind = m.group(1)
        # the text always says "Figure", never "Fig." -- that is for the
        # caption label only (your rule, 2026-09-28)
        if kind == "Fig." and not label:
            kind = "Figure"
        pieces.append((kind + NBSP + m.group(2), True,
                       _anchor(kind, m.group(2)), None))
        if m.group(4):
            pieces.append((m.group(3), True, None, None))
            pieces.append((m.group(4), True, _anchor(kind, m.group(4)),
                           None))
        last = m.end()
        # A panel list that follows -- "Figures 6(a), (c) and (d)",
        # "Figure 3(c) and (d)", "7(b)-(d)" -- is part of the reference:
        # bold, and linked to the same figure (2026-09-27).
        tail = PANELS.match(text, last)
        if tail and m.group(2)[-1:] == ")":
            pieces.append((tail.group(0), True,
                           _anchor(kind, m.group(4) or m.group(2)), None))
            last = tail.end()
    pieces.append((text[last:], None, None, None))
    for chunk, is_ref, link, mark in pieces:
        if not chunk:
            continue
        made = []
        for piece, script in _scripted(chunk):
            run = p.add_run(piece)
            _font(run, size, True if is_ref else bold, name)
            if script == "sub":
                run.font.subscript = True
            elif script == "sup":
                run.font.superscript = True
            made.append(run)
            # a citation "4)" or "1–3)" jumps to its entry in the reference
            # list -- to the first one of a range (your request, 2026-09-27)
            cite = CITE_SUP.match(piece) if script == "sup" else None
            if cite and not link:
                _link(p, [run], "ref_%s" % cite.group(1))
        if mark and made:
            _bookmark(p, made[0], mark)
        if link and made:
            _link(p, made, link)
    return p


def _para(doc, text, style, size=FONT_PT, bold=None, indent=True,
          align=WD_ALIGN_PARAGRAPH.JUSTIFY, name=FONT, label=False, left=0):
    """One paragraph in the JJAP body format.

    `left` indents the whole paragraph, which is how the items of a list
    are set: the template defines no list style, and an invented bullet
    glyph is not the journal's."""
    p = doc.add_paragraph(style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = LINE_SPACING
    pf.left_indent = Pt(left)
    pf.right_indent = Pt(0)
    pf.first_line_indent = Pt(INDENT_PT) if indent else Pt(0)
    _runs(p, text, size, bold, name, label=label)
    return p


def _border(pr, tag, edge, size=8):
    """A single horizontal rule on one edge, given a tblPr or a tcPr."""
    borders = pr.find(qn(tag))
    if borders is None:
        borders = pr.makeelement(qn(tag), {})
        pr.append(borders)
    line = borders.find(qn("w:" + edge))
    if line is None:
        line = borders.makeelement(qn("w:" + edge), {})
        borders.append(line)
    line.set(qn("w:val"), "single")
    line.set(qn("w:sz"), str(size))
    line.set(qn("w:color"), "000000")


def _rule(table, edge):
    # CT_Tbl exposes tblPr as a property, not a get_or_add_* method
    _border(table._tbl.tblPr, "w:tblBorders", edge)


def _row_rule(row, size=8):
    for cell in row.cells:
        _border(cell._tc.get_or_add_tcPr(), "w:tcBorders", "bottom", size)


def _cell(cell, text, bold=None):
    """Table cells are LEFT aligned and single spaced.

    They inherit the body format otherwise, and justified text in a narrow
    column stretches two words across the whole cell."""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    # through _runs, so a cell writes C_{dd} the way the text does
    _runs(p, text, FONT_PT, bold, FONT)


def _table(doc, caption, headers, rows, widths=None, rules_after=()):
    """A JJAP table: caption above it, Times New Roman 12 throughout.

    The template defines no table style beyond Normal Table, so the rule
    lines are drawn here rather than inherited."""
    _para(doc, caption, S_BODY, indent=False, label=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    # The template defines no table style but Normal Table, and a full grid
    # is not the journal's look anyway: rule above the header, below it, and
    # below the last row, with no vertical lines.
    _rule(t, "top")
    _rule(t, "bottom")
    _row_rule(t.rows[0])
    # a heavier rule under the last row of each block, so the four
    # capacitance matrices read as four groups rather than one list
    for i in rules_after:
        _row_rule(t.rows[i], size=12)
    for j, head in enumerate(headers):
        _cell(t.cell(0, j), head, bold=True)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row):
            _cell(t.cell(i, j), str(value))
    if widths:
        # Word ignores cell widths unless autofit is off, and it wants the
        # width set on EVERY cell of a column, not just the first.
        t.autofit = False
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    # Centred, and only as wide as its content needs.  Stretching a narrow
    # table to the full text width leaves a band of empty paper to the right
    # of the last column and makes the rules look like they overrun.
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    return t


# "3.1.2. Ground truth" is two levels down from "3. Method"; the depth is
# read off the number the heading already carries, so nothing has to be
# declared twice.  A part on its own numbers its sections 1., 2., 3. and
# every one of them is a top-level heading, exactly as before.
_DEPTH = re.compile(r"^\d+(\.\d+)*\.")


def _heading(doc, text):
    """A heading in the template's own two levels.

    SectionTitle (Arial bold 14) for `3. Method`, SubSectionTitle (bold 11,
    the size the template gives it) for anything numbered deeper."""
    m = _DEPTH.match(text)
    deep = bool(m) and m.group(0).count(".") > 1
    return _para(doc, text, S_SUBSECTION if deep else S_SECTION,
                 size=SUBHEAD_PT if deep else HEAD_PT, bold=True,
                 indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                 name=HEAD_FONT)


def stem(number, slug):
    """`3_1_Device_simulation_and_dataset` -- the folder and the file share it.

    `number` is a LABEL, not an integer: the parts run 0, 1, 2, then the
    four subsections of the method as 3_1 ... 3_4, then 4 for the results.
    It is written with an underscore in a name and a dot in prose."""
    return "%s_%s" % (number, slug[:1].upper() + slug[1:])


def shown(number):
    """The label as it is written in prose: 3_2 -> 3.2."""
    return str(number).replace("_", ".")


def part_dir(number, slug, create=True):
    """Each part gets its own folder, named exactly as the document is.

    The document, its notes and COPIES of the figures it uses all live
    there, so a part can be sent to someone as one self-contained folder.
    The four subsections of the method, 3_1 ... 3_4, share one folder,
    3_methods/, with one figures/ and one backups/ between them."""
    d = os.path.join(OUT_DIR, _figures.part_folder(stem(number, slug)))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _place_static(figures):
    """Copy any hand-drawn figure into the folder it is filed under.

    The generators write their own output there; these are drawn in
    Illustrator and live in docs/figures/, so they are copied in rather
    than produced."""
    for fname, _caption in figures:
        if fname not in _figures.STATIC:
            continue
        src = os.path.join(os.path.dirname(PKG), _figures.STATIC[fname])
        if not os.path.exists(src):
            raise SystemExit("%s is missing" % src)
        dst = _figures.out_path(fname)
        if (not os.path.exists(dst)
                or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copyfile(src, dst)


SIGNATURE = (
    "Ehsan Alizadeh Kashtiban, on behalf of all authors",
    "Department of Physics, Graduate School of Science, The University of Osaka",
    "E-mail: alizadeh21@sanken.osaka-u.ac.jp",
)


# ── never overwrite a document that was edited by hand ────────────────────
# The documents are generated, so a rebuild would normally just replace
# them.  But they are also Word files, and the obvious thing to do with a
# Word file is to open it and edit it.  So every document written here is
# fingerprinted, and a rebuild compares the file on disk with the
# fingerprint it was left with:
#
#   unchanged since the build  ->  replaced, and the old one backed up
#   changed (you edited it)    ->  NOT touched.  The new version is written
#                                  beside it as <stem>_rebuilt.docx and the
#                                  run says so, loudly.
#
# Nothing is ever deleted: the previous copy of a replaced document goes to
# <part>/backups/ with a timestamp.
STATE = os.path.join(OUT_DIR, ".build_state.json")


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _state():
    if not os.path.exists(STATE):
        return {}
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except ValueError:
        return {}


def _remember(path):
    st = _state()
    st[os.path.basename(path)] = _sha(path)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)


def hand_edited(path):
    """True if `path` is not byte-for-byte what this script last wrote.

    A document with no fingerprint at all counts as hand-edited: it is
    safer to treat an unknown file as someone's work than as scrap."""
    if not os.path.exists(path):
        return False
    known = _state().get(os.path.basename(path))
    return known is None or known != _sha(path)


KEEP_BACKUPS = 10


def _backup(path):
    """Keep the copy being replaced, timestamped, in <part>/backups/.

    The last KEEP_BACKUPS of a document are kept and older ones dropped, so
    the folder does not grow without bound while a bad rebuild is always
    one file away from being undone."""
    import time
    d = os.path.join(os.path.dirname(path), "backups")
    os.makedirs(d, exist_ok=True)
    root, ext = os.path.splitext(os.path.basename(path))
    dst = os.path.join(d, "%s_%s%s"
                       % (root, time.strftime("%Y%m%d_%H%M%S"), ext))
    shutil.copyfile(path, dst)
    old = sorted(f for f in os.listdir(d)
                 if f.startswith(root + "_") and f.endswith(ext))
    for name in old[:-KEEP_BACKUPS]:
        os.remove(os.path.join(d, name))
    return dst


def _place_at(work, out):
    """Move the freshly built `work` onto `out`, guard and all.

    Returns (path written, True if it had to go in beside a hand edit)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if hand_edited(out):
        alt = os.path.splitext(out)[0] + "_rebuilt.docx"
        shutil.move(work, alt)
        _remember(alt)
        return alt, True
    if os.path.exists(out):
        _backup(out)
        os.remove(out)
    shutil.move(work, out)
    _remember(out)
    return out, False


def _place(work, number, slug):
    """Move the freshly built `work` file into the part folder."""
    return _place_at(work, os.path.join(part_dir(number, slug),
                                        stem(number, slug) + ".docx"))


def _scratch(number, slug):
    """Where a document is built before it is allowed near the part folder."""
    d = os.path.join(OUT_DIR, ".build")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, stem(number, slug) + ".docx")


def _clone(doc, proto, text, size, bold=None, italic=None, name=FONT):
    """A copy of one of the template's paragraphs, with our words in it.

    The paragraph properties are COPIED, not rebuilt: the title page's
    spacing is a pile of little decisions (autospacing before the author
    line, 6 pt before each affiliation, a right indent of 4.5 characters,
    exactly 1.5 lines) that live in the template's own XML, and copying
    the element keeps every one of them.  Only the runs are replaced.
    {..} marks a superscript, as in the author and affiliation blocks."""
    element = copy.deepcopy(proto._p)
    # A hyperlink wraps its own runs, so dropping only <w:r> would leave
    # the template's mailto link sitting in front of our text.
    for tag in ("w:r", "w:hyperlink", "w:bookmarkStart", "w:bookmarkEnd"):
        for child in element.findall(qn(tag)):
            element.remove(child)
    doc.element.body.append(element)
    para = docx.text.paragraph.Paragraph(element, doc)
    for chunk in re.split(r"({[^}]*})", text):
        if not chunk:
            continue
        mark = chunk.startswith("{")
        run = para.add_run(chunk[1:-1] if mark else chunk)
        _font(run, size, bold, name)
        if italic is not None:
            run.font.italic = italic
        if mark:
            run.font.superscript = True
    return para


def build_title_page(number, slug, title, abstract, sections, figures,
                     **_ignored):
    """The title page, built from the title-page template.

    It carries what that template carries and nothing else: the title, the
    authors, the affiliations, the corresponding e-mail and the abstract,
    none of them under a heading.  Page size, margins, the three custom
    styles and every paragraph's spacing are the template's."""
    if not os.path.exists(TITLE_TEMPLATE):
        raise SystemExit("%s is missing" % TITLE_TEMPLATE)
    _check_abstract()
    part_dir(number, slug)
    work = _scratch(number, slug)
    shutil.copyfile(TITLE_TEMPLATE, work)
    doc = docx.Document(work)

    # The template's own paragraphs, used as prototypes: title, authors,
    # one affiliation, the e-mail, a blank line and the abstract.
    paras = doc.paragraphs
    assert paras[0].style.name == T_TITLE, paras[0].style.name
    assert paras[2].style.name == T_AFFIL, paras[2].style.name
    assert paras[12].style.name == T_EMAIL, paras[12].style.name
    proto = {"title": paras[0], "authors": paras[1], "affil": paras[2],
             "email": paras[12], "blank": paras[13], "abstract": paras[14]}
    _clear_body_after(doc, keep=0)

    # the title, in the paragraph the template already has
    first = paras[0]
    for run in first.runs[1:]:
        run._r.getparent().remove(run._r)
    first.runs[0].text = title
    _font(first.runs[0], TITLE_PT, True, TITLE_FONT)

    _clone(doc, proto["authors"], AUTHORS, FONT_PT)
    for line in AFFILIATIONS:
        # the italic is the style's; the template's runs only override the
        # size, to 12 pt
        _clone(doc, proto["affil"], line, FONT_PT)
    _clone(doc, proto["blank"], "", FONT_PT)
    _clone(doc, proto["email"], EMAIL, FONT_PT, italic=False)
    # The template's second blank line before the abstract.  It is kept
    # even when the abstract is long enough to run onto a second page:
    # the spacing on this page is the template's, and shortening the
    # abstract is the author's call, not this script's.
    _clone(doc, proto["blank"], "", FONT_PT)
    # the abstract: Normal, left aligned, single spaced -- the one block on
    # the page that is not 1.5 lines
    _clone(doc, proto["abstract"], abstract, FONT_PT)

    _keep_full_resolution(doc)
    doc.save(work)
    return _place(work, number, slug)


def build_letter(number, slug, title, abstract, sections, figures,
                 **_ignored):
    """The cover letter, built from the letter template.

    Nothing of the JJAP manuscript format applies here: no running title,
    no section headings, no 1.5 line spacing, no first-line indent.  The
    template's own page setup and face are kept exactly as they are and
    only the words are replaced, so what the editor receives looks like
    the letter that was sent with the previous paper."""
    if not os.path.exists(COVER_TEMPLATE):
        raise SystemExit("%s is missing" % COVER_TEMPLATE)
    part_dir(number, slug)
    work = _scratch(number, slug)
    shutil.copyfile(COVER_TEMPLATE, work)
    doc = docx.Document(work)

    # Keep the first paragraph as the template's, so its formatting is
    # inherited, and drop everything after it.
    _clear_body_after(doc, keep=0)
    body = [text for _heading, paras in sections for text in paras]
    doc.paragraphs[0].runs[0].text = body[0]          # "Dear Editor,"
    for run in doc.paragraphs[0].runs[1:]:
        run._r.getparent().remove(run._r)

    # The template's Normal already leaves 8 pt between paragraphs, so no
    # blank lines are inserted between them: with them the letter runs to a
    # second page, and a cover letter that does not fit on one page reads
    # like a manuscript.
    for text in body[1:]:
        doc.add_paragraph(text)
    # The sign-off is one block, not four paragraphs: the 8 pt the template
    # puts after a paragraph is dropped inside it, which also keeps the
    # letter on a single page.
    for line in ("Yours sincerely,",) + SIGNATURE:
        para = doc.add_paragraph(line)
        para.paragraph_format.space_after = Pt(0)

    _keep_full_resolution(doc)
    doc.save(work)
    return _place(work, number, slug)


def build(number, slug, title, abstract, sections, figures,
          references=(), front_matter=False, fig_offset=0, tbl_offset=0,
          out=None):
    """One document from the JJAP template.

    `fig_offset` / `tbl_offset` shift every number in it, so a part can be
    written with its own figures starting at 1 and still print the numbers
    the assembled manuscript uses.  `out` writes the document somewhere
    other than its part folder -- the assembled manuscript is not a part."""
    _place_static(figures)
    if out is None:
        part_dir(number, slug)
        work = _scratch(number, slug)
    else:
        work = os.path.join(OUT_DIR, ".build", os.path.basename(out))
        os.makedirs(os.path.dirname(work), exist_ok=True)
    shutil.copyfile(TEMPLATE, work)
    doc = docx.Document(work)

    # The template already has these, but a style copied from elsewhere
    # could change them silently; state them.
    for sec in doc.sections:
        sec.top_margin = Inches(MARGIN_IN["top"])
        sec.bottom_margin = Inches(MARGIN_IN["bottom"])
        sec.left_margin = Inches(MARGIN_IN["left"])
        sec.right_margin = Inches(MARGIN_IN["right"])
        sec.gutter = Inches(0)

    # The template's first five paragraphs are the title, the authors, the
    # two affiliations and the e-mail.  They are kept EXACTLY as the
    # template has them -- only the title text changes -- and everything
    # after them is replaced.
    paras = doc.paragraphs
    assert paras[0].style.name == S_TITLE, paras[0].style.name
    for run in paras[0].runs[1:]:
        run._r.getparent().remove(run._r)
    paras[0].runs[0].text = (title if number is None
                             else "%s. %s" % (shown(number), title))
    # ONLY the title is kept.  These are five parts of one manuscript, not
    # five manuscripts: repeating the authors, both affiliations and the
    # e-mail on every one of them is noise, and they belong on the
    # assembled paper instead.  _clear_body_after keeps the paragraph it
    # stops at, so keep=0 leaves the title and drops the rest.
    assert "Email" in paras[4].style.name, paras[4].style.name
    _clear_body_after(doc, keep=0)

    if front_matter:
        # the title page carries the authors and every affiliation
        _para_marked(doc, AUTHORS, "\u30b9\u30bf\u30a4\u30eb AUTHORS + \u53f3 :  4.5 \u5b57")
        for line in AFFILIATIONS:
            _para_marked(doc, line, "\u30b9\u30bf\u30a4\u30eb AFFILIATIONS + \u53f3 :  4.5 \u5b57", size=10)
        _para(doc, EMAIL, "\u30b9\u30bf\u30a4\u30eb Email + \u53f3 :  4.5 \u5b57",
              indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
        # always one empty line between the e-mail and the abstract, as
        # the title-page template has it (your request, 2026-09-27)
        _para(doc, "", "\u30b9\u30bf\u30a4\u30eb Email + \u53f3 :  4.5 \u5b57",
              indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    if abstract:
        _para(doc, renumber(abstract, fig_offset, tbl_offset), S_ABSTRACT,
              indent=False)
        # the paper's text starts on a page of its own (2026-09-27): the
        # title page ends with the abstract
        if front_matter:
            doc.add_page_break()

    # The body carries the TEXT only.  Tables declared inside a section are
    # collected and printed at the end, which is the order JJAP asks for:
    # text, then tables, then figures.  The text refers to them by number,
    # so nothing depends on where they physically sit.
    tables = []
    for heading, paragraphs in sections:
        # A heading of None is a section that is only its text: the part's
        # own title already names it, and a second line naming it again
        # would be a label, not a heading.
        if heading:
            _heading(doc, heading)
        for text in paragraphs:
            if isinstance(text, tuple):
                tables.append(renumber_table(text, fig_offset, tbl_offset))
            elif isinstance(text, str) and text.startswith(EQ):
                # a displayed equation: centred, on its own line, with the
                # body's first-line indent off
                _para(doc, renumber(text[len(EQ):], fig_offset, tbl_offset),
                      S_BODY, indent=False,
                      align=WD_ALIGN_PARAGRAPH.CENTER)
            elif isinstance(text, list):
                # a list of strings is a list of items, set indented and
                # without the body's first-line indent
                for item in text:
                    _para(doc, renumber(item, fig_offset, tbl_offset),
                          S_BODY, indent=False, left=INDENT_PT)
            else:
                _para(doc, renumber(text, fig_offset, tbl_offset), S_BODY)

    if references:
        # the reference list starts on a page of its own (2026-09-26)
        doc.add_page_break()
        _heading(doc, "References")
        for i, ref in enumerate(references, 1):
            # a part's own list is [(paper number, entry)], see
            # cited_references
            if isinstance(ref, tuple):
                i, ref = ref
            p = _para(doc, "%d) %s" % (i, ref), S_BODY, indent=False)
            # the target of every citation "i)" in the text
            if p.runs:
                _bookmark(p, p.runs[0], "ref_%d" % i)

    # ── tables, one per page ─────────────────────────────────────────────
    for i, spec in enumerate(tables):
        doc.add_page_break()
        if i == 0:
            _heading(doc, "Tables")
        _table(doc, *spec)

    # ── figures, one per page ────────────────────────────────────────────
    from PIL import Image
    for i, (fname, caption) in enumerate(figures, 1):
        doc.add_page_break()
        if i == 1:
            _heading(doc, "Figures")
        src = _figures.find(fname)
        iw, ih = Image.open(src).size
        width = TEXT_WIDTH_IN
        # a very tall figure would otherwise run off its page; the caption
        # needs room under it too
        height = width * ih / iw
        if height > 7.2:
            width = width * 7.2 / height
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(src, width=Inches(width))
        _para(doc, "Fig. %d. %s" % (i + fig_offset,
                                    renumber(caption, fig_offset, tbl_offset)),
              S_BODY, indent=False, label=True)

    _keep_full_resolution(doc)
    doc.save(work)

    # No copying: the figures this part uses are written straight into
    # <part>/figures/ by the generators, so they are already here.
    if out is not None:
        return _place_at(work, out)
    return _place(work, number, slug)


def numbering_span(fig_at, n_figs, tbl_at, n_tables):
    """`Fig. 4-7, Table I-II` -- what this part contributes to the paper."""
    bits = []
    if n_figs:
        bits.append("Fig. %d%s" % (fig_at + 1,
                                   "-%d" % (fig_at + n_figs) if n_figs > 1
                                   else ""))
    if n_tables:
        bits.append("Table %s%s"
                    % (roman(tbl_at + 1),
                       "-%s" % roman(tbl_at + n_tables) if n_tables > 1
                       else ""))
    return ", ".join(bits) or "no figures"


def main():
    print("template:", TEMPLATE)
    if not os.path.exists(TEMPLATE):
        raise SystemExit("JJAP template not found")
    print("headline: %d x %d, threshold %g, F1@1 %.3f at %.2f %% coverage"
          % (F["n_rays"], F["n_points"], F["threshold"], F["f1"][1],
             F["coverage"]))
    print()
    written, locked, beside = [], [], []
    # The figure and table numbers run on from part to part; see `renumber`.
    # The counters advance whether or not the part was actually written, so
    # one document open in Word cannot shift the numbering of the rest.
    fig_at, tbl_at = 0, 0
    _check_conclusions()
    for spec in parts():
        number, slug, title, abstract, sections, figures = spec[:6]
        extra = dict(spec[6]) if len(spec) > 6 else {}
        n_tables = sum(1 for _h, ps in sections for p in ps
                       if isinstance(p, tuple))
        make = build
        if extra.pop("letter", False):
            make = build_letter
        elif extra.pop("title_page", False):
            make = build_title_page
        else:
            extra["fig_offset"], extra["tbl_offset"] = fig_at, tbl_at
        if number == "4":
            # the Results, read on their own, end with the entries they cite
            extra["references"] = cited_references(sections)
        span = numbering_span(fig_at, len(figures), tbl_at, n_tables)
        fig_at += len(figures)
        tbl_at += n_tables
        if stem(number, slug) in _figures.METHOD_PARTS:
            # 3.1 ... 3.4 are not written one by one any more; they go out
            # together as 3_Methods.docx, below.  Still counted above, so
            # the parts after them keep the paper's numbers.
            continue
        try:
            path, edited = make(number, slug, title, abstract, sections,
                                figures, **extra)
        except PermissionError as exc:
            # Open in Word.  Skip it and keep going rather than abandoning
            # the parts after it; the file on disk is simply left as it was.
            locked.append(os.path.basename(exc.filename or "?"))
            continue
        print("  %-46s %d sections, %-22s%s"
              % (os.path.basename(path), len(sections), span,
                 "   <- your edited copy was kept" if edited else ""))
        written.append(path)
        if edited:
            beside.append((number, slug, path))
    import make_methods                     # here: it imports this module
    try:
        path = make_methods.main()
        written.append(path)
        if path.endswith("_rebuilt.docx"):
            beside.append(("3", "Methods", path))
    except PermissionError as exc:
        locked.append(os.path.basename(exc.filename or "?"))
    print()
    print("%d documents -> %s" % (len(written), OUT_DIR))
    if beside:
        print()
        print("EDITED BY HAND, so NOT overwritten.  The rebuild is beside")
        print("each one; merge what you want and delete the _rebuilt copy:")
        for number, slug, path in beside:
            print("    %s   kept" % (stem(number, slug) + ".docx"))
            print("    %s   new" % os.path.basename(path))
    if locked:
        print()
        print("NOT written, open in Word (close them and re-run):")
        for name in locked:
            print("   ", name)
    return written


if __name__ == "__main__":
    main()
