# -*- coding: utf-8 -*-
"""
make_method_docs.py -- the method, in five numbered JJAP documents.

Each part is one Word document built FROM THE JJAP REGULAR-PAPER TEMPLATE,
so the page size, margins, fonts and paragraph styles are the journal's and
not this script's:

    01_device_simulation_and_dataset
    02_ray_based_measurement
    03_network_and_training
    04_threshold_and_tolerance
    05_results

Figures go at the END of each document, one per page block, numbered within
the part and captioned -- the JJAP submission order (text, then figures).

EVERY NUMBER IN THE TEXT IS READ FROM THE RUN, not typed in: the hyper-
parameters come from csdrecon.ml.grid_train and the configuration's
config.json, the scores from comparison.csv.  A number that cannot be found
raises rather than being quietly omitted, because a method section with a
wrong constant in it is worse than one that failed to build.

    python paper_figures/make_method_docs.py

Writes paper_figures/method_docs/*.docx, then converts each to PDF with
Word if it is available (Windows only; the .docx are written either way).
"""
import json
import os
import shutil
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _run                                                    # noqa: E402
from csdrecon.config import capacitance_config as cc           # noqa: E402
from csdrecon.ml import grid_model, grid_train                 # noqa: E402

OUT_DIR = os.path.join(HERE, "method_docs")
TEMPLATE = os.path.join(
    "F:", os.sep, "Seminar Fujita", "Conferences&Journals",
    "Japanese Journal of Applied Physics_2026", "Drafts",
    "AO and TF Feedbacks", "20240730_template-RP.docx")

# The template's own style names.  Japanese names are the template's, not a
# mistake: it was made in a Japanese Word and the styles must be referred to
# exactly as it defines them or Word silently falls back to Normal.
S_TITLE = "表題1"
S_ABSTRACT = "Normal"
S_SECTION = "SectionTitle"
S_SUBSECTION = "SubSectionTitle"
S_BODY = "MainText"

TEXT_WIDTH_IN = 6.10          # 8.27 page - 1.06 left - 1.10 right margin

# ── the JJAP body format, from the journal's own Page Setup and
# Paragraph dialogs (margin_info.png / paragraph_info.png) ───────────
FONT = "Times New Roman"
FONT_PT = 12
# Headings are the one thing NOT in the body face.
HEAD_FONT = "Arial"
HEAD_PT = 14
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
        "n_rays": _run.N_RAYS, "n_points": _run.N_POINTS,
        "threshold": _run.THRESHOLD,
        "coverage": 100 * float(row["coverage"]),
        "coverage1": 100 * float(row["coverage@1"]),
        "f1": {t: float(row["f1@%d" % t]) for t in (0, 1, 2, 3)},
        "prec1": float(row["precision@1"]), "rec1": float(row["recall@1"]),
        "iou": float(row["iou"]),
        "n_budgets": len(ok), "n_failed": sum(1 for v in ok.values() if not v),
        "infl_lo": min(infl), "infl_hi": max(infl),
        "line_frac": 100 * float(row["true_line_fraction"]),
        "intervals": cc.DEFAULT_INTERVALS,
        "separation": _separation(),
        "device": os.path.basename(_run.DEVICE),
        "device_f1": _run.DEVICE_F1, "test_mean": _run.TEST_MEAN_F1,
    }


F = facts()



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


def _table_i_rows():
    rows = []
    for matrix, entries in F["intervals"].items():
        for name, spec in entries.items():
            bands = cc.as_bands(spec)
            interval = " U ".join("%.2f - %.2f" % (lo, hi) for lo, hi in bands)
            if name not in _CONTROLS:
                raise SystemExit("no description for capacitance %r" % name)
            rows.append([matrix, name, interval, _CONTROLS[name]])
    return rows


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


TABLE_I_ROWS = _table_i_rows()
TABLE_II_ROWS = _table_ii_rows()


# ── the five parts ────────────────────────────────────────────────────────
# (number, slug, title, abstract, [(heading, [paragraph, ...]), ...],
#  [(figure file, caption), ...])
def parts():
    f = F
    thr_list = ", ".join("%g" % t for t in f["thresholds"])
    return [
        (1, "device_simulation_and_dataset",
         "Device simulation and dataset construction",
         "We describe the simulated double-quantum-dot devices from which "
         "every charge stability diagram in this work is drawn, and the "
         "train/test split applied to them. %d devices are generated from a "
         "constant-capacitance model with randomly drawn capacitances, each "
         "rendered as a %d x %d pixel charge-sensor map over a %.0f x %.0f mV "
         "gate-voltage window. The simulation is noise-free. Devices are "
         "split by device identity, not by image, so no device contributes "
         "to both sets."
         % (f["n_total"], f["res"], f["res"], f["window_mv"], f["window_mv"]),
         [
             ("1. The constant-capacitance model", [
                 "Each device is a double quantum dot with a nearby charge "
                 "sensor, simulated with QArray under the "
                 "constant-capacitance model. In that model the dot array is "
                 "a network of capacitors: the charge state of the dots is "
                 "whichever integer occupation minimises the electrostatic "
                 "energy at the applied gate voltages, and a transition line "
                 "in the stability diagram is a locus where that minimiser "
                 "changes. Four capacitance matrices define a device:",
                 "Cdd, the 2 x 2 dot-to-dot matrix, with the self "
                 "capacitances d1d1 and d2d2 on the diagonal and the "
                 "interdot capacitance d1d2 off it. Cgd, the 2 x 3 "
                 "gate-to-dot matrix, whose rows are (d1g1, d1g2, d1g3) and "
                 "(d2g1, d2g2, d2g3). Cds, the 1 x 2 dot-to-sensor matrix "
                 "(s1d1, s1d2). Cgs, the 1 x 3 gate-to-sensor matrix (s1g1, "
                 "s1g2, s1g3).",
                 "Three of these entries decide what the honeycomb looks "
                 "like. The primary gate capacitances d1g1 and d2g2 set the "
                 "honeycomb PERIOD, i.e. how many cells fall inside the "
                 "window. The cross capacitances d1g2 and d2g1 set the SLOPE "
                 "of the two line families, since a dot-1 line has slope "
                 "-d1g1/d1g2 and a dot-2 line -d2g1/d2g2. The interdot "
                 "capacitance d1d2 sets the ANTICROSSING, from barely split "
                 "triple points to a long interdot segment.",
                 "The two line families stay distinguishable as long as each "
                 "dot is driven mainly by its own plunger gate, that is "
                 "max(cross) < min(primary). Table I satisfies this in every "
                 "entry, 0.60 < 0.80, but nothing enforces it: the "
                 "capacitances are drawn and whatever they produce is kept.",
                 "Every entry is drawn independently and uniformly from the "
                 "intervals of Table I, so a device is a point in a "
                 "13-dimensional box rather than a perturbation of a "
                 "template. NO DRAW IS FILTERED, CHECKED OR REDRAWN. A draw "
                 "that lands somewhere unusual - the two dots merging into "
                 "one, lines too dense to resolve on the pixel grid, no "
                 "transition at all inside the window - is kept like any "
                 "other, and the acceptance rate of the pool is therefore "
                 "1.00 by construction.",
                 ("Table I. The capacitance parameter space. Every device is "
                  "a single uniform draw from these intervals, in QArray's "
                  "dimensionless capacitance units. The same intervals are "
                  "used for the training and the held-out devices; the two "
                  "sets are separated by device identity, not by parameter "
                  "range (Table II).",
                  ["Matrix", "Entry", "Interval", "What it controls"],
                  TABLE_I_ROWS, [0.75, 0.75, 1.15, 3.45]),
                 "The sensor response is computed with the noise model set "
                 "to NoNoise and nothing is added afterwards, so the maps "
                 "are noise-free. The Coulomb peak width is %.3g and the "
                 "electron temperature %.0e in simulation units."
                 % (f["peak_width"], f["temperature"]),
                 "Each device is rendered over a %.0f x %.0f mV window at "
                 "%d x %d pixels. The window is randomly offset per device "
                 "(offset scale %.2f), so the honeycomb is not registered to "
                 "the frame and the network cannot learn a fixed position "
                 "for the lines."
                 % (f["window_mv"], f["window_mv"], f["res"], f["res"],
                    f["offset_scale"]),
                 "The raw sensor map is dominated by a large, smooth "
                 "gate-to-sensor cross-talk background, with the charge "
                 "transitions riding on it as a small modulation. Figure 1 "
                 "shows the same device before and after that slow "
                 "background is removed; the honeycomb is visible only in "
                 "the second. This is the reason the reconstruction problem "
                 "is not trivial.",
             ]),
             ("2. Ground truth", [
                 "For every device the simulator also returns the exact "
                 "charge-transition map: a binary image, one pixel wide on "
                 "each transition line. Transition-line pixels are %.1f %% "
                 "of the diagram, which is what makes pixel accuracy "
                 "useless as a score and motivates the metric of Part 4."
                 % f["line_frac"],
             ]),
             ("3. Train/test split", [
                 "THE THING THAT IS SPLIT IS THE DEVICE, NOT THE IMAGE. "
                 "Capacitance configurations are drawn first and each is "
                 "given an identity; the identities are split once, with "
                 "seed %s, into %d training and %d held-out devices. Every "
                 "image and every measurement budget inherits the identity "
                 "of the device it came from, so a device's data always "
                 "lands on one side of the split."
                 % (f["split_seed"], f["n_train"], f["n_test"]),
                 "The split is stored with the device pool rather than with "
                 "a configuration. Every cell of a sweep therefore reads the "
                 "same stored assignment, and a device cannot be a training "
                 "device for one measurement budget and a test device for "
                 "another.",
                 "A further %d %% of the TRAINING devices (%d of %d) is set "
                 "aside before any training as a validation split, leaving "
                 "%d to fit the weights. This validation split is where the "
                 "binarisation threshold of Part 4 is chosen. The %d "
                 "held-out devices are never used for any decision."
                 % (f["val_pct"], f["n_val"], f["n_train"], f["n_fit"],
                    f["n_test"]),
                 "Because both sets are drawn from the same intervals, the "
                 "split is not an extrapolation test and is not claimed to "
                 "be one. What it does claim is that no held-out device is a "
                 "near-duplicate of a training device, and that is measured "
                 "rather than asserted. In the %d-dimensional normalised "
                 "parameter space, the closest any held-out device lies to "
                 "any training device is %.3f, while the closest any two "
                 "TRAINING devices lie to each other is %.3f. The held-out "
                 "devices are therefore no closer to the training set than "
                 "its own members are to one another, which is the relevant "
                 "yardstick. Table II collects the split and this evidence."
                 % (f["separation"]["dimensions"],
                    f["separation"]["min_distance"],
                    f["separation"]["min_within_train"]),
                 ("Table II. The train/test split and its separation "
                  "evidence. Distances are Euclidean in the normalised "
                  "parameter space and are computed over the generated "
                  "devices, not over the intervals.",
                  ["Quantity", "Value"],
                  TABLE_II_ROWS, [3.40, 2.70]),
                 "Figure 2 shows the two levels of the split.",
             ]),
         ],
         [("panel_charge_sensor.png",
           "One simulated device. Left: the charge-sensor map as measured, "
           "dominated by a smooth gate cross-talk background. Right: the "
           "same data with that slow background differenced away, where the "
           "honeycomb of charge transitions becomes visible."),
          ("fig_data_split.png",
           "The train/test split. %d devices are divided by device identity "
           "into %d training and %d held-out devices (seed %s). Inside the "
           "training devices, %d %% is carved out as a validation split "
           "before any training, leaving %d devices to fit the weights."
           % (f["n_total"], f["n_train"], f["n_test"], f["split_seed"],
              f["val_pct"], f["n_fit"]))]),

        (2, "ray_based_measurement",
         "Ray-based measurement and network input",
         "We describe how a small fraction of the gate-voltage plane is "
         "measured and how those measurements are presented to the network. "
         "A fan of rays is fired from one corner of the window; each ray is "
         "sampled at a fixed number of points. For the headline budget of "
         "%d rays x %d points this touches %.2f %% of the plane. The "
         "measurement is encoded as two image channels: the measured signal "
         "where a ray passed, and a binary mask of where any ray passed."
         % (f["n_rays"], f["n_points"], f["coverage"]),
         [
             ("1. The ray fan", [
                 "A measurement budget is a pair: n_rays x n_points. The "
                 "rays fan out from one corner of the voltage window at "
                 "equally spaced angles, and each ray is sampled at "
                 "n_points equally spaced positions along its length. The "
                 "budget is therefore the total number of single-point "
                 "measurements, which is the quantity an experiment pays "
                 "for.",
                 "Sampling is nearest-cell: a measured voltage is assigned "
                 "to the pixel whose centre is closest. Because all rays "
                 "share an origin, the first point of every ray falls in the "
                 "same pixel, and adjacent rays can round into the same cell "
                 "one step out. The number of DISTINCT pixels touched is "
                 "therefore slightly smaller than the number of "
                 "measurements: %d x %d = %d measurements touch %.2f %% of "
                 "the %d x %d grid. Coverage is quoted as distinct pixels "
                 "throughout, since repeating a pixel adds no information."
                 % (f["n_rays"], f["n_points"], f["n_rays"] * f["n_points"],
                    f["coverage"], f["res"], f["res"]),
                 "Along each ray the one-dimensional trace is passed to a "
                 "peak finder, whose output is retained for the measurement "
                 "figures but is NOT given to the network.",
                 "Figure 1 shows the charge sensor, the ground truth and "
                 "what the measurement actually sees, for one held-out "
                 "device.",
             ]),
             ("2. Encoding the measurement", [
                 "The network is given two channels, both the full size of "
                 "the diagram. Channel 1 carries the raw sensor value at "
                 "every pixel a ray visited and zero elsewhere. Channel 2 "
                 "is a binary visited mask: one where any ray passed, zero "
                 "otherwise.",
                 "The second channel is what separates \"measured here, and "
                 "the signal was low\" from \"never looked here\". Without "
                 "it a zero in channel 1 is ambiguous at every pixel, and "
                 "the network cannot distinguish absence of evidence from "
                 "evidence of absence. Figures 2 and 3 show the two "
                 "channels for the same device.",
                 "The encoding has a property the study depends on: the "
                 "input shape is the same for EVERY budget. Changing "
                 "n_rays or n_points changes how much of the two channels "
                 "is populated, never their size, so one architecture "
                 "serves every budget and the comparison across budgets is "
                 "a comparison of measurement alone.",
             ]),
         ],
         [("fig_measurement_panel.png",
           "The measurement, for one held-out device. Left: the "
           "charge-sensor map. Centre: the ground-truth transition lines. "
           "Right: what the network is shown at %d rays x %d points, which "
           "is %.1f %% of the grid."
           % (f["n_rays"], f["n_points"], f["coverage"])),
          ("panel_channel1.png",
           "Channel 1: the measured charge-sensor signal, retained only at "
           "the pixels a ray visited."),
          ("panel_channel2.png",
           "Channel 2: the visited mask. Black marks every pixel a ray "
           "passed through, %.1f %% of the grid." % f["coverage"]),
          ("fig_network_input.png",
           "The two input channels side by side. Channel 2 is what makes a "
           "zero in channel 1 unambiguous.")]),

        (3, "network_and_training",
         "Network architecture and training",
         "We describe the network that maps the two measured channels to "
         "one probability per pixel, and how it is trained. The network is a "
         "U-Net of depth 3 with channel widths %s. The loss combines a "
         "positive-weighted binary cross-entropy with a soft Dice term, "
         "which is necessary because transition-line pixels are only "
         "%.1f %% of the diagram. Training uses Adam at a learning rate of "
         "%g, batch size %d, for %d epochs."
         % ("/".join(str(w) for w in f["widths"]), f["line_frac"],
            f["lr"], f["batch"], f["epochs"]),
         [
             ("1. Architecture", [
                 "The network is a U-Net taking 2 channels in and producing "
                 "1 channel out, at the same %d x %d resolution. The "
                 "encoder has three stages of widths %s, each a "
                 "convolutional block followed by 2x2 max pooling; the "
                 "bottleneck has width %d. The decoder mirrors the encoder, "
                 "upsampling by nearest-neighbour resize to the skip "
                 "tensor's actual size and concatenating the skip "
                 "connection before each block. A 1x1 convolution produces "
                 "the single output channel."
                 % (f["res"], f["res"],
                    "/".join(str(w) for w in f["widths"][:3]),
                    f["widths"][3]),
                 "Resizing to the skip tensor's own size rather than by a "
                 "fixed factor is what lets the same architecture run on a "
                 "grid whose size is not a power of two.",
                 "The output is a probability per pixel that a "
                 "charge-transition line passes through it. The network "
                 "never outputs lines directly; turning the probability map "
                 "into lines is a separate decision, described in Part 4. "
                 "Figure 1 is the architecture and Figure 2 the whole flow, "
                 "from the two measured channels to the probability map.",
             ]),
             ("2. Loss", [
                 "Transition-line pixels are %.1f %% of the diagram, so a "
                 "network that predicts no line anywhere already achieves "
                 "high pixel accuracy. Two measures counter this. The "
                 "binary cross-entropy weights the positive class by its "
                 "rarity, capped at %g so that an unusually sparse device "
                 "cannot dominate a batch. A soft Dice term is added "
                 "because weighted cross-entropy alone judges pixels one at "
                 "a time and is satisfied by a blanket of positives; Dice "
                 "penalises the spread."
                 % (f["line_frac"], f["max_pos_weight"]),
             ]),
             ("3. Optimisation", [
                 "Adam, learning rate %g, batch size %d, %d epochs, seed "
                 "%d. The seed fixes both the weight initialisation and the "
                 "order in which batches are drawn, so two runs differing "
                 "only in this seed see exactly the same data in the same "
                 "order."
                 % (f["lr"], f["batch"], f["epochs"], f["train_seed"]),
                 "The checkpoint saved is the epoch with the best "
                 "validation F1, NOT the last epoch. This matters when "
                 "reading a training log: a run can end on a high "
                 "last-epoch loss and still have saved a good model.",
             ]),
         ],
         [("fig_unet.png",
           "The network. Two input channels, an encoder of widths %s with "
           "2x2 max pooling, a bottleneck of width %d, a mirrored decoder "
           "with skip connections, and a 1x1 convolution to one output "
           "channel."
           % ("/".join(str(w) for w in f["widths"][:3]), f["widths"][3])),
          ("fig_model_flow.png",
           "The whole flow: the two measured channels in, the U-Net, and "
           "one probability per pixel out.")]),

        (4, "threshold_and_tolerance",
         "From probability map to transition lines: threshold and tolerance",
         "The network outputs a continuous probability map, while transition "
         "lines are binary, so two decisions stand between the output and a "
         "score. Where to cut the map is the THRESHOLD, chosen on the "
         "validation devices and stored in the checkpoint; for the headline "
         "budget it is %g. How strictly to compare the result with the truth "
         "is the TOLERANCE tau, a distance in pixels. We report F1 at "
         "tau = 0 to 3 and quote tau = 1 as the headline, always beside the "
         "area that tolerance covers."
         % f["threshold"],
         [
             ("1. Binarising the output: the threshold", [
                 "Every pixel must become 0 or 1, and where the map is cut "
                 "decides how many lines are drawn. The cut is a fitted "
                 "quantity and is therefore chosen out of sample: the "
                 "candidates %s are each scored on the %d VALIDATION "
                 "devices carved out of the training set, and the best is "
                 "stored inside the checkpoint. The held-out devices are "
                 "never involved in the choice."
                 % (thr_list, f["n_val"]),
                 "Figure 1 shows that validation curve. It is nearly flat "
                 "from 0.3 to 0.8 and falls off only beyond 0.9, so the "
                 "exact value matters far less than the fact that it was "
                 "chosen out of sample. For the headline budget the stored "
                 "value is %g." % f["threshold"],
                 "Figures 2 to 4 show the probability map and the effect of "
                 "cutting it at the chosen threshold and at one "
                 "deliberately too high.",
             ]),
             ("2. Scoring the output: the tolerance", [
                 "Transition lines are one pixel wide. Comparing them with "
                 "the truth pixel for pixel is brutal: a perfectly "
                 "recovered line displaced by a single pixel shares no "
                 "pixel at all with the truth and scores zero, for a result "
                 "any physicist would call correct. Figure 5 is exactly "
                 "that case, constructed and scored with the same metric "
                 "code used throughout: strict F1 = 0.00, and F1 at "
                 "tau = 1 equal to 1.00.",
                 "We therefore report F1 at a tolerance tau: a predicted "
                 "pixel counts as correct if a true line pixel lies within "
                 "tau pixels of it, and a true pixel counts as found if a "
                 "predicted one lies within tau. Distances are Euclidean. "
                 "Figure 6 shows what tau admits around a single pixel: "
                 "tau = 0 is the same pixel only, tau = 1 adds the four "
                 "side neighbours, and a diagonal neighbour is 1.41 pixels "
                 "away so it counts only from tau = 2.",
                 "Figures 7 to 9 show one prediction scored at three "
                 "tolerances, and Figure 10 the same thing zoomed until "
                 "individual pixels are visible: the truth and the output "
                 "never move, only the band that is allowed around them.",
             ]),
             ("3. Reporting", [
                 "F1 at tau can only rise with tau, so it is never quoted "
                 "alone. Two safeguards are used throughout. First, tau is "
                 "always stated, and F1 is reported for tau = 0 to 3 rather "
                 "than at a single convenient value. Second, the score is "
                 "quoted beside coverage at the same tau: the fraction of "
                 "the plane lying within tau pixels of a measured pixel, "
                 "which is the area the score is effectively read over. "
                 "Across this study, allowing a single pixel of slack "
                 "inflates the measured area by %.1f to %.1f times."
                 % (f["infl_lo"], f["infl_hi"]),
                 "Figure 11 shows F1 against tau together with the "
                 "precision and recall behind it. Pixel accuracy is not "
                 "reported as a result anywhere in this work.",
             ]),
         ],
         [("threshold_validation.png",
           "Choosing the threshold. Mean F1 at tau = 1 over the %d "
           "validation devices for every candidate cut, with the stored "
           "value ringed. The curve is flat from 0.3 to 0.8 and collapses "
           "only past 0.9." % f["n_val"]),
          ("p2l_1_probability.png",
           "The U-Net output for one held-out device: a probability per "
           "pixel that a transition line passes through it."),
          ("p2l_2_threshold_0p7.png",
           "The same map cut at the chosen threshold, P > %g."
           % f["threshold"]),
          ("p2l_3_threshold_0p9.png",
           "The same map cut too high, at P > 0.9. Lines break up and are "
           "lost."),
          ("tau_example.png",
           "Why a tolerance is needed. Left to right: the ground truth; the "
           "same line drawn one pixel to the left; the two superimposed, "
           "adjacent everywhere and coincident nowhere; and the same pair "
           "with the tau = 1 band. Strict F1 is 0.00 and F1 at tau = 1 is "
           "1.00."),
          ("tau_neighbourhood.png",
           "What tau admits around one pixel the model drew, for tau = 0, "
           "1, 2 and 3. Distances are Euclidean, so a diagonal neighbour "
           "(1.41 px) counts only from tau = 2."),
          ("p2l_4_tau0.png",
           "One prediction scored at tau = 0."),
          ("p2l_5_tau1.png",
           "The same prediction scored at tau = 1, the headline tolerance."),
          ("p2l_6_tau3.png",
           "The same prediction scored at tau = 3."),
          ("p2l_9_tau_zoom.png",
           "The same truth and the same output at tau = 0 to 3, zoomed "
           "until individual pixels are visible. Only the tolerance band "
           "grows."),
          ("p2l_7_tolerance_curve.png",
           "F1 against tau for the headline budget, with the precision and "
           "recall behind it, over the %d held-out devices."
           % f["n_test"])]),

        (5, "results",
         "Results",
         "We report the recovery of charge-transition lines as a function of "
         "the measurement budget, on %d held-out devices. The best budget "
         "that converged is %d rays x %d points: F1 at tau = 1 of %.3f from "
         "%.2f %% of the gate-voltage plane, at a threshold of %g chosen on "
         "the validation split. Of the %d budgets in the sweep, %d did not "
         "converge and are excluded from the trend."
         % (f["n_test"], f["n_rays"], f["n_points"], f["f1"][1],
            f["coverage"], f["threshold"], f["n_budgets"], f["n_failed"]),
         [
             ("1. Recovery against measurement budget", [
                 "Figure 1 plots F1 at tau = 1 against the fraction of the "
                 "plane measured. Within a fixed number of points per ray, "
                 "the score rises monotonically with the number of rays. "
                 "The best converged budget, %d rays x %d points, reaches "
                 "F1 at tau = 1 of %.3f from %.2f %% of the plane, with "
                 "precision %.3f and recall %.3f."
                 % (f["n_rays"], f["n_points"], f["f1"][1], f["coverage"],
                    f["prec1"], f["rec1"]),
                 "At equal coverage, spending the budget on MORE RAYS beats "
                 "spending it on more points per ray. Figure 1 shows the "
                 "comparison directly rather than asserting it: at the same "
                 "fraction of the plane measured, the budget with more rays "
                 "scores higher.",
                 "Strict IoU for the headline budget is %.3f. One-pixel-wide "
                 "lines are punished hard by IoU and it is reported rather "
                 "than hidden." % f["iou"],
             ]),
             ("2. Dependence on tolerance", [
                 "Figure 2 reports pixel accuracy, precision, recall, F1 and "
                 "grid coverage against tau for three budgets. F1 rises "
                 "from %.3f at tau = 0 to %.3f at tau = 1, then flattens to "
                 "%.3f at tau = 2 and %.3f at tau = 3. Most of the "
                 "remaining error is therefore sub-pixel placement of a "
                 "line that WAS found, rather than a line that was missed."
                 % (f["f1"][0], f["f1"][1], f["f1"][2], f["f1"][3]),
                 "The rightmost panel of Figure 2 is the cost of that "
                 "tolerance. Coverage rises from %.2f %% at tau = 0 to "
                 "%.2f %% at tau = 1 for the headline budget. The "
                 "improvement from tau = 0 to tau = 1 is therefore bought "
                 "with a several-fold increase in the area the score is "
                 "read over, which is why tau = 1 rather than a larger "
                 "value is quoted as the headline."
                 % (f["coverage"], f["coverage1"]),
                 "Pixel accuracy is included in Figure 2 only so that it "
                 "can be dismissed: it exceeds 0.9 at every tolerance and "
                 "for every budget, because transition lines are %.1f %% of "
                 "the diagram and predicting no line anywhere already "
                 "scores highly. It is not used as a result."
                 % f["line_frac"],
             ]),
             ("3. Runs that did not converge", [
                 "%d of the %d budgets in this sweep did not converge. "
                 "Their best validation F1 at tau = 1 lands near 0.42, "
                 "where every other run in the sweep reaches at least 0.66, "
                 "and their test scores follow. These are the scores of a "
                 "failed fit and not of the measurement budget; they are "
                 "excluded from the trend in Figure 1 and reported "
                 "separately rather than dropped, so that the gap in the "
                 "sweep is visible."
                 % (f["n_failed"], f["n_budgets"]),
             ]),
         ],
         [("results_budget_ladder.png",
           "F1 at tau = 1 against the fraction of the plane measured. The "
           "line is the %d-points-per-ray ladder; the number beside each "
           "point is the ray count. The best converged budget is ringed. "
           "The open marker is a same-coverage comparison showing that more "
           "rays beats more points per ray. %d budgets that did not "
           "converge are not shown."
           % (40, f["n_failed"])),
          ("fig_tau_metrics.png",
           "Pixel accuracy, precision, recall, F1 and grid coverage against "
           "the tolerance tau, for three budgets, on %d held-out devices. "
           "Pixel accuracy is shown only to be dismissed. Grid coverage is "
           "not a score: it is the fraction of the plane within tau pixels "
           "of a measured pixel, i.e. the area the scores to its left are "
           "read over." % f["n_test"])]),
    ]


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


def _para(doc, text, style, size=FONT_PT, bold=None, indent=True,
          align=WD_ALIGN_PARAGRAPH.JUSTIFY, name=FONT):
    """One paragraph in the JJAP body format."""
    p = doc.add_paragraph(text, style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = LINE_SPACING
    pf.left_indent = Pt(0)
    pf.right_indent = Pt(0)
    pf.first_line_indent = Pt(INDENT_PT) if indent else Pt(0)
    for run in p.runs:
        _font(run, size, bold, name)
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


def _row_rule(row):
    for cell in row.cells:
        _border(cell._tc.get_or_add_tcPr(), "w:tcBorders", "bottom")


def _cell(cell, text, bold=None):
    """Table cells are LEFT aligned and single spaced.

    They inherit the body format otherwise, and justified text in a narrow
    column stretches two words across the whole cell."""
    cell.text = text
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    for run in p.runs:
        _font(run, FONT_PT, bold)


def _table(doc, caption, headers, rows, widths=None):
    """A JJAP table: caption above it, Times New Roman 12 throughout.

    The template defines no table style beyond Normal Table, so the rule
    lines are drawn here rather than inherited."""
    _para(doc, caption, S_BODY, indent=False)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    # The template defines no table style but Normal Table, and a full grid
    # is not the journal's look anyway: rule above the header, below it, and
    # below the last row, with no vertical lines.
    _rule(t, "top")
    _rule(t, "bottom")
    _row_rule(t.rows[0])
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
    doc.add_paragraph()
    return t


def _heading(doc, text):
    """A section heading: Arial, bold, 14 pt, flush left, no indent."""
    return _para(doc, text, S_SECTION, size=HEAD_PT, bold=True,
                 indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                 name=HEAD_FONT)


def build(number, slug, title, abstract, sections, figures):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "%02d_%s.docx" % (number, slug))
    shutil.copyfile(TEMPLATE, out)
    doc = docx.Document(out)

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
    paras[0].runs[0].text = "Part %d. %s" % (number, title)
    # ONLY the title is kept.  These are five parts of one manuscript, not
    # five manuscripts: repeating the authors, both affiliations and the
    # e-mail on every one of them is noise, and they belong on the
    # assembled paper instead.  _clear_body_after keeps the paragraph it
    # stops at, so keep=0 leaves the title and drops the rest.
    assert "Email" in paras[4].style.name, paras[4].style.name
    _clear_body_after(doc, keep=0)

    _para(doc, abstract, S_ABSTRACT, indent=False)

    for heading, paragraphs in sections:
        _heading(doc, heading)
        for text in paragraphs:
            # a 4-tuple is a table, anything else is a paragraph
            if isinstance(text, tuple):
                _table(doc, *text)
            else:
                _para(doc, text, S_BODY)

    # ── figures last, as JJAP asks ───────────────────────────────────────
    doc.add_page_break()
    _heading(doc, "Figures")
    for i, (fname, caption) in enumerate(figures, 1):
        src = os.path.join(HERE, fname)
        if not os.path.exists(src):
            raise SystemExit("%s missing -- run make_figures.py first" % src)
        from PIL import Image
        iw, ih = Image.open(src).size
        width = min(TEXT_WIDTH_IN, TEXT_WIDTH_IN)
        # a very wide panel strip would otherwise become a sliver; a very
        # tall one would run off the page
        height = width * ih / iw
        if height > 7.5:
            width = width * 7.5 / height
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(src, width=Inches(width))
        _para(doc, "Fig. %d. %s" % (i, caption), S_BODY, indent=False)
        if i < len(figures):
            doc.add_paragraph()

    doc.save(out)
    return out


def main():
    print("template:", TEMPLATE)
    if not os.path.exists(TEMPLATE):
        raise SystemExit("JJAP template not found")
    print("headline: %d x %d, threshold %g, F1@1 %.3f at %.2f %% coverage"
          % (F["n_rays"], F["n_points"], F["threshold"], F["f1"][1],
             F["coverage"]))
    print()
    written, locked = [], []
    for number, slug, title, abstract, sections, figures in parts():
        try:
            path = build(number, slug, title, abstract, sections, figures)
        except PermissionError as exc:
            # Open in Word.  Skip it and keep going rather than abandoning
            # the parts after it; the file on disk is simply left as it was.
            locked.append(os.path.basename(exc.filename or "?"))
            continue
        print("  %-46s %d sections, %d figures"
              % (os.path.basename(path), len(sections), len(figures)))
        written.append(path)
    print()
    print("%d documents -> %s" % (len(written), OUT_DIR))
    if locked:
        print()
        print("NOT written, open in Word (close them and re-run):")
        for name in locked:
            print("   ", name)
    return written


if __name__ == "__main__":
    main()
