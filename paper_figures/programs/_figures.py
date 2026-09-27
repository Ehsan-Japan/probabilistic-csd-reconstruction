# -*- coding: utf-8 -*-
"""
_figures.py -- where each generated figure lives, and how to find it.

paper_figures/ used to be one flat folder of forty-odd PNGs.  They are now
filed by the part of the method they belong to, mirroring method_docs/:

    method_docs/3_methods/figures/     3_1 ... 3_4, the method, shared
                2_Introduction/figures/
                4_Results_and_Discussion/figures/
            deck/        the *_notitle variants, for the SSDM slides

ONE PLACE DECIDES, and both the writers and the readers use it:

    out_path(name)   where a generator should SAVE that figure
    find(name)       where a consumer should READ it from

find() searches every subfolder and then the old flat location, so a figure
that has not been regenerated yet is still found, and a consumer never has
to know which folder a figure ended up in.  A name that is not in LAYOUT is
an error rather than a silent drop into the root: a new figure should be
filed deliberately.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))   # paper_figures/programs
PKG = os.path.dirname(HERE)                         # paper_figures
DOCS_ROOT = os.path.join(PKG, "method_docs")
# Slide-only variants belong to no part of the method.
DECK_ROOT = os.path.join(PKG, "deck_figures")

# Parts 0, 1 and 2 (the cover letter, the title page and the introduction)
# carry no generated figures, so they have no constant here.  P1..P5 are the
# five parts that do, in their own order, not the manuscript's numbering.
# The introduction carries one figure -- the schematic of what the
# simulator is for -- so it needs a folder of its own like the rest.
P0 = "2_Introduction"
P1 = "3_1_Device_simulation_and_dataset"
P2 = "3_2_Ray_based_measurement"
P3 = "3_3_Network_and_training"
P4 = "3_4_Threshold_and_tolerance"
P5 = "4_Results_and_Discussion"
DECK = "deck"

FOLDERS = (P0, P1, P2, P3, P4, P5, DECK)

# The four subsections of the method share one folder on disk,
# method_docs/3_methods/, rather than one each.  P1..P4 stay as filing labels;
# this is only where they are put.
METHODS = "3_methods"
METHOD_PARTS = (P1, P2, P3, P4)


def part_folder(part):
    """The folder under method_docs/ that part `part` is written into."""
    return METHODS if part in METHOD_PARTS else part

# Figure (or sidecar) -> the folder it belongs in.  The five part folders
# hold what the corresponding method document uses; anything written only
# for the slides goes to deck/.
LAYOUT = {
    # ── the introduction ────────────────────────────────────
    "method_overview.png": P0,
    "method_overview.pdf": P0,
    "digital_twin.png": P0,
    # the tiles the Illustrator figure places
    "tile_diagram.png": P0,
    "tile_rays.png": P0,
    "tile_channels.png": P0,
    "tile_probability.png": P0,
    # fig_overview is not in the manuscript; it is still filed so that
    # make_figures can write it without a KeyError, and putting it back
    # is a one-line change in make_method_docs.
    "fig_overview.png": P0,
    "fig_overview.pdf": P0,
    # ── part 1 ───────────────────────────────────────────────────────────
    "dqd_model.png": P1,
    "dqd_model_full.png": P1,
    "dqd_model_full_bigb.png": P1,
    "dqd_model_full_bigb_v2.png": P1,
    "panel_charge_sensor.png": P1,
    "fig_inputs.png": P1,
    "panel_charge_sensor_big_b.png": P1,
    "fig_data_split.png": P1,
    "fig_data_split_notext.png": P1,
    "data_split_label.json": P1,
    # ── part 2 ───────────────────────────────────────────────────────────
    "fig_ray_fan.png": P2,
    "ray_fan_counts.json": P2,
    "fig_measurement_panel.png": P2,
    "panel_channel1.png": P2,
    "panel_channel2.png": P2,
    "fig_network_input.png": P2,
    "fig_network_input_slide.png": P2,
    # ── part 3 ───────────────────────────────────────────────────────────
    "fig_unet.png": P3,
    "fig_model_flow.png": P3,
    "panel_probability.png": P3,
    # ── part 4 ───────────────────────────────────────────────────────────
    "p2l_1_probability.png": P4,
    # named after the threshold they show: 0p7 was 8 x 50's, 0p6 is
    # 8 x 60's (the headline since 2026-09-27)
    "p2l_2_threshold_0p6.png": P4,
    "p2l_2_threshold_0p6_tau0.png": P4,
    "p2l_2_threshold_0p7.png": P4,
    "p2l_2_threshold_0p7_tau0.png": P4,
    "p2l_3_threshold_0p9.png": P4,
    "p2l_3_threshold_0p9_tau0.png": P4,
    "p2l_4_tau0.png": P4,
    "p2l_5_tau1.png": P4,
    "p2l_6_tau3.png": P4,
    "p2l_7_tolerance_curve.png": P4,
    "p2l_8_legend.png": P4,
    "p2l_9_tau_zoom.png": P4,
    "p2l_titles.json": P4,
    "tau_example.png": P4,
    "tau_neighbourhood.png": P4,
    "tau_layout.json": P4,
    "threshold_validation.png": P4,
    "fig_threshold_panels.png": P4,
    "threshold_validation.json": P4,
    "fig_probability_to_lines.png": P4,
    # ── part 5 ───────────────────────────────────────────────────────────
    "results_budget_ladder.png": P5,
    "fig_tau_metrics.png": P5,
    "prob_map_stats.json": P5,
    "results_f1_vs_coverage.png": P5,
}

# The *_notitle variants are built only so a slide can carry its own title
# box, so they are filed together rather than beside their titled twins.
for _name, _folder in list(LAYOUT.items()):
    if _name.endswith(".png"):
        LAYOUT.setdefault(_name.replace(".png", "_notitle.png"), DECK)


# Figures NOT produced by any generator here: drawn by hand in Adobe
# Illustrator and kept in docs/figures/.  make_method_docs copies them
# into the part that uses them, and no generator overwrites them.
STATIC = {
    # Drawn by hand, in PowerPoint for the SSDM talk, rather than
    # generated: copied in from docs/figures/ like the other two.
    # Drawn in Illustrator by docs/figures/build_overview.jsx, which
    # places the tile_*.png written from the run.
    "method_overview.png": os.path.join("docs", "figures",
                                        "method_overview.png"),
    "method_overview.pdf": os.path.join("docs", "figures",
                                        "method_overview.pdf"),
    "digital_twin.png": os.path.join("docs", "figures", "digital_twin.png"),
    "dqd_model.png": os.path.join("docs", "figures", "dqd_model.png"),
    "dqd_model_full.png": os.path.join("docs", "figures",
                                       "dqd_model_full.png"),
    # The same drawing with panel (b) enlarged: _bigb is 2.0x,
    # _bigb_v2 is 1.55x and is the one the document uses.
    "dqd_model_full_bigb.png": os.path.join("docs", "figures",
                                            "dqd_model_full_bigb.png"),
    "dqd_model_full_bigb_v2.png": os.path.join(
        "docs", "figures", "dqd_model_full_bigb_v2.png"),
}


def folder(name):
    """The folder `name` belongs in."""
    if name not in LAYOUT:
        raise KeyError(
            "%s is not filed in _figures.LAYOUT -- add it there so every "
            "figure has one home" % name)
    return LAYOUT[name]


def dir_for(name):
    """The folder `name` is written into.

    A part's figures sit beside that part's document, in
    method_docs/<part>/figures/, so the folder can be read or sent whole;
    the method's four parts share method_docs/3_methods/figures/.
    Slide-only variants have no part and go to deck_figures/."""
    f = folder(name)
    if f == DECK:
        return DECK_ROOT
    return os.path.join(DOCS_ROOT, part_folder(f), "figures")


def out_path(name, create=True):
    """Where a generator should write `name`."""
    d = dir_for(name)
    if create:
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def find(name, required=True):
    """Where `name` actually is, wherever it was filed."""
    candidates = []
    if name in LAYOUT:
        candidates.append(os.path.join(dir_for(name), name))
    candidates += [os.path.join(DOCS_ROOT, part_folder(f), "figures", name)
                   for f in FOLDERS if f != DECK]
    candidates.append(os.path.join(DECK_ROOT, name))
    # the layouts this folder had before
    candidates += [os.path.join(DOCS_ROOT, f, "figures", name)
                   for f in METHOD_PARTS]
    candidates += [os.path.join(PKG, "figures", f, name) for f in FOLDERS]
    candidates.append(os.path.join(PKG, name))
    if name in STATIC:
        candidates.append(os.path.join(os.path.dirname(PKG),
                                       STATIC[name]))
    for path in candidates:
        if os.path.isfile(path):
            return path
    if required:
        raise SystemExit(
            "%s not found under paper_figures/ -- run "
            "python paper_figures/make_figures.py" % name)
    return None
