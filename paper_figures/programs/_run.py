# -*- coding: utf-8 -*-
"""
_run.py — which run, which budget, which device every paper figure uses.

One place, so the figure scripts never disagree about what they are drawing.

THE HEADLINE BUDGET IS 8 rays x 60 points (since 2026-09-27).
At train_seed 0 four of the fifteen trainings never left their plateau
(best validation F1@1 0.416-0.438):

    5_rays_60_points   6_rays_50_points   7_rays_60_points   8_rays_60_points

scripts/run_10_retrain_collapsed.py retrained them, taking for each the
first of seeds 1, 2, 3 whose best validation F1@1 reaches COLLAPSE_VAL_F1,
and wrote its choice to retrain_collapsed.json in the run folder.  SEEDS
below reads that file, and model_dir() / eval_dir() point every reader at
the chosen seed's folders (model_seed<k>/, evaluation_seed<k>/).  With
them, all fifteen converged and 8 x 60 is the best (F1@1 0.817 at 4.7 %).
Before, 8 x 50 was the headline as the best budget that converged.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # paper_figures/programs
PKG = os.path.dirname(HERE)                         # paper_figures
ROOT = os.path.dirname(PKG)                         # the project root
sys.path.insert(0, os.path.join(ROOT, "src"))

RUN = os.environ.get("CSD_RUN", "4-5-6-7-8_rays_40-50-60_points_500_samples")
RUN_DIR = os.path.join(ROOT, "results", RUN)
CONFIG = os.environ.get("CSD_CONFIG", "8_rays_60_points_500_samples")
CFG_DIR = os.path.join(RUN_DIR, CONFIG)
POOL = os.path.join(ROOT, "data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")

# A run that never left its initial plateau tops out near 0.42 on the
# validation devices; the worst healthy run reaches 0.660.  Nothing in this
# sweep lies between, so the cut is unambiguous.
#
# This is deliberately a VALIDATION number, carved out of the training
# devices by grid_train before training: deciding which runs are usable must
# not look at the test set.  final_train_loss will NOT do -- it is the last
# epoch's loss, while the checkpoint is the best-validation epoch, so 4 x 50
# and 7 x 50 end high (1.755, 1.630) on perfectly good models.
COLLAPSE_VAL_F1 = 0.55

N_RAYS = int(CONFIG.split("_")[0])
N_POINTS = int(CONFIG.split("_rays_")[1].split("_")[0])


def cfg_json(name=CONFIG):
    with open(os.path.join(RUN_DIR, name, "config.json")) as fh:
        return json.load(fh)


def comparison_rows():
    with open(os.path.join(RUN_DIR, "comparison.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def _chosen_seeds():
    """{configuration: train_seed} for the cells run_10 retrained."""
    p = os.path.join(RUN_DIR, "retrain_collapsed.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as fh:
        cells = json.load(fh)["cells"]
    return {n: c["chosen_seed"] for n, c in cells.items() if c["chosen_seed"]}


SEEDS = _chosen_seeds()


def _suffix(name):
    seed = SEEDS.get(name, 0)
    return "" if seed == 0 else "_seed%d" % seed


def model_dir(name=CONFIG):
    return os.path.join(RUN_DIR, name, "model" + _suffix(name))


def eval_dir(name=CONFIG):
    return os.path.join(RUN_DIR, name, "evaluation" + _suffix(name))


def training_summary(name):
    p = os.path.join(model_dir(name), "training_summary.json")
    with open(p) as fh:
        return json.load(fh)


def best_val_f1(name):
    return float(training_summary(name)["best_val_f1"])


def converged(name=None):
    """True when that config's training actually fitted the data."""
    if name is None:
        return {r["configuration"]: best_val_f1(r["configuration"])
                >= COLLAPSE_VAL_F1 for r in comparison_rows()}
    return best_val_f1(name) >= COLLAPSE_VAL_F1


def threshold(name=CONFIG):
    row = next(r for r in comparison_rows() if r["configuration"] == name)
    return float(row["threshold"])


# How close to the test mean a device must score to count as representative.
REPRESENTATIVE = 0.02


def picked_device(name=CONFIG):
    """
    The held-out device the panels show.  Returns
    (sample_dir, its F1@1, the test mean, how many test devices).

    TWO conditions, in this order:

      1. REPRESENTATIVE -- its F1@1 is within REPRESENTATIVE of the test
         mean.  Never the best device: a panel has to show typical
         behaviour, not the run's luckiest draw.
      2. LEGIBLE -- of those, the one with the FEWEST true line pixels.

    Line density varies 17-fold across this test set (102 to 1715 pixels),
    and it is set by the device's charging energies, not by how well the
    model did.  A dense device fills the panel with a fine honeycomb that
    cannot be read from the back of a room, while telling the reader nothing
    a sparse one does not.  So density is chosen for legibility and the
    score is what is held representative.

    Set CSD_DEVICE=sample_157 to override.
    """
    path = os.path.join(eval_dir(name), "per_device.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    mean = sum(float(r["f1@1"]) for r in rows) / len(rows)
    want = os.environ.get("CSD_DEVICE")
    if want:
        row = next(r for r in rows if r["sample"] == want)
    else:
        near = [r for r in rows
                if abs(float(r["f1@1"]) - mean) <= REPRESENTATIVE]
        if not near:                      # nothing close: fall back to rank
            near = [sorted(rows, key=lambda r: float(r["f1@1"]))[len(rows) // 2]]
        row = min(near, key=lambda r: int(r["true_line_pixels"]))
    return (os.path.join(POOL, row["sample"]), float(row["f1@1"]), mean,
            len(rows))


DEVICE, DEVICE_F1, TEST_MEAN_F1, N_TEST = picked_device()
THRESHOLD = threshold()
