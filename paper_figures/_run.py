# -*- coding: utf-8 -*-
"""
_run.py — which run, which budget, which device every paper figure uses.

One place, so the figure scripts never disagree about what they are drawing.

THE HEADLINE BUDGET IS 8 rays x 50 points, NOT 8 x 60.
Four of the fifteen trainings in this sweep did not converge.  Their best
validation F1@1 lands at 0.416-0.438 where every healthy run reaches at
least 0.660, and their test F1@1 follows it down to ~0.43:

    5_rays_60_points   6_rays_50_points   7_rays_60_points   8_rays_60_points

8 x 60 is one of them, so it cannot carry the figures.  8 x 50 is the best
budget that did converge (F1@1 0.796 at 3.9 % of the plane).  `converged()`
re-derives that list from each config's training_summary.json rather than
hard-coding it, so the day those four are retrained the figures follow.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

RUN = os.environ.get("CSD_RUN", "4-5-6-7-8_rays_40-50-60_points_500_samples")
RUN_DIR = os.path.join(ROOT, "results", RUN)
CONFIG = os.environ.get("CSD_CONFIG", "8_rays_50_points_500_samples")
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


def training_summary(name):
    p = os.path.join(RUN_DIR, name, "model", "training_summary.json")
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
    path = os.path.join(RUN_DIR, name, "evaluation", "per_device.csv")
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
