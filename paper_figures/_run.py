# -*- coding: utf-8 -*-
"""
_run.py — which run, which budget, which device every paper figure uses.

One place, so the figure scripts never disagree about what they are drawing.

THE HEADLINE BUDGET IS 8 rays x 50 points, NOT 8 x 60.
Four of the fifteen trainings in this sweep did not converge -- they sit at
final train loss ~1.68 against ~0.37 for a healthy run, and their F1@1 lands
near 0.43 instead of ~0.8:

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

# A run that never left its initial plateau ends an order of magnitude above
# a healthy one; anything over this is not a weak result, it is a failure.
COLLAPSE_LOSS = 1.0

N_RAYS = int(CONFIG.split("_")[0])
N_POINTS = int(CONFIG.split("_rays_")[1].split("_")[0])


def cfg_json(name=CONFIG):
    with open(os.path.join(RUN_DIR, name, "config.json")) as fh:
        return json.load(fh)


def comparison_rows():
    with open(os.path.join(RUN_DIR, "comparison.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def train_loss(name):
    p = os.path.join(RUN_DIR, name, "model", "training_summary.json")
    with open(p) as fh:
        return float(json.load(fh)["final_train_loss"])


def converged(name=None):
    """True when that config's training actually fitted the data."""
    if name is None:
        return {r["configuration"]: train_loss(r["configuration"])
                < COLLAPSE_LOSS for r in comparison_rows()}
    return train_loss(name) < COLLAPSE_LOSS


def threshold(name=CONFIG):
    row = next(r for r in comparison_rows() if r["configuration"] == name)
    return float(row["threshold"])


def picked_device(name=CONFIG):
    """
    The held-out device the panels show: the MEDIAN performer of the test
    set, not the best one.  Returns (sample_dir, its F1@1, the test mean).

    Set CSD_DEVICE=sample_11 to override.
    """
    path = os.path.join(RUN_DIR, name, "evaluation", "per_device.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["f1@1"]))
    want = os.environ.get("CSD_DEVICE")
    row = (next(r for r in rows if r["sample"] == want) if want
           else rows[len(rows) // 2])
    mean = sum(float(r["f1@1"]) for r in rows) / len(rows)
    return (os.path.join(POOL, row["sample"]), float(row["f1@1"]), mean,
            len(rows))


DEVICE, DEVICE_F1, TEST_MEAN_F1, N_TEST = picked_device()
THRESHOLD = threshold()
