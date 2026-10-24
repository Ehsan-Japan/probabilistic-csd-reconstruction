"""
How few rays and points give virtual gates accurate to a few percent?

    python scripts/run_14_budget_slopes.py

For every measurement budget (rays x points per ray) this trains the
network at several seeds, measures the two loading-line slopes on every
held-out test device with the one fit of csdrecon/study/slopes.py, builds
the virtual gates they imply and records how much crosstalk is left.  The
two budget-free references — Canny on the full dense scan and the fit to
the true line map — and the budget-dependent "ray steps without a network"
are scored the same way.

Everything is chosen on VALIDATION devices only: the network's seed (the
best validation F1@1 of SEEDS — training at a fixed budget is seed-fragile,
so the spread over seeds is reported as well), every arm's Hough setting,
Canny's and the step detector's knobs.  The test devices are only scored.

Restartable: datasets, checkpoints and per-cell slope results already on
disk are reused.  Writes <run>/budget_slopes/: per_device.csv (every test
device, every cell, seed and arm), cells.csv (one row per cell, seed and
arm) and references.csv.
"""
import csv
import json
import os
import time
from dataclasses import replace

import numpy as np

import _common  # noqa: F401  (import path + headless plotting)

from csdrecon.config import log, paths
from csdrecon.ml import grid_train
from csdrecon.ml.ray_peaks import load_grid, load_ground_truth
from csdrecon.study import dataset, evaluation, training
from csdrecon.study import slopes as S
from csdrecon.study.dataset import load_split
from run_1_generate_dataset import CONFIG as TEMPLATE
from run_12_slope_study import CANNY_GRID, STEP_K

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

RAYS = [4, 6, 8, 12, 16, 24]
POINTS = [40, 60, 100]
SEEDS = [0, 1, 2]
N_TRAIN, N_TEST, EPOCHS = 500, 50, 50

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(
    ROOT, "results",
    f"{'-'.join(map(str, RAYS))}_rays_{'-'.join(map(str, POINTS))}"
    f"_points_{N_TRAIN}_samples")
OUT = os.path.join(RUN_DIR, "budget_slopes")

# ══════════════════════════════════════════════════════════════════════════

SHAPE = None


def cell_config(r, p):
    return replace(TEMPLATE, n_rays=r, n_points=p, n_train=N_TRAIN,
                   n_test=N_TEST, epochs=EPOCHS, save_device_figures=False,
                   figure_devices={})


def seeded(cfg, seed):
    c = replace(cfg, train_seed=seed)
    c._dir = cfg._dir
    return c


def validation_index(n):
    rng = np.random.default_rng(grid_train.SEED)
    n_val = max(1, int(grid_train.VAL_FRACTION * n))
    return sorted(int(i) for i in rng.permutation(n)[:n_val])


def device_facts(dirs):
    """Truth, lever arms and the before-gates crosstalk of each device."""
    out = []
    for d in dirs:
        t = S.device_truth(d)
        with open(os.path.join(d, "device.json")) as f:
            cap = json.load(f)["capacitance"]
        t["L2"] = S.lever_arms(cap["Cdd"], cap["Cgd"])[:, :2]
        out.append(t)
    return out


def score_arm(sets_val, facts_val, sets_test, facts_test):
    """Choose the arm's knob and Hough setting on validation; score test.

    -> (chosen, list of per-test-device records)
    """
    chosen = S.choose(sets_val, facts_val, SHAPE)
    recs = []
    for xy, t in zip(sets_test[chosen["knob"]], facts_test):
        fit = S.fit_slopes(xy, SHAPE, chosen["hough"])
        rec = {"sample": t["sample"], "points": int(len(xy))}
        for fam in S.FAMILIES:
            rec[f"{fam}_present"] = t["present"][fam]
            rec[f"{fam}_angle_err"] = S.angle_error(fit[fam], t["angle"][fam])
            rec[f"{fam}_slope_rel_err"] = (
                float("nan") if fit[fam] is None else
                abs(np.tan(np.radians(fit[fam])) - t["slope"][fam])
                / abs(t["slope"][fam]))
        al = S.alphas_from_angles(fit["dot1"], fit["dot2"])
        r12, r21 = ((float("nan"),) * 2 if al is None
                    else S.residual_crosstalk(t["L2"], al))
        rec.update(a12=float("nan") if al is None else al[0],
                   a21=float("nan") if al is None else al[1],
                   r12=r12, r21=r21,
                   crosstalk=max(abs(r12), abs(r21)) if al else float("nan"))
        recs.append(rec)
    return chosen, recs


def summarise_cell(recs):
    """One row of cells.csv from per-device records."""
    xt = np.array([r["crosstalk"] for r in recs], float)
    rel = np.array([r[f"{f}_slope_rel_err"] for r in recs
                    for f in S.FAMILIES if r[f"{f}_present"]], float)
    ang = np.array([r[f"{f}_angle_err"] for r in recs
                    for f in S.FAMILIES if r[f"{f}_present"]], float)
    miss = lambda v, t: float(np.mean(np.nan_to_num(v, nan=np.inf) <= t))  # noqa
    return {
        "n_devices": len(recs),
        "no_virtual_gates": int(np.isnan(xt).sum()),
        "crosstalk_median_%": 100 * float(np.nanmedian(xt)),
        "crosstalk_p90_%": 100 * float(np.nanpercentile(xt, 90)),
        "crosstalk_le_3%": miss(xt, 0.03),
        "crosstalk_le_5%": miss(xt, 0.05),
        "crosstalk_le_10%": miss(xt, 0.10),
        "slope_rel_err_median_%": 100 * float(np.nanmedian(rel)),
        "angle_err_median_deg": float(np.nanmedian(ang)),
        "slope_fits_failed": int(np.isnan(rel).sum()),
    }


def network_sets(cfg, idx_val, test_X):
    Xtr, _, _ = load_split(cfg.train_npz)
    net, ck = grid_train.load(cfg.checkpoint)
    thr = float(ck["threshold"])
    pv = grid_train.predict(net, Xtr[idx_val])
    pt = grid_train.predict(net, test_X)
    return ({"-": [S.mask_points(p > thr) for p in pv]},
            {"-": [S.mask_points(p > thr) for p in pt]})


def step_sets(dirs, r, p):
    return {f"k={k:g}": [S.ray_step_points(d, r, p, k) for d in dirs]
            for k in STEP_K}


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    global SHAPE
    t0 = time.time()
    paths.set_config_root(RUN_DIR)
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "_cells")
    os.makedirs(cache, exist_ok=True)

    cells_rows, device_rows = [], []
    refs_done = False
    for p in POINTS:
        for r in RAYS:
            cfg = cell_config(r, p)
            log.say(f"\n{'#' * 70}\n# {r} rays x {p} points  "
                    f"({(time.time() - t0) / 60:.0f} min so far)")
            dataset.build(cfg)
            Xte, _, test_dirs = load_split(cfg.test_npz)
            _, _, train_dirs = load_split(cfg.train_npz)
            SHAPE = Xte.shape[2:]
            iv = validation_index(len(train_dirs))
            val_dirs = [train_dirs[i] for i in iv]
            facts_val, facts_test = device_facts(val_dirs), device_facts(
                test_dirs)
            coverage = float(Xte[:, 1].mean())

            # the budget-free references, once
            if not refs_done:
                ref_path = os.path.join(cache, "references.json")
                if os.path.isfile(ref_path):
                    with open(ref_path) as f:
                        refs = json.load(f)
                else:
                    gv = [load_grid(d)[2] for d in val_dirs]
                    gt = [load_grid(d)[2] for d in test_dirs]
                    canny = lambda gs: {  # noqa: E731
                        f"sigma={s:g} q=({lo:g},{hi:g}) background={bg:g}px":
                        [S.canny_points(Z, s, lo, hi, bg) for Z in gs]
                        for s, lo, hi, bg in CANNY_GRID}
                    refs = {}
                    for arm, sv, st in (
                            ("canny", canny(gv), canny(gt)),
                            ("truth",
                             {"-": [S.mask_points(load_ground_truth(d))
                                    for d in val_dirs]},
                             {"-": [S.mask_points(load_ground_truth(d))
                                    for d in test_dirs]})):
                        ch, recs = score_arm(sv, facts_val, st, facts_test)
                        refs[arm] = {"chosen": ch, "devices": recs}
                    before = [max(abs(x) for x in S.residual_crosstalk(
                        t["L2"], None)) for t in facts_test]
                    refs["before"] = before
                    with open(ref_path, "w") as f:
                        json.dump(refs, f, indent=1)
                ref_rows = []
                for arm in ("canny", "truth"):
                    ref_rows.append({"arm": arm, **summarise_cell(
                        refs[arm]["devices"]),
                        "chosen": json.dumps(refs[arm]["chosen"])})
                    for rec in refs[arm]["devices"]:
                        device_rows.append({"arm": arm, "rays": "", "points": "",
                                            "seed": "", **rec})
                b = np.array(refs["before"])
                ref_rows.append({"arm": "before (no virtual gates)",
                                 "n_devices": len(b),
                                 "crosstalk_median_%": 100 * np.median(b),
                                 "crosstalk_p90_%": 100 * np.percentile(b, 90),
                                 "crosstalk_le_3%": float(np.mean(b <= .03)),
                                 "crosstalk_le_5%": float(np.mean(b <= .05)),
                                 "crosstalk_le_10%": float(np.mean(b <= .10))})
                write_csv(os.path.join(OUT, "references.csv"), ref_rows)
                refs_done = True

            # ray steps, no network: depends on the budget, not on a seed
            cpath = os.path.join(cache, f"{r}x{p}_ray_steps.json")
            if os.path.isfile(cpath):
                with open(cpath) as f:
                    steps = json.load(f)
            else:
                ch, recs = score_arm(step_sets(val_dirs, r, p), facts_val,
                                     step_sets(test_dirs, r, p), facts_test)
                steps = {"chosen": ch, "devices": recs}
                with open(cpath, "w") as f:
                    json.dump(steps, f, indent=1)
            cells_rows.append({"rays": r, "points": p, "coverage_%":
                               100 * coverage, "arm": "ray_steps", "seed": "",
                               **summarise_cell(steps["devices"]),
                               "chosen": json.dumps(steps["chosen"])})
            for rec in steps["devices"]:
                device_rows.append({"arm": "ray_steps", "rays": r,
                                    "points": p, "seed": "", **rec})

            # the network, every seed
            for seed in SEEDS:
                cs = seeded(cfg, seed)
                cpath = os.path.join(cache, f"{r}x{p}_network_seed{seed}.json")
                if os.path.isfile(cpath):
                    with open(cpath) as f:
                        net = json.load(f)
                else:
                    if not os.path.isfile(cs.checkpoint):
                        training.train(cs)
                    m = evaluation.run(cs)
                    with open(os.path.join(cs.model_dir,
                                           "training_summary.json")) as f:
                        val_f1 = json.load(f)["best_val_f1"]
                    sv, st = network_sets(cs, iv, Xte)
                    ch, recs = score_arm(sv, facts_val, st, facts_test)
                    net = {"chosen": ch, "devices": recs,
                           "best_val_f1": val_f1, "test_f1@1": m["f1@1"]}
                    with open(cpath, "w") as f:
                        json.dump(net, f, indent=1)
                cells_rows.append({"rays": r, "points": p,
                                   "coverage_%": 100 * coverage,
                                   "arm": "network", "seed": seed,
                                   "best_val_f1": net["best_val_f1"],
                                   "test_f1@1": net["test_f1@1"],
                                   **summarise_cell(net["devices"]),
                                   "chosen": json.dumps(net["chosen"])})
                for rec in net["devices"]:
                    device_rows.append({"arm": "network", "rays": r,
                                        "points": p, "seed": seed, **rec})
            write_csv(os.path.join(OUT, "cells.csv"), cells_rows)
            write_csv(os.path.join(OUT, "per_device.csv"), device_rows)

    log.say(f"\ndone in {(time.time() - t0) / 60:.0f} min -> "
            f"{os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
