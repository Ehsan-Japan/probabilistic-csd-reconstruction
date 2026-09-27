"""
Retrain the cells of a finished sweep whose training never left its plateau.

    python scripts/run_10_retrain_collapsed.py

At train_seed 0, four cells of the 500-sample sweep stayed at a training loss
of ~1.7 for all 50 epochs (best validation F1@1 0.42-0.44, where every other
cell reaches >= 0.66).  Nothing about the data is changed here: the same
train.npz, the same epochs, learning rate and loss -- only the training seed
(initial weights and batch order).

The rule, fixed before any result is seen and using validation data only:
try seeds 1, 2, 3 in order and keep the FIRST whose best validation F1@1
reaches COLLAPSE_VAL_F1.  Not the best of three -- that would give these
cells a best-of-N advantage the other cells never had.  The test devices
play no part in the choice.

Nothing is overwritten.  A seed other than 0 writes model_seed<k>/ and
evaluation_seed<k>/ beside the original model/ and evaluation/, and the
outcome of every attempt goes to retrain_collapsed.json in the run folder.
comparison.csv is NOT touched; merging the chosen seeds into the paper is a
separate, deliberate step.
"""
import json
import os
import time
from dataclasses import replace

import _common  # noqa: F401  (import path + headless plotting)
from csdrecon.config import log, paths
from csdrecon.study import evaluation, training
from csdrecon.study.config import StudyConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "results",
                       "4-5-6-7-8_rays_40-50-60_points_500_samples")

CELLS = ["5_rays_60_points_500_samples", "6_rays_50_points_500_samples",
         "7_rays_60_points_500_samples", "8_rays_60_points_500_samples"]
SEEDS = [1, 2, 3]

# The same cut paper_figures/programs/_run.py uses: a collapsed run tops out
# near 0.42 on the validation devices, the worst healthy one reaches 0.660.
COLLAPSE_VAL_F1 = 0.55


def _best_val_f1(cfg):
    with open(os.path.join(cfg.model_dir, "history.json")) as fh:
        return max(json.load(fh)["val_f1"])


def main():
    paths.set_config_root(RUN_DIR)
    out = os.path.join(RUN_DIR, "retrain_collapsed.json")
    record = {"rule": "first of seeds %s with best val F1@1 >= %g"
                      % (SEEDS, COLLAPSE_VAL_F1),
              "cells": {}}
    if os.path.isfile(out):
        with open(out) as fh:
            record = json.load(fh)

    t0 = time.time()
    for name in CELLS:
        base = StudyConfig.load(os.path.join(RUN_DIR, name))
        cell = record["cells"].setdefault(name, {"attempts": [],
                                                 "chosen_seed": None})
        if cell["chosen_seed"] is not None:
            log.say(f"{name}: seed {cell['chosen_seed']} already chosen")
            continue
        for seed in SEEDS:
            cfg = replace(base, train_seed=seed)
            cfg._dir = base._dir
            log.say(f"\n{'#' * 74}\n#  {name}  train_seed {seed}")
            t = time.time()
            if not os.path.isfile(cfg.checkpoint):
                training.train(cfg)
            f1 = _best_val_f1(cfg)
            ok = f1 >= COLLAPSE_VAL_F1
            attempt = {"seed": seed, "best_val_f1": f1, "converged": ok,
                       "minutes": round((time.time() - t) / 60, 1)}
            if ok:
                m = evaluation.run(cfg)
                attempt["test_f1@1"] = m["f1@1"]
                cell["chosen_seed"] = seed
            cell["attempts"] = [a for a in cell["attempts"]
                                if a["seed"] != seed] + [attempt]
            with open(out, "w") as fh:
                json.dump(record, fh, indent=2)
            log.say(f"  best val F1@1 {f1:.3f} -> "
                    f"{'converged' if ok else 'collapsed again'}")
            if ok:
                break

    log.say(f"\ntotal {(time.time() - t0) / 60:.1f} min -> {out}")
    for name, cell in record["cells"].items():
        log.say(f"  {name:32s} chosen seed {cell['chosen_seed']}  "
                + "  ".join(f"s{a['seed']}:{a['best_val_f1']:.3f}"
                            for a in cell["attempts"]))


if __name__ == "__main__":
    main()
