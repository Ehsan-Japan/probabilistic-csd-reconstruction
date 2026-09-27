"""
Rebuild a sweep's comparison table with the retrained cells in it.

    python scripts/run_11_merge_retrained.py

run_10_retrain_collapsed.py writes a model_seed<k>/ and evaluation_seed<k>/
for each cell whose seed-0 training never left its plateau, and records the
chosen seed in retrain_collapsed.json.  This script re-runs stage 4 over all
cells of the sweep, taking each cell at its chosen seed (seed 0 for every
cell that was not retrained), so comparison.csv, comparison.txt and the run's
figures describe the models the paper uses.

The seed-0 table is expected to have been copied aside first; nothing here
deletes a model or an evaluation.
"""
import json
import os
from dataclasses import replace

import _common  # noqa: F401  (import path + headless plotting)
from csdrecon.config import paths
from csdrecon.study import comparison
from csdrecon.study.config import StudyConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "results",
                       "4-5-6-7-8_rays_40-50-60_points_500_samples")
RAYS = [4, 5, 6, 7, 8]
POINTS = [40, 50, 60]


def main():
    paths.set_config_root(RUN_DIR)
    with open(os.path.join(RUN_DIR, "retrain_collapsed.json")) as fh:
        chosen = {n: c["chosen_seed"]
                  for n, c in json.load(fh)["cells"].items()}
    cfgs = []
    for p in POINTS:                     # the sweep's own order
        for r in RAYS:
            name = f"{r}_rays_{p}_points_500_samples"
            cfg = StudyConfig.load(os.path.join(RUN_DIR, name))
            seed = chosen.get(name) or 0
            if seed:
                d = cfg._dir
                cfg = replace(cfg, train_seed=seed)
                cfg._dir = d
            print(f"{name:32s} seed {seed}  <- {os.path.relpath(cfg.eval_dir, RUN_DIR)}")
            cfgs.append(cfg)
    comparison.run(out_dir=RUN_DIR, configs=cfgs)


if __name__ == "__main__":
    main()
