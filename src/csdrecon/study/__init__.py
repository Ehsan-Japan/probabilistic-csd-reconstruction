"""
csdrecon.study — the four-stage study, as a library.

The programs in scripts/ are settings blocks; everything they do lives here:

    config.py        StudyConfig — one configuration, and the folder it owns
    dataset.py       stage 1 — simulate, split by ID, cut rays, write the
                     dataset and its summary
    device_split.py  the train/test split: made once on device IDs, stored
                     with the pool so every sweep cell inherits it
    training.py      stage 2 — train the U-Net on one configuration
    evaluation.py    stage 3 — score the checkpoint and draw what it predicts
    comparison.py    stage 4 — put every configuration side by side
    sweep.py         stages 1-4 for a grid of budgets, behind run_0
    sampling.py      where the measured points go, at one fixed budget;
                     StudyConfig validates against these, but the ablation
                     that compared them was deleted on 2026-09-20
"""
