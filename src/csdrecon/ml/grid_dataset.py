"""
grid_dataset.py — (ray peaks -> transition-line map) pairs at a given budget.

    X : (3, H, W)  ch0 raw signal along rays, ch1 visited pixels (the network
                   input), ch2 ray peaks (baseline and figures only)
    Y : (H, W)     1.0 on a transition line

Y is the simulator's own ground_truth_labels.npy and does not depend on the
budget; only X gets sparser as rays or points are removed.  That is the
experiment: how sparse can X get before the reconstruction fails.

Datasets are cached to .npz keyed by budget, because re-cutting rays for ~1000
samples takes a minute and the sweep asks for the same budget on every epoch.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from .ray_peaks import load_ground_truth, measure, to_channels
from ..config import log


def build(sample_dirs: Sequence[str],
          n_rays: int,
          n_points: int,
          detector: Optional[object] = None,
          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(X, Y) for a list of samples at one budget.

    X : (N, 3, H, W) float32        Y : (N, H, W) float32

    Samples whose grids disagree in size are skipped with a warning rather
    than crashing the sweep: mixing runs of different image_res is easy to do
    by accident and should not cost an hour of training.
    """
    Xs, Ys = [], []
    shape = None
    for i, sdir in enumerate(sample_dirs, 1):
        y = load_ground_truth(sdir)
        if shape is None:
            shape = y.shape
        elif y.shape != shape:
            log.detail(f"  [skip] {sdir}: grid {y.shape} != {shape}")
            continue
        m = measure(sdir, n_rays, n_points, detector=detector)
        Xs.append(to_channels(m, shape))
        Ys.append(y)
        if verbose and i % 100 == 0:
            log.detail(f"  measured {i}/{len(sample_dirs)}")
    if not Xs:
        raise RuntimeError("no usable samples")
    return np.stack(Xs), np.stack(Ys)


