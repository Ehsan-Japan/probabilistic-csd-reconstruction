"""
How the gallery is split into bundles.  Pure bookkeeping — which budget
lands in which folder, and which stray folder is kept out — so it runs in a
second and needs no data, no model and no simulator.

The scoping test is the one that matters: existing_configs() searches all of
results/, so a leftover run next door (a different n_train) would otherwise
be bundled in beside this sweep's budgets and quietly change what every
figure means.
"""
import os

import pytest

from csdrecon.study import figure_bundles


class FakeConfig:
    """Just enough StudyConfig for grouping and scoping."""

    def __init__(self, n_rays, n_points, folder):
        self.n_rays = n_rays
        self.n_points = n_points
        self.name = "%d_rays_%d_points_500_samples" % (n_rays, n_points)
        self._folder = folder

    def path(self):
        return self._folder


def sweep(tmp, name="4-5-6_rays_40-50_points_500_samples"):
    return os.path.join(str(tmp), "results", name)


def budgets(root, pairs):
    return [FakeConfig(r, p, os.path.join(root, "%d_rays_%d_points" % (r, p)))
            for r, p in pairs]


def test_points_grouping_puts_every_ray_count_in_one_folder():
    cfgs = budgets("/s", [(4, 40), (8, 40), (5, 40), (4, 60), (8, 60)])
    got = figure_bundles.group(cfgs, by="points")
    assert [label for label, _ in got] == ["40_points_per_ray",
                                           "60_points_per_ray"]
    assert [c.n_rays for c in got[0][1]] == [4, 5, 8]      # sorted by ray
    assert [c.n_rays for c in got[1][1]] == [4, 8]


def test_rays_grouping_is_the_other_axis():
    cfgs = budgets("/s", [(4, 40), (4, 60), (4, 50), (8, 40)])
    got = figure_bundles.group(cfgs, by="rays")
    assert [label for label, _ in got] == ["4_rays", "8_rays"]
    assert [c.n_points for c in got[0][1]] == [40, 50, 60]


def test_grouping_key_must_be_known():
    with pytest.raises(ValueError):
        figure_bundles.group(budgets("/s", [(4, 40)]), by="samples")


def test_configs_in_keeps_only_this_sweeps_folders(tmp_path):
    mine = sweep(tmp_path)
    other = sweep(tmp_path, "4_rays_40_points_300_samples")
    cfgs = (budgets(mine, [(4, 40), (8, 50)])
            + budgets(other, [(4, 40)]))
    kept = figure_bundles.configs_in(mine, cfgs)
    assert [c.n_rays for c in kept] == [4, 8]
    assert all(c.path().startswith(mine) for c in kept)


def test_configs_in_does_not_match_a_sibling_with_the_same_prefix(tmp_path):
    """4-5-6_rays_... must not swallow 4-5-6_rays_..._v2/."""
    mine = sweep(tmp_path)
    cfgs = budgets(mine + "_v2", [(4, 40)])
    assert figure_bundles.configs_in(mine, cfgs) == []
