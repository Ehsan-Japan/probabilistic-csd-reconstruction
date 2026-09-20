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
    # the comparison figures are filed under the ray folder, not loose in it,
    # because the per-budget folders live there too
    assert [label.replace(os.sep, "/") for label, _ in got] == [
        "4_rays/4_rays_summary", "8_rays/8_rays_summary"]
    assert [c.n_points for c in got[0][1]] == [40, 50, 60]


def test_budget_grouping_nests_under_its_ray_count():
    cfgs = budgets("/s", [(8, 60), (4, 40), (8, 40)])
    got = figure_bundles.group(cfgs, by="budget")
    assert [label.replace(os.sep, "/") for label, _ in got] == [
        "4_rays/4_rays_40_points",
        "8_rays/8_rays_40_points",
        "8_rays/8_rays_60_points"]
    assert all(len(g) == 1 for _, g in got)


def test_a_ray_folder_holds_its_summary_and_its_budgets_side_by_side():
    """The two groupings must land in the same parent, never overwrite it."""
    cfgs = budgets("/s", [(8, 40), (8, 50)])
    summary = figure_bundles.group(cfgs, by="rays")[0][0]
    per_budget = [lbl for lbl, _ in figure_bundles.group(cfgs, by="budget")]
    assert os.path.dirname(summary) == "8_rays"
    assert all(os.path.dirname(p) == "8_rays" for p in per_budget)
    assert summary not in per_budget


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


def test_collapse_threshold_does_not_drift_between_modules():
    """
    readme_report duplicates this constant rather than importing it (that
    module writes markdown and must not pull in matplotlib).  If the two ever
    disagree, the README would mark a different set of runs as failed than
    the bundle READMEs do.
    """
    from csdrecon.study import readme_report

    assert (readme_report.COLLAPSE_VAL_F1
            == figure_bundles.COLLAPSE_VAL_F1)


def test_a_missing_training_summary_counts_as_converged(tmp_path):
    """An older run wrote no summary; that must not disqualify its budget."""
    from csdrecon.study import readme_report

    assert readme_report.converged(str(tmp_path),
                                   {"configuration": "nope"}) is True


def test_a_collapsed_run_is_marked_and_never_chosen_as_best(tmp_path):
    from csdrecon.study import readme_report

    run = tmp_path / "run"
    for name, val, f1 in (("good", 0.79, "0.780"), ("bad", 0.42, "0.900")):
        d = run / name / "model"
        d.mkdir(parents=True)
        (d / "training_summary.json").write_text(
            '{"best_val_f1": %s}' % val, encoding="utf-8")
    rows = [{"configuration": "good", "n_rays": 8, "n_points": 50,
             "coverage": "0.039", "f1@1": "0.780", "precision@1": "0.79",
             "recall@1": "0.81", "iou": "0.29", "threshold": "0.7"},
            {"configuration": "bad", "n_rays": 8, "n_points": 60,
             "coverage": "0.047", "f1@1": "0.900", "precision@1": "0.33",
             "recall@1": "0.74", "iou": "0.11", "threshold": "0.4"}]

    # the collapsed row scores higher, and must still not be the headline
    assert readme_report._best(rows, str(run))["configuration"] == "good"

    table = "\n".join(readme_report._budget_table(rows, str(run)))
    assert "8 × 60 †" in table
    assert "**8 × 50**" in table
    assert "did not converge" in table
