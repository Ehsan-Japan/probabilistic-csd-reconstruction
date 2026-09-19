"""
The README's results section is generated, so what is checked here is that
it cannot quietly go stale or eat the prose around it.
"""
import csv
import os

import pytest

from csdrecon.study import readme_report as rr


def write_run(tmp_path, **over):
    """A minimal run folder with one comparison.csv row."""
    row = {"configuration": "4_rays_40_points_300_samples", "n_rays": 4,
           "n_points": 40, "n_train": 300, "n_test": 50, "resolution": 100,
           "split_seed": 12345, "coverage": 0.0157, "threshold": 0.4,
           "iou": 0.187, "f1@1_std": 0.13, "f1@1_min": 0.18,
           "f1@1_max": 0.80, "true_line_fraction": 0.0714}
    for tau, f1 in zip((0, 1, 2, 3), (0.31, 0.63, 0.79, 0.87)):
        row[f"f1@{tau}"] = f1
        row[f"precision@{tau}"] = f1 - 0.05
        row[f"recall@{tau}"] = f1 + 0.05
        row[f"coverage@{tau}"] = 0.0157 * (tau + 1)
    row.update(over)
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return str(run)


def test_section_carries_the_numbers_from_the_csv(tmp_path):
    text = rr.render(write_run(tmp_path))
    assert "0.630" in text or "0.63" in text     # F1@1
    assert "1.57 %" in text                      # coverage
    assert "0.40" in text                        # the stored threshold
    assert text.startswith(rr.BEGIN) and text.rstrip().endswith(rr.END)


def test_a_run_that_was_never_compared_says_so(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No run has been compared yet" in rr.render(str(empty))


def test_update_replaces_only_between_the_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# title\n\nkeep me\n\n{rr.BEGIN}\nold\n{rr.END}\n\n"
                      f"keep me too\n", encoding="utf-8")
    rr.update(str(readme), rr.render(write_run(tmp_path)))
    out = readme.read_text(encoding="utf-8")
    assert "keep me" in out and "keep me too" in out
    assert "STALE_SECTION" not in out   # ("old" would match "threshold")
    assert out.count(rr.BEGIN) == 1 and out.count(rr.END) == 1


def test_update_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"{rr.BEGIN}\n{rr.END}\n", encoding="utf-8")
    section = rr.render(write_run(tmp_path))
    assert rr.update(str(readme), section) is True
    assert rr.update(str(readme), section) is False


def test_missing_markers_is_an_error_not_an_append(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rr.update(str(readme), rr.render(write_run(tmp_path)))
    assert readme.read_text(encoding="utf-8") == "# no markers here\n"


def test_the_best_budget_is_the_one_reported(tmp_path):
    run = write_run(tmp_path)
    rows = rr.read_rows(run)
    rows.append(dict(rows[0], n_rays=8, **{"f1@1": 0.85}))
    assert rr._best(rows)["n_rays"] == 8
