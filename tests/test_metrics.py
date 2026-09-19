"""
The metric the whole study is reported in, checked against cases whose
answer is known by hand.  These run in a second and need no data, no model
and no simulator — they are here so a change to grid_metrics.py that alters
what F1@tau means cannot pass silently.
"""
import numpy as np
import pytest

from csdrecon.ml.grid_metrics import coverage_at_tau, evaluate, iou, tolerant_f1


def line(row: int, n: int = 20) -> np.ndarray:
    """A one-pixel-wide horizontal line across an n x n map."""
    m = np.zeros((n, n), dtype=bool)
    m[row, :] = True
    return m


def test_identical_maps_score_one():
    a = line(10)
    for tau in (0, 1, 2, 3):
        m = tolerant_f1(a, a, tau)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)


def test_one_pixel_offset_is_zero_at_tau_0_and_one_at_tau_1():
    """
    The reason the study reports a tolerant score at all: a perfectly
    recovered line drawn one pixel too high is worthless to strict F1 and
    correct to a physicist.
    """
    assert tolerant_f1(line(11), line(10), 0)["f1"] == pytest.approx(0.0)
    assert tolerant_f1(line(11), line(10), 1)["f1"] == pytest.approx(1.0)


def test_tolerance_never_lowers_the_score():
    pred, true = line(13), line(10)
    scores = [tolerant_f1(pred, true, t)["f1"] for t in (0, 1, 2, 3)]
    assert scores == sorted(scores)


def test_empty_prediction_scores_zero_not_one():
    """An empty map must not be rewarded, however sparse the truth is."""
    m = tolerant_f1(np.zeros((20, 20), bool), line(10), 1)
    assert m["recall"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_both_empty_is_perfect():
    z = np.zeros((20, 20), bool)
    assert tolerant_f1(z, z, 0)["f1"] == pytest.approx(1.0)


def test_iou_is_strict():
    assert iou(line(10), line(10)) == pytest.approx(1.0)
    assert iou(line(11), line(10)) == pytest.approx(0.0)


def test_pixel_accuracy_is_high_for_predicting_nothing():
    """The reason pixel accuracy is reported only to be dismissed."""
    m = tolerant_f1(np.zeros((100, 100), bool), line(50, 100), 0)
    assert m["accuracy"] > 0.98


def test_coverage_at_tau_grows_with_tau():
    visited = np.zeros((100, 100))
    visited[50, :] = 1
    assert coverage_at_tau(visited, 0) == pytest.approx(0.01)
    assert coverage_at_tau(visited, 1) == pytest.approx(0.03)
    assert coverage_at_tau(visited, 3) == pytest.approx(0.07)
    assert coverage_at_tau(np.zeros((10, 10)), 2) == pytest.approx(0.0)


def test_evaluate_averages_per_device():
    """
    Per device, not pooled over pixels: one unusually dense honeycomb must
    not be able to dominate the reported number.
    """
    preds = np.stack([line(10), line(10)])
    trues = np.stack([line(10), line(14)])
    out = evaluate(preds, trues, taus=(0, 1))
    assert out["f1@0"] == pytest.approx(0.5)      # one perfect, one zero
