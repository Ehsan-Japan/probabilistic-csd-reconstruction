"""Ray-only normalisation and the slope study's fit — no data or model needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csdrecon.ml.ray_peaks import normalise_measured  # noqa: E402
from csdrecon.study import slopes as S  # noqa: E402


# ── normalisation ─────────────────────────────────────────────────────────

def test_normalisation_uses_only_the_measured_points():
    measured = np.array([2.0, 3.0, 4.0])
    trace = np.array([[4.0, 3.0, 2.0]])
    v, t = normalise_measured(measured, trace)
    assert np.allclose(v, [0.0, 0.5, 1.0])
    assert np.allclose(t, [[1.0, 0.5, 0.0]])


def test_normalisation_of_a_flat_measurement_is_finite():
    v, = normalise_measured(np.array([5.0, 5.0]))
    assert np.allclose(v, 0.0)


# ── the truth ─────────────────────────────────────────────────────────────

CDD = [[1.0758, 0.6961], [0.6961, 1.1928]]
CGD = [[5.2173, 0.5423, 0.0631], [0.5724, 3.8144, 0.0506]]


def test_true_slopes_steep_and_shallow():
    s = S.true_slopes(CDD, CGD)
    assert s["dot1"] < -1.0 < s["dot2"] < 0.0


def test_maxwell_matches_qarray():
    helpers = pytest.importorskip("qarray.DotArrays._helper_functions")
    cdd, cgd = S.maxwell(CDD, CGD)
    q_cdd, _, q_cgd = helpers.convert_to_maxwell(np.array(CDD), np.array(CGD))
    assert np.allclose(cdd, np.asarray(q_cdd))
    assert np.allclose(cgd, np.asarray(q_cgd))


def test_cross_coupling_inverts_the_slope():
    assert S.cross_coupling("dot1", -5.0) == pytest.approx(0.2)
    assert S.cross_coupling("dot2", -0.25) == pytest.approx(0.25)


# ── the fit ───────────────────────────────────────────────────────────────

def test_families_split_by_angle_and_interdot_ignored():
    segs = [(-80.0, 30.0), (-79.0, 20.0), (-20.0, 30.0), (-21.0, 25.0),
            (45.0, 50.0)]                       # the last one is interdot
    fam = S.assign_families(segs)
    assert -80.5 < fam["dot1"] < -78.5
    assert -21.5 < fam["dot2"] < -19.5


def test_a_single_group_goes_to_its_side_of_the_diagonal():
    fam = S.assign_families([(-70.0, 30.0), (-71.0, 30.0)])
    assert fam["dot2"] is None and fam["dot1"] is not None


def _line_pixels(angle_deg, offsets, n=100):
    t = np.tan(np.radians(angle_deg))
    mask = np.zeros((n, n), bool)
    if abs(t) > 1:
        for y in range(n):
            for c in offsets:
                x = int(round(c + y / t))
                if 0 <= x < n:
                    mask[y, x] = True
    else:
        for x in range(n):
            for c in offsets:
                y = int(round(c + t * x))
                if 0 <= y < n:
                    mask[y, x] = True
    return mask


def test_fit_recovers_two_families_of_parallel_lines():
    pytest.importorskip("skimage")
    mask = (_line_pixels(-75.0, [30, 60, 90, 120])
            | _line_pixels(-15.0, [20, 50, 80, 110]))
    fit = S.fit_slopes(S.mask_points(mask), mask.shape,
                       dict(threshold=5, line_length=10, line_gap=1))
    assert abs(fit["dot1"] - (-75.0)) < 0.5
    assert abs(fit["dot2"] - (-15.0)) < 0.5


def test_an_empty_point_set_is_a_failed_fit_not_a_guess():
    pytest.importorskip("skimage")
    fit = S.fit_slopes(np.empty((0, 2)), (100, 100),
                       dict(threshold=5, line_length=10, line_gap=1))
    assert fit == {"dot1": None, "dot2": None}
    assert np.isnan(S.angle_error(fit["dot1"], -70.0))


def test_step_rule_ignores_a_steep_smooth_flank():
    x = np.arange(60, dtype=float)
    flank = np.exp(-x / 8.0)                    # steep near 0, flat later
    trace = flank.copy()
    trace[40:] += 2 * (flank[39] - flank[40])   # a step twice the local slope,
                                                # 100x smaller than the flank's start
    r = S.step_residual(trace)
    assert int(np.argmax(r)) == 39


def test_summary_counts_failures_as_misses():
    s = S.summarise([0.2, 0.8, float("nan"), 3.0])
    assert s["n_failed"] == 1
    assert s["within_1deg"] == pytest.approx(0.5)
