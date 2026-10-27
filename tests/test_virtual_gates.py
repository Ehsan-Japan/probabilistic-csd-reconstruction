"""The virtual-gate matrix: the check device and the line fit — no data or model needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csdrecon.study import virtual_gates as V  # noqa: E402

CHECK = {"Cdd": [[0.6, 0.8], [0.8, 0.7]],
         "Cgd": [[4, 0.4, 0.05], [0.3, 3.5, 0.05]]}


def test_check_device():
    t = V.true_gates(CHECK)
    s = t["slopes"]
    assert s["dot1"] == pytest.approx(-4.38, abs=5e-3)
    assert s["dot2"] == pytest.approx(-0.238, abs=5e-4)
    assert s["interdot"] == pytest.approx(1.052, abs=5e-4)
    assert t["M12"] == pytest.approx(0.228, abs=5e-4)
    assert t["M21"] == pytest.approx(0.238, abs=5e-4)
    assert t["rho"] == pytest.approx(0.961, abs=5e-4)


def test_matrix_is_lever_arms_over_alpha11():
    a = V.lever_arms(CHECK)
    assert np.allclose(V.true_gates(CHECK)["G"], a / a[0, 0])


def _line(theta_deg, n=100, length=70):
    img = np.zeros((n, n), bool)
    t = np.linspace(-length / 2, length / 2, 4 * length)
    u = np.radians(theta_deg)
    c = np.rint(n / 2 + t * np.cos(u)).astype(int)
    r = np.rint(n / 2 + t * np.sin(u)).astype(int)
    img[r, c] = True
    return img


@pytest.mark.parametrize("theta", [20.0, 63.0, 101.0, 168.0])
def test_straight_line_angle(theta):
    fit = V.slopes_from_lines(_line(theta), 0.02, 0.02)
    got = [a for a in fit["angles"].values() if np.isfinite(a)]
    assert len(got) == 1
    assert abs(V.angle_error(got[0], theta)) < 0.5
