"""label_states on maps whose answer is known."""
import numpy as np

from csdrecon.study import charge_states as C


def _toy(n=60, cells=3.0, cross=0.3):
    """A small (n1, n2) map with hole convention ((0, 0) top right) and the
    line map drawn on its state boundaries."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    cs = np.stack([np.floor((1 - x) * cells + cross * (1 - y)),
                   np.floor((1 - y) * cells + cross * (1 - x))], -1).astype(int)
    Y = np.zeros((n, n), bool)
    Y[:, :-1] |= (cs[:, :-1] != cs[:, 1:]).any(-1)
    Y[:-1, :] |= (cs[:-1, :] != cs[1:, :]).any(-1)
    return cs, Y


def test_toy_map_labelled_exactly():
    cs, Y = _toy()
    res = C.label_states(Y, 0.02, 0.02)
    sc = C.score(res, cs, Y, 0.02, 0.02)
    assert sc["acc_rel"] == 1.0
    assert sc["acc_abs"] == 1.0          # corner really is (0, 0)
    assert sc["state_recall"] == 1.0
    assert sc["consistency"] == 1.0


def test_shift_is_found_when_the_corner_is_not_00():
    cs, Y = _toy()
    res = C.label_states(Y, 0.02, 0.02)
    sc = C.score(res, cs + np.array([2, 1]), Y, 0.02, 0.02)
    assert sc["acc_rel"] == 1.0
    assert sc["acc_abs"] == 0.0
    assert (sc["shift_n1"], sc["shift_n2"]) == (2, 1)


def test_gap_at_the_frame_is_closed():
    cs, Y = _toy()
    Y[:, -1] = False                     # lines stop one pixel short
    Y[-1, :] = False
    res = C.label_states(Y, 0.02, 0.02)
    assert C.score(res, cs, Y, 0.02, 0.02)["acc_rel"] == 1.0


def test_line_fit_bridges_a_gap():
    cs, Y = _toy()
    broken = Y.copy()
    # cut a 4-pixel gap in the dot-1 line at column ~20, between crossings
    broken[47:51, 17:24] = False
    before = C.score(C.label_states(broken, 0.02, 0.02), cs, Y, 0.02, 0.02)
    fixed = C.fit_lines(broken, min_len=3, max_ext=10, keep=1)
    after = C.score(C.label_states(fixed, 0.02, 0.02), cs, Y, 0.02, 0.02)
    assert before["acc_rel"] < 1.0
    assert after["acc_rel"] == 1.0


def test_line_fit_leaves_a_complete_map_alone():
    cs, Y = _toy()
    fixed = C.fit_lines(Y, min_len=3, max_ext=10, keep=1)
    assert C.score(C.label_states(fixed, 0.02, 0.02), cs, Y, 0.02,
                   0.02)["acc_rel"] == 1.0


def test_few_electron_scores():
    cs, Y = _toy(cells=5.0)
    res = C.label_states(Y, 0.02, 0.02)
    f = C.score_few(res, cs, Y, nmax=3)
    assert f["acc_few"] == 1.0 and f["bal_few"] == 1.0
    assert f["land_11"] == 1.0
    zero = dict(res, labels=np.zeros_like(res["labels"]))
    f0 = C.score_few(zero, cs, Y, nmax=3)
    assert f0["land_11"] == 0.0
    assert f0["bal_few"] == 1.0 / f0["n_states_few"]


def test_missing_line_merges_two_states():
    cs, Y = _toy()
    lab, n = C.regions(Y)
    # remove one whole region boundary: the two regions become one
    Y2 = Y.copy()
    Y2[:, 38:44] = False
    lab2, n2 = C.regions(Y2)
    assert n2 < n
