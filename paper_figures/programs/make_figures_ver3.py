"""
make_figures_ver3.py -- the figures of make_figures.py whose labels say
"held-out devices", redrawn with "test devices", into method_docs_ver2.

    python paper_figures/programs/make_figures_ver3.py

Nothing in make_figures.py is changed and method_docs/ is not written:
_figures.DOCS_ROOT is pointed at method_docs_ver2 for this run, and every
axis label passes through the same replacement as the manuscript text.

Redrawn: p2l_7_tolerance_curve (Fig. 10), results_budget_ladder (Fig. 11),
fig_tau_metrics (Fig. 12).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _figures                                              # noqa: E402

_figures.DOCS_ROOT = os.path.join(_figures.PKG, "method_docs_ver2")

import matplotlib.axes                                       # noqa: E402

WORDS = (("held-out test devices", "test devices"), ("held-out", "test"))


def _clean(s):
    for a, b in WORDS:
        s = s.replace(a, b)
    return s


for name in ("set_ylabel", "set_xlabel", "set_title"):
    orig = getattr(matplotlib.axes.Axes, name)

    def patched(self, label, *a, _orig=orig, **k):
        return _orig(self, _clean(label), *a, **k)
    setattr(matplotlib.axes.Axes, name, patched)

import make_figures as F                                     # noqa: E402

if __name__ == "__main__":
    F.fig_probability_panels()
    F.fig_budget_ladder()
    F.fig_tau_metrics(row=True)   # one row (user, 2026-10-26)
    F.fig_threshold_panels(row=True)   # one row (user, 2026-10-26)
