"""
THE RESULTS SECTION OF README.md — written from the run, not by hand.

    python scripts/run_9_update_readme.py

Reads a run's comparison.csv and rewrites the block between the
<!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> markers in README.md: the
table of every budget, the tolerance sweep at the best one, the per-device
spread, and the few cross-configuration figures (copied into docs/figures/,
because results/ is generated and not in git).

Run it after every study run — with a different number of rays, points or
devices, the README then says what is actually on disk instead of what was
true the first time.  Everything outside the markers is left alone.
"""
import _common
from csdrecon.study import readme_report

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which run to report.  None takes the most recently compared folder under
# results/; give a name to pin it, e.g.
#   RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
RUN = None

# Copy the README's figures into docs/figures/ and link them.  False writes
# the tables alone and leaves the committed figures as they are.
WITH_FIGURES = True

# ══════════════════════════════════════════════════════════════════════════


def main():
    import os
    from csdrecon.config import paths

    _common.banner("README — the results section, from the run")
    run_dir = os.path.join(paths.RESULTS, RUN) if RUN else None
    readme_report.run(run_dir=run_dir, with_figures=WITH_FIGURES)


if __name__ == "__main__":
    main()
