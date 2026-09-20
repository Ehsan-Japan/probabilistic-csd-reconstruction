"""
THE GALLERY, SPLIT INTO READABLE BUNDLES.

    python scripts/run_10_bundle_figures.py

run_4 already writes the gallery this way, one folder per bundle.  This
re-renders it — after editing a figure, or to group the sweep the other way.

A bundle is a slice in which only one thing varies, and there is no combined
copy: fifteen labelled series on one axes exhausts the palette, the lines
overlap and the point labels collide.

    results/<sweep>/figures/
                    40_points_per_ray/       4x40  5x40  6x40  7x40  8x40
                    50_points_per_ray/       4x50  5x50  6x50  7x50  8x50
                    60_points_per_ray/       4x60  5x60  6x60  7x60  8x60

Five series per plot instead of fifteen, and inside a bundle the only thing
that changes is the ray count — so every figure answers one question.  Each
folder also carries the comparison.csv for just those budgets and a
README.txt naming them, including any whose training did not converge.

Retrains nothing: it reads the metrics.json each configuration already has.
Safe to re-run at any time.
"""
import _common
from csdrecon.study import figure_bundles

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Which sweep to bundle.  None does every sweep under results/ that has a
# comparison.csv; give a folder name to pin it, e.g.
#   RUN = "4-5-6-7-8_rays_40-50-60_points_500_samples"
RUN = None

# What stays CONSTANT inside a bundle, and therefore what each bundle is
# about:
#   "points"  one folder per points-per-ray; the RAY COUNT varies inside it
#             -> 40_points_per_ray/ holds 4x40 … 8x40
#   "rays"    one folder per ray count; the POINTS PER RAY vary inside it
#             -> 4_rays/ holds 4x40, 4x50, 4x60
#   "budget"  one folder per configuration, a single model on every plot.
#             These NEST under their ray count, so the fifteen of them are
#             filed rather than spread flat:
#                 4_rays/4_rays_40_points/, 4_rays/4_rays_50_points/, …
#
# The three are independent slices of the same sweep and coexist happily:
# each writes its own folders and leaves the others alone.  List as many as
# you want; they are written in order, so put "rays" before "budget" if you
# want the ray folders to carry their own comparison figures as well as the
# per-budget folders nested inside them.
BY = ["points", "rays", "budget"]

# ══════════════════════════════════════════════════════════════════════════


def main():
    _common.banner("BUNDLE — the gallery, a few series at a time")
    import os

    from csdrecon.config import paths

    sweep = os.path.join(paths.RESULTS, RUN) if RUN else None
    groupings = [BY] if isinstance(BY, str) else list(BY)
    written = []
    for by in groupings:
        if len(groupings) > 1:
            print()
            print("grouped by %s" % by)
        written += figure_bundles.bundle(sweep_dir=sweep, by=by)
    if written:
        print()
        print("%d bundle folder(s) written." % len(written))


if __name__ == "__main__":
    main()
