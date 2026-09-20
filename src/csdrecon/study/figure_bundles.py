"""
figure_bundles.py — the comparison gallery, split into readable bundles.

results/<sweep>/figures/ draws every budget on every plot.  For a 2-D sweep
that is fifteen labelled series per axes, and the figures stop being
readable long before they stop being correct: the lines overlap, the point
labels collide, and a trend that IS there cannot be seen.

This re-renders the same gallery once per BUNDLE — a slice of the sweep in
which only one thing varies — into its own folder beside the full one:

    results/<sweep>/figures/                 all 15 budgets (unchanged)
                    40_points_per_ray/       4x40  5x40  6x40  7x40  8x40
                    50_points_per_ray/       4x50  5x50  6x50  7x50  8x50
                    60_points_per_ray/       4x60  5x60  6x60  7x60  8x60

Five series per plot instead of fifteen, and inside a bundle the only thing
that changes is the ray count — so every figure answers one question.

Nothing is recomputed and nothing existing is overwritten: this reads the
metrics.json each configuration already has and adds folders.  The full
gallery stays exactly where it was, because it is the one place that can
show a budget behaving unlike its neighbours.

Group the other way with by="rays" (4 rays at 40, 50 and 60 points, ...),
which isolates ray resolution instead of ray count.

    from csdrecon.study import figure_bundles
    figure_bundles.bundle(by="points")
"""
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import log, paths
from . import comparison, model_figures
from .config import StudyConfig, existing_configs

# A run that never left its initial plateau tops out near 0.42 on the
# validation devices while a healthy one reaches at least 0.66.  Bundles do
# not DROP those runs -- the gallery is a diagnostic, and a failed training
# is exactly the thing a diagnostic must show -- but each bundle's README
# names them, so nobody reads a collapsed curve as a result.
COLLAPSE_VAL_F1 = 0.55

# by -> (what stays constant inside a bundle, folder suffix, what varies)
GROUPINGS = {
    "points": ("n_points", "%d_points_per_ray", "ray count"),
    "rays": ("n_rays", "%d_rays", "points per ray"),
}


def sweep_dirs() -> List[str]:
    """Every sweep folder under results/ that has a comparison.csv."""
    if not os.path.isdir(paths.RESULTS):
        return []
    out = []
    for name in sorted(os.listdir(paths.RESULTS)):
        folder = os.path.join(paths.RESULTS, name)
        if os.path.isfile(os.path.join(folder, "comparison.csv")):
            out.append(os.path.abspath(folder))
    return out


def configs_in(sweep_dir: str,
               configs: Optional[Sequence[StudyConfig]] = None
               ) -> List[StudyConfig]:
    """
    The configurations that belong to THIS sweep.

    Scoped by folder, not by discovery: existing_configs() searches all of
    results/, so a leftover run next door (a different n_train, say) would
    otherwise be bundled in beside this sweep's budgets and quietly change
    what every figure means.
    """
    sweep = os.path.abspath(sweep_dir)
    out = []
    for cfg in (configs if configs is not None else existing_configs()):
        if os.path.abspath(cfg.path()).startswith(sweep + os.sep):
            out.append(cfg)
    return out


def best_val_f1(cfg: StudyConfig) -> Optional[float]:
    """The training's best validation F1@1, or None if it was not recorded."""
    path = os.path.join(cfg.path(), "model", "training_summary.json")
    try:
        with open(path) as fh:
            return float(json.load(fh)["best_val_f1"])
    except (OSError, KeyError, ValueError):
        return None


def group(configs: Sequence[StudyConfig], by: str = "points"
          ) -> List[Tuple[str, List[StudyConfig]]]:
    """[(folder name, its configurations)], in increasing order of the key."""
    if by not in GROUPINGS:
        raise ValueError("by must be one of %s, not %r"
                         % (sorted(GROUPINGS), by))
    attr, template, _varies = GROUPINGS[by]
    buckets: Dict[int, List[StudyConfig]] = {}
    for cfg in configs:
        buckets.setdefault(getattr(cfg, attr), []).append(cfg)
    other = "n_rays" if attr == "n_points" else "n_points"
    return [(template % key, sorted(buckets[key],
                                    key=lambda c: getattr(c, other)))
            for key in sorted(buckets)]


def _readme(path: str, label: str, varies: str, constant: str,
            cfgs: Sequence[StudyConfig], rows: Sequence[Dict],
            failed: Sequence[str]) -> None:
    by_name = {r["configuration"]: r for r in rows}
    lines = [
        "%s" % label,
        "=" * len(label),
        "",
        "The comparison gallery, restricted to the %d budgets that share "
        "%s." % (len(cfgs), constant),
        "Inside this folder the only thing that varies is the %s, so every"
        % varies,
        "figure answers one question.",
        "",
        "The full gallery, with every budget in the sweep on every plot, is",
        "one level up in figures/.  Nothing here replaces it.",
        "",
        "budgets",
        "-------",
    ]
    for cfg in cfgs:
        row = by_name.get(cfg.name)
        val = best_val_f1(cfg)
        mark = ""
        if val is not None and val < COLLAPSE_VAL_F1:
            mark = "   <-- DID NOT CONVERGE (best validation F1@1 %.3f)" % val
        if row is None:
            lines.append("  %-30s not evaluated" % cfg.name)
            continue
        lines.append("  %-30s coverage %5.2f %%   F1@1 %.3f%s"
                     % (cfg.name, 100 * float(row["coverage"]),
                        float(row["f1@1"]), mark))
    if failed:
        lines += [
            "",
            "WARNING",
            "-------",
            "%d of these %d trainings did not converge.  Their curves are "
            "drawn" % (len(failed), len(cfgs)),
            "anyway, because a gallery that hides a failed run is worse than "
            "one that",
            "shows it, but they are NOT results:",
        ]
        lines += ["  %s" % n for n in failed]
    lines += ["", "Rebuild:  python scripts/run_10_bundle_figures.py", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def bundle(sweep_dir: Optional[str] = None,
           by: str = "points",
           configs: Optional[Sequence[StudyConfig]] = None) -> List[str]:
    """
    Re-render the gallery once per bundle of *sweep_dir*.

    Returns the folders written.  With no sweep_dir, every sweep under
    results/ that has a comparison.csv is bundled.
    """
    targets = [sweep_dir] if sweep_dir else sweep_dirs()
    if not targets:
        log.warn("no sweep folder under results/ has a comparison.csv — "
                 "run scripts/run_4_compare_configs.py first")
        return []

    _attr, _template, varies = GROUPINGS[by]
    written: List[str] = []
    for sweep in targets:
        sweep = os.path.abspath(sweep)
        mine = configs_in(sweep, configs)
        if not mine:
            log.warn("no configuration folders inside %s" % sweep)
            continue
        rows, missing = comparison.collect(mine)
        if missing:
            log.detail("not yet evaluated, so left out: " + ", ".join(missing))
        if not rows:
            log.warn("nothing evaluated in %s" % sweep)
            continue
        by_name = {r["configuration"]: r for r in rows}

        log.say(os.path.basename(sweep))
        groups = group([c for c in mine if c.name in by_name], by)
        if len(groups) < 2:
            log.warn("  only one %s in this sweep — bundling would just "
                     "copy the gallery, so nothing was written" % varies)
            continue

        for label, cfgs in groups:
            out = os.path.join(sweep, "figures", label)
            os.makedirs(out, exist_ok=True)
            group_rows = [by_name[c.name] for c in cfgs]
            constant = ("%d points per ray" % cfgs[0].n_points
                        if by == "points" else "%d rays" % cfgs[0].n_rays)
            failed = [c.name for c in cfgs
                      if (best_val_f1(c) or 1.0) < COLLAPSE_VAL_F1]

            log.say("  %-22s %d budgets, %s varying"
                    % (label, len(cfgs), varies)
                    + ("   [%d did not converge]" % len(failed)
                       if failed else ""))

            with open(os.path.join(out, "comparison.csv"), "w",
                      newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
                w.writeheader()
                w.writerows(group_rows)
            _readme(os.path.join(out, "README.txt"), label, varies,
                    constant, cfgs, group_rows, failed)
            model_figures.render_all(list(cfgs), group_rows, out)
            written.append(out)
    return written
