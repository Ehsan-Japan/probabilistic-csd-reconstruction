"""
figure_bundles.py — the comparison gallery, split into readable bundles.

A gallery that draws every budget on every plot stops being readable long
before it stops being correct.  For a 2-D sweep that is fifteen labelled
series per axes: the palette runs out, the lines overlap, the point labels
collide, and a trend that IS there cannot be seen.

So the gallery is not drawn that way.  It is rendered once per BUNDLE — a
slice of the sweep in which only one thing varies — each in its own folder,
and there is no combined copy:

    results/<sweep>/figures/
                    40_points_per_ray/       4x40  5x40  6x40  7x40  8x40
                    50_points_per_ray/       4x50  5x50  6x50  7x50  8x50
                    60_points_per_ray/       4x60  5x60  6x60  7x60  8x60

Five series per plot instead of fifteen, and inside a bundle the only thing
that changes is the ray count — so every figure answers one question.  Each
folder also carries the comparison.csv for just those budgets and a
README.txt naming them, including any whose training did not converge.

A sweep with only ONE bundle goes straight into figures/: there is nothing
to separate, and a folder named after the thing every budget shares would
claim a distinction the sweep does not make.

Nothing is recomputed: this reads the metrics.json each configuration
already has.  comparison.run() calls it, so run_4 produces the bundles
directly; run_10 is for re-rendering them afterwards, or grouping the other
way with by="rays".

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
        "The other bundles of this sweep sit beside this folder.  There is",
        "no combined gallery: fifteen series on one axes is not readable.",
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
           configs: Optional[Sequence[StudyConfig]] = None,
           rows: Optional[Sequence[Dict]] = None) -> List[str]:
    """
    Render the gallery, one folder per bundle of *sweep_dir*.

    Returns the folders written.  With no sweep_dir, every sweep under
    results/ that has a comparison.csv is done.  Pass *rows* to reuse a
    comparison that has already been collected (comparison.run does).

    A sweep with only ONE bundle is rendered flat into figures/ instead:
    there is nothing to separate, and a folder called 40_points_per_ray that
    held the entire sweep would claim a distinction the sweep does not make.
    """
    # Both imported here rather than at module scope: comparison imports
    # this module, so a top-level import of it would close the cycle, and
    # model_figures is only needed to DRAW -- group(), configs_in() and
    # best_val_f1() above are pure bookkeeping and should stay importable
    # without it.
    #
    # This does NOT make the module matplotlib-free: csdrecon.config.__init__
    # imports figure_style, which imports pyplot, so `from ..config import
    # log` at the top pulls it in regardless.  Worth knowing before trimming
    # the CI install list.
    from . import comparison, model_figures
    targets = [sweep_dir] if sweep_dir else sweep_dirs()
    if not targets:
        log.warn("no sweep folder under results/ has a comparison.csv — "
                 "run scripts/run_4_compare_configs.py first")
        return []

    _attr, _template, varies = GROUPINGS[by]
    written: List[str] = []
    for sweep in targets:
        sweep = os.path.abspath(sweep)
        if rows is not None and configs is not None:
            # The caller has already decided what belongs together, and the
            # output folder need not physically contain the configuration
            # folders -- comparison.run() names it after the sweep, which for
            # a comparison spanning two runs is a third folder entirely.
            # Filtering by path here would silently drop every one of them.
            mine = list(configs)
        else:
            mine = configs_in(sweep, configs)
        if not mine:
            log.warn("no configuration folders inside %s" % sweep)
            continue
        if rows is None:
            mine_rows, missing = comparison.collect(mine)
            if missing:
                log.detail("not yet evaluated, so left out: "
                           + ", ".join(missing))
        else:
            mine_rows = list(rows)
        if not mine_rows:
            log.warn("nothing evaluated in %s" % sweep)
            continue
        by_name = {r["configuration"]: r for r in mine_rows}

        log.say(os.path.basename(sweep))
        groups = group([c for c in mine if c.name in by_name], by)
        if len(groups) < 2:
            flat = os.path.join(sweep, "figures")
            os.makedirs(flat, exist_ok=True)
            log.say("  one %s only — rendered flat into figures/" % varies)
            model_figures.render_all([c for c in mine if c.name in by_name],
                                     mine_rows, flat)
            written.append(flat)
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
