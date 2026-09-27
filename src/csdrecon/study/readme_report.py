"""
readme_report.py — the results section of README.md, written from the run.

A results table typed in by hand stops being true the next time the study is
run and nobody notices.  This writes that section from the run's own
comparison.csv, so after

    python scripts/run_9_update_readme.py

the README agrees with what is on disk, or says clearly that nothing has been
run yet.  The generated text is fenced between <!-- RESULTS:BEGIN --> and
<!-- RESULTS:END -->; everything outside the markers is left alone.

The figures named in FIGURES are copied out of results/ into docs/figures/,
because results/ is generated and not in git and a README cannot show a
picture that was never committed.
"""
import csv
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import log
from ..config import paths

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"

# Where the committed copies of the figures live, relative to the project
# root.  In git, unlike results/.
DOCS_FIGURES = os.path.join("docs", "figures")

# (name in docs/figures, path inside the run folder, caption).  A figure the
# run did not produce is skipped rather than linked broken.  Empty on purpose
# since 08_cost_and_tolerance.png left the gallery: the results section is
# tables-only until a figure replaces it.
FIGURES: Sequence[Tuple[str, str, str]] = ()

# Kept up to date in docs/figures/ so the file in the repo is never a stale
# picture, but not linked from the README's results section.  `None` as the
# path means the best cell's own copy.
COPIED_ONLY: Sequence[Tuple[str, Optional[str]]] = (
    ("probability.png", None),
)

TAUS = (0, 1, 2, 3)

# The same cut figure_bundles uses to mark a failed training.
# Deliberately duplicated rather than imported: that module
# pulls in matplotlib, and this one only writes markdown.
# tests/test_figure_bundles.py asserts the two agree.
COLLAPSE_VAL_F1 = 0.55


def _f(value, digits=3, default="—"):
    """A number for the table, or an em dash when the run did not record it."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _pct(value, digits=2, default="—"):
    try:
        return f"{100 * float(value):.{digits}f} %"
    except (TypeError, ValueError):
        return default


def read_rows(run_dir: str) -> List[Dict]:
    """The run's comparison.csv, one row per measurement budget."""
    path = os.path.join(run_dir, "comparison.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_run(results_root: Optional[str] = None) -> Optional[str]:
    """
    The most recently written run folder under results/, or None.

    "Most recent" is by the modification time of its comparison.csv, which is
    the file this module reads: a run that was never compared cannot be the
    one to report.
    """
    root = results_root or paths.RESULTS
    if not os.path.isdir(root):
        return None
    runs = []
    for name in os.listdir(root):
        csv_path = os.path.join(root, name, "comparison.csv")
        if os.path.isfile(csv_path):
            runs.append((os.path.getmtime(csv_path), os.path.join(root, name)))
    return max(runs)[1] if runs else None


def converged(run_dir: str, row: Dict) -> bool:
    """Did this budget's training actually fit the data?

    A run that never left its initial plateau still produces a full set of
    metrics and is indistinguishable in a table from a budget that is simply
    too small.  It is not: it is a failed fit, and publishing it unmarked
    invites the reader to conclude that more rays made things worse.

    Decided on the validation devices, so nothing here looks at the test set.
    A missing or unreadable summary counts as converged, so an older run that
    wrote none is never quietly disqualified.
    """
    path = os.path.join(_model_dir(run_dir, row), "training_summary.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return float(json.load(fh)["best_val_f1"]) >= COLLAPSE_VAL_F1
    except (OSError, KeyError, ValueError, TypeError):
        return True


def _retrained(run_dir: str) -> Dict[str, int]:
    """{configuration: train_seed} for the cells run_10 retrained."""
    path = os.path.join(run_dir, "retrain_collapsed.json")
    try:
        with open(path, encoding="utf-8") as fh:
            cells = json.load(fh)["cells"]
    except (OSError, KeyError, ValueError):
        return {}
    return {n: c["chosen_seed"] for n, c in cells.items()
            if c.get("chosen_seed")}


def _model_dir(run_dir: str, row: Dict) -> str:
    """model/, or model_seed<k>/ for a cell retrained at another seed."""
    name = str(row.get("configuration", ""))
    seed = _retrained(run_dir).get(name, 0)
    return os.path.join(run_dir, name,
                        "model" + (f"_seed{seed}" if seed else ""))


def _best(rows: Sequence[Dict], run_dir: Optional[str] = None) -> Dict:
    """The headline budget: the best F1@1 among the runs that converged."""
    usable = ([r for r in rows if converged(run_dir, r)] if run_dir
              else list(rows))
    return max(usable or list(rows), key=lambda r: float(r["f1@1"] or 0))


def _budget_table(rows: Sequence[Dict], run_dir: Optional[str] = None
                  ) -> List[str]:
    """Every budget, in the layout of the paper's results table.

    The same columns as the manuscript's Table II (make_method_docs.py,
    _table_results_rows): budget, coverage, the validation-chosen
    threshold, precision and recall at tau = 1, strict F1 at tau = 0 and
    F1 at tau = 1.  Grouped by points per ray, as the paper groups them.
    A budget retrained after its first training stalled carries a *.
    """
    out = ["| rays × points | coverage (%) | threshold | precision | "
           "recall | F1@0 | F1@1 |",
           "|---|---|---|---|---|---|---|"]
    best = _best(rows, run_dir)
    retrained = _retrained(run_dir) if run_dir else {}
    failed = n_star = 0
    order = sorted(rows, key=lambda r: (int(r["n_points"]), int(r["n_rays"])))
    for r in order:
        budget, mark = f"{r['n_rays']} × {r['n_points']}", ""
        if run_dir and not converged(run_dir, r):
            mark = " †"
            failed += 1
        elif r.get("configuration") in retrained:
            mark = "\*"                  # escaped: a bare * is emphasis
            n_star += 1
        f1 = _f(r["f1@1"])
        if r is best and len(rows) > 1:
            budget, f1 = f"**{budget}**", f"**{f1}**"
        budget += mark
        out.append(
            f"| {budget} | {_f(float(r['coverage']) * 100, 2)} | "
            f"{_f(r['threshold'], 1)} | {_f(r['precision@1'])} | "
            f"{_f(r['recall@1'])} | {_f(r.get('f1@0'))} | {f1} |")
    if n_star:
        # the paper's caption, word for word
        out += ["", "Budgets marked \* were retrained from a different random "
                    "initialisation."]
    if failed:
        out += [
            "",
            f"† {failed} of these {len(rows)} trainings did not converge "
            f"— they never left their initial plateau. Those rows are the "
            f"score of a failed fit, not of the measurement budget.",
        ]
    return out


def _tau_table(row: Dict) -> List[str]:
    out = ["| τ | precision@τ | recall@τ | F1@τ | coverage@τ |",
           "|---|---|---|---|---|"]
    for tau in TAUS:
        mark = "**" if tau == 1 else ""
        out.append(
            f"| {mark}{tau}{mark} | {_f(row.get(f'precision@{tau}'))} | "
            f"{_f(row.get(f'recall@{tau}'))} | "
            f"{mark}{_f(row.get(f'f1@{tau}'))}{mark} | "
            f"{_pct(row.get(f'coverage@{tau}'))} |")
    return out


def copy_figures(run_dir: str, best: Dict,
                 root: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Copy the README's figures out of the run into docs/figures/.

    Returns [(path relative to the project root, caption)] for the ones that
    were actually there.
    """
    root = root or paths.PROJECT_ROOT
    out_dir = os.path.join(root, DOCS_FIGURES)
    os.makedirs(out_dir, exist_ok=True)
    kept = []
    both = [(n, r, c) for n, r, c in FIGURES]
    both += [(n, r, None) for n, r in COPIED_ONLY]
    for name, rel, caption in both:
        if rel is None:                    # the best cell's own copy
            rel = os.path.join(best["configuration"], "evaluation", "figures",
                               name)
        src = os.path.join(run_dir, rel)
        if not os.path.isfile(src):
            log.detail(f"  [skip] {name}: {os.path.abspath(src)} is missing")
            continue
        shutil.copyfile(src, os.path.join(out_dir, name))
        if caption is None:                # copied, but not shown in the README
            continue
        kept.append((f"{DOCS_FIGURES.replace(os.sep, '/')}/{name}", caption))
    return kept


def render(run_dir: str, figures: Sequence[Tuple[str, str]] = ()) -> str:
    """The generated results section, as markdown."""
    rows = read_rows(run_dir)
    run = os.path.basename(run_dir.rstrip(os.sep))
    head = [BEGIN,
            "",
            "<!-- Generated by scripts/run_9_update_readme.py — do not edit "
            "by hand. -->",
            "",
            "## Results",
            ""]
    if not rows:
        return "\n".join(head + [
            "No run has been compared yet. `python scripts/run_0_full_sweep.py`"
            " writes one into `results/`, then `python "
            "scripts/run_9_update_readme.py` fills this section in.",
            "", END])

    best = _best(rows, run_dir)
    n_train, n_test = best.get("n_train"), best.get("n_test")
    res = best.get("resolution")
    plural = "s" if len(rows) != 1 else ""
    body = [
        f"From `results/{run}/`. "
        f"{len(rows)} measurement budget{plural}, "
        f"{n_train} training and {n_test} held-out devices "
        f"(disjoint by device ID, seed {best.get('split_seed')}), "
        f"{res} × {res} px diagrams.",
        "",
        "### Every budget",
        "",
    ]
    body += _budget_table(rows, run_dir)
    body += [
        "",
        f"The threshold is not 0.5 and is not tuned on the test devices: it is "
        f"chosen on a validation split carved out of the training devices and "
        f"stored in the checkpoint.",
        "",
        f"### Tolerance sweep — {best['n_rays']} rays × {best['n_points']} "
        f"points",
        "",
    ]
    body += _tau_table(best)
    body += [
        "",
        f"`coverage@τ` is how far the measurement itself reaches at that "
        f"tolerance: the fraction of the plane within τ pixels of a measured "
        f"pixel. At τ = 0 it is the plain coverage, {_pct(best['coverage'])}. "
        f"Read the two columns together — a tolerant score is only evidence "
        f"of recovery while the tolerance stays small against the plane.",
        "",
        f"Per-device spread at this budget: F1@1 mean {_f(best['f1@1'])}, "
        f"sd {_f(best.get('f1@1_std'))}, "
        f"min {_f(best.get('f1@1_min'))}, max {_f(best.get('f1@1_max'))} "
        f"over {n_test} held-out devices. "
        f"Pixel accuracy is not reported as a result — predicting no line "
        f"anywhere already scores "
        f"{_pct(1 - float(best.get('true_line_fraction') or 0), 1)}.",
    ]
    for rel, caption in figures:
        body += ["", f"![{caption}]({rel})", "", f"*{caption}*"]
    return "\n".join(head + body + ["", END])


def update(readme: str, section: str) -> bool:
    """
    Replace the fenced section in README.md.  True if the file changed.

    The markers must already be there: silently appending a results section
    to a README that never asked for one would put it in the wrong place.
    """
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        raise ValueError(
            f"{os.path.abspath(readme)} has no {BEGIN} / {END} markers — add "
            f"them where the results section belongs")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + section + tail
    if new == text:
        return False
    with open(readme, "w", encoding="utf-8") as f:
        f.write(new)
    return True


def run(run_dir: Optional[str] = None, readme: Optional[str] = None,
        with_figures: bool = True) -> str:
    """Copy the figures, render the section, write it into README.md."""
    run_dir = run_dir or latest_run()
    if not run_dir:
        raise FileNotFoundError(
            f"no compared run under {os.path.abspath(paths.RESULTS)} — run "
            f"scripts/run_0_full_sweep.py first")
    readme = readme or os.path.join(paths.PROJECT_ROOT, "README.md")
    rows = read_rows(run_dir)
    figures = (copy_figures(run_dir, _best(rows, run_dir))
               if (with_figures and rows) else [])
    changed = update(readme, render(run_dir, figures))
    log.say(f"  run     : {os.path.abspath(run_dir)}")
    log.say(f"  budgets : {len(rows)}")
    log.say(f"  figures : {len(figures)} -> {DOCS_FIGURES}")
    log.say(f"  README  : {'updated' if changed else 'already up to date'}")
    return readme
