# Part 5. Results

<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->

**Document**  `05_results.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
1. **1. Recovery against measurement budget** — 3 paragraphs
2. **2. Dependence on tolerance** — 3 paragraphs
3. **3. Runs that did not converge** — 1 paragraph

### Figures
1. `results_budget_ladder.png`
   > F1 at tau = 1 against the fraction of the plane measured. The line is the 40-points-per-ray ladder; the number beside each point is the ray count. The best converged budget is ringed. The open marker is a same-coverage comparison showing that more rays beats more points per ray. 4 budgets that did not converge are not shown.
2. `fig_tau_metrics.png`
   > Pixel accuracy, precision, recall, F1 and grid coverage against the tolerance tau, for three budgets, on 50 held-out devices. Pixel accuracy is shown only to be dismissed. Grid coverage is not a score: it is the fraction of the plane within tau pixels of a measured pixel, i.e. the area the scores to its left are read over.

<!-- INVENTORY:END  write your feedback below -->

## Notes

<!--
How to point at things:
  S2 P3        section 2, paragraph 3
  Fig. 4       figure 4 of this part
  ABSTRACT     the abstract
  FORMAT       font, spacing, margins, captions
Anything else in prose is fine too.

What can be changed, and where it lives:
  wording, a section, the abstract   paper_figures/make_method_docs.py
  a figure itself                    paper_figures/make_figures.py
                                     (or make_tau_grids.py /
                                      make_threshold_figure.py)
  which figures a part carries       the parts() list in make_method_docs.py
  fonts, spacing, margins            the format block in make_method_docs.py
-->

-

