# Part 4. From probability map to transition lines: threshold and tolerance

<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->

**Document**  `04_threshold_and_tolerance.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
1. **1. Binarising the output: the threshold** — 3 paragraphs
2. **2. Scoring the output: the tolerance** — 3 paragraphs
3. **3. Reporting** — 2 paragraphs

### Figures
1. `threshold_validation.png`
   > Choosing the threshold. Mean F1 at tau = 1 over the 75 validation devices for every candidate cut, with the stored value ringed. The curve is flat from 0.3 to 0.8 and collapses only past 0.9.
2. `p2l_1_probability.png`
   > The U-Net output for one held-out device: a probability per pixel that a transition line passes through it.
3. `p2l_2_threshold_0p7.png`
   > The same map cut at the chosen threshold, P > 0.7.
4. `p2l_3_threshold_0p9.png`
   > The same map cut too high, at P > 0.9. Lines break up and are lost.
5. `tau_example.png`
   > Why a tolerance is needed. Left to right: the ground truth; the same line drawn one pixel to the left; the two superimposed, adjacent everywhere and coincident nowhere; and the same pair with the tau = 1 band. Strict F1 is 0.00 and F1 at tau = 1 is 1.00.
6. `tau_neighbourhood.png`
   > What tau admits around one pixel the model drew, for tau = 0, 1, 2 and 3. Distances are Euclidean, so a diagonal neighbour (1.41 px) counts only from tau = 2.
7. `p2l_4_tau0.png`
   > One prediction scored at tau = 0.
8. `p2l_5_tau1.png`
   > The same prediction scored at tau = 1, the headline tolerance.
9. `p2l_6_tau3.png`
   > The same prediction scored at tau = 3.
10. `p2l_9_tau_zoom.png`
   > The same truth and the same output at tau = 0 to 3, zoomed until individual pixels are visible. Only the tolerance band grows.
11. `p2l_7_tolerance_curve.png`
   > F1 against tau for the headline budget, with the precision and recall behind it, over the 50 held-out devices.

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
