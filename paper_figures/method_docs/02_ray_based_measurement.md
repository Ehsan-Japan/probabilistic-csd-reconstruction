# Part 2. Ray-based measurement and network input

<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->

**Document**  `02_ray_based_measurement.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
1. **1. The ray fan** — 4 paragraphs
2. **2. Encoding the measurement** — 3 paragraphs

### Figures
1. `fig_measurement_panel.png`
   > The measurement, for one held-out device. Left: the charge-sensor map. Centre: the ground-truth transition lines. Right: what the network is shown at 8 rays x 50 points, which is 3.9 % of the grid.
2. `panel_channel1.png`
   > Channel 1: the measured charge-sensor signal, retained only at the pixels a ray visited.
3. `panel_channel2.png`
   > Channel 2: the visited mask. Black marks every pixel a ray passed through, 3.9 % of the grid.
4. `fig_network_input.png`
   > The two input channels side by side. Channel 2 is what makes a zero in channel 1 unambiguous.

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
