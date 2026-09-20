# Part 1. Device simulation and dataset construction

<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->

**Document**  `01_device_simulation_and_dataset.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
1. **1. Simulated devices** — 4 paragraphs
2. **2. Ground truth** — 1 paragraph
3. **3. Train/test split** — 4 paragraphs

### Figures
1. `panel_charge_sensor.png`
   > One simulated device. Left: the charge-sensor map as measured, dominated by a smooth gate cross-talk background. Right: the same data with that slow background differenced away, where the honeycomb of charge transitions becomes visible.
2. `fig_data_split.png`
   > The train/test split. 550 devices are divided by device identity into 500 training and 50 held-out devices (seed 12345). Inside the training devices, 15 % is carved out as a validation split before any training, leaving 425 devices to fit the weights.

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
