# Part 3. Network architecture and training

<!-- INVENTORY:BEGIN  regenerated; do not write below this line -->

**Document**  `03_network_and_training.docx` / `.pdf`
**Rebuild**   `python paper_figures/make_method_docs.py`

### Sections
1. **1. Architecture** — 3 paragraphs
2. **2. Loss** — 1 paragraph
3. **3. Optimisation** — 2 paragraphs

### Figures
1. `fig_unet.png`
   > The network. Two input channels, an encoder of widths 32/64/128 with 2x2 max pooling, a bottleneck of width 256, a mirrored decoder with skip connections, and a 1x1 convolution to one output channel.
2. `fig_model_flow.png`
   > The whole flow: the two measured channels in, the U-Net, and one probability per pixel out.

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
