# paper_figures

The **neat** figures — one point per panel, built for the JJAP manuscript
and for slides. Ported from the SSDM deck's figure suite
(`static_RBC_test_noise_9_19_ML_edited/paper_figures/make_figures.py`) and
repointed at this repo's run.

The dense diagnostic gallery in
`results/<run>/figures/` (23 plots, fifteen labelled series each) is *not*
what goes in the paper. It is for reading a sweep, not for showing one.

## Build

```bash
python paper_figures/make_figures.py          # everything below
python paper_figures/make_tau_grids.py        # the tolerance definition
python paper_figures/make_threshold_figure.py # the validation curve
```

Every figure is written twice: `.png` at 600 dpi and `.pdf` (vector).

## What it draws on

`_run.py` is the single place that decides. Override with environment
variables rather than editing it:

| variable | default | meaning |
|---|---|---|
| `CSD_RUN` | `4-5-6-7-8_rays_40-50-60_points_500_samples` | the sweep folder under `results/` |
| `CSD_CONFIG` | `8_rays_50_points_500_samples` | the headline budget |
| `CSD_DEVICE` | *median test device* | e.g. `sample_11` |
| `P2L_TITLES=off` | titles on | writes `*_notitle` copies, for a deck that carries its own text boxes |
| `SPLIT_LABEL=off` | label drawn | writes `fig_data_split_notext` + the label's slot |

### The headline budget is 8 × 50, not 8 × 60

**Four of the fifteen trainings in this sweep did not converge:**

| config | best val F1@1 | test F1@1 |
|---|---|---|
| `5_rays_60_points` | 0.419 | 0.431 |
| `6_rays_50_points` | 0.416 | 0.427 |
| `7_rays_60_points` | 0.421 | 0.432 |
| `8_rays_60_points` | 0.438 | 0.445 |

Every healthy run reaches at least 0.660 on validation, so the gap is
unambiguous. `8 × 60` is one of the four, so it cannot carry the figures;
`8 × 50` is the best budget that did converge — **F1@1 0.796 at 3.9 % of the
plane**, threshold 0.70.

`_run.converged()` re-derives that list from each config's
`model/training_summary.json` every run, so retraining those four is enough
to make the figures follow. Note that `final_train_loss` is **not** the
right test: the checkpoint saved is the best-validation epoch, so 4 × 50 and
7 × 50 end at a high last-epoch loss (1.755, 1.630) on perfectly good
models. The criterion is `best_val_f1`, a number decided inside the training
devices and never on the test set.

`results_f1_vs_coverage` does not hide the four: they are drawn as open grey
circles, off the trend lines, and labelled *did not converge* in the legend.

### The device in the panels

`sample_11`, the **median** performer of the 50 held-out test devices for
this budget (F1@1 0.839 against a test mean of 0.796) — not the best one,
and not the SSDM deck's `picked_device_33`, which does not exist in this
repo and was in any case simulated in a 0.8 mV window the network was never
trained on.

Axes carry the device's real window: 2 × 2 mV, randomly offset per device.

## Figures

| file | what it is |
|---|---|
| `fig_measurement_panel` | charge sensor │ ground truth │ what the network is shown |
| `fig_network_input`, `_slide` | the two input channels, with and without caption |
| `fig_unet` | the network schematic |
| `fig_model_flow` | input → network → output in one picture |
| `panel_channel1/2`, `panel_probability` | the same maps as standalone panels |
| `panel_charge_sensor` | raw sensor, and with its slow background removed |
| `p2l_1 … p2l_9` | probability map → threshold → tolerance, one file per panel |
| `fig_probability_to_lines` | the same story as one composite |
| `tau_example`, `tau_neighbourhood` | what τ *is* — text-free, labels belong in the caption |
| `threshold_validation` | the threshold chosen on the validation split |
| `fig_data_split` | 550 → 500 + 50, then 425 + 75 |
| `fig_tau_metrics` | F1 / precision / recall against τ, three budgets |
| `results_f1_vs_coverage` | the headline chart |

`p2l_titles.json`, `tau_layout.json`, `threshold_validation.json` carry the
titles, panel centres and numbers, for a deck that places its own text.

## Two things to say out loud

* On this device the **looser** cut scores better at τ = 1 (P > 0.4 →
  0.865, chosen P > 0.7 → 0.839). That is not a bug: the threshold is fitted
  on the validation devices, not tuned per device. `threshold_validation`
  shows why it barely matters — the validation curve is flat from 0.3 to
  0.8 and only collapses past 0.9.
* F1@τ rises with τ by construction. It is a unit to quote, never a knob to
  tune, which is why τ = 0…3 are all reported and τ = 1 is the headline.
