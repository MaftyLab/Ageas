# E09 · Sensitivity to ρ, τ_cut, τ_ret, α and N

**Serves:** R1.5 · A11 (portability of absolute thresholds) · **P1**

## Reviewer comment

- **R1.5.** "Please provide the AutoML search space and specify the values and selection criteria for ρ, τ_cut, τ_ret, the number of masking iterations N, and the IQR multiplier α."

The reply has three parts:
- **Report the values.** They are in `revision/protocol.json`, which is the frozen default for every revision run. The original runs used `kfold_selection_list=[2]` and `retention_point=0.5`; say so.
- **Report the search space** from the config folder (see the top-level README checklist).
- **Show robustness** with this experiment.

## What it does

- **Selection grid (offline).** Uses E03's full-panel outputs, so nothing is retrained. Grid:
  - ρ ∈ {0.25, 0.5, 0.75, 1.0};
  - τ_ret ∈ {0.8, 0.9, 0.95};
  - τ_cut ∈ {0.5, 0.6, 0.7};
  - plus relative-retention δ ∈ {0.01, 0.02, 0.05, 0.1} (the new portable gate).

  For each setting it reports the number of surviving configurations and the progenitor AUROC / macro F1, per seed.
- **Masking grid (`--masking`, retrains).** α ∈ {1.5, 3, 10} × N ∈ {1, 2, 3}. Reports the top-k Jaccard of each setting's features against the protocol default.

## Run

```bash
python revision/E03_ablation/train_full_panel.py --dataset larry   # once
python revision/E09_hparam_sensitivity/sensitivity_grid.py --dataset larry --masking
```

## How it goes into the paper

- A supplementary table: the parameter values plus the grid. The expected outcome is a flat AUROC across the grid.
- Methods text on Eq. 25 normalization (T4) and on the relative-retention option.
- If AUROC moves a lot with τ, adopt the relative gate as the default **before** the final E01 run, and say so.

**Status:** selection grid smoke-tested on synthetic data via E03 outputs. The masking grid is untested.
