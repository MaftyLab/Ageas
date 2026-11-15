# E08 · Iterative masking: diagnostics and redundancy of later-tier features

**Serves:** R1.3 (primary) · text edit T5 · **P1**

## Reviewer comment

- **R1.3.** "Later-emerging features are not necessarily deeper regulatory factors; they may also represent correlated substitutes for features removed in earlier iterations. Please provide basic robustness information for the iterative masking procedure, including the number of features removed and predictive performance across iterations, and examine whether major later-tier features are strongly correlated with earlier-tier features."

## Changes on this branch that this experiment relies on

`ageas.n_iter_extraction` now:
- L1-normalizes each iteration before summing, so iterations with more surviving models no longer dominate (R1.5);
- exposes the masking parameters `outlier_iqr_factor` (α) and `remove_outliers`;
- offers **correlation-aware masking** (`correlation_masking=0.6`), which also masks features correlated with each outlier;
- returns a per-iteration `history`.

Check which N and α were used for each figure: the code defaults are N = 1 and α = 10, while `scripts/test.py` used N = 2.

## What it does

1. For each masking iteration it reports:
   - features in;
   - models kept;
   - mean out-of-fold accuracy;
   - progenitor AUROC;
   - features masked (outliers, plus correlated features if enabled).
2. For each feature first masked in iteration ≥ 2 it computes, in terminal cells and in progenitors:
   - the maximum |Spearman| correlation with any earlier-tier feature;
   - the cross-validated R² of predicting it from all earlier-tier features (ridge).

   Both are compared with an expression-decile-matched random feature.

## Run

```bash
python revision/E08_masking_diagnostics/masking_diagnostics.py --dataset larry --iterations 3
python revision/E08_masking_diagnostics/masking_diagnostics.py --dataset larry --iterations 3 --correlation-masking 0.6
```

## Reading the result

- If progenitor AUROC stays flat after the dominant markers are masked, predictive information is highly redundant, which is the reviewer's point.
- Later-tier features with |ρ| ≥ 0.5 or a high R² are **substitutes**: label them that way.
- Rewrite "multi-tiered hierarchy of cell fate drivers" as "additional fate-discriminative features" (T5).
- Correlation-aware masking is the optional method change. Report how its tiers differ.

**Status:** code ready, not yet run on real data.
