# E12 · Per-class metrics and confusion matrices (classification benchmark)

**Serves:** R2.m1 (primary) · R1.2 (nested evaluation of the classification benchmark) · **P1**

## Reviewer comment

- **R2 minor 1.** "Please add per-class metric tables and confusion matrices for all four datasets, and report the per-class difference against the best baseline. This would help understand where the stated performance improvements are coming from (e.g. gains on abundant classes vs rare classes)."

## What it does

1. `run_ageas_cv.py` runs an outer stratified 5-fold CV, with Ageas's own k-fold selection **nested inside** each outer training fold. Check that the original Supp. Fig. 1 numbers were produced this way; if not, these runs replace them.
2. `per_class.py` reads every `predictions_*.csv` (Ageas plus the baselines from `benchmark/clf`, converted to the columns `cell, true, pred, fold, method`). It reports:
   - per-class precision, recall and F1 (mean ± s.d. over folds);
   - row-normalized confusion matrices;
   - per-class ΔF1 against the best baseline, next to class abundance.

## Registry entries

Add `zheng68k`, `fu`, `macparland` and `zhang` with:
- `terminal_classes` = all annotated cell types;
- `query_values: []`;
- `ground_truth: null`.

Fix the dataset naming at the same time:
- Zhang is **colon** cancer (GSE146771), not lung as Methods L1123 says.
- Supp. Table 1 / Supp. Fig. 1 call the Fu dataset "Liu".

## Run

```bash
python revision/E12_per_class_metrics/run_ageas_cv.py --dataset zheng68k --accelerator cuda
python revision/E12_per_class_metrics/per_class.py --dataset zheng68k
```

**Status:** both scripts smoke-tested on synthetic data. Not yet run on the four datasets; the baseline predictions need converting from the existing scANVI/SingleR/Seurat/TOSICA outputs.
