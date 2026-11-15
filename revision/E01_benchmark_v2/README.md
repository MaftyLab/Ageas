# E01 · Benchmark v2: seeds, confidence intervals, paired tests

**Serves:** R2.5 (primary) · R1.2 · R3.7 · every "outperforms" sentence in the Results (text edit T9) · **P0**

## Reviewer comments

- **R2.5.** "Ageas is a stochastic, seed-dependent ensemble with a stochastic selection step. However, no uncertainty estimates are provided, which is especially important given the small differences in performance (e.g. AUROC 0.882 (Ageas) vs 0.872 (CoSpar) in the hematopoiesis dataset)."
- **R1.2.** Asks for an independent final evaluation set. Here, lineage-traced progenitors are used *only* for scoring, under the frozen `revision/protocol.json`.

## What it does

1. `run_ageas.py` runs the frozen protocol for every seed (seeds change the k-fold splits, model initialization and therefore selection) and stores each seed's progenitor probabilities, per-unit probabilities and surviving units.
2. `score.py` scores Ageas (seed-averaged) and any other method on **the same progenitors** against the shared clonal ground truth (`benchmark/fate/ground_truth.py`, unipotent clones). It reports:
   - accuracy, macro F1, AUROC, Brier score and ECE, each with a 95% bootstrap CI over cells;
   - mean ± s.d. across seeds for Ageas;
   - DeLong's test for AUROC and paired bootstrap tests for accuracy and F1 against Ageas, Holm-adjusted.

## Run

```bash
python revision/E01_benchmark_v2/run_ageas.py --dataset larry --accelerator cuda
python revision/E01_benchmark_v2/score.py --dataset larry \
    --method CoSpar=revision/results/E02/larry/cospar_pseudotime2.csv \
    --method kNN30=revision/results/E02/larry/knn30.csv \
    --method FateID=revision/results/E02/larry/fateid.csv
```

Repeat for `iep_rna`, `iep_atac_peak` and `iep_atac_tf`. Other methods' CSVs must follow the shared prediction format: indexed by cell, with `prob_<class>` columns. E02 writes this format directly. For the original CoSpar/WOT/CellRank outputs, rename the probability columns.

## Outputs (`revision/results/E01/<dataset>/`)

`metrics.csv`, `comparisons.csv`, `ageas_seed*.csv`, `ageas_seed*_members.npz`, `ageas_seed*_units.csv`.

## How it goes into the paper

- Replot Figs 2D/2I/3D/3E as points with 95% CI error bars.
- Supp. Table 2 gains CI columns and a comparisons sheet.
- Where a CI overlaps, or the Holm-adjusted p ≥ 0.05, change "outperforms" to "comparable" (expected for the LARRY AUROC).
- Draft reply: response plan, R2.5.

**Status:** code ready, smoke-tested on synthetic data, not yet run on the real datasets.
