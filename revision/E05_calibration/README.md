# E05 · Calibration and continuous clonal ground truth

**Serves:** R2.3 · R2.8 · **P1**

## Reviewer comments

- **R2.3.** Framing the benchmark as binary classification "is biased in favor of Ageas … This should be stated explicitly as a scope limitation."
- **R2.8.** "Because the ground truth is binarised and no calibration is reported, the benchmark is structurally incapable of revealing this" (that an output of ≈0.5 conflates boundary, hybrid, unprimed and out-of-distribution cells).

## What it does

It scores every method's probabilities against the **clonal fate fraction** from `benchmark/fate/ground_truth.py`, rather than only against binarized labels:

| Stratum | Metric |
|---|---|
| Unipotent clones | Brier score, ECE, reliability table |
| All progenitors with ≥ 3 fate-annotated relatives | Spearman ρ (predicted P(positive fate) vs clonal fraction) |
| Mixed clones only | Mean absolute error (predicted probability vs clonal fraction) |

The continuous and mixed-clone analyses need a clone-matrix ground truth (LARRY). CellTag-multi can use them too if its clone matrix is registered.

## Run

```bash
python revision/E05_calibration/calibration.py --dataset larry \
    --method CoSpar=revision/results/E02/larry/cospar_pseudotime2.csv \
    --method kNN30=revision/results/E02/larry/knn30.csv
```

## How it goes into the paper

- A supplementary figure of reliability diagrams for all methods.
- A table with ECE, Brier score, Spearman ρ against clonal fraction, and mixed-clone MAE.
- The "Scope of the benchmark" paragraph (T7) cites these numbers.

If Ageas is over-confident (high ECE), say so. The uncertainty outputs in E06 are the response to that.

**Status:** code ready, not yet run on real data.
