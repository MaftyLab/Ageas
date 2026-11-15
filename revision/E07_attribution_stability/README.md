# E07 · Attribution stability, module-level aggregation and DE concordance

**Serves:** R2.9 (primary) · R1.3 · R3.3 · R3.8 (evidence tiers for hypotheses) · **P1**

## Reviewer comments

- **R2.9.** "In cases where predictors are highly collinear, feature attribution methods might not be identifiable … Lack of a stability analysis in the manuscript makes it unclear to what extend the presented features are complete and/or if they are reproducible."
- The repo's own open question (`Ageas.manuscript/README.md`): "Would extracted genes be easily identified with DEG analysis?"

## Before running

Run this only on the revision branch. Two fixes there change attributions:
- **A4:** binary logistic-regression and SVM attributions were assigned to the wrong class.
- **A6:** Integrated Gradients ran with dropout active.

On `main`, the stability numbers would partly measure those bugs.

## What it does

1. For each seed × stratified 80% subsample of the terminal cells, runs the frozen protocol and debriefs the deck (normalized OOF weights).
2. For each class it reports:
   - **Gene level:** pairwise top-k Jaccard index, rank-biased overlap, and per-feature selection frequency. Features in the top-k in ≥ 80% of runs are "stable".
   - **Module level:** features are clustered by |correlation| in terminal cells (average linkage, |r| ≥ 0.6), and each module's score is the sum of its features' positive attribution. Modules stay identifiable when correlated genes trade credit, so the expectation is module stability > gene stability.
   - **DE concordance:** overlap of the stable features with the top-k Wilcoxon DE features between terminal classes.

## Run

```bash
python revision/E07_attribution_stability/stability.py --dataset larry --n-subsamples 2 --top-k 50
```

With 10 seeds × 2 subsamples this gives 20 runs. Use `--seeds 0 1 2 3 4` to halve the cost.

## How it goes into the paper

- Mark stable features, or show modules, in Figs 2E, 2J, 3F and 5H.
- Add a supplementary figure of the Jaccard/RBO distributions, gene vs module.
- Add one sentence answering the DEG question explicitly.
- Stable and DE-concordant features feed Tier 1 of the R3.8 hypothesis table.

**Status:** code ready, not yet run on real data.
