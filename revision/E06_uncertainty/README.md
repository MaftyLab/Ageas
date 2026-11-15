# E06 · Per-cell uncertainty, out-of-distribution and fate-state calls

**Serves:** R2.8 (primary) · R3.6 · R2.7 (epiblast interpretation) · **P1**, a new method output

## Reviewer comments

- **R2.8.** "Ageas outputs P(fate | x) and discards P(x). Consequently an output of ≈0.5 can be produced by biologically distinct situations … 1. a cell lying on the decision boundary, 2. a cell co-expressing both lineage programs (e.g. hybrid primed states), 3. a cell expressing neither programs (e.g. unprimed progenitors), 4. out-of-distribution data points … Reporting per-cell predictive entropy and ensemble disagreement, alongside the prediction probability, would be valuable."
- **R3.6.** "How does it handle unlabelled cells that are terminally differentiated to an unseen fate … What about senescent cells?" The OOD score is the first line of defence; E13 tests it.

## Method (`ageas/tool/uncertainty.py`, part of the package)

| Output | Definition |
|---|---|
| `total_entropy` | H[mean over members of p_m] |
| `aleatoric` | Mean over members of H[p_m] |
| `epistemic` | `total_entropy` − `aleatoric` (mutual information = ensemble disagreement) |
| `knn_distance`, `reference_quantile`, `is_ood` | Mean distance to the k nearest terminal cells in a terminal-fitted PCA/SVD space, compared with the terminal-to-terminal distribution (flag above the 99th percentile) |
| `program_<fate>` | Mean z-score of that fate's top Ageas features |
| `state` | `ood` › `model_disagreement` › `confident:<fate>` › `hybrid` (both programs high) / `unprimed` (both low) / `boundary` |

Program thresholds come from the terminal cells: "high" is at or above the fate's median in its own terminal cells, and "low" is at or below its median in the other fate's terminal cells. `Deck.predict(..., return_members=True)` supplies the per-unit probabilities.

## Run

```bash
python revision/E06_uncertainty/run_uncertainty.py --dataset larry      # validation on mixed clones
python revision/E06_uncertainty/run_uncertainty.py --dataset cs8_embryo # epiblast: primed vs unprimed
```

## Validation and reading

- **LARRY:** `validation.csv` tests whether progenitors from mixed-fate clones have higher entropy or disagreement, and more hybrid or boundary calls, than unipotent ones. This is the evidence that the new outputs mean something.
- **Epiblast:** if most cells are `unprimed`, `boundary` or `ood`, report it. It is consistent with pluripotency and must temper the R2.7 claim.
- Thresholds (`--confident-prob 0.7`, `--max-epistemic 0.05`) belong in the frozen protocol. Choose them once, on terminal cells or LARRY, not on the epiblast.

**Status:** module unit-tested (`tests/test_uncertainty.py`), script smoke-tested on synthetic data, not yet run on real data.
