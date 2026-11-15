# Tests

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -B -m pytest tests -p no:cacheprovider   # ~4–5 min on CPU
```

Limit BLAS and OpenMP to a few threads. Unlimited, the tiny test models spread over every core and the v2 tests take many times longer. `-B` and `-p no:cacheprovider` keep caches out of synced folders.

| File | Covers | Audit item | Reviewer comment |
|---|---|---|---|
| `test_linear_attribution.py` | Logistic regression, linear SVC (binary and 3-class) and MNB attributions rank each class's marker gene first, with the correct sign | A4 | R2.9, R1.3 |
| `test_svc_scaler.py` | SVC probabilities are identical whether cells are scored together, in 10s or one by one; prediction does not refit the scaler | A5 | R2.5 |
| `test_nn_explain_and_order.py` | Integrated Gradients is deterministic and leaves the model in eval mode; the RNN's sequence axis has length 1; permuting genes together with the embedder's columns gives the same function | A6, A1 | R2.9, R1.4 |
| `test_selection_and_debrief.py` | OOF scores recorded; final gate is out-of-fold; never-empty fallback; relative retention; `debrief` = normalized weighted mean; separate runs don't share models; `predict(return_members=True)`; extraction history, per-iteration L1 normalization, caller's dataset untouched | A2, A3, A11 | R1.2, R1.5, R1.3, R2.5 |
| `test_ground_truth.py` | Clonal fate counts, mixed / no_info labels, later-only relatives, orientation check | A8 | R2.3, R2.5, R2.8 |
| `test_uncertainty.py` | Entropy decomposition, OOD flagging, program thresholds, state calls | E06 | R2.8, R3.6 |
| `test_revision_metrics.py` | DeLong matches sklearn AUROC and detects differences; bootstrap CI; paired bootstrap; ECE; Holm; offline selection gates replicate `n_kfold_selection` | E01, E03 | R2.5 |
| `test_revision_smoke.py` | Every experiment script E01–E15 runs end to end on synthetic lineage data | all | all |
| `test_package_exports.py` | Every name in the `__all__` of `ageas`, `ageas.nn`, `ageas.classical` and `ageas.ops` can be imported from it; `from ageas import *` brings the classifiers, network blocks and pipeline; top-level classes are the subpackages' own | — | — |

Ageas v2, the feature-routed pipeline (`ageas.run_pipeline`):

| File | Covers |
|---|---|
| `test_profile.py` | Far vs mixed progenitors; progenitors in a cluster of their own; ambiguity between classes; counts vs continuous input; a boundary no line can draw |
| `test_panel.py` | Every family the input allows, loadable as a hangar; each rule's effect on regularisation, capacity, naive Bayes, penalties and SVCs; every unit names the rules behind it; a range extended one step where validation prefers its edge; with the rules off, no profile rule shapes the panel |
| `test_shift.py` | Progenitor-like terminal cells weigh more; mean 1; near 1 without a shift; capped |
| `test_moe_selection.py` | The out-of-fold record covers every cell once and records which `ageas` it used. Folds equal R1's, and no label reaches its own out-of-fold prediction. Uniform weights equal ordinary validation, and weights decide what is integrated. Second-round units are routed in the same folds and integrated. Also covers experts, `predict` and `debrief`, and that nothing is written outside `cache_dir` |
| `test_synergy.py` | Complementary units kept; overconfident unit and duplicates left out; redundant early pick dropped; every kept unit is needed; cell weights decide what is worth combining |
| `test_committee.py` | Competent units within two points of the best, ranked by loss. Near-identical units count once, and a unit at odds with the rest is dropped. The vote is confident where members agree and hedges where they split; each model type can get one voice; a log-odds vote lets the surer member count for more. The no-harm mixture joins the committee and can count as much as all of it |
| `test_strategy.py` | Each rule borne out or contradicted; the two validations can disagree; overall agreement |
| `test_routing.py`, `test_explain.py` | Key-feature sets and the ladder search; network attributions don't depend on cell order |
| `test_graph.py` | Walks through unlabeled cells send progenitors to their own branch; LSI input; the fused ranking and the walk's share; the vote passes through without enough unlabeled cells or with more than two classes |
| `test_pipeline.py` | The whole pipeline on synthetic data, from profile to predictions, with and without unlabeled cells |

The behaviours guarded by the A4 and A5 tests were reproduced on `main` with the same synthetic data:
- A4: class 1's top gene is the class-0 marker;
- A5: SVC probabilities move by up to 0.975 with batching.
