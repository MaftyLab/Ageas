# Cell Systems R1 revision workspace

This directory holds every new analysis for the first-round revision. Each experiment has its own folder, and each folder's `README.md` names the reviewer comments it answers. The point-by-point plan lives in the manuscript repository (`Ageas.manuscript/reviewer_response_plan_R1.html`).

Branch: `revision/cell-systems-r1`.

## 1 · Fixes applied to the package on this branch

These fixes come first. Every new number must be produced **after** them, and some change published results (see §5).

| ID | Problem (on `main`) | Fix | Regression test | Reviewer |
|---|---|---|---|---|
| A4 | Binary logistic-regression / linear-SVM attributions assigned to the wrong class. On `main`, class 1's top gene is class 0's marker. The multi-class branch normalized over the wrong axis and skipped the class-contrastive step. MNB had the same axis error. | `Classifier_Template.class_coefficients()` orients coefficients by `classes_` (SVC: one-vs-one aggregation); per-class L1 + `class_contrast`. | `tests/test_linear_attribution.py` | R2.9, R1.3 |
| A5 | The SVC `StandardScaler` was refit on every prediction batch (`Deck.predict` batches 10 cells). On `main` the same cells' probabilities change by up to 0.98 with batching. | Scaler fitted in `forward()` only. | `tests/test_svc_scaler.py` | R2.5 |
| A6 | Integrated Gradients ran in `train()` mode, with dropout active. `stratify_dataset` also applied positions within each class's cells to the whole corpus, so IG explained classes on the wrong cells. | `eval()` mode, cuDNN disabled for the RNN backward pass. IG explains each class on its own correctly predicted cells; class subsets keep the parent's label maps. | `tests/test_nn_explain_and_order.py` | R2.9 |
| A1 | Not a code bug: the RNN recurs over the modality axis (length 1), not over genes. | Documented. Tests prove the sequence length is 1 and that the dense first layer imposes no gene order. | `tests/test_nn_explain_and_order.py` | R1.4 |
| A2 | The final retention gate scored refitted models on their own training data. | `final_gate='oof'` (default) uses out-of-fold scores; `'test'`, `'insample'` (legacy) and `'none'` also available. | `tests/test_selection_and_debrief.py` | R1.2 |
| A11 | Absolute accuracy gates; `'min'` metrics compared the wrong way; selection could return an empty ensemble. | Direction-aware gates, optional `relative_retention`, never-empty fallback, per-round OOF stored in `unit.report[op]['round_k']` and `['final']['oof']`. | same | R1.5 |
| A3 | Eq. 25 vs code: weights were 1/in-sample cross-entropy, unnormalized; iterations summed unnormalized. | `Deck.debrief` / `Deck.unit_weights`: OOF-score weights normalized to sum to 1. `n_iter_extraction` L1-normalizes each iteration. | same | R1.5, R1.3 |
| — | `Fake_Trainer` computed `CrossEntropyLoss` on probabilities (it re-applies softmax), flattening every classical model's `CEL`. | Log-loss on probabilities. | indirectly | R1.5 |
| — | `Hangar.sortie_generate` handed out the *same* `Unit` objects to every run, so a later seed or iteration overwrote earlier decks' models. | Deep copies per sortie. | `tests/test_selection_and_debrief.py` | R2.5 |
| — | `n_iter_extraction` mutated the caller's `exp_dataset`; masking parameters were hidden; there was no record of iterations. | Copies the input; exposes `outlier_iqr_factor`, `remove_outliers`, `correlation_masking`; `return_history=True`. | same | R1.3 |
| A8 | Benchmark ground truth: CoSpar and CellRank mapped `NeuMon_fate_bias` in opposite directions; mixed clones were forced into one class; CellRank columns were renamed in swapped order and used real days. | `benchmark/fate/ground_truth.py` (clone-matrix labels with mixed category). Both LARRY scripts use it. See `benchmark/fate/README.md`. | `tests/test_ground_truth.py` | R2.3, R2.5 |
| new | No per-cell uncertainty. | `ageas/tool/uncertainty.py`: entropy decomposition, OOD score, program scores, state calls. `Deck.predict(..., return_members=True)`. | `tests/test_uncertainty.py` | R2.8, R3.6 |

## 2 · Layout and conventions

```
revision/
  protocol.json            FROZEN selection protocol (seeds, K, rho, tau, alpha, N) — decide once
  datasets.example.json    registry template -> copy to datasets.json and fill paths
  common/                  registry/preparation, runner, metrics (CIs, DeLong, calibration), synthetic data
  E01_benchmark_v2/ ... E16_cancer_resistance/
  v2/                      Ageas v2 (ageas.run_pipeline): its protocol, benchmark scripts and the R2.4 ablation
  results/                 outputs (git-ignored): results/<EXP>/<dataset>/...
```

- **Prediction files:** CSV indexed by cell, with one `prob_<class>` column per terminal class and a `pred` column. Every method (Ageas, baselines, legacy outputs) is converted to this format and scored by the same code.
- **Ground truth:** always from `benchmark/fate/ground_truth.py` via the registry. It produces one label table per dataset, with `mixed` and `no_info` kept as separate categories and clonal fractions available.
- **No progenitor labels in any design choice.** Selection, thresholds and the search space come only from `protocol.json`. Lineage-traced progenitors are used for scoring only (R1.2).
- **Fresh hangar for every run** (`revision/common/runner.run_ageas`); `seed_everything` per seed.

## 3 · Experiment index

| Dir | What | Reviewer comments | Pri. | Status |
|---|---|---|---|---|
| [E01_benchmark_v2](E01_benchmark_v2) | Seeds, bootstrap CIs, DeLong / paired tests, shared ground truth | R2.5, R1.2, R3.7 | P0 | smoke-tested |
| [E02_baselines](E02_baselines) | Centroid / kNN / L2-LR; FateID, STEMNET; Palantir, CellRank 2 time-free; CoSpar real-time oracle | R1.1, R2.1, R2.2, R2.3 | P0 | reference mapping + R export smoke-tested; Palantir/CellRank/CoSpar/R scripts untested |
| [E03_ablation](E03_ablation) | Ablation ladder, configuration survival, gene-order permutation | R2.4, R1.4 | P0 | smoke-tested |
| [E04_transfer_gap](E04_transfer_gap) | Terminal OOF accuracy vs progenitor AUROC; terminal-only attribution mass | R2.6 | P1 | smoke-tested |
| [E05_calibration](E05_calibration) | Reliability, ECE, Brier; continuous clonal fraction; mixed clones | R2.3, R2.8 | P1 | smoke-tested |
| [E06_uncertainty](E06_uncertainty) | Entropy / disagreement / OOD / state calls; mixed-clone validation | R2.8, R3.6, R2.7 | P1 | smoke-tested |
| [E07_attribution_stability](E07_attribution_stability) | Seed × subsample stability, modules, DE concordance | R2.9, R1.3, R3.3 | P1 | smoke-tested |
| [E08_masking_diagnostics](E08_masking_diagnostics) | Per-iteration diagnostics; redundancy of later tiers | R1.3 | P1 | smoke-tested |
| [E09_hparam_sensitivity](E09_hparam_sensitivity) | ρ / τ grid (offline), α × N grid | R1.5 | P1 | selection grid smoke-tested |
| [E10_embryo_deconfounding](E10_embryo_deconfounding) | Position-matched training, within-slice, spillover, technical genes, position-only control | R2.7, R2.8 | **P0** | smoke-tested (synthetic) |
| [E11_progenitor_separability](E11_progenitor_separability) | Silhouette / LSI-only accuracy / terminal-neighbour share | R2.m2, R1.1 | P1 | smoke-tested |
| [E12_per_class_metrics](E12_per_class_metrics) | Nested CV; per-class P/R/F1, confusion, ΔF1 vs best baseline | R2.m1, R1.2 | P1 | smoke-tested |
| [E13_open_set](E13_open_set) | Unseen-fate progenitors, unlisted terminal cells, senescence | R3.6, R2.8 | P1 | smoke-tested |
| [E14_axolotl_temporal](E14_axolotl_temporal) | Raw (unscaled) probabilities, axis profile, wound-distance control | R3.5, R3.7 | P2 | smoke-tested |
| [E15_lineage_concordance](E15_lineage_concordance) | Oracle DE vs Ageas features; blind spot | R3.3, R3.8, R2.9 | P1 | smoke-tested |
| [E16_cancer_resistance](E16_cancer_resistance) | Optional cancer drug-resistance lineage dataset | R3.8, R3.1, R2.8 | P2 | plan only |
| [v2](v2) | Ageas v2 (`ageas.run_pipeline`): frozen protocol, benchmark on the lineage-traced and simulated datasets, comparison with the baselines, and the ablation of its components | R2.4 | P0 | done; ablation results in `Ageas.share/test.result/ablation` |

Comments answered by writing only (no experiment): R2.1/R2.2 citations, R3.1, R3.2, R3.4, R3.7 table, R3.8 use-case paragraph, and the Methods corrections T3/T4. Drafts are in the response plan.

## 4 · Setup and running order

```bash
pip install -e . scanpy pytest            # + cospar, palantir, cellrank, R FateID/STEMNET for E02
cp revision/datasets.example.json revision/datasets.json   # fill paths and FILL_ME fields
OMP_NUM_THREADS=4 python -m pytest tests  # unit + smoke tests, Ageas v2 included (~4–5 min on CPU; see tests/README.md)
```

Suggested order:
1. Fill `datasets.json` and **freeze `protocol.json`** (§6).
2. E01 + E02 + E03 on LARRY, iEP-RNA, iEP-ATAC.
3. E10 (it decides the abstract).
4. E06, E05, E13.
5. E07, E08, E09, E04, E11, E12, E15.
6. E14 and E16 if time allows.

## 5 · Effect on published results

- **Driver-gene figures (2E, 2J, 3F, 5H; Supp. Table 3).** These used ensembles that mixed IG/SHAP with sign-flipped linear attributions (A4), and their IG was stochastic and ran on the wrong cells (A6). Regenerate them all.
- **Progenitor predictions.** Any prediction from an ensemble that contained an SVC was affected by per-batch rescaling (A5).
- **Ensemble membership.** The final retention gate (A2) and the debrief weights (A3) change which models are kept and how features are ranked.
- **Legacy behaviour** is reproducible for the gate (`final_gate='insample'`) and approximately for the weights (`debrief(monitor_metric='vali.CEL', monitor_type='min', normalize_weights=False)`). A4 and A5 have no legacy switch, by design.
- **Response letter:** say that implementation issues were found and corrected during revision, and that all results were regenerated.

## 6 · Decisions needed from the authors before the real runs

1. **`protocol.json`:** K, monitor metric, ρ, τ_cut, τ_ret (or relative δ), α, N. The originals were K=2 and τ_ret=0.5. Note that with τ_cut=0.7 > τ_ret, selection reduced to "CV accuracy > 0.7".
2. **NN training budget:** every shipped config has `"max_epochs": 1` (A7).
3. **RNN depth:** configs use `"block_num_layer"`, which `RNN_Basic` ignores (its argument is `num_layers`). All recurrent configs therefore have depth 1. Fix the key or report depth 1.
4. **Axolotl training classes:** the terminal neurons named in the manuscript, or rIPC1 vs rIPC2 as in `docs/examples/spatial_example.ipynb`. Also correct the tissue (brain, per the cited paper) (E14).
5. **iEP ground-truth labels:** fill the `FILL_ME` mapping of `state_info` values in `datasets.json`.
6. **Explained cells (A10):** Integrated Gradients uses correctly classified cells of each true class, SHAP uses true-class cells, and linear models use global coefficients. Make the Methods text say exactly that. For IG this holds only after the A6 cell-selection fix.
