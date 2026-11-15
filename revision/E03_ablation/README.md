# E03 · Ablation ladder and gene-order permutation check

**Serves:** R2.4 (primary) · R1.4 · feeds E04 and E09 · **P0**

## Reviewer comments

- **R2.4.** "It is unclear which components carry the performance of the methods: the ensemble, the AutoML search, the iterative masking, the deep architectures? … Reporting per-model-class performance and ensemble-vs-best-single-model would help clarify this."
- **R1.4.** "Please clarify the specific rule used to order genes/features and discuss whether model performance depends on this ordering."

## Scripts

1. `train_full_panel.py` trains **every** configuration once per seed with all gates open. It saves each unit's out-of-fold score and progenitor probabilities, plus explanations if `--explain` is set (needed by E04).
2. `ladder.py` builds these variants offline from those outputs:
   - best per family;
   - best single;
   - classical-only;
   - NN-only;
   - all configurations;
   - Ageas protocol gates (uniform and OOF-weighted);
   - L2-LR from E02.

   It scores each variant against the ground truth and reports mean ± s.d. over seeds with a pooled bootstrap CI. It also reports how often each configuration survives the protocol gates.
3. `permutation_check.py` reruns the protocol with permuted gene order. Expected outcome: the predictions agree, because the recurrence runs along the modality axis (length 1) and the first layer is dense. `tests/test_nn_explain_and_order.py` proves the same invariance exactly.

Iterative masking does **not** change predictions. It is used only for feature discovery, so it is not a rung on the ladder; E08 covers it.

## Run

```bash
python revision/E03_ablation/train_full_panel.py --dataset larry --accelerator cuda --explain
python revision/E03_ablation/ladder.py --dataset larry
python revision/E03_ablation/permutation_check.py --dataset larry --rnn-only
```

## Before running

- The shipped configs set `"max_epochs": 1` for every neural network (audit A7). Decide on the epoch budget in `protocol.json` / the config folder **before** this run. Otherwise the NN families may look weak only because they are under-trained.
- RNN configs carry `"block_num_layer"`, which `RNN_Basic` silently ignores (its argument is `num_layers`). Every recurrent configuration therefore has depth 1. Either fix the key or describe the depth as 1 in the Methods.
- The original example notebooks ran with `kfold_selection_list=[2]` and `retention_point=0.5`. Because τ_cut = 0.7 > τ_ret, selection effectively reduced to "keep every configuration with CV accuracy > 0.7". The `survival.csv` output shows how selective the frozen protocol actually is.

## How it goes into the paper

- A new figure: the ladder, plus a heatmap of configuration survival across datasets (the evidence for "data-adaptive").
- The permutation result answers R1.4 together with the corrected Methods text (T3).
- If a single L2-LR is close, report it and lean on worst-case robustness across datasets (response plan R2.4).

**Status:** all three scripts are smoke-tested on synthetic data (the classical-only test panel is exactly order-invariant). Not yet run on real data.
