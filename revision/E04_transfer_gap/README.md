# E04 · Transfer gap: terminal classification vs fate inference

**Serves:** R2.6 · supports text edit T10 · **P1**

## Reviewer comment

- **R2.6.** "A model that maximally exploits terminal-state-specific features, that only appear after commitment, will classify terminal states best and transfer worst. The manuscript assumes a monotonic relationship between classification quality and fate-inference quality without testing it. Investigating in-distribution CV accuracy vs fate-inference accuracy across the model configurations could help elucidate this."

## What it does

Uses the full-panel outputs of E03, which contain every configuration × seed. For each configuration it pairs:

- the terminal-cell **out-of-fold** accuracy and macro F1, with
- the **progenitor** AUROC against lineage ground truth.

With E03 `--explain`, it also reports the share of each configuration's attribution that falls on **terminal-only features** (detected in fewer than 5% of progenitors), and correlates that share with transfer.

## Run

```bash
python revision/E03_ablation/train_full_panel.py --dataset larry --explain   # once
python revision/E04_transfer_gap/transfer_gap.py --dataset larry
```

## Reading the result

- **Positive, saturating relation:** classification accuracy is necessary but not sufficient. Rephrase L279–282 (T10).
- **Flat or negative at the top end, or terminal-only mass predicting worse transfer:** the reviewer's intuition holds. Report it, and consider one of the label-free method changes listed in the response plan (R2.6):
  - progenitor-aware feature filtering;
  - covariate-shift reweighting;
  - selecting on the terminal cells nearest the progenitors;
  - FateID-style self-training.

**Status:** code ready, not yet run on real data (depends on E03).
