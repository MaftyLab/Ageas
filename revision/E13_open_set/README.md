# E13 · Open-set tests: unseen fates, unlisted terminal cells, senescence

**Serves:** R3.6 (primary) · R2.8 (validation of the OOD output) · **P1**

## Reviewer comment

- **R3.6.** "Does Ageas assume that unlabelled cells are in fact going to give rise to one of whatever fates is labelled among the other cells? How does it handle unlabelled cells that are terminally differentiated to an unseen fate but not correctly labelled by the user? What about senescent cells?"

## What it does

Trains the frozen protocol on the listed fates (LARRY: Monocyte vs Neutrophil), then tests how well the E06 scores flag cells outside that closed world:

1. **Other-fate progenitors.** Progenitors whose clonal relatives are mostly an unlisted fate (Baso, Mast, Meg, Erythroid, …), identified from the clone matrix. Reports the AUROC of each score (OOD distance, entropy, epistemic, 1 − max prob) for separating them from listed-fate progenitors.
2. **Unlisted terminal cells.** Terminal cells of the unlisted types scored as query. Compares the fraction flagged OOD with the fraction of ordinary progenitors flagged.
3. **Senescence** (`--senescence-genes`, e.g. a SenMayo list in mouse or human symbols). Correlation of a senescence program score with each uncertainty score.

## Run

```bash
python revision/E13_open_set/open_set.py --dataset larry --senescence-genes path/to/senmayo_mouse.txt
```

The registry entry needs `extra_keep_values` listing the unlisted fates, and a clone-matrix ground truth.

## How it goes into the paper

- One supplementary panel.
- A Methods/Discussion sentence stating the closed-world assumption.
- The new "unassigned/ood" output (E06) is the mitigation.

If the AUROCs are near 0.5, say plainly that Ageas cannot detect unseen fates and that users must list all plausible fates.

**Status:** code ready, not yet run on real data.
