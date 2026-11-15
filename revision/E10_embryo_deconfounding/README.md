# E10 · Epiblast A–P result: deconfounding position from fate priming

**Serves:** R2.7 (primary) · R2.8 (epiblast part) · decides text edit T8 (abstract) · **P0**

## Reviewer comment

- **R2.7.** "In the embryo application, the A-P correlation is confounded by the definition of the training classes … Ageas's two training classes, Ecto and PS, are respectively an anterior population and a posterior population. A classifier discriminating them is therefore trained on an anterior-versus-posterior contrast. Applying it to epiblast cells and correlating the output with the A-P coordinate unsurprisingly recovers position, not fate."

A second warning sign is internal: the top Ecto-bias features in Supp. Table 3 are NCL, RPL28, RPL36A, SEM1, IGFBP2 and HBA1. These are ribosomal, housekeeping and hemoglobin genes, several of which correlate with A–P position.

## Analyses (`deconfound.py --analyses ...`)

| Analysis | Question | Key output |
|---|---|---|
| `provenance` | Where along A–P do the Ecto, PS and Epi training/query cells sit, and how much do the two terminal classes overlap? Also check *how* Ecto and Epi were annotated in Xiao et al. 2024 (manual step: record it in Methods). | `provenance.csv`, `terminal_axial_overlap` |
| `baseline` | Reproduces the manuscript's Pearson r (0.298) under the frozen protocol. | `baseline_pearson_r` |
| `position_matched` | Retrain only on Ecto/PS cells from the axial range where both occur, balanced per slice. Does the epiblast A–P trend survive? | `position_matched_pearson_r` |
| `within_position` | Among epiblast cells **at the same A–P position**, does P vary, and does that follow in-section geometry (distance to nearest PS / Ecto)? Partial Spearman with a within-slice permutation test. | `share_of_variance_within_position`, `within_position.csv` |
| `spillover` | Stereo-seq bins mix neighbours. Regress P on position + neighbour fractions of PS/Ecto + depth; correlate after dropping spots that touch PS/Ecto. | `spillover_regression.csv`, `spillover_free_pearson_r` |
| `technical` | Retrain without RPL/RPS/MT-/MRP/HB* genes; partial correlation with position given depth. | `no_technical_genes_pearson_r`, `partial_pearson_given_depth` |
| `position_control` | A classifier that only knows anterior vs posterior epiblast halves (cross-fitted). How similar are its predictions to Ageas? | `position_control_vs_ageas_spearman` |

The per-cell uncertainty calls for the epiblast come from E06 (`--dataset cs8_embryo`).

## Run

```bash
python revision/E10_embryo_deconfounding/deconfound.py --dataset cs8_embryo --accelerator cuda
```

The registry entry needs:
- `position_key` (slice index along A–P);
- `spatial_obsm` (in-section x, y);
- optionally `counts_metadata` (depth).

## Decision rule (agree on it before running)

- **Priming claim survives** if all three hold:
  - the A–P trend persists after position matching and after removing technical genes;
  - there is significant within-position variation that tracks in-section geometry;
  - the position-only control explains clearly less than Ageas.

  Keep the claim, reworded as "A–P graded states resembling ectoderm vs PS programs, with heterogeneity at fixed position" (T8, first variant).
- **Otherwise** demote the embryo to an illustrative application, change the abstract (T8, second variant), and state the confound as a limitation. Optional external check: Tyser et al. 2021 CS7 gastrula.

**Status:** code ready, smoke-tested on synthetic data (synthetic `slice` is deliberately fate-correlated), not yet run on the CS8 atlas.
