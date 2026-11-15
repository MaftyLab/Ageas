# E11 · Progenitor separability in unsupervised space

**Serves:** R2.m2 (primary) · R1.1 (context for the reference-mapping baseline) · **P1**

## Reviewer comment

- **R2 minor 2.** In Fig. 3A/B the scATAC Early-Dead and Early-Repr cells "form two cleanly separated groups with a visible gap, each sitting on the side of its own eventual terminal state … the authors should quantify how separable the Early-Dead and Early-Repr cells are in unsupervised chromatin space (silhouette score, or classification accuracy from LSI components alone) and temper the 'priming' language accordingly. If the progenitors already lie inside the terminal manifold, 'transfer of fate memory' is vacuous on this dataset."

## What it does

For each dataset, using progenitors with unipotent lineage labels and an embedding fitted without labels (PCA, or LSI without component 1):

- the silhouette of the two fate groups;
- the 5-fold cross-validated accuracy of logistic regression and kNN on the embedding alone;
- the share of each progenitor's 15 nearest terminal neighbours that belong to its eventual fate.

Running it on iEP-ATAC, iEP-RNA and LARRY places Fig. 3 against Fig. 2A (distinct Early cluster) and Fig. 2G (intermixed HSPCs).

## Run

```bash
python revision/E11_progenitor_separability/separability.py --datasets iep_atac_peak iep_rna larry
```

Also check the sampling time of the ATAC "Early" cells: if they include later days, that explains why they sit inside the terminal clusters.

## How it goes into the paper

- A supplementary table, plus one sentence in the ATAC section.
- If ATAC progenitors are highly separable (silhouette ≫ RNA, LSI-only accuracy close to Ageas), remove the "priming / transfer of fate memory" interpretation for that dataset. Present it as evidence of modality support and TF-activity-before-expression divergence (response plan R2.m2).

**Status:** code ready, not yet run on real data.
