# E15 · Concordance with lineage-tracing findings ("oracle" DE)

**Serves:** R3.3 (primary) · R3.8 (evidence tiers) · R2.9 (external check on attributions) · **P1**

## Reviewer comment

- **R3.3.** "There should be some comparison between the results of Ageas and the original lineage-tracing experiments. If the goal of lineage-tracing experiments was to identify informative or important features of fate-biased progenitors, which features did lineage-tracing identify that Ageas did not?"

## What it does

- **Oracle DE.** Wilcoxon rank-sum between progenitors of each ground-truth fate, i.e. what lineage tracing reveals (BH-adjusted).
- **Ageas features.** Selection frequency from E07 when available, otherwise a fresh debrief.
- **Terminal DE.** Wilcoxon between terminal classes. This defines what any transfer approach *can* see.
- Reports precision and recall of the Ageas top-k against the oracle top-k, and lists every oracle feature Ageas misses with a flag for whether it is DE between terminal states. Missed features that are *not* terminal-DE are the expected blind spot of transfer learning, and the paper should name them as such.

Also compare by hand with the factors reported in the original studies:
- Weinreb et al. 2020 (LARRY);
- Jindal et al. 2024 (CellTag-multi: fate-specific TF and chromatin changes in early cells), which is also needed for the AP-1 rewrite (R3.4).

## Run

```bash
python revision/E07_attribution_stability/stability.py --dataset larry   # optional, gives stable ranks
python revision/E15_lineage_concordance/oracle_de.py --dataset larry
python revision/E15_lineage_concordance/oracle_de.py --dataset iep_atac_tf
```

## How it goes into the paper

- A supplementary figure and table: precision/recall, plus the missed features annotated as "not terminal-DE".
- One Discussion paragraph on the blind spot.
- The response to R3.3.

**Status:** code ready, not yet run.
