# E02 · Baseline panel: reference mapping, snapshot fate methods, time-aware oracle

**Serves:** R1.1 · R2.1 · R2.2 · R2.3 · **P0**

## Reviewer comments

- **R1.1.** "Please include a simple and transparent reference-mapping baseline, such as nearest-centroid/correlation-based mapping or kNN label transfer, evaluated under the same progenitor/terminal split."
- **R2.1.** FateID trains classifiers on terminal fate clusters and applies them to progenitors. The reviewer asks what Ageas adds and requests FateID benchmarks.
- **R2.2.** "STEMNET, Palantir, and CellRank2 run with a CytoTRACE or pseudotime kernel (i.e. CellRank in its time-free mode)" are missing.
- **R2.3.** The benchmark should state that two fabricated timepoints handicap transition-map methods.

## Scripts

| Script | Methods | Notes |
|---|---|---|
| `reference_mapping.py` | `centroid_pearson`, `centroid_spearman`, `knn15`, `knn30`, `l2_logreg` | Fitted on terminal cells only, so query cells never shape the embedding. Also writes `ambiguous_by_similarity.csv`: the lowest quartile of the centroid-correlation gap, i.e. the cells that terminal-state similarity cannot place. |
| `export_for_r.py` + `fateid_stemnet.R` | FateID, STEMNET | `Rscript fateid_stemnet.R revision/results/E02/<dataset>`. Check argument names against the installed package versions. |
| `snapshot_methods.py` | Palantir, CellRank 2 (CytoTRACE / pseudotime / connectivity kernels) | Terminal states are fixed to the annotated classes. Needs `palantir`, `cellrank>=2`, `scanpy`. CytoTRACE is RNA-only. |
| `timeaware_oracle.py` | CoSpar with 2 pseudo-timepoints vs real sampling times | The real-time run is an upper bound, not a competitor (R2.3). |

L2-LR and the best single configuration from E03 also double as the simplest "is the ensemble needed?" baselines (R2.4).

## Run

```bash
python revision/E02_baselines/reference_mapping.py --dataset larry
python revision/E02_baselines/export_for_r.py --dataset larry && \
  Rscript revision/E02_baselines/fateid_stemnet.R revision/results/E02/larry
python revision/E02_baselines/snapshot_methods.py --dataset larry
python revision/E02_baselines/timeaware_oracle.py --dataset larry
# score everything together (E01), then the ambiguous stratum:
python revision/E01_benchmark_v2/score.py --dataset larry --method kNN30=revision/results/E02/larry/knn30.csv ...
python revision/E01_benchmark_v2/score.py --dataset larry --method ... \
    --subset revision/results/E02/larry/ambiguous_by_similarity.csv --tag _ambiguous
```

## How it goes into the paper

- Revised Fig. 2 and 3 with three groups: reference mapping, snapshot fate methods, and time-aware methods (two pseudo-timepoints, plus a real-time upper bound).
- The **ambiguous-by-similarity** result is the key R1.1 panel. If Ageas still separates fates among cells that centroid similarity cannot place, it captures more than terminal-state similarity.
- If kNN or FateID ≈ Ageas, report it and reposition the contribution (response plan R1.1/R2.1 contingencies).

**Status:** `reference_mapping.py` and `export_for_r.py` are smoke-tested on synthetic data. `fateid_stemnet.R`, `snapshot_methods.py` and `timeaware_oracle.py` are written but untested, because R, Palantir, CellRank and CoSpar aren't installed in the test environment. Expect small API fixes on the first real run.
