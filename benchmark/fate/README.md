# Fate-inference benchmark scripts

Scripts that ran the competitor methods (CoSpar, WOT, CellRank) for the
original submission. Paths point to the authors' server and are kept for
provenance. The revision re-runs every method through
[`revision/E01_benchmark_v2`](../../revision/E01_benchmark_v2) and
[`revision/E02_baselines`](../../revision/E02_baselines), which score all
methods against one shared ground-truth table.

## Shared ground truth — `ground_truth.py`

`clonal_fate_table()` derives each progenitor's clonal fate counts, fate
fractions and label (`<fate>` / `mixed` / `no_info`) directly from the
cell-by-clone matrix. Every method, including Ageas, must be scored against
this table (reviewer comments R2.3, R2.5, R2.8).

## Issues fixed on the revision branch

| Script | Issue | Fix |
|---|---|---|
| `CosPar_larry.py`, `Cellrank_larry.py` | Opposite mappings of `NeuMon_fate_bias > 0.999` (Neutrophil in CoSpar, Monocyte in CellRank); every clone ≤ 0.999 forced into the other class. | Both use `clonal_fate_table()`; `check_bias_orientation()` prints what `NeuMon_fate_bias` actually measures. |
| `Cellrank_larry.py` | Fate-probability columns were named twice, in opposite orders. | Names taken from the CellRank `Lineage` object. |
| `Cellrank_larry.py` | Used the real `day` column, while the manuscript says real time was withheld from all methods and CoSpar/WOT received two pseudo-timepoints. | `USE_REAL_TIME = False` by default (two pseudo-timepoints, as for the others); `True` is kept as the time-aware upper bound used in E02. |

## Known issues not yet addressed here

* `Cellrank_*.py` use a `RealTimeKernel` on pseudo-timepoints. The time-free
  CellRank 2 modes (CytoTRACE / pseudotime kernels) requested by R2.2 are in
  `revision/E02_baselines/snapshot_methods.py`.
* `CosPar_celltagmulti.py`, `WOT_*`, `Cellrank_celltagmulti.py` still read
  labels inline; E01 re-scores their saved predictions against the shared
  table.
* `docs/examples/spatial_example.ipynb` min-max rescales the ensemble
  probabilities over the whole matrix before plotting, so the plotted values
  are not probabilities (relevant to the axolotl Fig. 4E claim of
  "0.0–0.3 ventrally to >0.8 dorsally"). The same notebook trains on
  rIPC1 vs rIPC2 rather than on the terminal neurons named in the manuscript.
  E14 re-runs the axolotl analysis without rescaling.
