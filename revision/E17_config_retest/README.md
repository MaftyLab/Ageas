# E17 · Retest every configuration on the shared benchmark objects

Answers: which shipped model configurations, alone or as an Ageas ensemble, perform no worse than the manuscript reports on the benchmark objects in `Ageas.share/data`.

Related comments: R1.2 (which data choose the models), R2.4 (per-model-class and ensemble-vs-single performance), R2.5 (seed variation).

## Data

`revision/datasets.share.json` registers the objects described in `Ageas.share/data/README.md`, with that README's terminal/query splits and ground truth.

| Entry | Train | Query | Features |
|---|---|---|---|
| `iep_rna` | `celltype` dead-end / reprogramming, days 12 and 21 | 1,139 day-3 `early` cells; truth `lineage` | 1e4 + log1p, genes detected in ≥ 1% of cells |
| `iep_atac_peak` | `celltype` de / rep | 265 `early` cells; truth `lineage` | TF-IDF (Signac method 1, scale 1e4), all 50,000 peaks |
| `larry_bench` | `bench_group` Monocyte / Neutrophil | 3,078 HSPCs; truth `truth` | stored log-normalised X, genes detected in ≥ 1% of cells |
| `axolotl_d15` | Lineage-1 (rIPC1 + IMN + nptxEX) / Lineage-2 (rIPC2 + dpEX) | 327 reaEGCs; no truth | 1e4 + log1p, genes detected in ≥ 1% of all 2,182 cells |
| `axolotl_d15_published` | rIPC1 / rIPC2, as `docs/examples/spatial_example.ipynb` did for the manuscript figure | 327 reaEGCs; no truth | raw counts, 2,000 `seurat_v3` genes chosen on the six lineage clusters |
| `zheng68k_hvg2000` | all 11 cell types, stratified 5-fold cross-validation (`cv` block) | the held-out fold; truth `celltype` | 1e4 + log1p, 2,000 highly variable genes on all cells |

The `_hvg2000` entries use the manuscript Methods recipe instead (2,000 `seurat`-flavour highly variable genes). Zheng68K is not in `Ageas.share`: its object is `Dropbox/celltagmulti/anno/LiuBIB_5_Zheng68K(PBMC)_FACS_label.h5ad`, copied into `data/zheng68k/`.

## Method

1. `run_configs.py` pools every configuration in `data/configs`: 63 distinct among 72 files. A configuration shipped in several panels is trained once, and `pool_manifest.csv` records every panel it belongs to. Unit IDs are hashes of the configuration content.
   - **Legacy schema.** The `light_config` and `replay` panels (27 configurations) are written in an older schema and fail on the current package. They name the embedding width `latent_dim` and use the blocks `Basic_Block` and, for RNNs, `Residual_Encoder`. `translate_legacy()` maps these to `latent_fea_dim`, `ResNet_Basic` / `RNN_Basic` and `RNN_Residual_Encoder`. The mapping is inferred from the current panels, not from the old code, and the manifest's `translation` column flags every translated unit.
   - **Per-config runs.** For each configuration, seed 0 runs `n_kfold_selection` on that configuration alone with all gates open: 5-fold cross-validation on the terminal cells (the out-of-fold scores Ageas ranks by), then a refit on all terminal cells, whose query probabilities are saved. Further seeds can be single refits (`--no-cv`) to measure seed variation of the query predictions at a sixth of the cost.
2. `score.py` scores each configuration on the query cells and rebuilds ensembles offline for each panel:
   - `protocol` keeps the units that pass the frozen `protocol.json` gates on out-of-fold terminal accuracy, then the final out-of-fold gate. This is what Ageas does, and it never sees query labels. Seeds run with `--no-cv` use each unit's out-of-fold accuracy from the cross-validated seed.
   - `uniform` averages every unit of the panel.
   - `query_selected` averages the configurations that pass on the query cells.
3. `targets.json` holds the manuscript's metrics. How a configuration "passes" depends on the dataset:
   - **Lineage-traced datasets:** accuracy, macro F1 and AUROC are all at least the reported values (seed mean). A second check puts weighted F1 in place of macro F1, because the reported iEP ATAC F1 exceeds its accuracy, which two-class macro F1 cannot do. `configs.csv` also counts the seeds in which each check passes.
   - **Axolotl**, which has no ground truth: the mean P(Lineage-2) is ≤ 0.3 in the most ventral fifth of reaEGCs and > 0.8 in the most dorsal fifth. This is checked on raw probabilities and, as the published figure did, after min-max rescaling.
   - **Zheng68K:** the median accuracy and median macro F1 over the five held-out folds reach .828 and .716. Each configuration is fitted once per fold (`--no-cv`), so there are no out-of-fold scores and no `protocol` ensemble, only per-configuration and `uniform` results. SVC configurations are left out, because libsvm does not scale to 55,000 training cells. So are neural configurations with batch size below 20 (`--min-batch-size 20`), which need 19,000–41,000 optimizer steps per fit.

**Choosing configurations by their query performance uses the evaluation labels.** Numbers from `query_selected`, or from any configuration picked off `configs.csv`, are optimistic. `optimism.csv` measures by how much: it chooses the best configuration on a random half of the query cells and scores it on the other half. Only `protocol` numbers are free of this.

## Running

```bash
R="--registry revision/datasets.share.json --results-root <outside the shared folder>"
python revision/E17_config_retest/run_configs.py --dataset iep_rna $R --seeds 0 1 2 --accelerator cuda
python revision/E17_config_retest/score.py --dataset iep_rna $R
```

Lightning logs go to `<results-root>/E17/<dataset>/lightning/w<worker>`. Each worker needs its own folder, because workers sharing one pick the same version folder and break each other's metric logs. `--types` or `--units` restrict the pool, and `--type-order` runs cheap model types first. Interrupted runs resume.

`run_configs.py` limits BLAS, OpenMP and PyTorch to `--threads` (default 1) per worker, because parallelism comes from running several workers. Two problems made this necessary:
- **BLAS:** scikit-learn's libsvm computes dense kernel entries with one BLAS `ddot` each, so multi-threaded OpenBLAS turned a linear SVC on 540 axolotl cells into more than 15 minutes on 24 spinning threads. With one thread it takes about 70 s.
- **PyTorch:** with one 24-thread OpenMP pool per worker, the `torch.stack` that builds every batch stalled. One MNB configuration on 2,000 iEP ATAC cells took 586 s, against 11 s with one thread.

Several workers sharing one GPU also make tiny batches wait. `--cpu-max-batch 20` trains MLP and RNN configurations with batches of 20 or less on the CPU instead: a batch-10 MLP refit on iEP RNA took 51 s there, against 560 s on the shared GPU. The convolutional `resnet` type stays on the GPU, because on one core it was about ten times slower. CPU and GPU runs of these small-batch models differ about as much as two seeds do. The iEP ATAC runs keep everything on the GPU, where the 50,000-peak input layer is faster. The scikit-learn models run on one CPU core each, and the shipped logistic-regression configurations are slow. They use `saga` with an L1 penalty, which hits its 100-iteration cap without converging: about 60 s per fit on iEP RNA, so about 6 min per configuration and seed with 5 folds plus the refit. To share the pool between processes, build it once, then start the workers:

```bash
python revision/E17_config_retest/run_configs.py --dataset iep_rna $R --units none
for i in 0 1 2; do   # seed 0 with cross-validation
  python revision/E17_config_retest/run_configs.py --dataset iep_rna $R --seeds 0 --accelerator cuda \
    --reuse-pool --worker $i --workers 3 &
done; wait
for i in 0 1 2; do   # seeds 1 and 2 as single refits
  python revision/E17_config_retest/run_configs.py --dataset iep_rna $R --seeds 1 2 --no-cv --accelerator cuda \
    --reuse-pool --worker $i --workers 3 &
done; wait
```

Each worker with CUDA initialised holds about 3 GB of host memory, plus up to 3.5 GB while a 12,000-feature SVC fits. On iEP ATAC (50,000 features) a single SVC or `saga` fit holds several GB more.

`--dynamic` replaces the static split: every worker walks the whole list and claims each unit through a `probs/<unit>_seed<k>.claim` file, so workers can be added while others run. After killing workers, restart one with `--clear-claims` to release the units they had claimed. Zheng68K runs one held-out fold at a time:

```bash
for k in 0 1 2 3 4; do python revision/E17_config_retest/run_configs.py --dataset zheng68k_hvg2000 $R --units none --fold $k; done
for k in 0 1 2 3 4; do   # start this loop in as many workers as memory allows (--worker 0, 1, ...)
  python revision/E17_config_retest/run_configs.py --dataset zheng68k_hvg2000 $R --fold $k --seeds 0 --no-cv \
    --reuse-pool --dynamic --worker 0 --types logreg mlp mnb resnet rnn xgb --type-order mnb xgb resnet rnn mlp logreg \
    --accelerator cuda
done
python revision/E17_config_retest/score.py --dataset zheng68k_hvg2000 $R
```

## Outputs (`<results-root>/E17/<dataset>/`)

`pool_manifest.csv`, `runs.csv`, `probs/`, `configs.csv`, `ensembles.csv`, `ensembles_per_seed.csv`, `optimism.csv`, `summary.md`.
