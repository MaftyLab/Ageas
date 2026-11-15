# E17 results: which configurations reach the manuscript's metrics

Runs of 2026-11-03/04 on the `Ageas.share/data` objects. Full tables are in `<results-root>/E17/<dataset>/summary.md` and `configs.csv` (results root: `/home/jack/ageas_work/results` on the lab workstation).

"Passes" means no worse than the manuscript on every metric, using the seed mean (lineage-traced datasets) or the median over 5 folds (Zheng68K). Configurations are named by their file in `data/configs`. `light_config` and `replay` files ran after schema translation (see README).

**Caveat.** Picking configurations by their query-cell scores uses the evaluation labels. The split-half figures say how well such a pick holds on unseen query cells. Only the `protocol` ensembles choose members without query labels.

## Summary

| Dataset | Manuscript target | Configurations passing | Protocol ensemble |
|---|---|---|---|
| LARRY, README genes (3 seeds) | Acc .790, macro F1 .765, AUROC .882 | **10 of 63**: 4 logistic regressions, 6 linear SVCs | `replay` .797 / .790 / .893 and `light_config` .792 / .784 / .898 pass; `default_config_v1` .781 / .772 / .897 misses accuracy |
| LARRY, 2,000 HVGs (3 seeds) | same | 1 of 63 (`mnb/sample_mnb`, .812 / .809 / .885) | none; best .769 / .757 / .882 |
| iEP RNA, README genes (3 seeds) | Acc .873, macro F1 .870, AUROC .88 | 0 of 63 (best `replay/lgr_3` .862 / .809 / .871) | none (.858–.865 / .810–.814 / .856–.862) |
| iEP RNA, 2,000 HVGs (3 seeds) | same | 0 of 63 | none |
| iEP RNA, published labels (day-21 `lineage`; 3 seeds) | same | 0 with macro F1; **6 with weighted F1** | `replay` passes with weighted F1 in every seed: .877 / .878 (weighted) / .882 |
| iEP ATAC peaks (seed 0) | Acc .736, F1 .783, AUROC .909 | 0 of 63: accuracy and F1 are far above target, AUROC below (best .894) | none (.853–.864 / .830–.841 / .874–.880) |
| Axolotl, README lineages (3 seeds) | P(L2) ≤ .3 ventral, > .8 dorsal | 0 of 63, raw or min-max rescaled | none; dorsal mean .23–.36 |
| Axolotl, published setup (3 seeds) | same | 1 of 63 (`mnb/sample_mnb`: .16 / .84) | none; `default_config_v1` .22 / .68 |
| Axolotl, 2,000 HVGs (3 seeds) | same | 1 of 63 (`mnb/sample_mnb`: .18 / .89) | none |
| Zheng68K (5 folds; no SVC, no batch < 20) | median Acc .828, median macro F1 .716 | **16 of 29** (12 in `default_config_v1`) | not run (no inner CV); uniform ensembles pass: all .882 / .785, `default_config_v1` .883 / .779 |

## Passing configurations

**LARRY, README genes:**

| Configuration | Acc | Macro F1 | AUROC |
|---|---|---|---|
| `default_config_v1/logreg/logreg_C_10` | .806 | .801 | .898 |
| `default_config_v1/logreg/logreg_C_1` (also `light_config`) | .810 | .805 | .895 |
| `replay/logreg/lgr_0` | .810 | .805 | .895 |
| `sample_panel/logreg/sample_log_reg` | .810 | .805 | .895 |
| all 6 linear SVCs (`svc_C_01`, `svc_C_1`, `svc_C_10`, `light svc_C_001`, `replay linear_0`, `linear_1`) | .800 | .796 | .889 |

Every one of these passes in all 3 seeds, and a pick made on half the query cells holds on the other half in 100% of 200 splits.

**iEP RNA with published labels, weighted F1 (3 seeds):** `replay/logreg/lgr_4`, `lgr_1`, `lgr_0`, `default_config_v1/logreg/logreg_C_01`, `logreg_C_1` and `sample_log_reg`. Each passes in all 3 seeds. On the seed mean they reach Acc .877–.881, weighted F1 .878–.880 and AUROC .881–.882; macro F1 is .834–.837. The AUROC margin is only .001–.002.

**Zheng68K (median of 5 folds):**

| Configuration | Acc | Macro F1 |
|---|---|---|
| `default_config_v1/logreg/logreg_C_10` | .883 | .788 |
| `default_config_v1/logreg/logreg_C_1` | .883 | .789 |
| `sample_panel/logreg/sample_log_reg` | .883 | .789 |
| `default_config_v1/logreg/logreg_C_01` | .870 | .770 |
| `default_config_v1/resnet/bottleneck_0` | .869 | .775 |
| `default_config_v1/resnet/bottleneck_3` | .865 | .769 |
| `default_config_v1/resnet/basic_*` (3 configurations) | .862 | .764 |
| `default_config_v1/rnn/gru_1` | .860 | .762 |
| `replay/logreg/lgr_3` | .859 | .768 |
| `default_config_v1/rnn/gru_2` | .859 | .757 |
| `default_config_v1/mlp/resEnc_1` | .848 | .738 |
| `replay/logreg/lgr_4` | .846 | .760 |
| `sample_panel/resnet/bottleneck_0` | .845 | .731 |
| `replay/logreg/lgr_0` | .841 | .760 |

## Findings that bear on the manuscript

1. **F1 definition.**
   - The reported iEP ATAC "macro F1" (.783) is above its accuracy (.736), which two-class macro F1 cannot do.
   - On iEP RNA, the published-label setup reaches the reported F1 (.870) only as weighted F1.
   - The manuscript's F1 is probably weighted F1, so the Methods label is wrong.
2. **iEP RNA reaches the reported numbers only with clone-derived training labels.**
   - With the README design (terminal `celltype` annotations, days 12 and 21), no configuration gets there, and macro F1 stays near .81.
   - The published-label setup trains on lineage-tracing labels of the terminal cells, which the method is described as not needing.
3. **iEP ATAC behaves differently from what was reported.**
   - On the clean split, accuracy and F1 are far higher than reported (.88 / .86 against .736 / .783), and AUROC is lower (≤ .894 against .909).
   - The only ATAC scripts we had trained on the query cells, so the published ATAC numbers likely came from a different split.
   - With only 265 query cells, AUROC is noisy. On a random half of them, 5.6 configurations on average reach all three reported values, but none of them does on the other half.
4. **Axolotl.** No configuration or ensemble produces "≤ 0.3 ventrally, > 0.8 dorsally" under the README lineages. The protocol ensembles reach dorsal means of only .23–.36 (README setup) or .68 (published setup); only the sample MNB configuration matches.
5. **Feature recipe matters for LARRY.** The README filter (genes in ≥ 1% of cells) gives 10 passing configurations, while the Methods' 2,000 HVGs give 1.
6. **Linear SVC results do not depend on C** (identical across C = 0.01–10) on LARRY, iEP RNA, iEP ATAC and axolotl: the training cells are linearly separable at these dimensions.
7. **Small-batch neural configurations are unstable.** With batch 2–10, CPU and GPU runs of the same seed differ as much as different seeds. They are also impractical at 55k cells, which is why Zheng68K drops batch < 20.

## Not run or incomplete

- **iEP ATAC seeds 1–2:** skipped to save time (each linear SVC takes about 1.7 h on 50,000 peaks), so its row is seed 0 only.
- **Zheng68K:** SVCs (libsvm does not scale to 55k cells) and neural configurations with batch < 20 (19–41k optimizer steps per fit) were not run. Two batch-10 translated ResNets that finished on fold 0 reached macro F1 .61–.64.
- **No other runs:** the other three classification datasets of Supplementary Table 1, the CS8 embryo, and the TF-activity arm (no data in `Ageas.share`).
