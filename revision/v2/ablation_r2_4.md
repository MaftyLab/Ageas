# Ablation for reviewer comment R2.4: which components carry Ageas v2's performance

**Reviewer R2.4.** "In the current state of the manuscript it is unclear which components carry the
performance of the methods: the ensemble, the AutoML search, the iterative masking, the deep
architectures? Currently the method is presented monolithically. Reporting per-model-class
performance and ensemble-vs-best-single-model would help clarify this."

Agreed with the user on 2026-11-12: the ablation takes apart **v2** (the method the revision presents),
on the three lineage-traced real datasets (LARRY, iEP scRNA-seq, iEP scATAC-seq) and all 130 simulated
datasets, one run per setup (seed 0) with bootstrap intervals, and ends in a shareable report page.
Anything the saved runs cannot answer gets a new run.

## The reviewer's components in v2

| Reviewer's term | v2 component | Ablation |
|---|---|---|
| the ensemble | committee vote (log-odds, one voice per model type) joined by the no-harm mixture | vote alone; mixture alone; plain average of every unit |
| ensemble vs best single model | the final output against each unit | best single unit by v2's own validation (lowest shift-weighted out-of-fold log-loss); best single unit chosen with progenitor labels, shown only as an optimistic ceiling |
| per model class | 7 classes: logreg, svc, mnb, xgb (classical); mlp, resnet, rnn (deep) | each class's validation-best unit; every unit's score by class |
| the deep architectures | mlp, resnet, rnn units | v2 rebuilt from classical units only, and from deep units only |
| the AutoML search | the data profile's panel rules and the second round that extends a family past its edge | without the second-round units (offline); a fixed panel with rules and second round off (**new run**) |
| the iterative masking | routing: each unit narrowed to the features its attributions rank highest, as far as validation allows | routing off: every unit on all features (**new run**) |
| (v2 only) | graph stage on data with unlabeled intermediate cells | with and without (simulated data; never runs on the real data) |

## Variants and where they come from

Offline, from the saved round-5 runs (real: `v2/refine/round5`; simulated: `v2/sim` plus the graph
stage's `v2/graph`): every unit's progenitor probabilities (`v2_seed0_members.npz`), out-of-fold
records (`v2_seed0_oof.npz`: probabilities, labels, shift weights, folds) and the run record
(committee, no-harm weights, panel with each unit's `shaped_by`). Rebuilt with v2's own functions
(`competent_units`, `select_committee`, `vote`, `no_harm_selection`, `graph_stage`). The rebuild of
the full integration must reproduce the saved vote and mixture before any variant is trusted.

1. **Full v2**: the final output (`v2_final`).
2. **Without the graph stage**: the vote joined by the mixture (simulated data only; identical on real data).
3. **Vote alone**: the committee's vote without the mixture.
4. **Mixture alone**: the no-harm mixture.
5. **Plain average**: every unit, equal weight, no selection.
6. **Best single unit (validation)**: the unit with the lowest shift-weighted out-of-fold log-loss.
7. **Best single unit (labels)**: the unit with the best progenitor AUROC; optimistic, a ceiling only.
8. **Best unit of each class (validation)**: one variant per class.
9. **Classical units only** / **deep units only**: competence, mixture, committee and vote rebuilt on that subset, then the graph stage where it runs.
10. **Without second-round units**: the same rebuild without the units the second round added.

New runs (one per setup, seed 0, frozen code, through the usual queue and the graph stage):

11. **Routing off**: protocol `routing.mass_ladder = []`; everything else as the protocol.
12. **Fixed panel**: protocol `pipeline.rules = false` (no profile rule shapes the panel: middle
    regularisation and capacity, L2 penalties, one naive-Bayes smoothing; data-type checks and the
    network schedule by cell count stay) and `pipeline.refine = false` (no second round).

## Scoring

Accuracy, macro F1 (threshold 0.5) and AUROC with `binary_scores`, the code that scores the baselines.

- **Real datasets**: 1,000 bootstrap resamples of the query cells (seed 0, the same for every variant);
  each variant's 95% interval and its paired difference from full v2 with a 95% interval.
- **Simulated datasets**: per-dataset scores; per modality the mean and the paired difference from
  full v2, with a Wilcoxon signed-rank test over datasets and Holm correction across variants. The graph
  stage was tuned on seeds 1–5, so seeds 6–10 are also reported on their own.
- **Per class**: every unit's score with its class and validation loss.

Outputs in `/home/jack/ageas_work/v2/ablation/scores/`: `variants_real.csv` (bootstrap per real
dataset), `variants_sim.csv` (per simulated dataset), `summary_sim.csv`, `units.csv` and `checks.csv`
(the rebuild checks).

## Report page

1. The answer first: which components carry v2's performance, filled in from the data.
2. Ladder: each variant's change from full v2 with its interval, per dataset group, metric switch.
3. Per class: every unit's score by class, each class's validation-best unit marked, full v2 as a line.
4. Ensemble vs best single unit: a table with paired differences.
5. What each variant switches off.
6. A draft response paragraph for R2.4.

## Code

- `ageas.design_panel(profile, use_rules=True)` and `ageas.run_pipeline(rules=True)`; protocol
  `pipeline.rules` (default true). Tested at the panel seam.
- `revision/v2/run_job.py --protocol` and `revision/v2/run_queue.py --protocol`, so an arm runs its own
  protocol copy without touching `protocol.json`.
- `revision/v2/ablate.py`: the offline variants and the scoring; `--predictions` writes their per-cell predictions.

## Limits

- One seed: intervals reflect resampling cells, not training randomness.
- "Fixed panel" is v2 without its search, not R1's panel; R1's own ladder (`Ageas/revision/E03_ablation`) is not run.
- Best single unit by labels is optimistic by construction and is never presented as achievable.

## Results (2026-11-13)

All 266 new runs finished (routing off and fixed panel, 133 setups each, no failures). Rebuilding full
v2 from the saved runs reproduces its mixture, committee, vote and final output on all 133 setups.

Changes from full v2, AUROC unless stated. Real datasets in the order LARRY / iEP scRNA-seq /
iEP scATAC-seq, with 95% bootstrap intervals; simulated values are means over all 10 seeds.

- **Ensemble vs best single unit.** v2's own validation pick: −0.011 [−0.015, −0.007] / −0.010
  [−0.017, −0.001] / +0.019 [+0.007, +0.034], but iEP scATAC-seq accuracy −0.023. The unit chosen with
  labels (ceiling): −0.002 / +0.001 / +0.019. Simulated: validation pick −0.001 (RNA) and −0.006 (ATAC);
  ceiling +0.013 and +0.014.
- **How the ensemble combines.** Plain average: LARRY accuracy −0.036, macro F1 −0.046. Vote alone:
  iEP scRNA-seq accuracy −0.007. Mixture alone: LARRY AUROC −0.008.
- **Per model class.** Every class's best unit falls significantly below full v2 on at least one real
  dataset. The closest class differs by dataset: RNN on LARRY (−0.003), naive Bayes on iEP scRNA-seq
  (−0.002), RNN on iEP scATAC-seq (+0.019).
- **Deep architectures.** Classical units only: +0.000 / +0.000 / −0.002 (LARRY accuracy +0.011).
  Deep units only: −0.001 / −0.006 / +0.007 (iEP scATAC-seq accuracy −0.023).
- **AutoML search.** Fixed panel: −0.000 / −0.001 / +0.000; accuracy −0.005 / ±0.000 / +0.004. Without
  the second-round units: LARRY accuracy −0.020 (AUROC +0.000).
- **Iterative masking.** Routing off: −0.001 / −0.000 / +0.005; accuracy −0.008 / −0.005, and identical
  calls on iEP scATAC-seq. Routing takes most of the run time. With 7 jobs in parallel, a simulated
  dataset took a mean of 5.9 min with routing off, against 57.9 min for the fixed panel, which keeps
  routing.
- **Graph stage.** Off: −0.044 [−0.052, −0.037] on simulated RNA, −0.006 [−0.019, +0.008] on ATAC.
  Because the stage runs after every variant, the other variants barely move the simulated means.

In short: on the real datasets the ensemble's selection carries performance. On two of the three it
matches the label-chosen best unit, which v2 could not pick by itself. The panel search, routing and
deep units each change little. On simulated data the graph stage carries the ranking.

The report page stores its tables with six decimals so that its three-decimal display rounds only once.
