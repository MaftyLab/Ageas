# E14 · Axolotl: raw probabilities, axis profile, injury-site control

**Serves:** R3.5 (primary) · R3.7 · fixes the axolotl dataset description (audit M1) · **P2** (but the description fix is P0)

## Reviewer comments

- **R3.5.** "Can the authors enhance their discussion of the axolotl results with more literature context? To what extent are more dorsal reaEGCs actually Lineage-2-generating?"
- **R3.7.** "The case where Ageas performs 'better' (i.e., axolotl) does not appear to have the same ground-truth lineage tracing."

## Problems found while preparing this

1. **Tissue.** The cited source (Wei et al., *Science* 2022) is titled "…axolotl **brain** regeneration". The manuscript says *spinal cord* and "15 days post-amputation". Check the tissue, the injury model, the time unit, and the IMN/nptxEX/dpEX expansions against the paper.
2. **Rescaled probabilities.** `docs/examples/spatial_example.ipynb` min-max rescales the ensemble probabilities over the whole matrix before plotting. The values reported as "0.0–0.3 ventrally to >0.8 dorsally" may therefore be stretched, not probabilities, while CoSpar's were shown unscaled.
3. **Training classes.** The same notebook overwrote the training set with **rIPC1 vs rIPC2** (intermediate progenitors), not the terminal neurons the manuscript names. Decide which design is intended and state it in the Methods (registry field `terminal_classes`).

## What it does

- Writes **raw** P(lineage) for the reaEGCs under the frozen protocol.
- Profiles P along a section axis (binned), next to CoSpar if given.
- With `--distance-to`, gives the partial correlation with the axis after controlling for distance to the injury site. This tests whether the "dorsal" enrichment is just wound proximity.

Optional extension: the source study has several timepoints. Run `E02/timeaware_oracle.py` on the time series to get a real-time reference for the 15-DPI predictions.

## Run

```bash
python revision/E14_axolotl_temporal/axolotl_temporal.py --dataset axolotl_d15 \
    --cospar revision/results/E02/axolotl_d15/cospar_pseudotime2.csv --distance-to wound_distance
```

## How it goes into the paper

- Corrected dataset description (M1).
- An unscaled Fig. 4E.
- "Consistent with the source study" rather than "outperforming" (no lineage ground truth).
- A short literature paragraph on where rIPC1/rIPC2 and dpEX arise.

**Status:** smoke-tested on synthetic data, not yet run on the axolotl section.
