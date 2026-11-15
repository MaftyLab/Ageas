# E16 · (Optional) Cancer drug-resistance lineage dataset

**Serves:** R3.8 (utility beyond development) · R3.1 (therapy-resistance priming) · R2.8 (hybrid states) · **P2**, do only if time allows

## Reviewer comments

- **R3.8.** "How do the authors envision this model being most useful? … What are the examples from the datasets the authors have already analyzed that represent new insights into the trajectories studied? Any testable hypotheses of particularly high confidence?"
- **R3.1.** Literature on primed states "in the contexts of development, cellular reprogramming/transdifferentiation, and cancer therapy resistance".
- **R2.8** mentions hybrid luminal/basal tumour states as a case where ≈0.5 outputs matter.

## Proposal

Add one barcoded cancer dataset where pre-treatment cells are linked to resistant or non-resistant fates. This tests Ageas, and the E06 uncertainty outputs, outside development. Candidates (verify that data and clone tables are public):

- **FateMap:** Goyal et al., *Nature* 2023, melanoma resistant-fate diversity;
- **Watermelon:** Oren et al., *Nature* 2021, cycling vs non-cycling persisters;
- the two lineage-tracing studies cited by Reviewer 3 (PMID 39930084, 41862498). Look them up first, since they may be the most persuasive choice.

## How to run it with the existing code

1. Add a registry entry:
   - `terminal_classes`: the post-treatment fates (e.g. resistant subtypes);
   - `query_values`: the pre-treatment cells;
   - `ground_truth`: a clone matrix linking them.
2. Run, in order:
   - E01 (`run_ageas.py` and `score.py`);
   - E02 `reference_mapping.py`;
   - E06 `run_uncertainty.py`, for hybrid vs unprimed pre-treatment states;
   - E15 `oracle_de.py`.

No new code is needed.

**Status:** not started; needs a dataset decision.
