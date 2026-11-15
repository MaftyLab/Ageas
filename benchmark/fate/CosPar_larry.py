import numpy as np
import pandas as pd
import cospar as cs
import scanpy as sc
from sklearn.metrics import accuracy_score

from ground_truth import binary_eval_view, check_bias_orientation, clonal_fate_table

FATES = ('Monocyte', 'Neutrophil')

adata = sc.read_h5ad('/data/jiangjunyao/public/cospar data/LARRY_adata_preprocessed.h5ad')

# Ground truth from the clone matrix (shared with every other method), computed
# before time_info is replaced by the two pseudo-timepoints below.
gt = clonal_fate_table(
    adata,
    fates=FATES,
    progenitor_mask=adata.obs['state_info'] == 'undiff',
    clone_obsm='X_clone',
    state_key='state_info',
    time_key='time_info',
)
print('NeuMon_fate_bias vs clonal Neutrophil fraction (Spearman):',
      check_bias_orientation(gt, adata.obs['NeuMon_fate_bias'], 'Neutrophil'))
gt.to_csv('/data/jiangjunyao/fa_result/larry/larry_ground_truth.csv')

cs.pp.get_highly_variable_genes(adata)
time_info = np.array(adata.obs["state_info"])
time_info[time_info != 'undiff'] = "late"
time_info[time_info == 'undiff'] = "early"
adata.obs["time_info"] = time_info
adata = cs.pp.initialize_adata_object(adata)
adata_2 = cs.tmap.infer_Tmap_from_state_info_alone(
    adata,
    initial_time_points=["early"],
    later_time_point="late",
    initialize_method="OT",
    OT_cost="GED",
    smooth_array=[20, 15, 10],
    max_iter_N=[1, 3],
    sparsity_threshold=0.2,
    use_full_Smatrix=True,
)

cs.tl.progenitor(adata_2, selected_fates=list(FATES))
obs = adata_2.obs
prob = pd.DataFrame({
    fate: obs[f'fate_map_transition_map_{fate}'] for fate in FATES
})
prob = prob.div(prob.sum(axis=1), axis=0)
pred = pd.Series(np.nan, index=obs.index, dtype=object)
for fate in FATES:
    pred[obs[f'progenitor_transition_map_{fate}'] == 1] = fate

result = pd.concat([prob.add_prefix('prob_'), pred.rename('pred')], axis=1)
result.to_csv('/data/jiangjunyao/fa_result/larry/cospar_larry_pred.csv')

evaluable = binary_eval_view(gt, FATES).join(result, how='inner')
evaluable = evaluable[evaluable['pred'].notna()]
print('CoSpar accuracy (unipotent clones):',
      accuracy_score(evaluable['binary_label'], evaluable['pred']))
