import numpy as np
import cospar as cs
import scanpy as sc


adata = sc.read_h5ad('/data/jiangjunyao/public/cospar data/LARRY_adata_preprocessed.h5ad')
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

cs.tl.progenitor(adata_2,selected_fates=['Neutrophil','Monocyte'])
adata.obs['pred'] = np.nan
condition = adata_2.obs['progenitor_transition_map_Neutrophil'] == 1
adata_2.obs.loc[condition, 'pred'] = 'Neutrophil'
condition = adata_2.obs['progenitor_transition_map_Monocyte'] == 1
adata_2.obs.loc[condition, 'pred'] = 'Monocyte'
adata_final = adata_2[adata_2.obs['NeuMon_fate_bias']!=0.5]
adata_final = adata_final[adata_final.obs['NeuMon_mask']]
adata_final.obs['lineage']='Monocyte'
adata_final.obs.loc[adata_final.obs['NeuMon_fate_bias'] > 0.999, 'lineage'] = 'Neutrophil'

from sklearn.metrics import accuracy_score, recall_score, f1_score
adata_final = adata_final[~adata_final.obs['pred'].isna(), :]
acc = accuracy_score(adata_final.obs['lineage'].values, adata_final.obs['pred'].values)
acc

#adata_2.write('/data/jiangjunyao/AEGAS_analysis/pred_result/cospar_larry.h5ad')
adata_final.obs.to_csv('/data/jiangjunyao/fa_result/larry/cospar_larry_obs.csv')