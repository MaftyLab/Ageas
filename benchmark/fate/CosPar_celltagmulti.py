import numpy as np
import scanpy as sc
import cospar as cs



adata = sc.read_h5ad('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep.prcessed.h5ad')
cs.pp.get_highly_variable_genes(adata)
adata.obs['time_info'] = 'late'
adata.obs.loc[adata.obs['celltype'].isin(['early']), 'time_info'] = 'early'
adata.obsm['X_emb'] = adata.obsm['X_umap']
adata.obs['state_info'] = adata.obs['celltype']
adata.obsm['X_pca'] = adata.obsm['X_scVI']
cs.pp.get_X_clone(adata,clone_data_barcode_id=adata.obs['barcodes'],clone_data_cell_id=adata.obs_names)
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
    use_full_Smatrix=True,compute_new=True
)

cs.tl.progenitor(adata_2,selected_fates=['dead-end','reprogramming'])
adata.obs['pred'] = np.nan
condition = adata_2.obs['progenitor_transition_map_reprogramming'] == 1
adata_2.obs.loc[condition, 'pred'] = 'reprogramming'
condition = adata_2.obs['progenitor_transition_map_dead-end'] == 1
adata_2.obs.loc[condition, 'pred'] = 'dead-end'
adata_final = adata_2[adata_2.obs.time_info.isin(['early','trans'])]
df = adata_final.obs
sum_prob = (
    df['fate_map_transition_map_reprogramming']
    + df['fate_map_transition_map_dead-end']
)


# Normalize probabilities
df['reprogramming_norm'] = (
    df['fate_map_transition_map_reprogramming'] / sum_prob
)
df['deadend_norm'] = (
    df['fate_map_transition_map_dead-end'] / sum_prob
)

df.to_csv('/data/jiangjunyao/fa_result/celltag_multi/cospar_celltagmulti.csv')
