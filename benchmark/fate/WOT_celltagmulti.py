import wot
import numpy as np
import scanpy as sc




adata = sc.read_h5ad('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep.prcessed.h5ad')
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True,flavor='seurat_v3')
adata.obs['day'] = 2
adata.obs.loc[adata.obs['celltype'] == 'early', 'day'] = 1

ot_model = wot.ot.OTModel(adata,epsilon = 0.05, lambda1 = 1,lambda2 = 50) 
#tmaps = ot_model.compute_transport_map(2,1,tmap_out='/data/jiangjunyao/easyGRN/processed_data/mouse_hematopoiesis/ot/')
ot_model.compute_all_transport_maps(tmap_out='/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep/ot1/')
celltype_dict = adata.obs.groupby('celltype').apply(lambda x: x.index.tolist()).to_dict()
print(celltype_dict.keys())
tmap_model = wot.tmap.TransportMapModel.from_directory('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep/ot1/')
target_destinations = tmap_model.population_from_cell_sets(celltype_dict, at_time=2)
fate_ds = tmap_model.fates(target_destinations)
max_indices = np.argmax(fate_ds.X, axis=1)

max_var_names = [fate_ds.var_names[idx] for idx in max_indices]

result = list(max_var_names)
fate_ds.obs['pred'] = result

common_index = adata.obs.index.intersection(fate_ds.obs.index)
fate_data = fate_ds.obs.loc[common_index, 'pred']
adata.obs.loc[common_index, 'pred'] = fate_data
adata.obs['prob1'] = fate_ds.X[:,0]
adata.obs['prob2'] = fate_ds.X[:,1]

adata.obs.to_csv('/data/jiangjunyao/fa_result/celltag_multi/wot_celltagmulti.csv')

