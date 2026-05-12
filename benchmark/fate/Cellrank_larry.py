import os
import scanpy as sc
import pandas as pd
import cellrank as cr
from cellrank.kernels import RealTimeKernel
from moscot.problems.time import TemporalProblem



# Set number of threads for parallel computation

os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["VECLIB_MAXIMUM_THREADS"] = "40"
os.environ["NUMEXPR_NUM_THREADS"] = "40"

adata1 = sc.read_h5ad('/data/jiangjunyao/fa_result/config_tunning/larry_train_hvg.h5ad')
adata2 = sc.read_h5ad('/data/jiangjunyao/fa_result/config_tunning/larry_test_hvg.h5ad')
adata = adata1.concatenate(adata2)
sc.pl.embedding(adata,color=['state_info'],basis='X_emb')
adata.obs['day']

tp = TemporalProblem(adata)
tp = tp.prepare(time_key="day")
tp = tp.solve()
tmk = RealTimeKernel.from_moscot(tp)
tmk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")
g = cr.estimators.GPCCA(tmk)
g.fit(cluster_key="state_info", n_states=2)
g.set_terminal_states(states=["Neutrophil", "Monocyte"])
g.plot_macrostates(which="terminal", legend_loc="right", size=100)
g.compute_fate_probabilities()

df1=pd.DataFrame(g.adata.obsm['lineages_fwd'],columns=['Neutrophil','Monocyte'])
adata.obs['NeuMon_mask'] = adata.obs['NeuMon_mask'].astype(bool)
df1=pd.DataFrame(g.adata.obsm['lineages_fwd'],columns=['Monocyte','Neutrophil'])
max_columns = df1.idxmax(axis=1)
adata.obs['pred'] = max_columns.tolist()
adata_test2 = adata[adata.obs['state_info']=='undiff']
adata_test2 = adata_test2[adata_test2.obs['NeuMon_mask']]
adata_test2 = adata_test2[adata_test2.obs['NeuMon_fate_bias']!=0.5]
adata_test2.obs['lineage']='Neutrophil'
adata_test2.obs.loc[adata_test2.obs['NeuMon_fate_bias'] > 0.999, 'lineage'] = 'Monocyte'

adata.write('/data/jiangjunyao/AEGAS_analysis/pred_result/cellrank_larry.h5ad')
adata.obs.to_csv('/data/jiangjunyao/AEGAS_analysis/pred_result/cellrank_larry_obs.csv')