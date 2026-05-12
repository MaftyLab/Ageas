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

adata = sc.read_h5ad('/data/jiangjunyao/easyGRN/processed_data/celltag_multi_iep.prcessed.h5ad')
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True,flavor='seurat_v3')
adata.obs['day'] = 2
adata.obs.loc[adata.obs['celltype'] == 'early', 'day'] = 1

tp = TemporalProblem(adata)
tp = tp.prepare(time_key="day")
tp = tp.solve()
tmk = RealTimeKernel.from_moscot(tp)
tmk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")
g = cr.estimators.GPCCA(tmk)
g.fit(cluster_key="celltype", n_states=2)
g.set_terminal_states(states=["dead-end", "reprogramming"])
g.compute_fate_probabilities()

df1=pd.DataFrame(g.adata.obsm['lineages_fwd'],columns=['dead-end','reprogramming'])
max_columns = df1.idxmax(axis=1)
adata.obs['pred'] = max_columns.tolist()
adata.obs.to_csv('/data/jiangjunyao/AEGAS_analysis/pred_result/cellrank_celltag_multi_rna_obs.csv')