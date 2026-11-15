import os
import numpy as np
import scanpy as sc
import pandas as pd
import cellrank as cr
from cellrank.kernels import RealTimeKernel
from moscot.problems.time import TemporalProblem

from ground_truth import binary_eval_view, clonal_fate_table

# Set number of threads for parallel computation
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["VECLIB_MAXIMUM_THREADS"] = "40"
os.environ["NUMEXPR_NUM_THREADS"] = "40"

FATES = ('Monocyte', 'Neutrophil')
# The manuscript states that real sampling times were withheld from all
# methods. The original script used the real `day` column; keep that only as
# the time-aware "oracle" upper bound (revision experiment E02).
USE_REAL_TIME = False

adata1 = sc.read_h5ad('/data/jiangjunyao/fa_result/config_tunning/larry_train_hvg.h5ad')
adata2 = sc.read_h5ad('/data/jiangjunyao/fa_result/config_tunning/larry_test_hvg.h5ad')
adata = adata1.concatenate(adata2)

if not USE_REAL_TIME:
    adata.obs['day'] = np.where(adata.obs['state_info'] == 'undiff', 1, 2)

tp = TemporalProblem(adata)
tp = tp.prepare(time_key="day")
tp = tp.solve()
tmk = RealTimeKernel.from_moscot(tp)
tmk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")
g = cr.estimators.GPCCA(tmk)
g.fit(cluster_key="state_info", n_states=2)
g.set_terminal_states(states=list(FATES))
g.compute_fate_probabilities()

# Take column names from the Lineage object itself; the previous script
# assigned names twice in opposite orders.
lineages = g.fate_probabilities
prob = pd.DataFrame(np.asarray(lineages), columns=list(lineages.names), index=adata.obs_names)
prob = prob[list(FATES)]
pred = prob.idxmax(axis=1)

result = pd.concat([prob.add_prefix('prob_'), pred.rename('pred')], axis=1)
suffix = 'realtime' if USE_REAL_TIME else 'pseudotime2'
result.to_csv(f'/data/jiangjunyao/AEGAS_analysis/pred_result/cellrank_larry_{suffix}_pred.csv')

# Score against the shared ground truth (clone matrix required in obsm).
if 'X_clone' in adata.obsm:
    gt = clonal_fate_table(
        adata, fates=FATES,
        progenitor_mask=adata.obs['state_info'] == 'undiff',
        clone_obsm='X_clone', state_key='state_info', time_key='time_info',
    )
    evaluable = binary_eval_view(gt, FATES).join(result, how='inner')
    print('CellRank accuracy (unipotent clones):',
          (evaluable['binary_label'] == evaluable['pred']).mean())
