#!/usr/bin/env python3
"""E02 · Time-aware methods run with two pseudo-timepoints and with real times.

R2.3 asks to state that collapsing samples into two fabricated timepoints
handicaps transition-map methods. This script runs CoSpar (state-only mode)
in both settings so the paper can report the time-free comparison *and* the
upper bound reached when real sampling times are available:

* ``cospar_pseudotime2.csv`` — progenitors = "early", everything else
  "late" (the setting of the original benchmark);
* ``cospar_realtime.csv`` — the registry's ``time_key`` (oracle).

Requires ``cospar``. No clonal information is given to CoSpar here; its
lineage-aware modes would use the ground truth itself.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, save_predictions  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402


def run_cospar(adata, classes, time_info, initial, later):
    import cospar as cs
    adata = adata.copy()
    adata.obs['time_info'] = np.asarray(time_info).astype(str)
    adata.obs['state_info'] = adata.obs['_state'].astype(str)
    if 'X_emb' not in adata.obsm:
        adata.obsm['X_emb'] = adata.obsm.get('X_umap', adata.obsm['X_pca'][:, :2])
    adata = cs.pp.initialize_adata_object(adata)
    tmap = cs.tmap.infer_Tmap_from_state_info_alone(
        adata, initial_time_points=initial, later_time_point=later,
        initialize_method='OT', OT_cost='GED', smooth_array=[20, 15, 10],
        max_iter_N=[1, 3], sparsity_threshold=0.2, use_full_Smatrix=True,
    )
    cs.tl.fate_map(tmap, selected_fates=classes, source='transition_map')
    return pd.DataFrame({c: tmap.obs[f'fate_map_transition_map_{c}'] for c in classes})


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    registry, _, _, _ = resolve(args)
    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E02', args.dataset, args.results_root)

    import scanpy as sc
    adata = data.adata.copy()
    adata.obs['_state'] = adata.obs[data.spec['label_key']].astype(str)
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
    query = adata.obs['_state'].isin(data.spec['query_values']).values

    settings = {'pseudotime2': (np.where(query, 'early', 'late'), ['early'], 'late')}
    time_key = data.spec.get('time_key')
    if time_key:
        times = adata.obs[time_key].astype(str).values
        ordered = sorted(pd.unique(times), key=lambda t: float(t) if t.replace('.', '', 1).isdigit() else t)
        settings['realtime'] = (times, ordered[:-1], ordered[-1])

    for name, (time_info, initial, later) in settings.items():
        fate = run_cospar(adata, data.classes, time_info, initial, later)
        fate = fate.loc[adata.obs_names[query]]
        probs = fate.values / np.clip(fate.values.sum(axis=1, keepdims=True), 1e-12, None)
        save_predictions(probs_frame(probs, fate.index, data.classes),
                         os.path.join(out, f'cospar_{name}.csv'), f'cospar_{name}')
        print('wrote', name)


if __name__ == '__main__':
    main()
