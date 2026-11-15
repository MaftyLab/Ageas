#!/usr/bin/env python3
"""E02 · Time-free snapshot fate methods: Palantir and CellRank 2.

Runs in an environment with ``palantir`` and ``cellrank>=2`` installed (not
Ageas dependencies). Terminal states are fixed to the annotated terminal
classes, as for Ageas; no sampling time is used.

* ``palantir`` — diffusion maps + Palantir pseudotime from a start cell
  (the query cell with the highest CytoTRACE score if available, else the
  query cell closest to the query centroid); fate probabilities = Palantir
  branch probabilities toward the terminal classes.
* ``cellrank_cytotrace`` — CellRank 2 ``CytoTRACEKernel`` (RNA only).
* ``cellrank_pseudotime`` — CellRank 2 ``PseudotimeKernel`` on Palantir
  pseudotime.
* ``cellrank_connectivity`` — CellRank 2 ``ConnectivityKernel`` (undirected
  kNN graph, absorption into the terminal classes).

Each method writes ``<method>.csv`` in the shared prediction format.
"""
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir, save_predictions  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import probs_frame  # noqa: E402

TERMINAL_KEY = '_terminal'


def _prepare_graph(adata, n_pcs=50):
    import scanpy as sc
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    return adata


def _start_cell(adata, query_mask):
    if 'ct_score' in adata.obs:
        scores = adata.obs.loc[query_mask, 'ct_score']
        return scores.idxmax()
    pcs = adata.obsm['X_pca'][query_mask]
    centre = pcs.mean(axis=0)
    return adata.obs_names[query_mask][np.argmin(np.linalg.norm(pcs - centre, axis=1))]


def run_palantir(adata, classes, query_mask):
    """Palantir (>= 1.3 AnnData API) with one central terminal cell per class."""
    import palantir
    palantir.utils.run_diffusion_maps(adata, n_components=10)
    palantir.utils.determine_multiscale_space(adata)
    multiscale = adata.obsm['DM_EigenVectors_multiscaled']
    terminal = {}
    for c in classes:
        mask = (adata.obs[TERMINAL_KEY] == c).values
        sub = np.asarray(multiscale[mask])
        terminal[adata.obs_names[mask][np.argmin(np.linalg.norm(sub - sub.mean(axis=0), axis=1))]] = c
    start = _start_cell(adata, query_mask)
    palantir.core.run_palantir(adata, start, terminal_states=pd.Series(terminal))
    return adata.obsm['palantir_fate_probabilities'][classes].values


def run_cellrank(adata, classes, kernel_name):
    import cellrank as cr
    if kernel_name == 'cytotrace':
        kernel = cr.kernels.CytoTRACEKernel(adata).compute_cytotrace()
    elif kernel_name == 'pseudotime':
        kernel = cr.kernels.PseudotimeKernel(adata, time_key='palantir_pseudotime')
    else:
        kernel = cr.kernels.ConnectivityKernel(adata)
    kernel.compute_transition_matrix()
    estimator = cr.estimators.GPCCA(kernel)
    estimator.set_terminal_states({c: adata.obs_names[adata.obs[TERMINAL_KEY] == c].tolist()
                                   for c in classes})
    estimator.compute_fate_probabilities()
    lineages = estimator.fate_probabilities
    frame = pd.DataFrame(np.asarray(lineages), columns=list(lineages.names), index=adata.obs_names)
    return frame[classes].values


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--methods', nargs='*',
                        default=['palantir', 'cellrank_cytotrace', 'cellrank_pseudotime',
                                 'cellrank_connectivity'])
    args = parser.parse_args(argv)
    registry, _, _, _ = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E02', args.dataset, args.results_root)
    adata = data.adata.copy()
    states = adata.obs[data.spec['label_key']].astype(str)
    adata.obs[TERMINAL_KEY] = states.where(states.isin(data.classes))
    query_mask = states.isin(data.spec['query_values']).values
    adata = _prepare_graph(adata)

    probs = {}
    for method in args.methods:
        try:
            if method == 'palantir' or method == 'cellrank_pseudotime':
                if 'palantir_pseudotime' not in adata.obs:
                    probs['palantir'] = run_palantir(adata, data.classes, query_mask)
                if method == 'cellrank_pseudotime':
                    probs[method] = run_cellrank(adata, data.classes, 'pseudotime')
            elif method == 'cellrank_cytotrace':
                if data.spec['modality'] not in ('rna', 'spatial_rna'):
                    print('CytoTRACE kernel needs RNA counts; skipped for', data.spec['modality'])
                    continue
                probs[method] = run_cellrank(adata, data.classes, 'cytotrace')
            elif method == 'cellrank_connectivity':
                probs[method] = run_cellrank(adata, data.classes, 'connectivity')
        except Exception:  # keep going: one failing method must not block the others
            traceback.print_exc()

    for method, p in probs.items():
        p = np.asarray(p)[query_mask]
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        save_predictions(probs_frame(p, adata.obs_names[query_mask], data.classes),
                         os.path.join(out, f'{method}.csv'), method)
        print('wrote', method)


if __name__ == '__main__':
    main()
