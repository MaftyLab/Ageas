"""Synthetic lineage-traced dataset for smoke-testing the experiment scripts.

Generates two terminal fates (A, B), a third fate (C, for open-set tests)
and progenitors that are partially primed toward their clonal fate, with a
clone matrix, sampling days, an axial ``slice`` coordinate correlated with
fate (to exercise the E10 deconfounding code) and 2D spatial coordinates.

Usage::

    python revision/common/synthetic.py revision/results/synthetic/synthetic.h5ad
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_synthetic(n_genes: int = 60, n_clones: int = 120, seed: int = 0) -> ad.AnnData:
    """Build the synthetic AnnData (log-normalised-like values in ``X``)."""
    rng = np.random.default_rng(seed)
    programs = {'A': np.arange(0, 8), 'B': np.arange(8, 16), 'C': np.arange(16, 24)}

    rows, obs, clone_ids = [], [], []
    for clone in range(n_clones):
        fate = rng.choice(['A', 'B', 'C'], p=[0.45, 0.45, 0.10])
        mixed = rng.uniform() < 0.1
        priming = rng.uniform(0.2, 0.9)
        # one progenitor on day 2
        x = rng.normal(1.0, 0.5, n_genes)
        x[programs[fate]] += priming
        rows.append(x)
        obs.append(('progenitor', 2, fate, priming))
        clone_ids.append(clone)
        # descendants on day 4
        for _ in range(rng.integers(2, 5)):
            child_fate = rng.choice(['A', 'B']) if mixed else fate
            x = rng.normal(1.0, 0.5, n_genes)
            x[programs[child_fate]] += 2.5
            rows.append(x)
            obs.append((child_fate, 4, child_fate, 1.0))
            clone_ids.append(clone)

    X = np.clip(np.vstack(rows), 0, None)
    obs = pd.DataFrame(obs, columns=['state', 'day', 'true_fate', 'priming'])
    obs.index = [f'cell_{i}' for i in range(len(obs))]
    is_b = (obs['true_fate'] == 'B').values
    obs['slice'] = np.clip(np.round(rng.normal(np.where(is_b, 30, 20), 6)), 1, 50).astype(int)
    obs['total_counts'] = rng.normal(5000, 800, len(obs))

    adata = ad.AnnData(X.astype(np.float32), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(n_genes)]
    adata.var['name'] = adata.var_names
    adata.obsm['X_clone'] = sparse.csr_matrix(
        (np.ones(len(clone_ids)), (np.arange(len(clone_ids)), clone_ids)),
        shape=(len(clone_ids), n_clones),
    )
    adata.obsm['spatial'] = rng.uniform(0, 100, size=(len(obs), 2))
    return adata


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'synthetic', 'synthetic.h5ad',
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make_synthetic().write_h5ad(out)
    print(out)
