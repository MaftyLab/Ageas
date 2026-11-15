#!/usr/bin/env python3
"""E02 · Export a prepared dataset for the R baselines (FateID, STEMNET).

Writes to ``revision/results/E02/<dataset>/r_input/``:

* ``matrix.mtx`` — features × cells (Matrix Market, as R expects);
* ``features.txt`` and ``cells.txt``;
* ``obs.csv`` — per cell: ``state`` (terminal class or ``query``) and
  ``is_query``.
"""
import os
import sys

import pandas as pd
from scipy import io, sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    registry, _, _, _ = resolve(args)
    data = prepare_dataset(args.dataset, registry)
    out = os.path.join(results_dir('E02', args.dataset, args.results_root), 'r_input')
    os.makedirs(out, exist_ok=True)

    adata = data.adata
    states = adata.obs[data.spec['label_key']].astype(str)
    is_query = states.isin(data.spec['query_values'])
    keep = is_query | states.isin(data.classes)
    adata = adata[keep.values]

    io.mmwrite(os.path.join(out, 'matrix.mtx'), sparse.csr_matrix(adata.X).T.tocoo())
    pd.Series(adata.var_names).to_csv(os.path.join(out, 'features.txt'), index=False, header=False)
    pd.Series(adata.obs_names).to_csv(os.path.join(out, 'cells.txt'), index=False, header=False)
    pd.DataFrame({
        'state': states[keep].where(~is_query[keep], 'query').values,
        'is_query': is_query[keep].values,
    }, index=adata.obs_names).to_csv(os.path.join(out, 'obs.csv'))
    print('exported', adata.shape, 'to', out)


if __name__ == '__main__':
    main()
