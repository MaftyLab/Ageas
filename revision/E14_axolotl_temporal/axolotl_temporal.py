#!/usr/bin/env python3
"""E14 · Axolotl: raw (unscaled) Ageas probabilities and a temporal cross-check.

1. Runs the frozen protocol on the 15-DPI section and writes the *raw*
   ensemble probabilities of the query reaEGCs, without the global min-max
   rescaling used in ``docs/examples/spatial_example.ipynb`` (that rescaling
   stretches the values to [0, 1], so they are not probabilities).
2. Profiles P(lineage 2) along an axis of the section (``--axis-obsm-dim``,
   0 = x, 1 = y), binned, with the CoSpar profile if ``--cospar`` points to
   E02 predictions for the same cells.
3. If ``--distance-to`` gives an obs column with each spot's distance to the
   injury site, reports the partial Spearman correlation of P with the
   section axis given that distance (is the "dorsal" enrichment just
   proximity to the wound?).

Outputs in ``revision/results/E14/<dataset>/``.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import load_predictions, predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--axis-obsm-dim', type=int, default=1)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--cospar', default=None, help='E02 CoSpar predictions for the same cells.')
    parser.add_argument('--distance-to', default=None, help='obs column: distance to injury site.')
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E14', args.dataset, args.results_root)
    positive = data.classes[1]
    deck, _ = run_ageas(data.train, data.classes, config_dir, seeds[0], protocol['selection'],
                        args.accelerator)
    probs, _ = predict_query(deck, data.query, data.classes)
    probs.to_csv(os.path.join(out, 'ageas_raw_probabilities.csv'))

    axis = pd.Series(np.asarray(data.query.obsm[data.spec['spatial_obsm']])[:, args.axis_obsm_dim],
                     index=data.query.obs_names, name='axis')
    profile = pd.DataFrame({'axis': axis, 'ageas': probs[f'prob_{positive}']})
    if args.cospar:
        cospar = load_predictions(args.cospar)
        profile['cospar'] = cospar.reindex(profile.index)[f'prob_{positive}']
    profile['bin'] = pd.qcut(profile['axis'], args.bins, labels=False, duplicates='drop')
    summary = profile.groupby('bin').agg(['mean', 'std', 'count'])
    summary.to_csv(os.path.join(out, 'axis_profile.csv'))

    rows = [{'method': m, 'spearman_axis': spearmanr(profile['axis'], profile[m], nan_policy='omit').correlation,
             'range_of_bin_means': float(np.ptp(profile.groupby('bin')[m].mean().values))}
            for m in ('ageas', 'cospar') if m in profile]
    if args.distance_to:
        dist = data.query.obs[args.distance_to].astype(float)
        res = lambda v: v - np.polyval(np.polyfit(dist, v, 1), dist)  # noqa: E731
        rows.append({'method': 'ageas', 'partial_spearman_axis_given_wound_distance':
                     spearmanr(res(profile['ageas'].values), res(profile['axis'].values)).correlation})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'axis_summary.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
