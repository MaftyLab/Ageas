#!/usr/bin/env python3
"""E06 · Per-cell uncertainty, OOD and fate-state calls.

Runs the frozen protocol (first seed), then, for every query cell:

* decomposes the ensemble's predictive entropy into aleatoric and
  epistemic (member-disagreement) parts;
* computes an out-of-distribution score relative to the terminal cells;
* scores each fate's program (top ``--program-size`` Ageas features of
  that class) and calls the cell confident / boundary / hybrid / unprimed /
  model_disagreement / ood (``ageas.tool.uncertainty.call_fate_states``).

With a clone-matrix ground truth it tests whether these outputs separate
mixed-fate clones from unipotent clones (the R2.8 validation).

Outputs in ``revision/results/E06/<dataset>/``: ``cell_uncertainty.csv``,
``programs.json`` and ``validation.csv``.
"""
import json
import os
import sys

import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import predict_query, prepare_dataset, results_dir, run_ageas  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.registry import LABEL_COL  # noqa: E402
from ageas.tool.uncertainty import (  # noqa: E402
    call_fate_states, decompose_uncertainty, ood_scores, program_scores, program_thresholds,
)


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--program-size', type=int, default=50)
    parser.add_argument('--confident-prob', type=float, default=0.7)
    parser.add_argument('--max-epistemic', type=float, default=0.05)
    args = parser.parse_args(argv)
    registry, protocol, config_dir, seeds = resolve(args)

    data = prepare_dataset(args.dataset, registry)
    out = results_dir('E06', args.dataset, args.results_root)
    deck, corpus = run_ageas(data.train, data.classes, config_dir, seeds[0],
                             protocol['selection'], args.accelerator)
    probs, members = predict_query(deck, data.query, data.classes)
    cells = data.query.obs_names

    uncertainty = decompose_uncertainty(members, index=cells)
    ood = ood_scores(data.train.X, data.query.X, query_index=cells)

    explanation = deck.debrief(exp_dataset=corpus, verbose=False, **protocol['debrief'])
    programs = {
        c: explanation[f'Class_{i}_Scores'].sort_values(ascending=False)
        .head(args.program_size).index.tolist()
        for i, c in enumerate(data.classes)
    }
    names = data.train.var_names
    train_scores = program_scores(data.train.X, names, programs, index=data.train.obs_names)
    query_scores = program_scores(data.query.X, names, programs, reference=data.train.X, index=cells)
    thresholds = program_thresholds(train_scores, data.train.obs[LABEL_COL].astype(str).values)

    prob_cols = [f'prob_{c}' for c in data.classes]
    states = call_fate_states(
        probs[prob_cols].set_axis(data.classes, axis=1), uncertainty, ood,
        query_scores if len(data.classes) == 2 else None, thresholds,
        confident_prob=args.confident_prob, max_epistemic=args.max_epistemic,
    )
    table = pd.concat([probs, uncertainty, ood, query_scores.add_prefix('program_'),
                       states.rename('state')], axis=1)
    table.to_csv(os.path.join(out, 'cell_uncertainty.csv'))
    with open(os.path.join(out, 'programs.json'), 'w') as fh:
        json.dump(programs, fh, indent=1)
    print(table['state'].value_counts().to_string())

    gt = data.ground_truth
    if gt is not None and 'label' in gt:
        joined = table.join(gt['label'], how='inner')
        mixed = joined[joined['label'] == 'mixed']
        unipotent = joined[joined['label'].isin(data.classes)]
        rows = []
        for metric in ('total_entropy', 'epistemic', 'aleatoric'):
            if len(mixed) and len(unipotent):
                stat = mannwhitneyu(mixed[metric], unipotent[metric], alternative='greater')
                rows.append({'metric': metric, 'mixed_median': mixed[metric].median(),
                             'unipotent_median': unipotent[metric].median(), 'p_greater': stat.pvalue})
        for group, frame in (('mixed', mixed), ('unipotent', unipotent)):
            for state, frac in frame['state'].value_counts(normalize=True).items():
                rows.append({'metric': f'state_fraction:{state}', 'group': group, 'value': frac})
        pd.DataFrame(rows).to_csv(os.path.join(out, 'validation.csv'), index=False)


if __name__ == '__main__':
    main()
