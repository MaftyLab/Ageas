#!/usr/bin/env python3
"""Registry of the simulated benchmark (``data/simulation``) for ``run_job.py``.

One entry per dataset and modality: 90 RNA and 40 ATAC datasets from
scMultiSim, each with two terminal states (``progeny_A``, ``progeny_B``) as
the training classes and the progenitors as the query. The unlabeled context
cells (``context_values``) are neither trained on nor scored; only v2's
graph stage walks through them (the real-data setups have none). Features follow the
registry's recipes for each modality: RNA is normalised to 10,000 counts per
cell and log-transformed, keeping genes detected in at least 1% of cells
(all 1,000 genes are fewer than the 2,000 HVGs of the manuscript recipe);
ATAC is TF-IDF transformed (Signac method 1), as for iEP ATAC.

The query fates (``truth.csv``, A or B) are written as ``progeny_A`` /
``progeny_B`` labels to ``<out>/truth/<setup>.csv``, which the registry
points to. Jobs only copy them next to the results, for scoring; no method
sees them. Each entry also carries its scenario (``simulation``) for the
summary.

usage: sim_registry.py --out <bench dir> [--data <simulation data dir>]
"""
import argparse
import json
import os

import pandas as pd

CLASSES = ['progeny_A', 'progeny_B']
PREPROCESS = {
    'rna': {'normalize_total': True, 'target_sum': 10000, 'log1p': True, 'min_detect_frac': 0.01},
    'atac': {'tfidf': True, 'scale_factor': 10000},
}
MODALITY = {'rna': 'rna', 'atac': 'atac_peak'}


def setup_name(modality: str, dataset: str) -> str:
    """``sim_<modality>_<non-lineage level>_<difficulty>_<rep>``, e.g. ``sim_rna_nl160_hard_rep1``."""
    return f'sim_{modality}_' + dataset.replace('/', '_')


def build(index: pd.DataFrame, data: str, truth_dir: str) -> dict:
    """Registry entries for every dataset and modality in ``index``; writes the truth tables."""
    registry = {'_comment': __doc__.split('\n\n')[0].replace('\n', ' ')}
    for row in index.itertuples():
        for modality in ('rna', 'atac'):
            if not getattr(row, f'has_{modality}'):
                continue
            name = setup_name(modality, row.dataset)
            truth = pd.read_csv(os.path.join(data, 'datasets', row.dataset, 'truth.csv'), index_col=0)
            assert set(truth['fate']) <= {'A', 'B'}, f'{row.dataset}: unexpected fates {set(truth["fate"])}'
            table = pd.DataFrame({'label': 'progeny_' + truth['fate']}, index=truth.index.rename('cell'))
            path = os.path.join(truth_dir, name + '.csv')
            table.to_csv(path)
            registry[name] = {
                'h5ad': os.path.join('..', 'data', 'simulation', 'datasets', row.dataset, f'{modality}.h5ad'),
                'modality': MODALITY[modality],
                'label_key': 'celltype',
                'terminal_classes': CLASSES,
                'query_values': ['progenitor'],
                'context_values': ['unlabeled'],
                'preprocess': PREPROCESS[modality],
                'ground_truth': {'type': 'csv', 'path': path},
                'simulation': {
                    'dataset': row.dataset, 'modality': modality,
                    'non_lineage_factors': int(row.non_lineage_factors), 'difficulty': row.difficulty,
                    'seed': int(row.seed), 'split': row.split,
                },
            }
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Benchmark folder; writes registry.json and truth/.')
    parser.add_argument('--data', default='/home/jack/ageas_work/data/simulation',
                        help='The simulation folder (a local copy of Ageas.share/data/simulation).')
    args = parser.parse_args(argv)

    truth_dir = os.path.join(args.out, 'truth')
    os.makedirs(truth_dir, exist_ok=True)
    index = pd.read_csv(os.path.join(args.data, 'dataset_index.csv'))
    registry = build(index, args.data, truth_dir)
    with open(os.path.join(args.out, 'registry.json'), 'w', encoding='utf-8') as fh:
        json.dump(registry, fh, indent=1)
    n = {m: sum(1 for k in registry if k.startswith(f'sim_{m}_')) for m in ('rna', 'atac')}
    print(f'{len(registry) - 1} setups ({n["rna"]} RNA, {n["atac"]} ATAC) in {args.out}/registry.json')


if __name__ == '__main__':
    main()
