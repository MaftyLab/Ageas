#!/usr/bin/env python3
"""One benchmark job: Ageas v2 or Ageas (R1) on one dataset and seed (or held-out fold).

v2 (:func:`ageas.run_pipeline`) profiles the dataset without progenitor labels,
designs its own panel and runs under ``revision/v2/protocol.json``. One run
gives both of its integrations: ``v2`` (the no-harm mixture) and ``v2_vote``
(the committee's vote, which the mixture joins when the protocol's committee
sets ``with_mixture``). R1 runs its ``default_config_v1`` panel under
``revision/protocol.json``. Both see the same terminal cells, features and
folds. The code is imported from the Ageas checkout this script sits in, so a
frozen copy of the checkout (``snapshot.py``) runs as a unit.

Datasets with a ``cv`` block (Zheng68K) run one held-out fold at a time
(``--fold k``, as E17 did): the other folds are the terminal cells and fold
``k`` is the query. Outputs in ``<out>/results/<dataset>/``, with ``<rep>``
``seed<k>`` or ``fold<k>``:

* ``<method>_<rep>.csv`` — query probabilities in the R1 prediction format
  (``v2_<rep>.csv``, ``v2_vote_<rep>.csv`` and ``v2_final_<rep>.csv`` from a
  v2 run; ``v2_final`` is v2's output: the graph stage's fusion when the
  registry names unlabeled context cells (``context_values``) and there are
  enough of them, the vote otherwise);
* ``v2_vote_<rep>_consistency.csv`` — per query cell, the share of committee
  members backing the call and their disagreement in bits;
* ``<method>_<rep>_members.npz`` — per-unit (R1) or per-expert (v2) probabilities;
* ``r1_<rep>_units.csv`` — R1's surviving units and their out-of-fold scores;
* ``v2_<rep>_oof.npz`` — every unit's out-of-fold probabilities on the terminal
  cells, their labels, the shift-aware cell weights and the folds;
* ``truth.csv`` (``truth_fold<k>.csv``) — the query labels, for scoring only;
* ``<method>_<rep>.json`` — the run record, written last: it marks the job done.
  For v2 it holds the profile, the rules that fired, the panel, the strategy
  check, the no-harm weights and the committee; nothing in it is scored on
  query labels.

usage: run_job.py --out <bench dir> --dataset <registry key> --seed <k> [--fold <k>] --method {r1,v2}
                  [--registry <registry json>]
"""
import argparse
import json
import logging
import os
import platform
import resource
import shutil
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the Ageas checkout
CODE = os.path.dirname(REPO)  # the folder holding it: Ageas.share, or a benchmark's frozen ``code/``
PROTOCOL = os.path.join(REPO, 'revision', 'v2', 'protocol.json')
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import ageas.tool.trainer as ageas_trainer  # noqa: E402
from ageas import Hangar, n_kfold_selection  # noqa: E402
from ageas.tool.corpus_loader import get_all_data  # noqa: E402
from revision.common import load_protocol, load_registry, prepare_dataset, save_predictions  # noqa: E402
from revision.common.registry import STATE_COL  # noqa: E402
from revision.common.runner import _corpus, probs_frame, unit_table  # noqa: E402
from revision.E17_config_retest.run_configs import cv_fold  # noqa: E402

from ageas import run_pipeline  # noqa: E402
from ageas.panel import active_rules  # noqa: E402
from ageas.tool.synergy import log_loss  # noqa: E402

METHODS = ('r1', 'v2')
NAMES = {'r1': 'Ageas', 'v2': 'Ageas v2 (no-harm)', 'v2_vote': 'Ageas v2 (committee vote)',
         'v2_final': 'Ageas v2'}
#: R1 units E17 could not run at 55,000 training cells: libsvm SVCs, and
#: networks with batches under 20 (19,000-41,000 optimizer steps per fit).
R1_LIMITS = {
    'zheng68k_hvg2000': {'types': ('logreg', 'mlp', 'mnb', 'resnet', 'rnn', 'xgb'), 'min_batch_size': 20},
}


def training_loader(dataset, *args, **kwargs) -> DataLoader:
    """R1's trainer loaders, except that a final training batch of one cell is dropped.

    R1's trainer keeps every last batch. When a training set is one cell more
    than a multiple of the batch size, a BatchNorm network cannot train on
    that batch and the run fails (R1 on Zheng68K, 2026-11-08). Every other
    loader is built exactly as before. Applied to R1 and v2 alike, which both
    train through R1's trainer.
    """
    batch_size = kwargs.get('batch_size', 1)
    if kwargs.get('shuffle') and batch_size and len(dataset) % batch_size == 1:
        kwargs['drop_last'] = True
    return DataLoader(dataset, *args, **kwargs)


ageas_trainer.DataLoader = training_loader


def quiet() -> None:
    """Silence per-fit Lightning banners and solver convergence warnings."""
    for name in ('pytorch_lightning', 'lightning_fabric', 'lightning.pytorch', 'lightning.fabric'):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')


def predict_query(deck, query, classes) -> tuple:
    """R1's ``predict_query`` on the in-memory corpus (same items, as in E17).

    R1's version wraps the query in the AnnData-backed corpus, which builds an
    AnnData view per cell for every unit: about 10 s per unit on LARRY.
    """
    mean, _, members = deck.predict(
        _corpus(query, classes, dense=True), batch_size=512, num_workers=0, return_members=True,
    )
    return probs_frame(mean, query.obs_names, classes), members


def r1_units(hangar, units=None, limits=None) -> list:
    """R1's units within ``limits`` (``types`` kept, networks' ``min_batch_size``), or ``None`` for all."""
    if not limits:
        return units
    keep = []
    for unit_id, unit in hangar.units.items():
        batch = (unit.config.get('train_config') or {}).get('batch_size')
        if unit.type in limits['types'] and (batch is None or batch >= limits['min_batch_size']):
            keep.append(unit_id)
    return [u for u in keep if units is None or u in units]


def run_r1(data, seed, accelerator, stem, units=None, limits=None) -> dict:
    """E01's ``run_ageas``, trained from the in-memory corpus E17 uses (same items, faster).

    Every unit builds its batches in-process: the data are served from memory,
    so loader worker processes only add start-up time.
    """
    from pytorch_lightning import seed_everything

    repo = REPO
    protocol = load_protocol(os.path.join(repo, 'revision', 'protocol.json'))
    hangar = Hangar(os.path.join(repo, protocol['config_dir']))
    for unit in hangar.units.values():
        unit.config['dataloader_workers'] = 0
    units = r1_units(hangar, units, limits)
    selection = dict(protocol['selection'], **({'using_model_list': units} if units else {}))
    seed_everything(seed, workers=True)
    deck = n_kfold_selection(
        hangar=hangar, operation_name='trail', query_dataset=_corpus(data.train, data.classes, dense=True),
        accelerator=accelerator, seed=seed, **selection,
    )
    probs, members = predict_query(deck, data.query, data.classes)
    save_predictions(probs, stem + '.csv', NAMES['r1'], seed)
    np.savez_compressed(stem + '_members.npz', cells=np.asarray(data.query.obs_names), **members)
    unit_table(deck).to_csv(stem + '_units.csv')
    return {'kept_units': list(deck.squad), 'r1_limits': limits, 'r1_candidates': units}


def _summary(spec) -> dict:
    """A panel unit's identity, settings and the rules behind them."""
    return {
        'unit_id': spec.unit_id, 'type': spec.type, 'family': spec.family,
        'model_params': spec.config.get('model_params'), 'train_config': spec.config.get('train_config'),
        'max_epochs': spec.config.get('max_epochs'), 'shaped_by': spec.shaped_by,
        'favoured_by': spec.favoured_by, 'disfavoured_by': spec.disfavoured_by,
    }


def graph_input(data) -> dict:
    """Every cell for v2's graph stage, or ``None`` when the registry names no unlabeled context cells.

    The cells are read and preprocessed together (terminal, query and context,
    the registry's recipe), apart from the terminal and query cells the units
    train and predict on, which stay exactly as before. Labels are the
    terminal classes' indices and ``-1`` for every other cell.
    """
    context = data.spec.get('context_values')
    if not context:
        return None
    view = prepare_dataset(data.name, {data.name: dict(data.spec, extra_keep_values=list(context))})
    index = {c: i for i, c in enumerate(view.classes)}
    labels = np.array([index.get(s, -1) for s in view.adata.obs[STATE_COL].astype(str)])
    query_index = pd.Index(view.adata.obs_names).get_indexer(data.query.obs_names)
    assert (query_index >= 0).all(), 'every query cell must be in the graph view'
    kind = 'lsi' if (data.spec.get('preprocess') or {}).get('tfidf') else 'pca'
    return {'features': view.adata.X, 'labels': labels, 'query_index': query_index, 'kind': kind}


def save_final(final, data, seed, final_stem, stage) -> dict:
    """Write v2's output (``v2_final``) and return the graph stage's record."""
    save_predictions(probs_frame(final, data.query.obs_names, data.classes), final_stem + '.csv',
                     NAMES['v2_final'], seed)
    return {k: v for k, v in stage.items() if k not in ('probs', 'walk')}


def run_v2(data, seed, accelerator, workdir, stem, vote_stem, unit_types=None, protocol_path=None) -> dict:
    """The v2 pipeline under its protocol (``revision/v2/protocol.json`` unless an ablation passes its own copy)."""
    with open(protocol_path or PROTOCOL, encoding='utf-8') as fh:
        protocol = json.load(fh)
    terminal = _corpus(data.train, data.classes, dense=True)
    query_x = get_all_data(_corpus(data.query, data.classes, dense=True))[0].numpy()  # features only
    graph = graph_input(data)
    result = run_pipeline(
        terminal, query_x, workdir, selection=protocol['selection'], routing=protocol['routing'],
        synergy=protocol['synergy'], committee=protocol['committee'],
        validation=protocol['pipeline']['validation'], refine=protocol['pipeline']['refine'],
        rules=protocol['pipeline'].get('rules', True),
        accelerator=accelerator, unit_types=unit_types, seed=seed,
        graph=graph, graph_settings=protocol.get('graph'),
    )
    deck, query, cells = result.deck, result.query, data.query.obs_names
    save_predictions(probs_frame(query['no_harm'], cells, data.classes), stem + '.csv', NAMES['v2'], seed)
    save_predictions(probs_frame(query['vote'], cells, data.classes), vote_stem + '.csv', NAMES['v2_vote'], seed)
    final_stem = os.path.join(os.path.dirname(stem), 'v2_final_' + os.path.basename(stem)[len('v2_'):])
    graph_record = save_final(query['final'], data, seed, final_stem, result.graph)
    pd.DataFrame({'agreement': query['agreement'], 'disagreement_bits': query['disagreement']},
                 index=pd.Index(cells, name='cell')).to_csv(vote_stem + '_consistency.csv')
    np.savez_compressed(stem + '_members.npz', cells=np.asarray(cells), **query['members'])

    oof, weights = deck.oof, result.cell_weights
    folds = {f'fold{k}_{part}': fold[part] for k, fold in enumerate(oof['folds']) for part in fold}
    np.savez_compressed(
        stem + '_oof.npz', cells=np.asarray(data.train.obs_names), labels=oof['labels'], cell_weights=weights,
        **folds, **{f'oof_{unit_id}': p for unit_id, p in oof['probs'].items()},
    )
    labels = oof['labels']
    ensemble = sum(w * oof['probs'][u] for u, w in deck.weights.items())
    committee_oof = np.mean([oof['probs'][u] for u in result.committee], axis=0)
    return {
        'protocol': protocol_path or 'revision/v2/protocol.json',
        'pipeline': protocol['pipeline'],
        'routing': protocol['routing'],
        'validation': protocol['pipeline']['validation'],
        'profile': result.profile,
        'rules': active_rules(result.profile),
        'panel': [_summary(spec) for spec in result.panel],
        'cell_weights': {
            'effective_share': float(weights.sum() ** 2 / (weights ** 2).sum() / len(weights)),
            'max': float(weights.max()),
        },
        'strategy': result.report,
        'weights': deck.weights,
        'competent': deck.competent,
        'committee': result.committee,
        'committee_settings': protocol['committee'],
        'graph': graph_record,
        'graph_settings': protocol.get('graph'),
        'query_consistency': {
            'mean_agreement': float(query['agreement'].mean()),
            'share_unanimous': float((query['agreement'] == 1).mean()),
            'mean_disagreement_bits': float(query['disagreement'].mean()),
        },
        'experts': {
            unit_id: {'level': expert.level, 'n_features': len(expert.features),
                      'features': None if expert.level is None else list(expert.features)}
            for unit_id, expert in deck.experts.items()
        },
        'fold_levels': oof['levels'],
        'fold_n_features': {u: [len(f) for f in fs] for u, fs in oof['features'].items()},
        'fold_valid_logloss': {u: [{str(k): v for k, v in d.items()} for d in ds] for u, ds in oof['losses'].items()},
        'ensemble_oof_logloss': {'ordinary': log_loss(ensemble, labels), 'shift': log_loss(ensemble, labels, weights)},
        'committee_oof_logloss': {'ordinary': log_loss(committee_oof, labels),
                                  'shift': log_loss(committee_oof, labels, weights)},
        'ageas_import': deck.provenance,
    }


def _write_json(path, record) -> None:
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1, default=str)
    os.replace(path + '.tmp', path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Benchmark folder (results/, cache/ go inside).')
    parser.add_argument('--dataset', required=True, help='Key in the dataset registry.')
    parser.add_argument('--registry', default=os.path.join(REPO, 'revision', 'datasets.share.json'),
                        help="Dataset registry (default: R1's registry of Ageas.share/data; the simulated "
                             'datasets have their own, from sim_registry.py).')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, default=None,
                        help='For registry entries with a cv block: the fold held out as the query.')
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--accelerator', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=1,
                        help='CPU threads for BLAS, OpenMP and PyTorch (E17: more threads per worker '
                             'oversubscribed the CPU).')
    parser.add_argument('--units', nargs='*', default=None,
                        help='R1 only: these unit IDs (for a quick check; benchmark runs use the whole panel).')
    parser.add_argument('--unit-types', nargs='*', default=None,
                        help='v2 only: these unit types of the designed panel (for a quick check).')
    parser.add_argument('--protocol', default=None,
                        help="v2 only: a protocol file in place of revision/v2/protocol.json (an ablation arm's copy).")
    args = parser.parse_args(argv)

    pool_limits = threadpool_limits(limits=args.threads)  # noqa: F841 (kept for the process lifetime)
    torch.set_num_threads(args.threads)
    quiet()

    rep = f'seed{args.seed}' if args.fold is None else f'fold{args.fold}'
    out_dir = os.path.join(args.out, 'results', args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'{args.method}_{rep}')
    vote_stem = os.path.join(out_dir, f'v2_vote_{rep}')
    if os.path.exists(stem + '.json'):
        print(f'{stem}.json exists; nothing to do.')
        return
    workdir = os.path.join(args.out, 'cache', f'{args.dataset}_{args.method}_{rep}')
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    os.chdir(workdir)  # R1 writes its Lightning logs to the working directory

    registry = load_registry(args.registry)
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    data = prepare_dataset(args.dataset, registry)
    if args.fold is not None:
        assert 'cv' in data.spec, f'{args.dataset} has no cv block; --fold does not apply.'
        data = cv_fold(data, args.fold)
    print(f'{args.dataset}: {data.train.n_obs} terminal cells, {data.query.n_obs} query cells, '
          f'{data.train.n_vars} features; {args.method} {rep}', flush=True)
    truth = os.path.join(out_dir, 'truth.csv' if args.fold is None else f'truth_fold{args.fold}.csv')
    if data.ground_truth is not None and not os.path.exists(truth):  # query labels, for scoring only
        data.ground_truth.to_csv(truth + f'.{os.getpid()}')
        os.replace(truth + f'.{os.getpid()}', truth)
    start = time.time()
    if args.method == 'r1':
        details = run_r1(data, args.seed, args.accelerator, stem, units=args.units,
                         limits=R1_LIMITS.get(args.dataset))
    else:
        details = run_v2(data, args.seed, args.accelerator, workdir, stem, vote_stem, unit_types=args.unit_types,
                         protocol_path=args.protocol)
    minutes = (time.time() - start) / 60

    snapshot = os.path.join(CODE, 'SNAPSHOT.json')
    record = {
        'dataset': args.dataset,
        'registry': args.registry,
        'seed': args.seed,
        'fold': args.fold,
        'method': args.method,
        'minutes': minutes,
        'started': started,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_terminal': int(data.train.n_obs),
        'n_query': int(data.query.n_obs),
        'n_features': int(data.train.n_vars),
        'classes': list(data.classes),
        'accelerator': args.accelerator,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'threads': args.threads,
        'unit_filter': args.units or args.unit_types,
        'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        'host': platform.node(),
        'code': CODE,
        'snapshot': json.load(open(snapshot, encoding='utf-8')) if os.path.exists(snapshot) else None,
        **details,
    }
    if args.method == 'v2':  # the vote comes from the same run; its record is written first
        _write_json(vote_stem + '.json', dict(record, method='v2_vote', same_run_as=os.path.basename(stem) + '.json'))
    _write_json(stem + '.json', record)
    os.chdir(args.out)
    shutil.rmtree(workdir, ignore_errors=True)
    print(f'{args.method} {rep} done in {minutes:.1f} min', flush=True)


if __name__ == '__main__':
    main()
