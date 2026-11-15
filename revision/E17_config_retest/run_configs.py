#!/usr/bin/env python3
"""E17 · Train every model configuration on its own and save its query predictions.

Pools the configurations of every panel in ``data/configs`` (a configuration
shipped in several panels is trained once; configurations written in the
older schema are translated, see :func:`translate_legacy`) and, for each configuration and
seed, runs Ageas selection on that configuration alone with every gate open:
K-fold cross-validation on the terminal cells, which gives the out-of-fold
scores Ageas ranks by, then a refit on all terminal cells whose query-cell
probabilities are saved. ``score.py`` builds the Ageas ensembles offline from
these outputs. Finished (configuration, seed) pairs are skipped, so an
interrupted run resumes where it stopped.

Outputs in ``<results-root>/E17/<dataset>/``:

* ``pool/`` and ``pool_manifest.csv`` — pooled configurations and, for each
  pooled unit, the panels and files it came from;
* ``runs.csv`` — one row per (unit, seed): out-of-fold metrics, seconds,
  status (``runs.w<i>.csv`` for worker ``i`` of ``--workers``);
* ``probs/<unit>_seed<k>.npz`` — query-cell probabilities.

The scikit-learn models are single-threaded, so ``--worker i --workers n``
lets ``n`` processes share the pool (unit ``j`` goes to worker ``j mod n``).
With ``--dynamic`` every worker walks the whole list instead and claims each
(unit, seed) through a ``probs/<unit>_seed<k>.claim`` file, so workers can be
added at any time. A worker that is killed leaves its claim behind; restart
with ``--clear-claims`` (only when no other worker of that dataset runs).

Registry entries with a ``cv`` block (the Zheng68K classification benchmark)
are run one held-out fold at a time with ``--fold k``: the other folds train,
fold ``k`` is the query, and its labels are written to ``truth.csv`` in
``<results-root>/E17/<dataset>_fold<k>/``.
"""
import gc
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from revision.common import REPO_ROOT, prepare_dataset, results_dir  # noqa: E402
from revision.common.cli import base_parser, resolve  # noqa: E402
from revision.common.runner import _corpus  # noqa: E402

CONFIG_ROOT = os.path.join(REPO_ROOT, 'data', 'configs')
OPEN_GATES = dict(selection_ratio=1.0, retention_point=-np.inf, cutoff_point=-np.inf,
                  relative_retention=None, final_gate='none')
RUN_COLUMNS = ['unit_id', 'type', 'seed', 'status', 'seconds',
               'oof.accuracy', 'oof.f1', 'oof.auroc', 'oof.CEL']


def find_configs(root: str = CONFIG_ROOT) -> list:
    """``(panel, type, path)`` for every configuration JSON under ``root``.

    Panels hold ``<type>/<file>.json``. A folder holding JSON files directly
    (``data/configs/mnb``) is a one-type panel named after that type.
    """
    found = []
    for panel in sorted(os.listdir(root)):
        panel_dir = os.path.join(root, panel)
        if not os.path.isdir(panel_dir):
            continue
        for entry in sorted(os.listdir(panel_dir)):
            path = os.path.join(panel_dir, entry)
            if entry.endswith('.json'):
                found.append((panel, panel, path))
            elif os.path.isdir(path):
                found += [(panel, entry, os.path.join(path, f))
                          for f in sorted(os.listdir(path)) if f.endswith('.json')]
    return found


#: Block names of the older configuration schema used by ``light_config`` and
#: ``replay``, per model type, and their current equivalents.
LEGACY_BLOCKS = {
    'resnet': {'Basic_Block': 'ResNet_Basic'},
    'rnn': {'Basic_Block': 'RNN_Basic', 'Residual_Encoder': 'RNN_Residual_Encoder'},
}


def translate_legacy(unit_type: str, config: dict) -> tuple:
    """Map the older configuration schema onto the current one.

    The ``light_config`` and ``replay`` panels name the embedding width
    ``latent_dim`` (now ``latent_fea_dim``) and use block names the current
    ``ageas.nn.blocks`` no longer has, so they fail as shipped. The mapping is
    inferred from the current panels, not from the old code, and every
    translated unit is flagged in the manifest.

    Returns:
        ``(config, changes)`` where ``changes`` lists what was renamed.
    """
    params = dict(config.get('model_params') or {})
    changes = []
    if 'latent_dim' in params and 'latent_fea_dim' not in params:
        params['latent_fea_dim'] = params.pop('latent_dim')
        changes.append('latent_dim->latent_fea_dim')
    new_block = LEGACY_BLOCKS.get(unit_type, {}).get(params.get('block'))
    if new_block:
        changes.append(f"{params['block']}->{new_block}")
        params['block'] = new_block
    return (dict(config, model_params=params) if changes else config), changes


def build_pool(pool_dir: str) -> pd.DataFrame:
    """Write each distinct configuration once to ``pool_dir/<type>/<hash>.json``.

    Units are named by a hash of their (translated) content, so a unit ID
    always means the same configuration, whatever the panel order.

    Returns:
        Manifest with one row per source file: pooled ``unit_id``, ``type``,
        ``panel``, the repository-relative ``source`` path and any legacy
        ``translation`` applied.
    """
    shutil.rmtree(pool_dir, ignore_errors=True)
    rows = []
    for panel, unit_type, path in find_configs():
        with open(path, encoding='utf-8') as fh:
            config, changes = translate_legacy(unit_type, json.load(fh))
        tail = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        target = os.path.join(pool_dir, unit_type, tail + '.json')
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'w', encoding='utf-8') as fh:
                json.dump(config, fh, indent=2)
        rows.append(dict(unit_id=f'{unit_type}_{tail}', type=unit_type, panel=panel,
                         source=os.path.relpath(path, REPO_ROOT).replace(os.sep, '/'),
                         translation='; '.join(changes)))
    return pd.DataFrame(rows)


def load_runs_table(out: str):
    """All run rows of a dataset (``runs.csv`` and worker files), or ``None``."""
    paths = sorted(glob.glob(os.path.join(out, 'runs*.csv')))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None


def cv_fold(data, fold: int):
    """Hold out one stratified fold of a registry entry with a ``cv`` block.

    The other folds become the training cells and this fold the query; its
    labels become the ground truth.
    """
    from sklearn.model_selection import StratifiedKFold
    from revision.common.registry import LABEL_COL

    cv = data.spec['cv']
    labels = data.train.obs[LABEL_COL].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=cv.get('n_folds', 5), shuffle=True, random_state=cv.get('seed', 0))
    train_idx, test_idx = list(splitter.split(labels, labels))[fold]
    data.query = data.train[test_idx].copy()
    data.ground_truth = pd.DataFrame({'label': labels[test_idx]}, index=data.query.obs_names)
    data.train = data.train[train_idx].copy()
    return data


def quiet() -> None:
    """Silence per-fit Lightning banners and solver convergence warnings."""
    for name in ('pytorch_lightning', 'lightning_fabric', 'lightning.pytorch', 'lightning.fabric'):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')


def main(argv=None):
    parser = base_parser(__doc__)
    parser.add_argument('--types', nargs='*', default=None, help='Only these model types.')
    parser.add_argument('--units', nargs='*', default=None, help='Only these pooled unit IDs.')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Folds for the out-of-fold scores (default: protocol).')
    parser.add_argument('--worker', type=int, default=0, help='This worker\'s index.')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers.')
    parser.add_argument('--reuse-pool', action='store_true',
                        help='Use the existing pool instead of rebuilding it (for parallel workers, '
                             'after one call has built it).')
    parser.add_argument('--no-cv', action='store_true',
                        help='Skip cross-validation: one fit on all terminal cells (no out-of-fold '
                             'scores). For extra seeds when only query predictions are needed.')
    parser.add_argument('--type-order', nargs='*', default=None,
                        help='Run model types in this order (e.g. cheap ones first); unlisted types last.')
    parser.add_argument('--fold', type=int, default=None,
                        help='For registry entries with a cv block: the fold held out as the query.')
    parser.add_argument('--dynamic', action='store_true',
                        help='Claim units through lock files instead of the static --workers split.')
    parser.add_argument('--clear-claims', action='store_true',
                        help='Remove claims left by killed workers (unfinished units) before starting.')
    parser.add_argument('--min-batch-size', type=int, default=None,
                        help='Skip neural configurations with a smaller training batch (their optimizer '
                             'steps grow with the number of cells; used for Zheng68K).')
    parser.add_argument('--min-batch-types', nargs='*', default=None,
                        help='Model types --min-batch-size applies to (default: all with a batch size).')
    parser.add_argument('--threads', type=int, default=1,
                        help='CPU threads per worker for BLAS, OpenMP and PyTorch (default 1).')
    parser.add_argument('--cpu-max-batch', type=int, default=None,
                        help='Train neural configurations of --cpu-types with at most this batch size on the '
                             'CPU. With several workers sharing one GPU, tiny batches wait on the GPU far '
                             'longer than they take to compute on one core.')
    parser.add_argument('--cpu-types', nargs='*', default=['mlp', 'rnn'],
                        help='Model types --cpu-max-batch may move to the CPU. The convolutional "resnet" '
                             'type stays on the GPU: on one core it was ten times slower.')
    args = parser.parse_args(argv)
    registry, protocol, _, seeds = resolve(args)

    import torch
    from pytorch_lightning import seed_everything
    from threadpoolctl import threadpool_limits
    from ageas import Deck, Hangar, n_kfold_selection
    quiet()  # after the imports: Lightning sets its own log level when imported
    # Parallelism comes from running several workers, so each worker keeps its CPU thread pools small.
    # scikit-learn's libsvm computes dense kernels with one BLAS ddot per pair of cells; with
    # multi-threaded BLAS a linear SVC on 540 cells spun 24 threads for over a quarter of an hour.
    # PyTorch builds every batch with torch.stack; with one 24-thread OpenMP pool per worker the pools
    # oversubscribe the cores, and stacking 2,000 ATAC rows took 9 s instead of a fraction of a second.
    pool_limits = threadpool_limits(limits=args.threads)  # noqa: F841 (kept for the process lifetime)
    torch.set_num_threads(args.threads)

    name = args.dataset if args.fold is None else f'{args.dataset}_fold{args.fold}'
    out = results_dir('E17', name, args.results_root)
    pool_dir = os.path.join(out, 'pool')
    if not (args.reuse_pool and os.path.exists(os.path.join(out, 'pool_manifest.csv'))):
        build_pool(pool_dir).to_csv(os.path.join(out, 'pool_manifest.csv'), index=False)
    hangar = Hangar(pool_dir)
    for unit in hangar.units.values():
        # Data are served from memory, so loader worker processes only add start-up time.
        unit.config['dataloader_workers'] = 0
        # One Lightning log folder per worker: parallel workers sharing one would pick the same
        # version folder and break each other's CSV metric logs.
        unit.config['default_root_dir'] = os.path.join(out, 'lightning', f'w{args.worker}')
    # Ordering by the content hash interleaves model types, so parallel workers do not all reach
    # the memory-hungry SVCs at the same time; --type-order puts whole types first instead.
    type_rank = {t: i for i, t in enumerate(args.type_order or [])}

    def batch_ok(unit) -> bool:
        batch = (unit.config.get('train_config') or {}).get('batch_size')
        if args.min_batch_types is not None and unit.type not in args.min_batch_types:
            return True
        return args.min_batch_size is None or batch is None or batch >= args.min_batch_size

    unit_ids = sorted((u for u in hangar.units
                       if (args.types is None or hangar.units[u].type in args.types)
                       and (args.units is None or u in args.units) and batch_ok(hangar.units[u])),
                      key=lambda u: (type_rank.get(hangar.units[u].type, len(type_rank)),
                                     u.rsplit('_', 1)[-1]))
    if not args.dynamic:
        unit_ids = unit_ids[args.worker::args.workers]

    data = prepare_dataset(args.dataset, registry)
    if args.fold is not None:
        data = cv_fold(data, args.fold)
        data.ground_truth.to_csv(os.path.join(out, 'truth.csv'))
    print(f'{name}: {data.train.n_obs} terminal cells, {data.query.n_obs} query cells, '
          f'{data.train.n_vars} features; {len(unit_ids)} units x {len(seeds)} seeds', flush=True)
    train = _corpus(data.train, data.classes, dense=True)
    query = _corpus(data.query, data.classes, dense=True)
    cells = np.asarray(data.query.obs_names)
    selection = dict(protocol['selection'], **OPEN_GATES)
    if args.kfold:
        selection['kfold_selection_list'] = [args.kfold]

    probs_dir = os.path.join(out, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    if args.clear_claims:
        for claim in glob.glob(os.path.join(probs_dir, '*.claim')):
            if not os.path.exists(claim[:-len('.claim')] + '.npz'):
                os.remove(claim)
    single = args.workers == 1 and not args.dynamic
    runs_path = os.path.join(out, 'runs.csv' if single else f'runs.w{args.worker}.csv')
    done = set()
    finished = load_runs_table(out)
    if finished is not None:
        ok = finished[finished['status'] == 'ok']
        done = set(zip(ok['unit_id'], ok['seed']))

    for seed in seeds:
        for unit_id in unit_ids:
            if (unit_id, seed) in done:
                continue
            claim = os.path.join(probs_dir, f'{unit_id}_seed{seed}.claim')
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue  # another worker has this unit
            row = dict(unit_id=unit_id, type=hangar.units[unit_id].type, seed=seed)
            batch = (hangar.units[unit_id].config.get('train_config') or {}).get('batch_size')
            accelerator = ('cpu' if args.cpu_max_batch is not None and batch is not None
                           and batch <= args.cpu_max_batch and hangar.units[unit_id].type in args.cpu_types
                           else args.accelerator)
            start = time.time()
            try:
                seed_everything(seed, workers=True)
                if args.no_cv:
                    deck = Deck(squad=hangar.sortie_generate(unit_list=[unit_id]), n_dataloader_workers=0,
                                accelerator=accelerator)
                    deck.sortie(train_data=train, vali_data=train, n_classes=len(data.classes),
                                fea_names=list(data.train.var_names), save_model=True, verbose=False)
                else:
                    deck = n_kfold_selection(hangar=hangar, operation_name='trail', query_dataset=train,
                                             accelerator=accelerator, seed=seed,
                                             using_model_list=[unit_id], **selection)
                unit = deck.squad[unit_id]
                if isinstance(unit.model, torch.nn.Module):
                    unit.model.cpu()  # Deck.predict feeds CPU batches
                probs, _ = deck.predict(query, batch_size=512, num_workers=0)
                np.savez_compressed(os.path.join(probs_dir, f'{unit_id}_seed{seed}.npz'),
                                    cells=cells, probs=probs, classes=np.asarray(data.classes))
                if not args.no_cv:
                    row.update(unit.report['trail']['final']['oof'])
                row['status'] = 'ok'
                del deck, unit
            except Exception as exc:  # one failing configuration must not stop the search
                row['status'] = f'error: {type(exc).__name__}: {exc}'.replace('\n', ' ')[:300]
                os.remove(claim)  # a later run may retry it
            row['seconds'] = round(time.time() - start, 1)
            pd.DataFrame([row]).reindex(columns=RUN_COLUMNS).to_csv(
                runs_path, mode='a', index=False, header=not os.path.exists(runs_path))
            print(f"seed {seed} {unit_id}: {row['status']} {row['seconds']}s "
                  f"oof.accuracy={row.get('oof.accuracy', float('nan')):.3f}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
