#!/usr/bin/env python3
"""R2.4 ablation: which components carry Ageas v2's performance (``revision/v2/ablation_r2_4.md``).

Every offline variant is rebuilt from the saved round-5 runs with v2's own
functions (no-harm mixture, competence, committee, vote, graph stage): each
unit's query probabilities (``v2_seed0_members.npz``), out-of-fold records
(``v2_seed0_oof.npz``) and run record. Where the graph stage runs (the
simulated datasets), it is applied to every variant, so each variant is
"v2 with one component changed". The two re-run arms (routing off, fixed
panel) add their own final outputs. Rebuilding the full set must reproduce
the saved mixture, vote and final output (``checks.csv``).

Variants: ``full``, ``no_graph_stage``, ``vote_alone``, ``mixture_alone``,
``plain_average``, ``best_single_validation`` (lowest shift-weighted
out-of-fold log-loss), ``best_single_labels`` (best progenitor AUROC: an
optimistic ceiling), ``classical_only``, ``deep_only``,
``no_second_round``, ``best_<class>`` (each class's validation-best unit),
``no_routing`` and ``fixed_panel`` (re-run arms).

Scores: accuracy, macro F1 (threshold 0.5) and AUROC (``binary_scores``).
Real datasets get 1,000 bootstrap resamples of the query cells (seed 0, the
same for every variant) for intervals and paired differences from ``full``;
simulated datasets get per-dataset scores and, per modality and seed half,
the mean difference from ``full`` with a 95% interval from resampling
datasets, and a Wilcoxon signed-rank test with Holm correction across
variants.

Writes to ``--out``: ``variants_real.csv``, ``variants_sim.csv``,
``summary_sim.csv``, ``units.csv``, ``checks.csv`` (and ``walks/`` caches).
With ``--predictions <dir>`` it also writes each variant's query predictions
in the benchmark's prediction format, ``<dir>/<setup>/<variant>.csv``
(``prob_<class>``, ``pred`` at P >= 0.5, ``method``), with ``classes.json``
and ``truth.csv``.

usage: ablate.py --out <dir> [--arms <ablation dir>] [--setups ...] [--predictions <dir>]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_job  # noqa: E402  (puts the code it sits in on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from ageas.tool.committee import competent_units, select_committee, vote  # noqa: E402
from ageas.tool.graph import fuse, graph_fate  # noqa: E402
from ageas.tool.synergy import log_loss, no_harm_selection  # noqa: E402
from revision.common.metrics import binary_scores  # noqa: E402

METRICS = ['auroc', 'accuracy', 'f1_macro']
CLASSICAL = ('logreg', 'svc', 'mnb', 'xgb')
DEEP = ('mlp', 'resnet', 'rnn')
ARMS = ('no_routing', 'fixed_panel')
REAL = {'larry_bench_hvg2000': 'LARRY', 'iep_rna_hvg2000': 'iEP scRNA-seq', 'iep_atac_peak': 'iEP scATAC-seq'}


def scores(y, p) -> dict:
    s = binary_scores(y, p)
    return {m: float(s[m]) for m in METRICS}


def aligned(path, cells, classes) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    probs = frame.loc[cells, [f'prob_{c}' for c in classes]].to_numpy(float)
    return probs / probs.sum(axis=1, keepdims=True)


class Setup:
    """One dataset's saved round-5 run, and v2's integration rebuilt on any subset of its units."""

    def __init__(self, folder, settings):
        self.folder = folder
        self.record = json.load(open(os.path.join(folder, 'v2_seed0.json'), encoding='utf-8'))
        self.classes = self.record['classes']
        z = np.load(os.path.join(folder, 'v2_seed0_members.npz'), allow_pickle=True)
        self.cells = z['cells'].astype(str)
        self.members = {u: z[u] / z[u].sum(axis=1, keepdims=True) for u in z.files if u != 'cells'}
        oof = np.load(os.path.join(folder, 'v2_seed0_oof.npz'), allow_pickle=True)
        self.oof = {k[len('oof_'):]: oof[k] for k in oof.files if k.startswith('oof_')}
        self.labels, self.weights = oof['labels'], oof['cell_weights']
        self.types = {u: u.split('_')[0] for u in self.members}
        self.second_round = {s['unit_id'] for s in self.record['panel'] if s.get('shaped_by') == ['validation_edge']}
        self.committee_settings, self.synergy = settings['committee'], settings['synergy']
        truth = pd.read_csv(os.path.join(folder, 'truth.csv'), index_col=0)
        truth.index = truth.index.astype(str)
        column = 'label' if 'label' in truth.columns else truth.columns[0]
        self.truth = truth.loc[self.cells, column].astype(str).to_numpy()
        self.y = (self.truth == self.classes[1]).astype(int)
        self.saved_vote = aligned(os.path.join(folder, 'v2_vote_seed0.csv'), self.cells, self.classes)
        self.saved_mixture = aligned(os.path.join(folder, 'v2_seed0.csv'), self.cells, self.classes)

    def oof_loss(self, unit) -> float:
        return log_loss(self.oof[unit], self.labels, self.weights)

    def integrate(self, units) -> dict:
        """The no-harm mixture, committee and vote over ``units``, as the pipeline builds them."""
        c = self.committee_settings
        oof = {u: self.oof[u] for u in units}
        weights = no_harm_selection(oof, self.labels, min_relative_gain=self.synergy['min_relative_gain'],
                                    max_steps=self.synergy['max_steps'], weights=self.weights)
        mixture = sum(w * self.members[u] for u, w in weights.items())
        ranked = [u for u in competent_units(oof, self.labels, self.weights, c['tolerance']) if u in self.members]
        chosen = select_committee({u: self.members[u] for u in ranked}, ranked,
                                  max_jsd=c['max_jsd'], min_majority=c['min_majority'])
        groups = {u: self.types[u] for u in chosen} if c['balance'] == 'type' else None
        members = {u: self.members[u] for u in chosen}
        voted = vote(members, rule=c['rule'], groups=groups, mixture=mixture if c['with_mixture'] else None)['probs']
        alone = vote(members, rule=c['rule'], groups=groups)['probs']
        return {'vote': voted, 'vote_alone': alone, 'mixture': mixture, 'weights': weights, 'committee': chosen}


def walk_for(name, spec, data_dir, cache, cells) -> np.ndarray:
    """The graph stage's walk for the query cells (cached), matched to ``cells`` by name."""
    path = os.path.join(cache, f'{name}.npz')
    if not os.path.exists(path):
        spec = dict(spec, h5ad=spec['h5ad'].replace('../data', data_dir))
        data = run_job.prepare_dataset(name, {name: spec})
        graph = run_job.graph_input(data)
        settings = json.load(open(run_job.PROTOCOL, encoding='utf-8'))['graph']
        walk = graph_fate(graph['features'], graph['labels'], graph['kind'], settings['neighbors'],
                          settings['diffusion_components'], settings['n_components'], seed=0)[graph['query_index']]
        np.savez_compressed(path, walk=walk, cells=np.asarray(data.query.obs_names).astype(str))
    saved = np.load(path)
    index = pd.Index(saved['cells'].astype(str)).get_indexer(cells)
    assert (index >= 0).all(), f'{name}: query cells of the walk and the run differ'
    return saved['walk'][index]


def variants_for(name, setup, final_dir, arms_dir, walk, weight) -> tuple:
    """Every variant's query probabilities, the unit table and the rebuild check."""
    post = (lambda p: fuse(p, walk, weight)) if walk is not None else (lambda p: p)
    units = list(setup.members)
    full = setup.integrate(units)
    final = aligned(os.path.join(final_dir, name, 'v2_final_seed0.csv'), setup.cells, setup.classes)
    check = {
        'setup': name,
        'mixture_max_diff': float(np.abs(full['mixture'] - setup.saved_mixture).max()),
        'vote_max_diff': float(np.abs(full['vote'] - setup.saved_vote).max()),
        'final_max_diff': float(np.abs(post(full['vote']) - final).max()),
        'weights_equal': {u: round(w, 6) for u, w in full['weights'].items()}
                         == {u: round(w, 6) for u, w in setup.record['weights'].items()},
        'committee_equal': full['committee'] == setup.record['committee'],
    }
    by_validation = sorted(units, key=setup.oof_loss)
    own = {u: scores(setup.y, setup.members[u][:, 1]) for u in units}
    best_labels = max(units, key=lambda u: own[u]['auroc'])
    out = {
        'full': final,
        'no_graph_stage': setup.saved_vote,
        'vote_alone': post(full['vote_alone']),
        'mixture_alone': post(full['mixture']),
        'plain_average': post(np.mean([setup.members[u] for u in units], axis=0)),
        'best_single_validation': post(setup.members[by_validation[0]]),
        'best_single_labels': post(setup.members[best_labels]),
    }
    for label, kinds in (('classical_only', CLASSICAL), ('deep_only', DEEP)):
        subset = [u for u in units if setup.types[u] in kinds]
        if subset:
            out[label] = post(setup.integrate(subset)['vote'])
    if setup.second_round:
        out['no_second_round'] = post(setup.integrate([u for u in units if u not in setup.second_round])['vote'])
    for kind in CLASSICAL + DEEP:
        members = [u for u in by_validation if setup.types[u] == kind]
        if members:
            out[f'best_{kind}'] = post(setup.members[members[0]])
    for arm in ARMS:
        path = os.path.join(arms_dir, arm, 'results', name, 'v2_final_seed0.csv')
        if os.path.exists(path):
            out[arm] = aligned(path, setup.cells, setup.classes)
    table = pd.DataFrame([dict(setup=name, unit=u, unit_class=setup.types[u],
                               round=2 if u in setup.second_round else 1, oof_logloss=setup.oof_loss(u),
                               validation_rank=by_validation.index(u) + 1, in_committee=u in full['committee'],
                               mixture_weight=full['weights'].get(u, 0.0), **own[u]) for u in units])
    picks = {'best_single_validation': by_validation[0], 'best_single_labels': best_labels}
    return out, table, dict(check, **{f'unit_{k}': v for k, v in picks.items()})


def write_predictions(folder, setup, variants) -> None:
    """Each variant's query probabilities in the benchmark's prediction format, with class order and truth."""
    os.makedirs(folder, exist_ok=True)
    for v, p in variants.items():
        frame = pd.DataFrame(p, index=setup.cells, columns=[f'prob_{c}' for c in setup.classes])
        frame['pred'] = np.where(p[:, 1] >= 0.5, setup.classes[1], setup.classes[0])  # binary_scores' cut-off
        frame['method'] = v
        frame.to_csv(os.path.join(folder, f'{v}.csv'))
    with open(os.path.join(folder, 'classes.json'), 'w', encoding='utf-8') as handle:
        json.dump({'classes': list(setup.classes), 'positive': setup.classes[1], 'n_query': len(setup.cells)},
                  handle, indent=1)
    pd.DataFrame({'truth': setup.truth}, index=setup.cells).to_csv(os.path.join(folder, 'truth.csv'))


def bootstrap_real(dataset, y, variants, n_boot=1000) -> list:
    rng = np.random.default_rng(0)
    resamples = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if len(np.unique(y[i])) == 2]
    boot = {v: pd.DataFrame([scores(y[i], p[i, 1]) for i in resamples]) for v, p in variants.items()}
    rows = []
    for v, p in variants.items():
        est = scores(y, p[:, 1])
        for m in METRICS:
            diff = boot[v][m] - boot['full'][m]
            rows.append(dict(dataset=dataset, variant=v, metric=m, estimate=est[m],
                             lo=boot[v][m].quantile(0.025), hi=boot[v][m].quantile(0.975),
                             diff=est[m] - scores(y, variants['full'][:, 1])[m],
                             diff_lo=diff.quantile(0.025), diff_hi=diff.quantile(0.975), n_cells=len(y),
                             n_boot=len(resamples)))
    return rows


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adjusted, running = {}, 0.0
    for k, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * p[idx]))
        adjusted[idx] = running
    return pd.Series(adjusted)[p.index]


def summarise_sim(frame: pd.DataFrame) -> pd.DataFrame:
    halves = {'all': frame.seed > 0, 'dev': frame.seed <= 5, 'held_out': frame.seed >= 6}
    rows = []
    for modality, (half, mask) in ((m, h) for m in ('RNA', 'ATAC') for h in halves.items()):
        part = frame[(frame.modality == modality) & mask]
        if part.empty:
            continue
        full = part[part.variant == 'full'].set_index('setup')
        for m in METRICS:
            block = []
            for v, rows_v in part.groupby('variant'):
                paired = rows_v.set_index('setup')[m].dropna()
                common = paired.index.intersection(full.index)
                diffs = paired[common] - full.loc[common, m]
                p = (wilcoxon(diffs).pvalue if v != 'full' and len(common) > 5 and (diffs != 0).any() else np.nan)
                rng = np.random.default_rng(0)  # 95% interval of the mean difference, resampling datasets
                means = [diffs.to_numpy()[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(1000)]
                block.append(dict(modality=modality, half=half, variant=v, metric=m, n=len(common),
                                  mean=float(paired[common].mean()), diff=float(diffs.mean()),
                                  diff_lo=float(np.quantile(means, 0.025)), diff_hi=float(np.quantile(means, 0.975)),
                                  ahead=int((diffs > 0).sum()), behind=int((diffs < 0).sum()), p=p))
            block = pd.DataFrame(block)
            tested = block.p.notna()
            block.loc[tested, 'p_holm'] = holm(block.loc[tested, 'p']).values
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sim-runs', default='/home/jack/ageas_work/v2/sim/results')
    parser.add_argument('--real-runs', default='/home/jack/ageas_work/v2/refine/round5/results')
    parser.add_argument('--final', default='/home/jack/ageas_work/v2/graph/results',
                        help="v2's final outputs and graph records (graph_stage.py).")
    parser.add_argument('--arms', default='/home/jack/ageas_work/v2/ablation')
    parser.add_argument('--registry', default='/home/jack/ageas_work/v2/ablation/registry.json')
    parser.add_argument('--data', default='/home/jack/ageas_work/data')
    parser.add_argument('--setups', nargs='*', default=None)
    parser.add_argument('--predictions', default=None,
                        help="Also write each variant's query predictions to <dir>/<setup>/<variant>.csv.")
    args = parser.parse_args(argv)

    os.makedirs(os.path.join(args.out, 'walks'), exist_ok=True)
    protocol = json.load(open(run_job.PROTOCOL, encoding='utf-8'))
    registry = run_job.load_registry(args.registry)
    real_rows, sim_rows, unit_tables, checks = [], [], [], []
    for name in args.setups or list(registry):
        spec = registry[name]
        runs = args.real_runs if name in REAL else args.sim_runs
        if not os.path.exists(os.path.join(runs, name, 'v2_seed0.json')):
            print(f'{name}: no saved run, skipped', flush=True)
            continue
        setup = Setup(os.path.join(runs, name), protocol)
        graph_record = json.load(open(os.path.join(args.final, name, 'graph_seed0.json'), encoding='utf-8'))
        walk = (walk_for(name, spec, args.data, os.path.join(args.out, 'walks'), setup.cells)
                if graph_record['active'] else None)
        variants, table, check = variants_for(name, setup, args.final, args.arms, walk, protocol['graph']['weight'])
        unit_tables.append(table)
        checks.append(check)
        if args.predictions:
            write_predictions(os.path.join(args.predictions, name), setup, variants)
        if name in REAL:
            real_rows += bootstrap_real(REAL[name], setup.y, variants)
        else:
            sim = spec['simulation']
            base = dict(setup=name, modality=sim['modality'].upper(), non_lineage=int(sim['non_lineage_factors']),
                        difficulty=sim['difficulty'], seed=int(sim['seed']))
            sim_rows += [dict(base, variant=v, **scores(setup.y, p[:, 1])) for v, p in variants.items()]
        print(f"{name}: {len(variants)} variants; rebuild max diff vote {check['vote_max_diff']:.1e}, "
              f"final {check['final_max_diff']:.1e}", flush=True)
    pd.DataFrame(checks).to_csv(os.path.join(args.out, 'checks.csv'), index=False)
    pd.concat(unit_tables, ignore_index=True).to_csv(os.path.join(args.out, 'units.csv'), index=False)
    if real_rows:
        pd.DataFrame(real_rows).to_csv(os.path.join(args.out, 'variants_real.csv'), index=False)
    if sim_rows:
        frame = pd.DataFrame(sim_rows)
        frame.to_csv(os.path.join(args.out, 'variants_sim.csv'), index=False)
        summarise_sim(frame).to_csv(os.path.join(args.out, 'summary_sim.csv'), index=False)


if __name__ == '__main__':
    main()
