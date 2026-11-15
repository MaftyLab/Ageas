#!/usr/bin/env python3
"""Apply v2's graph stage to finished v2 runs, without rerunning their units.

The graph stage comes after the committee vote and reads nothing else from a
run: its inputs are the vote and every cell's features. For every setup of a
registry with a finished v2 run in ``--runs``, this script builds the
all-cells view (``run_job.graph_input``), applies the protocol's graph stage
to the saved vote, and writes in ``<out>/results/<setup>/``:

* ``v2_final_<rep>.csv`` — v2's output (the vote, unchanged, where the stage
  does not run);
* ``graph_<rep>.json`` — the stage's record (whether it ran and why, the
  numbers of unlabeled and query cells, the walk's share, the weight);
* ``truth.csv`` — copied from the run, for scoring only.

The units are the saved run's own; a fresh run of the same protocol differs
from this only where the ResNets do (GPU nondeterminism).

usage: graph_stage.py --runs <bench with v2 runs> --out <bench> [--registry <json>] [--setups ...] [--rep seed0]
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_job  # noqa: E402  (puts the code it sits in on sys.path)

import pandas as pd  # noqa: E402

from ageas.tool.graph import graph_stage  # noqa: E402


def stage_for(name, registry, runs, out, rep, settings) -> str:
    """Apply the stage to one setup's saved vote; returns what happened."""
    run_dir = os.path.join(runs, 'results', name)
    out_dir = os.path.join(out, 'results', name)
    if not os.path.exists(os.path.join(run_dir, f'v2_{rep}.json')):
        return 'no finished v2 run'
    if os.path.exists(os.path.join(out_dir, f'graph_{rep}.json')):
        return 'done already'
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    seed = int(rep[len('seed'):]) if rep.startswith('seed') else 0
    data = run_job.prepare_dataset(name, registry)
    saved = pd.read_csv(os.path.join(run_dir, f'v2_vote_{rep}.csv'), index_col=0)
    saved.index = saved.index.astype(str)
    vote = saved.loc[data.query.obs_names.astype(str), [f'prob_{c}' for c in data.classes]].to_numpy(float)
    graph = run_job.graph_input(data)
    if graph is None:
        stage = {'probs': vote, 'active': False, 'reason': 'no unlabeled cells given', 'n_context': 0,
                 'n_query': len(vote), 'walk': None, 'share': None, 'weight': None}
    else:
        stage = graph_stage(vote, graph['features'], graph['labels'], graph['query_index'], settings,
                            kind=graph['kind'], seed=seed)
    record = run_job.save_final(stage['probs'], data, seed, os.path.join(out_dir, f'v2_final_{rep}'), stage)
    truth = os.path.join(run_dir, 'truth.csv')
    if os.path.exists(truth):
        shutil.copyfile(truth, os.path.join(out_dir, 'truth.csv'))
    record.update(setup=name, rep=rep, runs=run_dir, settings=settings, kind=None if graph is None else graph['kind'],
                  seconds=round(time.time() - start, 1))
    run_job._write_json(os.path.join(out_dir, f'graph_{rep}.json'), record)
    return f"{'ran' if record['active'] else 'passed the vote through'} ({record['reason']})"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', required=True, help='Bench whose results/ hold the finished v2 runs.')
    parser.add_argument('--out', required=True, help='Bench to write results/<setup>/ into.')
    parser.add_argument('--registry', default=os.path.join(run_job.REPO, 'revision', 'datasets.share.json'))
    parser.add_argument('--setups', nargs='*', default=None, help='Registry keys (default: all).')
    parser.add_argument('--rep', default='seed0')
    args = parser.parse_args(argv)

    registry = run_job.load_registry(args.registry)
    with open(run_job.PROTOCOL, encoding='utf-8') as fh:
        settings = json.load(fh)['graph']
    for name in args.setups or list(registry):
        print(f'{name}: {stage_for(name, registry, args.runs, args.out, args.rep, settings)}', flush=True)


if __name__ == '__main__':
    main()
