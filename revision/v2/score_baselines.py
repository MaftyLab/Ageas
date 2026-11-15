"""Score Ageas v2 exactly like the baseline methods, for the dashboard.

Simulation (test.result/simulation_benchmark): every baseline run is rescored from its
predictions with ``binary_scores`` (positive progeny_B, threshold 0.5) and checked against
``scores_per_dataset.csv``; R1 and v2 are scored the same way on the same cells. v2 is its
final output (``v2_final``, from ``graph_stage.py``): the graph stage's fusion where the data
holds unlabeled cells (the simulated datasets), the committee vote otherwise (the real ones).
One row per dataset x modality x method.

Real data (test.result/real_data_baselines): the same scores for the 8 re-run baselines and
v2 (seed 0) on LARRY, iEP scRNA-seq and iEP scATAC-seq, plus a 1,000-resample
bootstrap over cells (seed 0, as ``bootstrap_ci``; the same resamples for every method of a
dataset). Box statistics per method x dataset x metric: quartiles of the resamples and their
2.5-97.5% interval (the README's 95% CI). Manuscript-only methods (CoSpar, WOT, CellRank
real-time kernel) are added as points. R1 is not compared on real data.

usage: score_baselines.py <Ageas.share root> <out dir>   (run with python -I -B; writes sim_scores.csv and real_boxes.csv)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SHARE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(SHARE, 'Ageas'))
from revision.common.metrics import binary_scores  # noqa: E402

SIM_BENCH = '/home/jack/ageas_work/v2/sim'
FINAL = '/home/jack/ageas_work/v2/graph/results'  # v2_final_seed0.csv per setup (graph_stage.py)
V2_REAL = '/home/jack/ageas_work/v2/refine/round5/results'
SIM_BASE = os.path.join(SHARE, 'test.result', 'simulation_benchmark')
REAL_BASE = os.path.join(SHARE, 'test.result', 'real_data_baselines')
METRICS = ['auroc', 'accuracy', 'f1_macro']
FILES = ['centroid', 'knn15', 'knn30', 'fateid', 'stemnet', 'cospar', 'wot', 'cr_realtime',
         'cr_pseudotime', 'cr_cytotrace', 'palantir']
REAL = [('larry_bench_hvg2000', 'LARRY'), ('iep_rna_hvg2000', 'iEP scRNA-seq'), ('iep_atac_peak', 'iEP scATAC-seq')]
os.makedirs(OUT, exist_ok=True)


def scores(y, p):
    s = binary_scores(y, p)
    return {m: float(s[m]) for m in METRICS}


def positive_prob(path, positive, cells):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    missing = set(cells) - set(frame.index)
    assert not missing, f'{path}: {len(missing)} query cells without a prediction'
    return frame.loc[cells, f'prob_{positive}'].to_numpy(dtype=float), frame


# ---------------------------------------------------------------- simulation
registry = {k: v for k, v in json.load(open(f'{SIM_BENCH}/registry.json')).items() if not k.startswith('_')}
given = pd.read_csv(f'{SIM_BASE}/scores_per_dataset.csv')
given['modality'] = given['modality'].str.lower()
given = given.set_index(['non_lineage', 'difficulty', 'seed', 'modality', 'method'])
rows, worst = [], 0.0
for setup, spec in registry.items():
    sim = spec['simulation']
    truth = pd.read_csv(f'{SIM_BENCH}/results/{setup}/truth.csv', index_col=0)['label'].astype(str)
    truth.index = truth.index.astype(str)
    cells = list(truth.index)
    y = (truth == 'progeny_B').to_numpy().astype(int)
    base = dict(modality=sim['modality'], non_lineage=int(sim['non_lineage_factors']),
                difficulty=sim['difficulty'], seed=int(sim['seed']), split=sim['split'], n=len(cells))
    for name in FILES:
        p, frame = positive_prob(f"{SIM_BASE}/predictions/{sim['dataset']}/{sim['modality']}/{name}.csv",
                                 'progeny_B', cells)
        method = str(frame['method'].iloc[0])
        s = scores(y, p)
        ref = given.loc[(base['non_lineage'], base['difficulty'], base['seed'], base['modality'], method)]
        assert int(ref['n']) == len(cells), (setup, method, ref['n'], len(cells))
        worst = max(worst, max(abs(s[m] - float(ref[m])) for m in METRICS))
        rows.append(dict(base, method=method, **s))
    for label, path in (('Ageas R1', f'{SIM_BENCH}/results/{setup}/r1_seed0.csv'),
                        ('Ageas v2', f'{FINAL}/{setup}/v2_final_seed0.csv')):
        p, _ = positive_prob(path, 'progeny_B', cells)
        rows.append(dict(base, method=label, **scores(y, p)))
sim_scores = pd.DataFrame(rows)
print(f'simulation: {len(sim_scores)} rows; baselines rescored, largest difference from '
      f'scores_per_dataset.csv {worst:.2e}')
out = sim_scores.copy()
out['modality'] = out['modality'].str.upper()
for m in METRICS:
    out[m] = out[m].round(4)
out = out[['modality', 'non_lineage', 'difficulty', 'seed', 'split', 'method', 'n'] + METRICS]
out.to_csv(f'{OUT}/sim_scores.csv', index=False)

# ---------------------------------------------------------------- real data
manuscript = pd.read_csv(f'{REAL_BASE}/scores_with_manuscript_methods.csv')
manuscript = manuscript[manuscript['source'].str.startswith('Manuscript')]
given_real = pd.read_csv(f'{REAL_BASE}/scores.csv')
box_rows, point_rows, missing = [], [], []
for key, label in REAL:
    info = json.load(open(f'{REAL_BASE}/predictions/{key}/classes.json'))
    positive = info['positive']
    truth = pd.read_csv(f'{V2_REAL}/{key}/truth.csv', index_col=0)
    truth.index = truth.index.astype(str)
    col = 'label' if 'label' in truth.columns else truth.columns[0]
    truth = truth[col].astype(str)
    cells = list(truth.index)
    assert len(cells) == info['n_query'], (key, len(cells), info['n_query'])
    y = (truth == positive).to_numpy().astype(int)
    methods = {}
    for name in [f for f in FILES if os.path.exists(f'{REAL_BASE}/predictions/{key}/{f}.csv')]:
        p, frame = positive_prob(f'{REAL_BASE}/predictions/{key}/{name}.csv', positive, cells)
        methods[str(frame['method'].iloc[0])] = ('New run (baseline)', p)
    v2_path = f'{FINAL}/{key}/v2_final_seed0.csv'
    if os.path.exists(v2_path):
        methods['Ageas v2'] = ('This benchmark, seed 0', positive_prob(v2_path, positive, cells)[0])
    else:
        missing.append(label)
    rng = np.random.default_rng(0)
    resamples = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    resamples = [i for i in resamples if len(np.unique(y[i])) == 2]
    for method, (source, p) in methods.items():
        est = scores(y, p)
        if source.startswith('New run'):
            ref = given_real[(given_real['registry_key'] == key) & (given_real['method'] == method)].iloc[0]
            assert all(abs(est[m] - float(ref[m])) < 1e-9 for m in METRICS), (key, method)
        boot = pd.DataFrame([scores(y[i], p[i]) for i in resamples])
        for m in METRICS:
            q = np.quantile(boot[m], [0.025, 0.25, 0.5, 0.75, 0.975])
            box_rows.append(dict(dataset=label, method=method, source=source, metric=m, estimate=est[m],
                                 lo=q[0], q1=q[1], median=q[2], q3=q[3], hi=q[4], n_cells=len(y),
                                 n_boot=len(resamples)))
    for _, r in manuscript.iterrows():
        if r['dataset'].split(' (')[0].replace('iEP CellTag-multi', 'iEP') != label.split(' ')[0] and not (
                label.startswith('iEP') and r['dataset'].startswith('iEP')):
            continue
        if label.startswith('iEP') and ('scATAC' in r['dataset']) != ('scATAC' in label):
            continue
        name = str(r['method'])
        if name.startswith('Ageas'):
            continue  # R1 is not compared on real data
        name = {'CellRank RealTime': 'CellRank RealTime'}.get(name, name)
        for m in ('auroc', 'accuracy'):
            box_rows.append(dict(dataset=label, method=name, source='Manuscript value', metric=m,
                                 estimate=float(r[m]), lo=np.nan, q1=np.nan, median=np.nan, q3=np.nan,
                                 hi=np.nan, n_cells=len(y), n_boot=0))
real = pd.DataFrame(box_rows)
for c in ['estimate', 'lo', 'q1', 'median', 'q3', 'hi']:
    real[c] = real[c].round(4)
real.to_csv(f'{OUT}/real_boxes.csv', index=False)
print(f'real data: {len(real)} rows; v2 missing on: {missing or "none"}')

# ---------------------------------------------------------------- quick look
pd.set_option('display.width', 220)
mean = sim_scores.groupby(['modality', 'non_lineage', 'difficulty', 'method'])[METRICS].mean().reset_index()
for m in METRICS:
    mean[f'rank_{m}'] = mean.groupby(['modality', 'non_lineage', 'difficulty'])[m].rank(ascending=False)
print('\nmean rank over scenarios (1 = best of 13):')
print(mean.groupby(['modality', 'method'])[[f'rank_{m}' for m in METRICS]].mean().round(1)
      .sort_values(['modality', 'rank_auroc']).to_string())
print('\nsimulation mean over datasets:')
print(sim_scores.groupby(['modality', 'method'])[METRICS].mean().round(3).sort_values(['modality', 'auroc'], ascending=[True, False]).to_string())
print('\nreal data point estimates:')
print(real.pivot_table(index='method', columns=['dataset', 'metric'], values='estimate').round(3).to_string())
