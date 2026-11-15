"""The whole v2 pipeline end to end on synthetic data.

Profile the data without progenitor labels, design the panel from the
profile, weight the terminal cells toward the progenitors, route and
integrate under that validation, and check the strategy.
"""
import inspect

import numpy as np
import torch

from ageas.tool.corpus_loader import get_all_data

from ageas import run_pipeline
from ageas.tool.feature_corpus import Feature_Corpus
from conftest import make_fake_corpus

RUN = dict(
    selection={'kfold': 3, 'valid_fraction': 0.25},
    routing={'mass_ladder': [0.9, 0.6], 'floor': 1, 'relative_delta': 0.5,
             'exp_sample_limit': 10, 'explain_args': {'exp_step': 4}},
    synergy={'min_relative_gain': 0.01},
    seed=0,
)


def test_the_pipeline_runs_from_profile_to_predictions_without_progenitor_labels(tmp_path):
    corpus = make_fake_corpus()
    x, _ = get_all_data(corpus)
    rng = np.random.default_rng(0)
    progenitors = x.numpy()[:40] + rng.normal(0.0, 0.5, size=x[:40].shape).astype(np.float32)

    result = run_pipeline(corpus, progenitors, workdir=str(tmp_path), **RUN)

    assert 'labels' not in ' '.join(inspect.signature(run_pipeline).parameters)
    assert result.profile['n_query'] == 40
    assert {spec.type for spec in result.panel} >= {'logreg', 'xgb', 'mlp'}
    assert abs(result.deck.oof['cell_weights'].mean() - 1.0) < 1e-6
    assert set(result.report['agreement']) == {'ordinary', 'shift'}
    assert result.report['rules']
    # The second round added units past the edges validation preferred, and
    # they have out-of-fold records like the rest.
    added = [spec for spec in result.panel if spec.shaped_by == ['validation_edge']]
    assert added
    assert all(spec.unit_id in result.deck.oof['probs'] for spec in result.panel)

    unlabelled = Feature_Corpus(torch.as_tensor(progenitors), torch.zeros(40, dtype=torch.long),
                                corpus.features, corpus.label_dict)
    probs, _ = result.deck.predict(query_dataset=unlabelled, num_workers=0)
    assert probs.shape == (40, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    # One run gives both integrations of the progenitors: the no-harm mixture
    # and the vote of a committee drawn from the competent units.
    query = result.query
    np.testing.assert_allclose(query['no_harm'], probs, atol=1e-6)
    assert result.committee and set(result.committee) <= set(result.deck.competent)
    np.testing.assert_allclose(query['vote'], np.mean([query['members'][u] for u in result.committee], axis=0))
    np.testing.assert_allclose(query['vote'].sum(axis=1), 1.0, atol=1e-6)
    assert query['agreement'].shape == (40,)
    assert ((query['agreement'] > 0) & (query['agreement'] <= 1)).all()
    assert (query['disagreement'] >= 0).all()

    # Without unlabeled cells to walk through, the final output is the vote.
    assert result.graph['active'] is False
    np.testing.assert_array_equal(query['final'], query['vote'])


GRAPH = {'min_context_per_query': 1.0, 'neighbors': 10, 'diffusion_components': 4, 'n_components': 8,
         'weight': 0.6}


def test_with_unlabeled_cells_the_pipeline_walks_them_and_fuses_the_walk_into_the_vote(tmp_path):
    corpus = make_fake_corpus()
    x, y = get_all_data(corpus)
    x, y = x.numpy(), y.numpy()
    rng = np.random.default_rng(0)
    progenitors = x[:40] + rng.normal(0.0, 0.5, size=x[:40].shape).astype(np.float32)
    flat_x, flat_progenitors = x.reshape(len(x), -1), progenitors.reshape(40, -1)  # model input is (cells, 1, genes)
    mix = rng.uniform(0.2, 0.8, size=(80, 1))
    context = mix * flat_progenitors[rng.integers(0, 40, 80)] + (1 - mix) * flat_x[rng.integers(0, len(x), 80)]
    graph = {'features': np.vstack([flat_x, flat_progenitors, context]),
             'labels': np.concatenate([y, np.full(120, -1)]),
             'query_index': np.arange(len(x), len(x) + 40), 'kind': 'pca'}

    result = run_pipeline(corpus, progenitors, workdir=str(tmp_path), graph=graph, graph_settings=GRAPH, **RUN)

    assert result.graph['active'] is True
    assert result.graph['n_context'] == 80
    query = result.query
    assert query['final'].shape == (40, 2) and query['walk'].shape == (40, 2)
    np.testing.assert_allclose(query['final'].sum(axis=1), 1.0, atol=1e-6)
    assert (query['final'][:, 1] > 0.5).sum() == round(query['walk'][:, 1].mean() * 40)
