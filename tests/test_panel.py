"""Panel design from a dataset's label-free profile.

The rules are stated before any run: how far the progenitors sit from the
mature cells, how many terminal cells there are, how curved their class
boundary is and what the input looks like decide which model families
enter the panel and how much capacity and regularisation each gets. Every
unit says which rules shaped it, so validation can later check them.
"""
from ageas import Hangar

from ageas import design_panel, extend_panel, write_panel

#: A middle-of-the-road dataset: no rule fires strongly.
BASE = {
    'n_terminal': 3000, 'n_query': 500, 'n_features': 2000, 'n_classes': 2, 'min_class_share': 0.4,
    'sparsity': 0.6, 'nonnegative': True, 'integer_share': 0.2,
    'nn_distance_ratio': 1.3, 'coverage': 0.7, 'domain_auc': 0.8,
    'n_clusters': 3, 'query_own_clusters': 0.3, 'terminal_purity': 0.9,
    'query_ambiguity': 0.3, 'terminal_linear_accuracy': 0.93, 'terminal_nonlinear_gain': 0.02,
}


def test_every_family_the_input_allows_is_in_the_panel_and_loads_into_a_hangar(tmp_path):
    counts = dict(BASE, sparsity=0.9, integer_share=1.0)
    signed = dict(BASE, nonnegative=False)

    panel = design_panel(counts)
    hangar = Hangar(write_panel(panel, tmp_path / 'hangar'))

    assert {spec.type for spec in panel} == {'logreg', 'svc', 'mnb', 'xgb', 'mlp', 'resnet', 'rnn'}
    assert set(hangar.units) == {spec.unit_id for spec in panel}
    # Naive Bayes models counts, so it needs inputs that are never negative.
    assert 'mnb' not in {spec.type for spec in design_panel(signed)}


def test_with_the_rules_off_no_profile_rule_shapes_the_panel(monkeypatch):
    # Far progenitors, few and wide cells, counts, ambiguity, a straight boundary: many rules fire.
    busy = dict(BASE, n_terminal=800, n_features=3000, domain_auc=0.97, coverage=0.3, query_own_clusters=0.7,
                integer_share=1.0, query_ambiguity=0.8, terminal_linear_accuracy=0.99, terminal_nonlinear_gain=0.0)

    off = design_panel(busy, use_rules=False)
    on = design_panel(busy)
    import ageas.panel as panel_module
    monkeypatch.setattr(panel_module, 'active_rules', lambda profile: [])
    neutral = design_panel(busy)

    # The ablation's fixed panel: what the panel is when no rule fires; data-type checks still apply.
    assert [(s.unit_id, s.config) for s in off] == [(s.unit_id, s.config) for s in neutral]
    assert [(s.unit_id, s.config) for s in on] != [(s.unit_id, s.config) for s in off]
    assert not any(s.shaped_by or s.favoured_by or s.disfavoured_by for s in off)


def _of(panel, unit_type):
    return [spec for spec in panel if spec.type == unit_type]


def _strongest(panel, unit_type, key):
    return max(spec.config['model_params'][key] for spec in _of(panel, unit_type))


def test_progenitors_far_from_the_mature_cells_call_for_more_regularisation():
    far = design_panel(dict(BASE, domain_auc=0.97, coverage=0.3, query_own_clusters=0.7))
    near = design_panel(dict(BASE, domain_auc=0.6, coverage=0.95, query_own_clusters=0.0))

    assert _strongest(far, 'logreg', 'C') < _strongest(near, 'logreg', 'C')
    assert _strongest(far, 'xgb', 'max_depth') < _strongest(near, 'xgb', 'max_depth')
    for unit_type in ('mlp', 'resnet'):
        assert _strongest(far, unit_type, 'dropout') > _strongest(near, unit_type, 'dropout')
        assert (max(s.config['train_config']['weight_decay'] for s in _of(far, unit_type))
                > max(s.config['train_config']['weight_decay'] for s in _of(near, unit_type)))
    # The rule says which of its units it expects to validate best.
    logreg = sorted(_of(far, 'logreg'), key=lambda s: s.config['model_params']['C'])
    assert 'far_progenitors' in logreg[0].favoured_by
    assert 'far_progenitors' in logreg[-1].disfavoured_by
    assert all('far_progenitors' in s.shaped_by for s in far if s.type != 'mnb' and 'light' not in s.name)


def test_every_panel_carries_light_networks_trained_for_one_epoch():
    # R1's lightly trained, wide-latent networks (its resEnc_1 MLP and GRUs)
    # rank progenitors best (refinement round 1). Every panel carries them on
    # R1's recipe, whatever the profile; the committee decides if they vote.
    for profile in (BASE, dict(BASE, n_terminal=60000), dict(BASE, domain_auc=0.97, coverage=0.3)):
        light = {s.unit_id: s for s in design_panel(profile) if 'light' in s.name}
        assert set(light) == {'mlp_light', 'rnn_gru_light', 'rnn_gru_light_small'}
        for spec in light.values():
            assert spec.config['max_epochs'] == 1
            assert spec.config['train_config']['batch_size'] == 200
            assert spec.shaped_by == []
        assert light['mlp_light'].config['model_params']['latent_fea_dim'] == 128
        assert light['rnn_gru_light'].config['model_params']['block_layer_type'] == 'GRU'
        assert light['rnn_gru_light'].config['model_params']['latent_fea_dim'] == 128
        assert light['rnn_gru_light_small'].config['model_params']['latent_fea_dim'] == 64


def _width(spec):
    params = spec.config['model_params']
    return params['latent_fea_dim'] if spec.type == 'mlp' else params['latent_planes']


def test_few_terminal_cells_call_for_smaller_models_and_many_for_larger():
    few = design_panel(dict(BASE, n_terminal=400))
    many = design_panel(dict(BASE, n_terminal=8000))

    for unit_type in ('mlp', 'resnet'):
        assert max(_width(s) for s in _of(few, unit_type)) < max(_width(s) for s in _of(many, unit_type))
    assert (max(s.config['train_config']['num_boost_round'] for s in _of(few, 'xgb'))
            < max(s.config['train_config']['num_boost_round'] for s in _of(many, 'xgb')))
    # Smaller batches for fewer cells, so every network still takes enough steps.
    assert (_of(few, 'mlp')[0].config['train_config']['batch_size']
            < _of(many, 'mlp')[0].config['train_config']['batch_size'])
    roomy_few = next(s for s in _of(few, 'mlp') if s.name == 'roomy')
    assert 'few_cells' in roomy_few.disfavoured_by


def test_a_boundary_no_line_draws_calls_for_flexible_models_and_a_straight_one_for_linear():
    curved = design_panel(dict(BASE, terminal_linear_accuracy=0.85, terminal_nonlinear_gain=0.08))
    straight = design_panel(dict(BASE, terminal_linear_accuracy=0.99, terminal_nonlinear_gain=0.0))

    assert _strongest(curved, 'xgb', 'max_depth') > _strongest(straight, 'xgb', 'max_depth')
    assert max(_width(s) for s in _of(curved, 'mlp')) > max(_width(s) for s in _of(straight, 'mlp'))
    for spec in _of(curved, 'logreg') + _of(curved, 'svc'):
        assert 'curved_boundary' in spec.disfavoured_by
    for spec in _of(straight, 'logreg') + _of(straight, 'svc'):
        assert 'straight_boundary' in spec.favoured_by
    flexible_xgb = next(s for s in _of(curved, 'xgb') if s.name == 'flexible')
    assert 'curved_boundary' in flexible_xgb.favoured_by


def test_raw_counts_bring_a_naive_bayes_smoothing_grid():
    counts = design_panel(dict(BASE, sparsity=0.9, integer_share=1.0))
    scaled = design_panel(dict(BASE, integer_share=0.1))

    assert sorted(s.config['model_params']['alpha'] for s in _of(counts, 'mnb')) == [0.1, 1.0, 10.0]
    assert all('counts' in s.favoured_by and 'counts' in s.shaped_by for s in _of(counts, 'mnb'))
    # Non-negative but not counts (e.g. log-normalised): one naive Bayes, no expectation.
    assert [s.config['model_params']['alpha'] for s in _of(scaled, 'mnb')] == [1.0]
    assert _of(scaled, 'mnb')[0].favoured_by == []


def test_inputs_with_more_features_than_cells_get_sparse_linear_models_and_feature_sampling_trees():
    wide = design_panel(dict(BASE, n_features=20000))
    narrow = design_panel(BASE)

    assert {s.config['model_params']['penalty'] for s in _of(wide, 'logreg')} == {'l1'}
    assert {s.config['model_params']['solver'] for s in _of(wide, 'logreg')} == {'saga'}
    assert {s.config['model_params']['penalty'] for s in _of(narrow, 'logreg')} == {'l2'}
    assert _strongest(wide, 'xgb', 'colsample_bytree') < _strongest(narrow, 'xgb', 'colsample_bytree')
    assert all('wide' in s.shaped_by for s in _of(wide, 'logreg') + _of(wide, 'xgb'))


def test_progenitors_that_look_halfway_between_classes_call_for_smoother_models():
    ambiguous = design_panel(dict(BASE, query_ambiguity=0.7))

    assert _strongest(ambiguous, 'logreg', 'C') < _strongest(design_panel(BASE), 'logreg', 'C')
    logreg = sorted(_of(ambiguous, 'logreg'), key=lambda s: s.config['model_params']['C'])
    assert 'ambiguous_progenitors' in logreg[0].favoured_by


def test_validation_that_favours_the_edge_of_a_range_extends_the_range_one_step_that_way(tmp_path):
    panel = design_panel(BASE)
    c_values = sorted(s.config['model_params']['C'] for s in _of(panel, 'logreg'))
    losses = {s.unit_id: 1.0 for s in panel}
    # Logistic regression: the weakest regularisation validates best.
    losses[next(s.unit_id for s in _of(panel, 'logreg') if s.config['model_params']['C'] == c_values[-1])] = 0.1
    # XGBoost: the deepest variant; MLP: the smallest; ResNet: the middle one, so no edge to follow.
    losses['xgb_flexible'] = 0.1
    losses['mlp_compact'] = 0.1
    losses['resnet_standard'] = 0.1

    added = extend_panel(panel, losses)

    by_type = {s.type: s for s in added}
    assert set(by_type) == {'logreg', 'xgb', 'mlp'}
    assert by_type['logreg'].config['model_params']['C'] == 10 * c_values[-1]
    flexible = next(s for s in _of(panel, 'xgb') if s.name == 'flexible')
    assert by_type['xgb'].config['model_params']['max_depth'] > flexible.config['model_params']['max_depth']
    assert (by_type['xgb'].config['train_config']['num_boost_round']
            > flexible.config['train_config']['num_boost_round'])
    compact = next(s for s in _of(panel, 'mlp') if s.name == 'compact')
    assert _width(by_type['mlp']) < _width(compact)
    # New units carry no rule's expectation, only that validation asked for them.
    assert all(s.shaped_by == ['validation_edge'] and not s.favoured_by and not s.disfavoured_by for s in added)
    # They load next to the panel without clashing with its units.
    hangar = Hangar(write_panel(panel + added, tmp_path / 'hangar'))
    assert len(hangar.units) == len(panel) + len(added)


def test_svcs_are_left_out_where_one_fit_would_take_hours():
    # A linear SVC with probability calibration on 6,500 cells x 50,000 peaks
    # took about 100 minutes per fit, and routing refits every unit per fold.
    huge = design_panel(dict(BASE, n_terminal=6500, n_features=50000))

    assert 'svc' not in {s.type for s in huge}
    assert 'svc' in {s.type for s in design_panel(BASE)}
