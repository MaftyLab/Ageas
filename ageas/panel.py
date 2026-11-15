#!/usr/bin/env python3
"""Panel design from a dataset's label-free profile.

The rules here are stated before any run and never chosen by progenitor
scores. A profile (:func:`~ageas.profile_dataset`) decides which model
families enter the panel and how much capacity and regularisation each
gets; every unit records the rules that shaped it, so that validation can
check the strategy afterwards (:func:`~ageas.check_strategy`).
"""
import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np

#: Family of each unit type.
FAMILIES = {
    'logreg': 'linear', 'svc': 'linear', 'mnb': 'bayes', 'xgb': 'tree',
    'mlp': 'nn', 'rnn': 'nn', 'resnet': 'nn',
}

_LOGREG = {
    'penalty': 'l2', 'dual': False, 'tol': 1e-4, 'C': 1.0, 'fit_intercept': True,
    'intercept_scaling': 1.0, 'class_weight': None, 'random_state': None, 'solver': 'lbfgs',
}
_SVC = {
    'scaler_copy': True, 'scaler_with_mean': True, 'scaler_with_std': True, 'C': 1.0,
    'kernel': 'linear', 'degree': 3, 'gamma': 'scale', 'coef0': 0.0, 'shrinking': True, 'tol': 1e-3,
    'cache_size': 200, 'class_weight': None, 'verbose': False, 'max_iter': -1,
    'decision_func_shape': 'ovr', 'break_ties': False, 'random_state': None,
}
_MNB = {'alpha': 1.0, 'force_alpha': True, 'fit_prior': True, 'class_prior': None}
_NN_TRAIN = {
    'batch_size': 64, 'precision': 32, 'num_nodes': 1, 'log_every_n_steps': 1,
    'accumulate_grad_batches': 1, 'gradient_clip_val': 0.5, 'gradient_clip_algorithm': 'value',
    'ckpt_every_n_epochs': 1, 'save_last': True, 'monitor': 'vali.accuracy', 'save_top_k_ckpt': 1,
    'gradient_accum_schedule': None, 'loss_reduction': 'mean', 'optimizer': 'adamw',
    'learning_rate': 0.01, 'weight_decay': 1e-3, 'betas': [0.9, 0.999], 'momentum': 0.9,
    'scheduler': 'cosine', 'sch_T_0': 10, 'sch_T_mult': 2, 'sch_eta_min': 0.001,
}
_MLP = {
    'block': 'Residual_Encoder', 'block_nums': [2], 'block_dims': [32], 'bias': True,
    'dropout': 0.1, 'norm_layer': 'LayerNorm', 'latent_fea_dim': 64,
}
_RESNET = {
    'latent_planes': 64, 'block': 'ResNet_Basic', 'block_nums': [1, 1], 'block_planes': [32, 8],
    'dropout': 0.1, 'zero_init_residual': False, 'groups': 1, 'dilation': 1, 'width_per_group': 64,
    'replace_stride_with_dilation': False, 'norm_layer': 'BatchNorm1d', 'latent_fea_dim': 10,
    'global_conv': True,
}
_RNN = {
    'block': 'RNN_Residual_Encoder', 'block_layer_type': 'LSTM', 'block_num_layer': 1,
    'block_nums': [1], 'block_dims': [16], 'nonlinearity': 'tanh', 'bias': True, 'dropout': 0.0,
    'bidirectional': True, 'proj_size': 0, 'norm_layer': 'LayerNorm', 'latent_fea_dim': 32,
}


@dataclass
class UnitSpec:
    """One generated unit of the panel.

    :ivar name: Unit name within its type (the config file's stem).
    :ivar type: Ageas model type (``'logreg'``, ``'xgb'``, ...).
    :ivar config: The Ageas unit configuration.
    :ivar shaped_by: Rules that set this unit's settings.
    :ivar favoured_by: Rules that expect this unit to validate well.
    :ivar disfavoured_by: Rules that expect this unit to validate poorly.
    :ivar axis: The ordered setting this unit varies with others of its
        family (``'logreg'``: C; ``'mnb'``: smoothing; ``'xgb'``: depth and
        rounds; ``'mlp'``, ``'resnet'``: size), or ``None``.
    :ivar position: The unit's place on its axis (log10 C, log10 alpha,
        boosting step or network size).
    """
    name: str
    type: str
    config: dict
    shaped_by: list = field(default_factory=list)
    favoured_by: list = field(default_factory=list)
    disfavoured_by: list = field(default_factory=list)
    axis: str = None
    position: float = None

    @property
    def unit_id(self) -> str:
        """The unit's ID in a hangar written by :func:`write_panel`."""
        return f'{self.type}_{self.name}'

    @property
    def family(self) -> str:
        return FAMILIES[self.type]


def _nn(model_params: dict) -> dict:
    return {'max_epochs': 5, 'model_params': model_params, 'train_config': copy.deepcopy(_NN_TRAIN)}


def _number(value: float) -> str:
    """``value`` for a unit name: Ageas cuts IDs at the first dot, so ``0.1`` becomes ``0p1``."""
    return f'{value:g}'.replace('.', 'p')


#: What each rule expects: the unit roles it favours and those it disfavours.
#: Roles: ``regularised``/``unregularised`` (the ends of the logistic
#: regression's C grid), ``simple``/``flexible`` (the smaller and larger
#: variant of trees and networks).
RULES = {
    'far_progenitors': (('regularised', 'simple'), ('unregularised', 'flexible')),
    'near_progenitors': (('unregularised', 'flexible'), ('regularised', 'simple')),
    'few_cells': (('simple',), ('flexible',)),
    'many_cells': (('flexible',), ('simple',)),
    'curved_boundary': (('flexible',), ('linear',)),
    'straight_boundary': (('linear',), ('flexible',)),
    'counts': (('bayes',), ()),
    'wide': ((), ()),
    'ambiguous_progenitors': (('regularised',), ('unregularised',)),
}
#: Rules that move the regularisation level, and those that move capacity.
REGULARISATION_RULES = ('far_progenitors', 'near_progenitors', 'ambiguous_progenitors')
CAPACITY_RULES = (
    'far_progenitors', 'near_progenitors', 'few_cells', 'many_cells', 'curved_boundary', 'straight_boundary',
)
#: Optimizer steps a network should take in one fit, whatever the cell count.
STEPS_PER_FIT = 500
#: Largest cells x features for a linear SVC: its probability calibration
#: took about 100 minutes per fit on 6,500 cells x 50,000 peaks.
SVC_MAX_SIZE = 1e8

#: Inverse regularisation of the logistic regressions at each level (light, medium, strong).
_C_GRID = {0: (1.0, 10.0, 100.0), 1: (0.1, 1.0, 10.0), 2: (0.01, 0.1, 1.0)}
_DROPOUT = {0: 0.05, 1: 0.1, 2: 0.3}
_WEIGHT_DECAY = {0: 1e-4, 1: 1e-3, 2: 1e-2}
#: Network sizes. The panel spans one size either side of the capacity level
#: (0 to 2), so -1 to 3; -2 and 4 are only reached when validation asks for
#: more than the rules gave (:func:`extend_panel`).
_MLP_SIZE = {
    -2: ([1], [4], 8), -1: ([1], [8], 16), 0: ([1], [16], 32), 1: ([2], [32], 64),
    2: ([2, 2], [64, 32], 128), 3: ([3, 3], [128, 64], 256), 4: ([4, 4], [256, 128], 512),
}
_RESNET_SIZE = {
    -2: ([1], [4], 4), -1: ([1], [4], 8), 0: ([1], [8], 16), 1: ([1, 1], [32, 8], 64),
    2: ([2, 2], [64, 16], 128), 3: ([3, 3], [128, 32], 256), 4: ([4, 4], [256, 64], 512),
}
_RNN_SIZE = {0: ([1], [16]), 1: ([2], [16]), 2: ([2, 2], [16, 8])}
#: Light networks on R1's recipe (its ``mlp/resEnc_1`` and ``rnn/gru_1``, ``gru_2``):
#: a wide latent layer, one epoch at batch 200. They ranked progenitors best of
#: R1's units on LARRY and iEP (refinement round 1, 2026-11-08), and every
#: panel carries them, whatever its rules.
_LIGHT_TRAIN = {'batch_size': 200, 'learning_rate': 0.01, 'weight_decay': 1e-3}
_LIGHT = {
    ('mlp', 'light'): dict(_MLP, block_nums=[2], block_dims=[16], dropout=0.1, latent_fea_dim=128),
    ('rnn', 'gru_light'): dict(_RNN, block_layer_type='GRU', block_nums=[2], block_dims=[16], latent_fea_dim=128),
    ('rnn', 'gru_light_small'): dict(_RNN, block_layer_type='GRU', block_nums=[2], block_dims=[16],
                                     latent_fea_dim=64),
}
#: Boosting steps: step s grows trees of depth 1 + s for 50 * 2**s rounds.
_XGB_STEPS = range(0, 6)


def _xgb_step(step: int) -> tuple:
    """Tree depth and boosting rounds at one step of the XGBoost axis."""
    return 1 + step, 50 * 2 ** step


def active_rules(profile: dict) -> list:
    """The panel rules a profile triggers.

    * ``far_progenitors``: the progenitors sit apart from the mature cells
      (neighbourhood AUC at least 0.9, under half of them within reach of a
      terminal cell, or half in clusters of their own). Models fitted to
      the terminal cells must extrapolate, which simpler, more regularised
      models do more conservatively.
    * ``near_progenitors``: the progenitors sit among the mature cells (AUC
      under 0.75, at least 80% within reach, under a fifth in clusters of
      their own), so what fits the terminal cells is likely to carry over.
    * ``few_cells`` (under 1,000 terminal cells) and ``many_cells`` (5,000
      or more): capacity has to match the data a model learns from.
    * ``curved_boundary``: a nearest-neighbour classifier beats a linear
      model on the terminal cells by at least 0.03, or the linear model's
      accuracy is under 0.9, so the classes need a flexible boundary.
    * ``straight_boundary``: the linear model reaches 0.97 and a flexible
      one adds under 0.01, so capacity beyond a line buys nothing.
    * ``counts``: non-negative whole numbers, the data multinomial naive
      Bayes models; its smoothing is varied.
    * ``wide``: more than twice as many features as terminal cells. The
      logistic regressions get an L1 penalty, which keeps few features,
      and the trees sample features.
    * ``ambiguous_progenitors``: a linear model of the mature classes is
      unsure about the progenitors (mean scaled entropy at least 0.5).
      Their calls rest on probabilities near the boundary, which smoother,
      more regularised models estimate more reliably.
    """
    p = profile
    tests = {
        'far_progenitors': p['domain_auc'] >= 0.9 or p['coverage'] < 0.5 or p['query_own_clusters'] >= 0.5,
        'near_progenitors': p['domain_auc'] < 0.75 and p['coverage'] >= 0.8 and p['query_own_clusters'] < 0.2,
        'few_cells': p['n_terminal'] < 1000,
        'many_cells': p['n_terminal'] >= 5000,
        'curved_boundary': p['terminal_nonlinear_gain'] >= 0.03 or p['terminal_linear_accuracy'] < 0.9,
        'straight_boundary': p['terminal_linear_accuracy'] >= 0.97 and p['terminal_nonlinear_gain'] < 0.01,
        'counts': bool(p['nonnegative']) and p['integer_share'] >= 0.99,
        'wide': p['n_features'] > 2 * p['n_terminal'],
        'ambiguous_progenitors': p['query_ambiguity'] >= 0.5,
    }
    return [name for name, hit in tests.items() if hit]


def _levels(rules: list) -> tuple:
    """Regularisation and capacity levels, each 0 (light/small) to 2 (strong/large)."""
    on = set(rules)
    regularisation = (
        1 + ('far_progenitors' in on) - ('near_progenitors' in on) + ('ambiguous_progenitors' in on)
    )
    capacity = (
        1 - ('few_cells' in on) + ('many_cells' in on)
        - ('far_progenitors' in on) + ('near_progenitors' in on)
        + ('curved_boundary' in on) - ('straight_boundary' in on)
    )
    return min(max(regularisation, 0), 2), min(max(capacity, 0), 2)


def _schedule(n_terminal: int) -> tuple:
    """Batch size and epochs that give a network about :data:`STEPS_PER_FIT` steps per fit."""
    batch = 32 if n_terminal < 1000 else 64 if n_terminal < 5000 else 128
    epochs = int(np.clip(np.ceil(STEPS_PER_FIT * batch / n_terminal), 3, 30))
    return batch, epochs


def _tag(spec: UnitSpec, roles: tuple, rules: list, shaping: tuple) -> UnitSpec:
    """Record which active rules (of those in ``shaping``) set ``spec`` and what they expect of it."""
    spec.shaped_by = [r for r in rules if r in shaping]
    spec.favoured_by = [r for r in rules if set(RULES[r][0]) & set(roles)]
    spec.disfavoured_by = [r for r in rules if set(RULES[r][1]) & set(roles)]
    return spec


def _mlp_params(size: int) -> dict:
    blocks, dims, latent = _MLP_SIZE[size]
    return dict(_MLP, block_nums=blocks, block_dims=dims, latent_fea_dim=latent)


def _resnet_params(size: int) -> dict:
    blocks, planes, latent = _RESNET_SIZE[size]
    return dict(_RESNET, block_nums=blocks, block_planes=planes, latent_planes=latent)


def _nn_config(model_params: dict, regularisation: int, n_terminal: int) -> dict:
    config = _nn(model_params)
    config['model_params']['dropout'] = _DROPOUT[regularisation]
    config['train_config']['weight_decay'] = _WEIGHT_DECAY[regularisation]
    config['train_config']['batch_size'], config['max_epochs'] = _schedule(n_terminal)
    return config


def design_panel(profile: dict, use_rules: bool = True) -> list:
    """The units a dataset's profile calls for.

    :param profile: Output of :func:`~ageas.profile_dataset`.
    :param use_rules: If ``False``, no rule shapes the panel (middle
        regularisation and capacity, L2 penalties, one naive-Bayes
        smoothing): the fixed panel of the R2.4 ablation. Data-type checks
        (naive Bayes for non-negative input, the SVC size limit) and the
        networks' schedule by cell count still apply.
    :returns: :class:`UnitSpec` list, one per unit.
    """
    rules = active_rules(profile) if use_rules else []
    regularisation, capacity = _levels(rules)
    wide = 'wide' in rules
    linear_shaping = REGULARISATION_RULES + ('wide',)
    tree_shaping = REGULARISATION_RULES + CAPACITY_RULES + ('wide',)
    nn_shaping = REGULARISATION_RULES + CAPACITY_RULES
    panel = []

    grid = _C_GRID[regularisation]
    penalty = {'penalty': 'l1', 'solver': 'saga'} if wide else {'penalty': 'l2', 'solver': 'lbfgs'}
    for c in grid:
        roles = ('linear',) + (('regularised',) if c == grid[0] else ('unregularised',) if c == grid[-1] else ())
        spec = UnitSpec(f'C_{_number(c)}', 'logreg', {'max_epochs': 1, 'model_params': dict(_LOGREG, C=c, **penalty)},
                        axis='logreg', position=float(np.log10(c)))
        panel.append(_tag(spec, roles, rules, linear_shaping))
    if profile['n_terminal'] * profile['n_features'] <= SVC_MAX_SIZE:
        spec = UnitSpec(f'C_{_number(grid[1])}', 'svc', {'max_epochs': 1, 'model_params': dict(_SVC, C=grid[1])})
        panel.append(_tag(spec, ('linear',), rules, REGULARISATION_RULES))
    if profile['nonnegative']:
        alphas = (0.1, 1.0, 10.0) if 'counts' in rules else (1.0,)
        for alpha in alphas:
            spec = UnitSpec(f'alpha_{_number(alpha)}', 'mnb', {'max_epochs': 1, 'model_params': dict(_MNB, alpha=alpha)},
                            axis='mnb' if len(alphas) > 1 else None, position=float(np.log10(alpha)))
            panel.append(_tag(spec, ('bayes',), rules, ('counts',)))

    # Trees and networks get three variants: the capacity the rules chose and
    # one step either side, so validation can find its best inside the range.
    eta = 0.05 if regularisation == 2 else 0.1
    colsample = 0.3 if wide else 0.8
    for name, roles, step in (('conservative', ('simple',), capacity), ('standard', (), capacity + 1),
                              ('flexible', ('flexible',), capacity + 2)):
        depth, rounds = _xgb_step(step)
        spec = UnitSpec(name, 'xgb', {
            'max_epochs': 1,
            'model_params': {'booster': 'gbtree', 'max_depth': depth, 'eta': eta,
                             'subsample': 0.8, 'colsample_bytree': colsample},
            'train_config': {'num_boost_round': rounds, 'early_stopping_rounds': None, 'verbose_eval': False},
        }, axis='xgb', position=step)
        panel.append(_tag(spec, roles, rules, tree_shaping))

    n = profile['n_terminal']
    for name, roles, size in (('compact', ('simple',), capacity - 1), ('standard', (), capacity),
                              ('roomy', ('flexible',), capacity + 1)):
        panel.append(_tag(UnitSpec(name, 'mlp', _nn_config(_mlp_params(size), regularisation, n),
                                   axis='mlp', position=size), roles, rules, nn_shaping))
        panel.append(_tag(UnitSpec(name, 'resnet', _nn_config(_resnet_params(size), regularisation, n),
                                   axis='resnet', position=size), roles, rules, nn_shaping))
    blocks, dims = _RNN_SIZE[max(capacity - 1, 0)]
    rnn = dict(_RNN, block_nums=blocks, block_dims=dims)
    panel.append(_tag(UnitSpec('compact', 'rnn', _nn_config(rnn, regularisation, n)), ('simple',), rules, nn_shaping))
    for (unit_type, name), params in _LIGHT.items():
        config = {'max_epochs': 1, 'model_params': dict(params),
                  'train_config': dict(copy.deepcopy(_NN_TRAIN), **_LIGHT_TRAIN)}
        panel.append(UnitSpec(name, unit_type, config))
    return panel


def _beyond(spec: UnitSpec, position) -> UnitSpec:
    """A copy of ``spec`` moved to ``position`` on its axis, or ``None`` past the axis's end."""
    config = copy.deepcopy(spec.config)
    params = config['model_params']
    if spec.axis == 'logreg':
        params['C'] = float(10.0 ** position)
        name = f'C_{_number(params["C"])}'
    elif spec.axis == 'mnb':
        params['alpha'] = float(10.0 ** position)
        name = f'alpha_{_number(params["alpha"])}'
    elif spec.axis == 'xgb':
        if position not in _XGB_STEPS:
            return None
        params['max_depth'], config['train_config']['num_boost_round'] = _xgb_step(position)
        name = f'step_{position}'
    elif spec.axis in ('mlp', 'resnet'):
        sizes = _MLP_SIZE if spec.axis == 'mlp' else _RESNET_SIZE
        if position not in sizes:
            return None
        fresh = _mlp_params(position) if spec.axis == 'mlp' else _resnet_params(position)
        config['model_params'] = dict(fresh, dropout=params['dropout'])
        name = f'size_{_number(position)}'
    else:
        return None
    return UnitSpec(name, spec.type, config, shaped_by=['validation_edge'], axis=spec.axis, position=position)


def extend_panel(panel: list, losses: dict) -> list:
    """Units one step past the edge of a range where validation prefers that edge.

    The rules choose each family's range (logistic regression's C, naive
    Bayes smoothing, XGBoost depth and rounds, network size). When the
    family's unit with the lowest validation loss sits at an end of that
    range, the range may stop short of what the data want, so one more unit
    is offered past that end; it records only that validation asked for it.

    :param panel: The designed :class:`UnitSpec` list.
    :param losses: ``unit_id -> validation loss`` of the panel's units.
    :returns: The new :class:`UnitSpec` list (empty when no family ends at an edge).
    """
    added = []
    for axis in sorted({s.axis for s in panel if s.axis is not None}):
        members = [s for s in panel if s.axis == axis and s.unit_id in losses]
        if len(members) < 2:
            continue
        positions = sorted(s.position for s in members)
        step = 1 if axis in ('logreg', 'mnb', 'xgb', 'mlp', 'resnet') else None
        best = min(members, key=lambda s: losses[s.unit_id])
        if best.position == positions[-1]:
            spec = _beyond(best, best.position + step)
        elif best.position == positions[0]:
            spec = _beyond(best, best.position - step)
        else:
            spec = None
        if spec is not None and spec.unit_id not in {s.unit_id for s in panel}:
            added.append(spec)
    return added


def write_panel(panel: list, folder) -> str:
    """Write ``panel`` as an Ageas config folder; returns its path."""
    for spec in panel:
        os.makedirs(os.path.join(folder, spec.type), exist_ok=True)
        with open(os.path.join(folder, spec.type, f'{spec.name}.json'), 'w', encoding='utf-8') as fh:
            json.dump(spec.config, fh, indent=1)
    return str(folder)
