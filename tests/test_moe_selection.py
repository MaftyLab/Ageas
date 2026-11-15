"""moe_selection end to end on synthetic data.

Every unit is routed inside each fold (train on the fold's training part,
rank its features, judge the pruning ladder on the fold's validation split)
and predicts the fold's held-out cells with the version it chose. The
no-harm step then integrates units on those out-of-fold predictions.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from ageas import Hangar
from ageas.tool import kfold_random_split
from ageas.tool.corpus_loader import get_all_data

from ageas import moe_selection, no_harm_selection
from conftest import AGEAS_ROOT, CLASSICAL_UNITS, LOGREG_PARAMS, MLP_UNITS, make_fake_corpus, write_hangar

RUN = dict(
    kfold=3, valid_fraction=0.25, mass_ladder=[0.9, 0.6], floor=1, relative_delta=0.5,
    min_relative_gain=0.01, exp_sample_limit=20, explain_args={"exp_step": 8}, seed=0,
)


@pytest.fixture(scope="module")
def classical_hangar(tmp_path_factory):
    return write_hangar(tmp_path_factory.mktemp("hangar"), CLASSICAL_UNITS)


@pytest.fixture(scope="module")
def classical_run(classical_hangar, tmp_path_factory):
    corpus = make_fake_corpus()
    deck = moe_selection(
        hangar=Hangar(classical_hangar), query_dataset=corpus,
        cache_dir=str(tmp_path_factory.mktemp("cache")), **RUN,
    )
    return deck, corpus


def test_oof_record_covers_every_terminal_cell_once(classical_run):
    deck, corpus = classical_run
    n_cells = corpus.adata.n_obs
    test_cells = np.concatenate([fold["test"] for fold in deck.oof["folds"]])
    assert np.array_equal(np.sort(test_cells), np.arange(n_cells))
    for unit_id, probs in deck.oof["probs"].items():
        assert probs.shape == (n_cells, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert len(deck.oof["levels"][unit_id]) == RUN["kfold"]
    # The ladder narrowed at least one unit in at least one fold.
    assert any(level is not None for levels in deck.oof["levels"].values() for level in levels)


def test_run_records_which_ageas_it_imported(classical_run):
    # The WSL env also holds a pre-revision install, so every run says which
    # copy it used and the git state of that checkout.
    deck, _ = classical_run
    record = deck.provenance
    assert os.path.samefile(record["ageas_path"], os.path.join(AGEAS_ROOT, "ageas"))
    assert len(record["ageas_commit"]) == 40
    assert isinstance(record["ageas_dirty"], bool)


def test_folds_are_the_ones_r1_uses_for_the_same_seed(classical_hangar, tmp_path):
    # Stratified, as in the frozen protocol, so v2 and R1 runs stay paired.
    corpus = make_fake_corpus()
    deck = moe_selection(
        hangar=Hangar(classical_hangar), query_dataset=corpus, cache_dir=str(tmp_path),
        using_model_list=["logreg_lr_strong"],
        stratified_kfold_test=True, stratified_kfold_valid=True, **RUN,
    )
    r1 = kfold_random_split(
        corpus, n_splits=RUN["kfold"], valid_fraction=RUN["valid_fraction"],
        stratified_test=True, stratified_valid=True, random_seed=RUN["seed"],
    )
    x, _ = get_all_data(corpus)
    for k, fold in enumerate(deck.oof["folds"]):
        for part, r1_parts in zip(("train", "valid", "test"), r1):
            assert torch.equal(x[fold[part]], r1_parts[k].data)


def test_no_cell_label_reaches_its_own_oof_prediction(classical_hangar, classical_run, tmp_path):
    deck, _ = classical_run
    cells = deck.oof["folds"][0]["test"]
    flipped = make_fake_corpus()
    obs = flipped.adata.obs
    labels = obs["celltype"].astype(int).to_numpy()
    labels[cells] = 1 - labels[cells]
    obs["celltype"] = pd.Categorical(labels, categories=obs["celltype"].cat.categories)

    deck_flipped = moe_selection(
        hangar=Hangar(classical_hangar), query_dataset=flipped, cache_dir=str(tmp_path), **RUN,
    )
    others = np.setdiff1d(np.arange(len(labels)), cells)
    # The flip reaches the folds that trained on those labels ...
    assert any(
        not np.allclose(deck_flipped.oof["probs"][unit_id][others], probs[others])
        for unit_id, probs in deck.oof["probs"].items()
    )
    # ... but never the flipped cells' own out-of-fold predictions.
    for unit_id, probs in deck.oof["probs"].items():
        np.testing.assert_allclose(deck_flipped.oof["probs"][unit_id][cells], probs[cells])


def test_ordinary_validation_is_validation_with_every_cell_weighted_alike(classical_hangar, classical_run, tmp_path):
    deck, corpus = classical_run
    uniform = moe_selection(
        hangar=Hangar(classical_hangar), query_dataset=make_fake_corpus(), cache_dir=str(tmp_path),
        cell_weights=np.ones(corpus.adata.n_obs), **RUN,
    )
    assert uniform.weights == pytest.approx(deck.weights)
    assert uniform.oof["levels"] == deck.oof["levels"]


def test_shift_aware_weights_decide_which_units_are_integrated(classical_hangar, tmp_path):
    # Twenty cells carry the wrong label, and they are the ones most like the
    # progenitors. The sharp units call them confidently wrong; the nearly
    # flat logistic regression hedges. Counting every cell alike the sharp
    # units win; weighting those twenty cells, the hedging unit does.
    corpus = make_fake_corpus()
    obs = corpus.adata.obs
    labels = obs["celltype"].astype(int).to_numpy()
    odd = np.random.default_rng(0).choice(len(labels), size=20, replace=False)
    labels[odd] = 1 - labels[odd]
    obs["celltype"] = pd.Categorical(labels, categories=obs["celltype"].cat.categories)
    weights = np.ones(len(labels))
    weights[odd] = 10.0

    ordinary = moe_selection(hangar=Hangar(classical_hangar), query_dataset=corpus,
                             cache_dir=str(tmp_path / "o"), **RUN)
    shifted = moe_selection(hangar=Hangar(classical_hangar), query_dataset=corpus,
                            cache_dir=str(tmp_path / "s"), cell_weights=weights, **RUN)

    assert shifted.weights.get("logreg_lr_weak", 0) > ordinary.weights.get("logreg_lr_weak", 0)
    np.testing.assert_array_equal(shifted.oof["cell_weights"], weights)
    assert ordinary.oof["cell_weights"] is None


def test_units_added_after_the_first_round_are_routed_in_the_same_folds_and_can_be_integrated(
        classical_hangar, tmp_path):
    extra = Hangar(write_hangar(tmp_path / "extra", {
        ("logreg", "lr_extra"): {"max_epochs": 1, "dataloader_workers": 0,
                                 "model_params": dict(LOGREG_PARAMS, C=10.0)},
    }))
    seen = {}

    def extend(losses):
        seen.update(losses)
        return {"logreg_lr_extra": extra.units["logreg_lr_extra"]}

    deck = moe_selection(hangar=Hangar(classical_hangar), query_dataset=make_fake_corpus(),
                         cache_dir=str(tmp_path / "cache"), extend=extend, **RUN)

    # The first round's validation losses decide what to add ...
    assert set(seen) == {"logreg_lr_strong", "logreg_lr_weak", "svc_svc"}
    # ... and the added unit gets an out-of-fold record from the same folds ...
    probs = deck.oof["probs"]["logreg_lr_extra"]
    assert probs.shape == (120, 2) and not np.isnan(probs).any()
    assert len(deck.oof["levels"]["logreg_lr_extra"]) == RUN["kfold"]
    # ... and competes with the first round's units for integration.
    assert deck.weights == pytest.approx(no_harm_selection(
        deck.oof["probs"], deck.oof["labels"], min_relative_gain=RUN["min_relative_gain"]))


def test_best_ladder_choice_routes_every_unit_to_its_best_validated_version(classical_hangar, tmp_path):
    deck = moe_selection(hangar=Hangar(classical_hangar), query_dataset=make_fake_corpus(),
                         cache_dir=str(tmp_path), ladder_choice='best', **RUN)

    for unit_id, levels in deck.oof['levels'].items():
        for level, losses in zip(levels, deck.oof['losses'][unit_id]):
            chosen = losses['full' if level is None else level]
            assert chosen == min(losses.values()), (unit_id, level, losses)


def test_competent_units_are_refitted_alongside_the_integrated_ones(classical_run):
    # The committee votes with every competent unit, so each is refitted on
    # all terminal cells, like an integrated one.
    deck, _ = classical_run
    assert deck.competent
    assert set(deck.experts) == set(deck.weights) | set(deck.competent)


def test_every_routed_unit_can_be_refitted(classical_hangar, tmp_path):
    # With refit='all' each routed unit gets an expert, so any committee can
    # be drawn from the run's predictions afterwards.
    deck = moe_selection(hangar=Hangar(classical_hangar), query_dataset=make_fake_corpus(),
                         cache_dir=str(tmp_path), refit='all', **RUN)

    assert set(deck.experts) == set(deck.oof['probs']) == {'logreg_lr_strong', 'logreg_lr_weak', 'svc_svc'}


def test_experts_work_on_their_own_features_and_predict_as_a_weighted_mean(classical_run):
    deck, corpus = classical_run
    assert set(deck.weights) <= set(deck.experts)
    assert sum(deck.weights.values()) == pytest.approx(1.0)
    informative = {f for f in corpus.features if not f.startswith("fake_gene")}
    for expert in deck.experts.values():
        assert expert.level in [None] + RUN["mass_ladder"]
        assert set(expert.features) <= set(corpus.features)
        assert informative & set(expert.features)

    probs, labels, members = deck.predict(query_dataset=corpus, return_members=True)
    assert probs.shape == (corpus.adata.n_obs, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(probs, sum(deck.weights[u] * members[u] for u in deck.weights))
    assert np.array_equal(labels, deck.oof["labels"])


def test_debrief_covers_every_feature_and_is_zero_outside_the_experts(classical_run):
    deck, corpus = classical_run
    table = deck.debrief(exp_dataset=corpus)
    assert list(table.columns) == ["Class_0_Scores", "Class_1_Scores"]
    assert list(table.index) == corpus.features

    used = set().union(*(deck.experts[unit_id].features for unit_id in deck.weights))
    unused = [f for f in corpus.features if f not in used]
    assert unused
    assert (table.loc[unused] == 0).all().all()
    assert (table.loc[sorted(used)].abs().sum(axis=1) > 0).all()


def test_nn_run_writes_nothing_outside_the_cache_dir(tmp_path, monkeypatch):
    # Lightning would otherwise drop logs into the working directory, which
    # for this package is a Dropbox folder shared with collaborators.
    hangar_dir = write_hangar(tmp_path / "hangar", MLP_UNITS)
    cache = tmp_path / "cache"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    corpus = make_fake_corpus()

    deck = moe_selection(hangar=Hangar(hangar_dir), query_dataset=corpus, cache_dir=str(cache), **RUN)
    probs, _ = deck.predict(query_dataset=corpus)
    table = deck.debrief(exp_dataset=corpus, exp_step=8)

    assert set(deck.experts) == {"mlp_mlp"}
    assert probs.shape == (corpus.adata.n_obs, 2)
    assert list(table.index) == corpus.features
    assert list(workdir.iterdir()) == []
    assert any(cache.iterdir())
