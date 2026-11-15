"""A2/A3/A11 and squad isolation.

* A2 (supports R1.2): the final retention gate uses out-of-fold scores, not
  the refitted models' scores on their own training data.
* A11 (supports R1.5): relative retention and a never-empty fallback.
* A3 (supports R1.5): explanation weights are normalised performance
  weights, so ``A_c`` is a weighted mean.
* Squad isolation: separate runs from one hangar must not share models
  (needed for multi-seed experiments, R2.5).
"""
import numpy as np
import pytest

from ageas import Hangar, n_iter_extraction, n_kfold_selection


def _select(hangar_dir, corpus, **kwargs):
    args = dict(hangar=Hangar(hangar_dir), query_dataset=corpus,
                kfold_selection_list=[3], seed=0, n_dataloader_workers=0,
                retention_point=0.0, cutoff_point=0.0, selection_ratio=1.0)
    args.update(kwargs)
    return n_kfold_selection(**args)


def test_oof_scores_are_recorded(tiny_hangar_dir, fake_corpus):
    deck = _select(tiny_hangar_dir, fake_corpus)
    for unit in deck.squad.values():
        final = unit.report["trail"]["final"]
        assert 0.0 <= final["oof"]["oof.accuracy"] <= 1.0
        assert "round_1" in unit.report["trail"]


def test_final_gate_is_out_of_fold(tiny_hangar_dir, fake_corpus):
    deck = _select(tiny_hangar_dir, fake_corpus, retention_point=0.75)
    for unit in deck.squad.values():
        assert unit.report["trail"]["final"]["oof"]["oof.accuracy"] >= 0.75


def test_fallback_keeps_best_unit(tiny_hangar_dir, fake_corpus):
    with pytest.warns(UserWarning):
        deck = _select(tiny_hangar_dir, fake_corpus, retention_point=1.01)
    assert len(deck.squad) == 1


def test_relative_retention(tiny_hangar_dir, fake_corpus):
    deck = _select(tiny_hangar_dir, fake_corpus, retention_point=1.01,
                   cutoff_point=0.0, selection_ratio=0.0, relative_retention=0.5)
    assert len(deck.squad) >= 1
    scores = [u.report["trail"]["final"]["oof"]["oof.accuracy"] for u in deck.squad.values()]
    assert min(scores) >= max(scores) - 0.5


def test_invalid_gate_rejected(tiny_hangar_dir, fake_corpus):
    with pytest.raises(ValueError):
        _select(tiny_hangar_dir, fake_corpus, final_gate="test")


def test_debrief_weights_are_normalised(tiny_hangar_dir, fake_corpus):
    deck = _select(tiny_hangar_dir, fake_corpus)
    weights = deck.unit_weights()
    assert pytest.approx(sum(weights.values())) == 1.0

    exp = deck.debrief(exp_dataset=fake_corpus, verbose=False)
    single = {uid: unit.model.explain(dataset=fake_corpus) for uid, unit in deck.squad.items()}
    expected = sum(weights[uid] * single[uid] for uid in single)
    expected = expected.drop(columns=[c for c in expected.columns if "Std" in c])
    np.testing.assert_allclose(exp.values, expected.loc[exp.index, exp.columns].values)


def test_runs_do_not_share_models(tiny_hangar_dir, fake_corpus):
    hangar = Hangar(tiny_hangar_dir)
    deck_a = n_kfold_selection(hangar=hangar, query_dataset=fake_corpus, kfold_selection_list=[2],
                               seed=0, n_dataloader_workers=0, retention_point=0.0, cutoff_point=0.0)
    deck_b = n_kfold_selection(hangar=hangar, query_dataset=fake_corpus, kfold_selection_list=[2],
                               seed=1, n_dataloader_workers=0, retention_point=0.0, cutoff_point=0.0)
    for unit_id in set(deck_a.squad) & set(deck_b.squad):
        assert deck_a.squad[unit_id] is not deck_b.squad[unit_id]
        assert deck_a.squad[unit_id].model is not deck_b.squad[unit_id].model
    assert all(unit.model is None for unit in hangar.units.values())


def test_predict_returns_members(tiny_hangar_dir, fake_corpus):
    deck = _select(tiny_hangar_dir, fake_corpus)
    mean, labels, members = deck.predict(fake_corpus, num_workers=0, return_members=True)
    assert set(members) == set(deck.squad)
    np.testing.assert_allclose(mean, np.mean(np.stack(list(members.values())), axis=0))
    assert len(labels) == len(mean)


def test_extraction_history_and_normalisation(tiny_hangar_dir, fake_corpus):
    exp_corpus = fake_corpus.copy()
    n_vars = exp_corpus.adata.n_vars
    top, final_exp, history = n_iter_extraction(
        hangar=Hangar(tiny_hangar_dir), query_dataset=fake_corpus, exp_dataset=exp_corpus,
        max_extraction_iter=2, outlier_iqr_factor=1.5, extract_top_n=3,
        use_gene_names=False, return_history=True, seed=0,
        selection_args=dict(kfold_selection_list=[2], n_dataloader_workers=0,
                            retention_point=0.0, cutoff_point=0.0),
    )
    assert exp_corpus.adata.n_vars == n_vars          # caller's dataset untouched
    assert 1 <= len(history) <= 2
    for record in history:
        np.testing.assert_allclose(record["explanation"].abs().sum().values, 1.0)
    np.testing.assert_allclose(final_exp.abs().sum().values, 1.0)
    assert "Outlier_Iter" in top.columns
