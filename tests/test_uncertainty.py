"""E6: per-cell uncertainty / OOD outputs (supports R2.8, R3.6, R2.7)."""
import numpy as np
import pandas as pd

from ageas.tool.uncertainty import (
    call_fate_states,
    decompose_uncertainty,
    ood_scores,
    program_scores,
    program_thresholds,
)


def test_agreeing_members_have_no_epistemic_uncertainty():
    p = np.array([[0.5, 0.5], [0.9, 0.1]])
    members = np.stack([p, p, p])
    u = decompose_uncertainty(members)
    np.testing.assert_allclose(u["epistemic"], 0.0, atol=1e-9)
    assert u.loc[0, "total_entropy"] > u.loc[1, "total_entropy"]


def test_disagreeing_members_raise_epistemic_uncertainty():
    members = np.array([[[0.99, 0.01]], [[0.01, 0.99]]])
    u = decompose_uncertainty({"a": members[0], "b": members[1]})
    assert u.loc[0, "epistemic"] > 0.5
    assert u.loc[0, "aleatoric"] < 0.1


def test_far_query_is_ood():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 20))
    query = np.vstack([rng.normal(size=(5, 20)), rng.normal(loc=8.0, size=(5, 20))])
    ood = ood_scores(reference, query, n_neighbors=5, n_components=5)
    assert not ood["is_ood"].iloc[:5].any()
    assert ood["is_ood"].iloc[5:].all()


def test_state_calls():
    probs = pd.DataFrame({"A": [0.95, 0.5, 0.5, 0.5, 0.5], "B": [0.05, 0.5, 0.5, 0.5, 0.5]})
    unc = pd.DataFrame({"epistemic": [0.0, 0.0, 0.0, 0.0, 0.3]})
    ood = pd.DataFrame({"is_ood": [False, False, False, True, False]})
    programs = pd.DataFrame({"A": [2.0, 2.0, -2.0, 0.0, 0.0], "B": [-2.0, 2.0, -2.0, 0.0, 0.0]})
    thresholds = {"A": (-1.0, 1.0), "B": (-1.0, 1.0)}
    states = call_fate_states(probs, unc, ood, programs, thresholds).astype(str).tolist()
    assert states == ["confident:A", "hybrid", "unprimed", "ood", "model_disagreement"]


def test_program_thresholds_from_terminal_cells():
    matrix = np.array([[5.0, 0.0], [6.0, 1.0], [0.0, 5.0], [1.0, 6.0]])
    scores = program_scores(matrix, ["gA", "gB"], {"A": ["gA"], "B": ["gB"]})
    th = program_thresholds(scores, ["A", "A", "B", "B"])
    assert th["A"][1] > th["A"][0]
    assert th["B"][1] > th["B"][0]
