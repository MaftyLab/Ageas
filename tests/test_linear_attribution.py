"""A4: linear-model attributions must point toward the class they favour.

Supports reviewer comments R2.9 (attribution reliability) and R1.3 (feature
tiers); before the fix, binary logistic-regression/SVM attributions were
assigned to the wrong class and could cancel the IG/SHAP attributions.
"""
import numpy as np
import pytest

from ageas.classical import LogReg_Classifier, MNB_Classifier, SVM_Classifier
from conftest import LOGREG_PARAMS, SVC_PARAMS, marker_data


def _fit(cls, params, n_classes, x, y, names):
    model = cls(fea_names=names, model_params=dict(params, num_class=n_classes))
    model(x, y)
    return model


@pytest.mark.parametrize("cls, params", [
    (LogReg_Classifier, LOGREG_PARAMS),
    (SVM_Classifier, SVC_PARAMS),
])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_marker_ranks_first_for_its_class(cls, params, n_classes):
    x, y, names = marker_data(n_classes=n_classes)
    exp = _fit(cls, params, n_classes, x, y, names).explain()

    for c in range(n_classes):
        col = exp[f"Class_{c}_Scores"]
        assert col.idxmax() == f"marker_{c}"
        assert col[f"marker_{c}"] > 0


def test_binary_scores_are_opposite():
    x, y, names = marker_data(n_classes=2)
    exp = _fit(LogReg_Classifier, LOGREG_PARAMS, 2, x, y, names).explain()
    np.testing.assert_allclose(exp["Class_0_Scores"], -exp["Class_1_Scores"])
    assert exp.loc["marker_1", "Class_0_Scores"] < 0


def test_mnb_marker_ranks_first():
    x, y, names = marker_data(n_classes=3)
    x = x - x.min()  # MultinomialNB needs non-negative counts
    model = MNB_Classifier(fea_names=names, model_params={
        "alpha": 1.0, "force_alpha": True, "fit_prior": True,
        "class_prior": None, "num_class": 3,
    })
    model(x, y)
    exp = model.explain()
    for c in range(3):
        assert exp[f"Class_{c}_Scores"].idxmax() == f"marker_{c}"
