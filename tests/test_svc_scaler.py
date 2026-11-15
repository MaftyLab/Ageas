"""A5: SVC predictions must not depend on how query cells are batched.

Supports R2.5 (seed/batch variance). Before the fix the StandardScaler was
refit on every prediction batch (``Deck.predict`` defaults to 10 cells).
"""
import numpy as np

from ageas.classical import SVM_Classifier
from conftest import SVC_PARAMS, marker_data


def test_prediction_is_batch_invariant():
    x, y, names = marker_data(n_classes=2)
    model = SVM_Classifier(fea_names=names, model_params=dict(SVC_PARAMS, num_class=2))
    model(x, y)

    query, _, _ = marker_data(n_classes=2, seed=7)
    whole = model.predict(query)
    batched = np.concatenate([model.predict(query[i:i + 10]) for i in range(0, len(query), 10)])
    single = np.concatenate([model.predict(query[i:i + 1]) for i in range(len(query))])

    np.testing.assert_allclose(whole, batched, atol=1e-10)
    np.testing.assert_allclose(whole, single, atol=1e-10)


def test_scaler_uses_training_statistics():
    x, y, names = marker_data(n_classes=2)
    model = SVM_Classifier(fea_names=names, model_params=dict(SVC_PARAMS, num_class=2))
    model(x, y)
    mean_before = model.scaler.mean_.copy()
    model.predict(x[:5] + 100.0)
    np.testing.assert_array_equal(model.scaler.mean_, mean_before)
