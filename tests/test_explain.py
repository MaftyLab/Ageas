"""NN attributions come from the right cells.

Integrated Gradients for class ``c`` must run on cells of class ``c`` that
the model predicts correctly, so reordering the cells cannot change the
result. R1's ``Basic_Clf_Explainer.stratify_dataset`` indexes the full
dataset with positions taken from the class subset, so which cells it
explains depends on their order.
"""
import numpy as np
import torch

from ageas.nn import NN_Classifier
from ageas.tool import Trainer_Maker

from ageas import explain_unit
from ageas.tool.feature_corpus import Feature_Corpus
from conftest import MLP_PARAMS, MLP_TRAIN_CONFIG


def _marker_data(n_per_class=60, n_genes=12, seed=0):
    """Class-1 cells first, then class 0; gene ``marker_c`` is raised in class ``c``."""
    rng = np.random.default_rng(seed)
    y = np.repeat([1, 0], n_per_class)
    x = rng.normal(0.0, 1.0, size=(len(y), 1, n_genes))
    for c in (0, 1):
        x[y == c, 0, c] += 4.0
    names = ["marker_0", "marker_1"] + [f"noise_{i}" for i in range(2, n_genes)]
    return x, y, names


def _corpus(x, y, names):
    return Feature_Corpus(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y), names,
        label_dict={0: "a", 1: "b"},
    )


def test_attributions_do_not_depend_on_cell_order(tmp_path):
    x, y, names = _marker_data()
    corpus = _corpus(x, y, names)
    order = np.argsort(y, kind="stable")
    reordered = _corpus(x[order], y[order], names)

    torch.manual_seed(0)
    model, _ = Trainer_Maker().make(
        model_type="mlp", clf_object=NN_Classifier, device=1, accelerator="cpu",
        train_data=corpus, vali_data=corpus, n_classes=2,
        max_epochs=5, dataloader_workers=0, default_root_dir=str(tmp_path),
        model_params=dict(MLP_PARAMS), train_config=dict(MLP_TRAIN_CONFIG),
    )
    table = explain_unit(model, "mlp", corpus, device="cpu", exp_step=16)
    again = explain_unit(model, "mlp", reordered, device="cpu", exp_step=16)

    assert list(table.columns) == ["Class_0_Scores", "Class_1_Scores"]
    np.testing.assert_allclose(table.values, again.loc[table.index].values, atol=1e-6)

    # In a binary problem one marker is enough, and which one the network
    # leans on depends on its initialisation; the other class is then
    # "marker absent", which a zero-baseline IG cannot credit. So only the
    # strongest attribution is pinned: a marker, pushing towards its class.
    top = table.abs().max(axis=1).idxmax()
    assert top in ("marker_0", "marker_1")
    assert table.loc[top, f"Class_{top[-1]}_Scores"] > 0
