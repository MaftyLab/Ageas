"""A1/A6: neural-network attributions and the recurrent block.

* A6 (supports R2.9): Integrated Gradients must be deterministic for a fixed
  model; before the fix it ran in train mode with dropout active.
* A6 (supports R2.9): each class must be explained on its own cells that the
  model predicts correctly; before the fix, positions within a class's cells
  were applied to the whole corpus, so IG ran on other cells.
* A1 (supports R1.4): the recurrent block runs along the modality axis
  (length 1 for unimodal data), and the dense first layer imposes no
  ordering on genes.
"""
import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from ageas.nn import NN_Classifier
from ageas.tool import Basic_Clf_Explainer, Multimodal_Corpus, make_fake_adata
from conftest import MLP_TRAIN_CONFIG, marker_data


def _model(block, dropout, n_genes, extra=None):
    params = {
        "len_in": n_genes, "inplanes": 1, "num_classes": 2,
        "block": block, "block_nums": [1], "block_dims": [8],
        "bias": True, "dropout": dropout, "norm_layer": "LayerNorm",
        "latent_fea_dim": 16,
    }
    params.update(extra or {})
    torch.manual_seed(0)
    return NN_Classifier(model_params=params, train_config=dict(MLP_TRAIN_CONFIG))


def _corpus(n_genes):
    adata = make_fake_adata(n_cells=30, n_genes=n_genes, n_class=2,
                            n_clusters_per_class=1, seed=3)
    return Multimodal_Corpus(adata=adata, label_key="celltype", backed=False)


@pytest.fixture(scope="module", params=[2, 3])
def marker_setup(request):
    """Trained MLP plus a marker corpus with the last class first.

    Every 10th cell gets the next class's label but keeps its own profile, so
    the model (trained on the true labels) misclassifies it. Each class then
    has 60 labelled cells, 54 of them correctly predicted.
    """
    n_classes = request.param
    x, y, names = marker_data(n_classes=n_classes)
    model = _model("Residual_Encoder", dropout=0.0, n_genes=len(names),
                   extra={"num_classes": n_classes})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(100):
        optimizer.zero_grad()
        model.criterion(model(x), y).backward()
        optimizer.step()
    model.eval()

    labels = y.clone()
    labels[::10] = (labels[::10] + 1) % n_classes
    x, labels = x.flip(0), labels.flip(0)
    adata = ad.AnnData(csr_matrix(x[:, 0].numpy()),
                       obs=pd.DataFrame(index=[f"cell_{i}" for i in range(len(labels))]),
                       var=pd.DataFrame(index=names))
    adata.obs["celltype"] = pd.Categorical(labels.numpy(), categories=range(n_classes))
    return model, Multimodal_Corpus(adata=adata, label_key="celltype", backed=False)


def _explained_cells(explainer, class_index, sample_limit=None):
    """Label, prediction and ``class_index`` loss of each cell selected for it."""
    target = torch.tensor([class_index])
    rows = []
    with torch.no_grad():
        for x, y in explainer.stratify_dataset(class_index=class_index, sample_limit=sample_limit):
            out = explainer.model(x)
            rows.append((y.item(), out.argmax(-1).item(), explainer.model.criterion(out, target).item()))
    return pd.DataFrame(rows, columns=["label", "pred", "loss"])


def test_integrated_gradients_are_deterministic():
    corpus = _corpus(10)
    model = _model("Residual_Encoder", dropout=0.5, n_genes=10)
    kwargs = dict(dataset=corpus, device="cpu", exp_step=16, n_dataloader_workers=0)
    first = model.explain(**kwargs)
    second = model.explain(**kwargs)
    np.testing.assert_allclose(first.values, second.values, atol=1e-6)
    assert not model.training


def test_explained_cells_are_correctly_predicted_cells_of_the_class(marker_setup):
    model, corpus = marker_setup
    explainer = Basic_Clf_Explainer(model=model, dataset=corpus, device="cpu",
                                    n_dataloader_workers=0)
    for c in corpus.label_dict:
        kept = _explained_cells(explainer, c)
        assert len(kept) == 54
        assert (kept["label"] == c).all() and (kept["pred"] == c).all()

        # sample_limit: evenly spaced in loss order over the same cells.
        part = _explained_cells(explainer, c, sample_limit=5)
        assert len(part) == 5
        assert set(part["loss"]) <= set(kept["loss"])
        assert part["loss"].min() == kept["loss"].min()
        assert part["loss"].max() == kept["loss"].max()


@pytest.mark.parametrize("exp_sample_limit", [None, 10])
def test_nn_marker_ranks_first_for_its_class(marker_setup, exp_sample_limit):
    model, corpus = marker_setup
    exp = model.explain(dataset=corpus, exp_sample_limit=exp_sample_limit, device="cpu",
                        exp_step=16, exp_batch_size=64, n_dataloader_workers=0)
    for c in corpus.label_dict:
        col = exp[f"Class_{c}_Scores"]
        assert col.idxmax() == f"marker_{c}"
        assert col[f"marker_{c}"] > 0


def test_rnn_sequence_axis_is_modality():
    n_genes = 10
    model = _model("RNN_Residual_Encoder", dropout=0.0, n_genes=n_genes,
                   extra={"block_layer_type": "GRU", "nonlinearity": "tanh",
                          "bidirectional": True, "proj_size": 0})
    seen = {}

    def record_input(module, inputs, output):
        seen["shape"] = tuple(inputs[0].shape)

    model.blocks[0][0].layer.register_forward_hook(record_input)

    model.eval()
    model(torch.randn(4, 1, n_genes))
    batch, seq_len, features = seen["shape"]
    assert (batch, seq_len, features) == (4, 1, 16)


def test_feature_order_is_not_imposed():
    """Permuting genes and the embedder's input columns gives the same function."""
    n_genes = 10
    model = _model("RNN_Residual_Encoder", dropout=0.0, n_genes=n_genes,
                   extra={"block_layer_type": "LSTM", "nonlinearity": "tanh",
                          "bidirectional": True, "proj_size": 0})
    model.eval()
    perm = torch.randperm(n_genes, generator=torch.Generator().manual_seed(1))

    permuted = copy.deepcopy(model)
    with torch.no_grad():
        permuted.embedder[0].weight.copy_(model.embedder[0].weight[:, perm])

    x = torch.randn(8, 1, n_genes)
    with torch.no_grad():
        np.testing.assert_allclose(model(x).numpy(), permuted(x[:, :, perm]).numpy(), atol=1e-5)
