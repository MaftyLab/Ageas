#!/usr/bin/env python3
"""Per-unit attributions used to rank the features a unit relies on.

Classical and tree models use their own ``explain``. Neural networks use
Integrated Gradients from :class:`ageas.tool.Basic_Clf_Explainer`, run on
the cells of each class that the model predicts correctly. Those cells are
picked here: R1's ``Basic_Clf_Explainer.stratify_dataset`` applies
positions from the class subset to the full dataset and explains the
wrong cells.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ageas.tool import Basic_Clf_Explainer, class_contrast, l1_normalize
from ageas.tool.corpus_loader import get_all_data

#: Unit types explained with Integrated Gradients.
NN_TYPES = ('mlp', 'rnn', 'resnet')


def explain_unit(
    model,
    unit_type: str,
    dataset,
    exp_sample_limit: int = None,
    device: str = 'cpu',
    **kwargs,
) -> pd.DataFrame:
    """Per-class, class-contrastive attribution of one trained unit.

    :param model: Trained classifier of the unit.
    :param unit_type: The unit's model type key (e.g. ``'mlp'``, ``'logreg'``).
    :param dataset: Labelled corpus exposing ``features`` and ``label_dict``.
    :param exp_sample_limit: For neural networks, the largest number of
        correctly predicted cells explained per class, spread evenly over
        their loss order. ``None`` explains all of them.
    :param device: Device for Integrated Gradients.
    :param kwargs: Forwarded to :class:`ageas.tool.Basic_Clf_Explainer`
        for neural networks (e.g. ``exp_step``), otherwise to the model's
        ``explain``.
    :returns: Table indexed by feature with one ``Class_{i}_Scores`` column
        per class.
    """
    if unit_type in NN_TYPES:
        return _explain_nn(model, dataset, exp_sample_limit, device, **kwargs)
    table = model.explain(dataset=dataset, **kwargs)
    return table[[c for c in table.columns if str(c).endswith('_Scores')]]


def _explain_nn(model, dataset, exp_sample_limit, device, batch_size=512, **kwargs):
    """Integrated Gradients on each class's correctly predicted cells."""
    x, y = get_all_data(dataset)
    explainer = Basic_Clf_Explainer(model, device=device, n_dataloader_workers=0, **kwargs)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([
            model(x[i:i + batch_size].to(device)).cpu()
            for i in range(0, len(x), batch_size)
        ])
    loss = F.cross_entropy(logits, y.long(), reduction='none')
    decision = logits.argmax(dim=-1)
    n_class = len(dataset.label_dict) if dataset.label_dict else logits.shape[-1]

    means = []
    for c in range(n_class):
        cells = torch.where((y == c) & (decision == c))[0]
        if exp_sample_limit is not None and len(cells) > exp_sample_limit:
            cells = cells[torch.argsort(loss[cells], stable=True)]
            cells = cells[np.linspace(0, len(cells) - 1, exp_sample_limit).astype(int)]
        loader = DataLoader(
            TensorDataset(x[cells], y[cells]),
            batch_size=explainer.exp_batch_size,
            shuffle=False,
        )
        # cuDNN only implements the RNN backward pass in training mode.
        with torch.backends.cudnn.flags(enabled=False):
            mean, _ = explainer(c, dataloader=loader)
        mean = np.zeros(x.shape[-1]) if mean is None else mean.cpu().numpy()
        means.append(l1_normalize(mean, mode='numpy'))
    model.to('cpu')

    scores = class_contrast(np.array(means))
    return pd.DataFrame(
        scores.T,
        index=dataset.features,
        columns=[f'Class_{c}_Scores' for c in range(n_class)],
    )
