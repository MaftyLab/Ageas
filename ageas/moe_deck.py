#!/usr/bin/env python3
"""MoE deck for Ageas v2.

The deck holds the refitted experts: each integrated or competent unit,
refitted on all terminal cells and narrowed to its own key features. The
integrated ones carry the weight the no-harm selection gave them; the
competent ones are the committee's candidates.
"""
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ageas.tool.corpus_loader import decomp_batch, get_all_data

from .tool.feature_corpus import Feature_Corpus
from .tool.unit_explain import explain_unit


def predict_probs(model, corpus, batch_size: int = 512) -> np.ndarray:
    """Class probabilities of ``model`` for every cell of ``corpus``."""
    parts = [
        np.asarray(model.predict(x, y))
        for x, y in DataLoader(corpus, batch_size=batch_size, shuffle=False)
    ]
    return np.concatenate(parts, axis=0)


class Expert:
    """One integrated unit, refitted on all terminal cells.

    :ivar unit: The :class:`~ageas.Unit` it came from.
    :ivar level: Mass level it was pruned to, or ``None`` for all features.
    :ivar features: Names of the features it works on.
    :ivar model: Trained classifier.
    """

    def __init__(self, unit, level, features: list, model) -> None:
        self.unit = unit
        self.level = level
        self.features = list(features)
        self.model = model


class MoE_Deck:
    """Integrated experts of an Ageas v2 run.

    :ivar experts: Mapping ``unit_id -> Expert`` for every integrated or
        competent unit.
    :ivar weights: Mapping ``unit_id -> weight`` from the no-harm selection;
        the weights sum to one.
    :ivar competent: Units whose out-of-fold balanced accuracy is close to
        the best, ranked by out-of-fold loss: the committee's candidates.
    :ivar oof: Out-of-fold record of the run: per-unit probabilities on
        every terminal cell (``'probs'``), the labels, the fold indices, and
        each unit's per-fold level, features and validation losses.
    :ivar features: Feature names of the terminal corpus.
    :ivar provenance: Which ``ageas`` the run imported (``ageas_path``) and
        the git state of that checkout (``ageas_commit``, ``ageas_dirty``).
    """

    def __init__(
        self,
        experts: dict,
        weights: dict,
        oof: dict,
        features: list,
        provenance: dict = None,
        competent: list = None,
    ) -> None:
        self.experts = experts
        self.weights = weights
        self.oof = oof
        self.features = list(features)
        self.provenance = provenance if provenance is not None else {}
        self.competent = list(competent) if competent is not None else []

    def _feature_index(self, dataset) -> dict:
        """Positions of each expert's features in ``dataset.features``."""
        names = pd.Index(dataset.features)
        index = {}
        for unit_id, expert in self.experts.items():
            positions = names.get_indexer(expert.features)
            assert (positions >= 0).all(), \
                f"{unit_id} needs features that the dataset does not have."
            index[unit_id] = positions
        return index

    def predict(
        self,
        query_dataset=None,
        batch_size: int = 10,
        num_workers: int = 1,
        return_members: bool = False,
    ) -> tuple:
        """Ensemble-predict with the no-harm weights.

        Each expert sees only its own features, looked up by name in
        ``query_dataset.features``, and the experts' probabilities are
        averaged with :attr:`weights`.

        :param query_dataset: Dataset to score.
        :param batch_size: Batch size for the prediction dataloader.
        :param num_workers: Number of dataloader workers.
        :param return_members: If ``True``, also return the probabilities of
            every expert, weighted or not.
        :returns: ``(probs, labels)``, or ``(probs, labels, member_probs)``
            with ``return_members=True``. ``labels`` is ``None`` for an
            unlabelled dataset.
        """
        index = self._feature_index(query_dataset)
        members = {unit_id: [] for unit_id in self.experts}
        labels = []
        for batch in DataLoader(
            query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ):
            x, y = decomp_batch(batch)
            for unit_id, expert in self.experts.items():
                members[unit_id].append(
                    np.asarray(expert.model.predict(x[..., index[unit_id]], y))
                )
            if y is not None:
                labels.append(np.asarray(y))

        members = {unit_id: np.concatenate(parts, axis=0) for unit_id, parts in members.items()}
        probs = sum(weight * members[unit_id] for unit_id, weight in self.weights.items())
        labels = np.concatenate(labels, axis=0) if labels else None
        if return_members:
            return probs, labels, members
        return probs, labels

    def debrief(
        self,
        exp_dataset=None,
        exp_sample_limit: int = None,
        device: str = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Integrated per-class attribution over all features.

        Each expert explains ``exp_dataset`` on its own features, so the
        features it does not use score zero for it. The experts' tables are
        combined as ``A_c = sum_e w_e d_{e,c}`` with the no-harm weights.

        :param exp_dataset: Labelled dataset to explain.
        :param exp_sample_limit: Cells per class explained by Integrated
            Gradients for neural-network experts.
        :param device: Device for Integrated Gradients. ``None`` uses each
            expert's own accelerator.
        :param kwargs: Forwarded to :func:`~ageas.explain_unit`.
        :returns: Table indexed by :attr:`features` with one
            ``Class_{i}_Scores`` column per class.
        """
        x, y = get_all_data(exp_dataset)
        corpus = Feature_Corpus(
            x, y, exp_dataset.features, exp_dataset.label_dict,
            getattr(exp_dataset, 'label_key', None),
        )
        total = None
        for unit_id in self.weights:
            expert = self.experts[unit_id]
            table = explain_unit(
                expert.model, expert.unit.type, corpus.restrict(expert.features),
                exp_sample_limit=exp_sample_limit,
                device=device or ('cuda' if expert.unit.accelerator == 'cuda' else 'cpu'),
                **kwargs,
            )
            table = table.reindex(self.features, fill_value=0.0) * self.weights[unit_id]
            total = table if total is None else total + table
        return total
