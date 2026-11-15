"""Running Ageas and storing predictions in one shared format.

Prediction files (CSV) are indexed by cell and hold one ``prob_<class>``
column per terminal class, a ``pred`` column and, optionally, ``method`` and
``seed`` columns. Every experiment writes and reads this format, so any
method's output can be scored by ``revision/common/metrics.py``.
"""
import os

import numpy as np
import pandas as pd

from .registry import LABEL_COL

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir(experiment: str, dataset: str, root: str = None) -> str:
    """``revision/results/<experiment>/<dataset>/``, created if needed."""
    path = os.path.join(root or os.path.join(_HERE, 'results'), experiment, dataset)
    os.makedirs(path, exist_ok=True)
    return path


def _dense_corpus_class():
    """``Multimodal_Corpus`` subclass serving rows from a dense copy of ``X``.

    ``Multimodal_Corpus`` slices the AnnData once per cell, which dominates
    run time when every configuration is cross-validated. This subclass
    returns the same ``(1, n_features)`` float32 rows and integer labels from
    arrays held in memory.
    """
    from ageas.tool import Multimodal_Corpus

    class Dense_Corpus(Multimodal_Corpus):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            x = self.adata.X
            self._x = np.asarray(x.toarray() if hasattr(x, 'toarray') else x, dtype=self.dtype)[:, None, :]
            self._y = self.adata.obs[self.label_key].cat.codes.to_numpy(dtype=np.int64)

        def __getitem__(self, idx):
            return self._x[idx], int(self._y[idx])

    return Dense_Corpus


def _corpus(adata, classes, dense: bool = False):
    """Wrap AnnData in a Multimodal_Corpus with the terminal label categories.

    Query cells have no terminal label; they receive the first class as a
    placeholder so the dataloader yields ``(x, y)`` pairs (the label is
    never used for prediction). ``dense=True`` serves rows from memory (see
    :func:`_dense_corpus_class`); the items are identical.
    """
    from ageas.tool import Multimodal_Corpus

    adata = adata.copy()
    if LABEL_COL not in adata.obs or adata.obs[LABEL_COL].isna().any():
        adata.obs[LABEL_COL] = classes[0]
    adata.obs[LABEL_COL] = pd.Categorical(adata.obs[LABEL_COL].astype(str), categories=classes)
    corpus_class = _dense_corpus_class() if dense else Multimodal_Corpus
    return corpus_class(adata=adata, label_key=LABEL_COL, backed=False)


def run_ageas(
    train,
    classes: list,
    config_dir: str,
    seed: int,
    selection: dict,
    accelerator: str = 'cpu',
    operation_name: str = 'trail',
):
    """Run Ageas model selection on terminal cells with a fresh hangar.

    Args:
        train: Terminal-cell AnnData with ``obs['_ageas_label']``.
        classes: Terminal classes in label order.
        config_dir: Model-panel config folder (e.g.
            ``data/configs/default_config_v1``).
        seed: Seed for the k-fold splits and for model initialisation.
        selection: Keyword arguments for :func:`ageas.n_kfold_selection`
            (from ``revision/protocol.json`` unless deliberately varied).
        accelerator: ``'cpu'`` or ``'cuda'``.
        operation_name: Report key for this run.

    Returns:
        ``(deck, corpus)`` — the selected deck and the training corpus.
    """
    from pytorch_lightning import seed_everything
    from ageas import Hangar, n_kfold_selection

    seed_everything(seed, workers=True)
    corpus = _corpus(train, classes)
    deck = n_kfold_selection(
        hangar=Hangar(config_dir),
        operation_name=operation_name,
        query_dataset=corpus,
        accelerator=accelerator,
        seed=seed,
        **selection,
    )
    return deck, corpus


def predict_query(deck, query, classes: list, batch_size: int = 512, num_workers: int = 0):
    """Ensemble and per-unit class probabilities for query cells.

    Returns:
        ``(probs, members)`` — a DataFrame of ensemble probabilities with
        ``prob_<class>`` columns and ``pred``, and a dict mapping unit ID to
        that unit's probability array.
    """
    corpus = _corpus(query, classes)
    mean, _, members = deck.predict(
        corpus, batch_size=batch_size, num_workers=num_workers, return_members=True
    )
    return probs_frame(mean, query.obs_names, classes), members


def probs_frame(probs: np.ndarray, index, classes: list) -> pd.DataFrame:
    """Standard prediction table from a probability matrix."""
    frame = pd.DataFrame(np.asarray(probs), index=index, columns=[f'prob_{c}' for c in classes])
    frame['pred'] = [classes[i] for i in np.argmax(frame.values, axis=1)]
    return frame


def unit_table(deck, operation_name: str = 'trail') -> pd.DataFrame:
    """Per-unit type and out-of-fold scores of a deck."""
    rows = []
    for unit_id, unit in deck.squad.items():
        oof = unit.report[operation_name].get('final', {}).get('oof', {})
        rows.append(dict(unit_id=unit_id, type=unit.type, **oof))
    return pd.DataFrame(rows).set_index('unit_id')


def save_predictions(frame: pd.DataFrame, path: str, method: str = None, seed=None) -> None:
    """Write a prediction table, tagging method and seed."""
    frame = frame.copy()
    if method is not None:
        frame['method'] = method
    if seed is not None:
        frame['seed'] = seed
    frame.to_csv(path)


def load_predictions(path: str) -> pd.DataFrame:
    """Read a prediction table written by :func:`save_predictions`."""
    return pd.read_csv(path, index_col=0)


def apply_selection_gates(
    oof: pd.Series,
    selection_ratio: float = 0.5,
    retention_point: float = 0.9,
    cutoff_point: float = 0.7,
    relative_retention: float = None,
    monitor_type: str = 'max',
) -> list:
    """Offline replica of the per-round gate in ``n_kfold_selection``.

    Lets ablation/sensitivity experiments re-select from one fully trained
    panel without retraining. Mirrors the single-round case exactly.

    Args:
        oof: Out-of-fold score per unit.
        selection_ratio, retention_point, cutoff_point, relative_retention,
        monitor_type: As in :func:`ageas.n_kfold_selection`.

    Returns:
        Surviving unit IDs (never empty).
    """
    ranked = oof.sort_values(ascending=monitor_type != 'max')
    better = (lambda a, b: a > b) if monitor_type == 'max' else (lambda a, b: a < b)
    retention = retention_point
    if relative_retention is not None:
        best = ranked.iloc[0]
        retention = best - relative_retention if monitor_type == 'max' else best + relative_retention
    expect = max(int(len(ranked) * selection_ratio), 1)
    keep = [
        uid for rank, (uid, score) in enumerate(ranked.items())
        if (rank < expect or better(score, retention)) and better(score, cutoff_point)
    ]
    return keep or [ranked.index[0]]
