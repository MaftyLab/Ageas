"""Dataset registry and preparation.

Every experiment reads datasets through ``revision/datasets.json`` (copy
``datasets.example.json`` and fill in the paths). A dataset entry declares
which ``obs`` column holds cell states, which states are terminal training
classes and which are query progenitors, how to preprocess, and where the
lineage ground truth lives. See ``revision/README.md`` for the schema.
"""
import json
import os
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ground_truth import clonal_fate_table, labels_from_obs

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(_HERE, 'datasets.json')
DEFAULT_PROTOCOL = os.path.join(_HERE, 'protocol.json')
LABEL_COL = '_ageas_label'
STATE_COL = '_ageas_state'


@dataclass
class Prepared:
    """A dataset ready for fate inference.

    Attributes:
        name: Registry key.
        spec: The registry entry.
        adata: Preprocessed AnnData restricted to terminal + query cells.
        train: Terminal cells with categorical ``obs['_ageas_label']``.
        query: Query (progenitor) cells.
        classes: Terminal class names in training-label order.
        ground_truth: Ground-truth table indexed by query cell (may be
            ``None`` for datasets without lineage tracing).
    """
    name: str
    spec: dict
    adata: ad.AnnData
    train: ad.AnnData
    query: ad.AnnData
    classes: list
    ground_truth: pd.DataFrame = None
    extras: dict = field(default_factory=dict)


def load_registry(path: str = None) -> dict:
    """Read the dataset registry JSON."""
    path = path or DEFAULT_REGISTRY
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy revision/datasets.example.json to "
            "revision/datasets.json and fill in the dataset paths."
        )
    with open(path, encoding='utf-8') as fh:
        registry = json.load(fh)
    return {k: v for k, v in registry.items() if not k.startswith('_')}


def load_protocol(path: str = None) -> dict:
    """Read the frozen selection protocol shared by all experiments."""
    with open(path or DEFAULT_PROTOCOL, encoding='utf-8') as fh:
        protocol = json.load(fh)
    return {k: v for k, v in protocol.items() if not k.startswith('_')}


def resolve_path(spec: dict) -> str:
    """Absolute path of a registry entry's h5ad (relative paths are repo-relative)."""
    path = spec['h5ad']
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(_HERE), path)


def _tfidf(matrix, scale_factor: float = 1e4) -> sp.csr_matrix:
    """Signac ``RunTFIDF(method = 1)``: ``log1p(TF * IDF * scale_factor)``.

    TF divides each cell's counts by the cell's total; IDF is the number of
    cells divided by each peak's total count over all cells.
    """
    counts = sp.csr_matrix(matrix, dtype=np.float64)
    cell_totals = np.asarray(counts.sum(axis=1)).ravel()
    peak_totals = np.asarray(counts.sum(axis=0)).ravel()
    tf = sp.diags(1.0 / np.maximum(cell_totals, 1e-12)) @ counts
    out = (tf @ sp.diags(counts.shape[0] / np.maximum(peak_totals, 1e-12))).tocsr()
    out.data = np.log1p(out.data * scale_factor)
    return out


def _preprocess(adata: ad.AnnData, options: dict) -> ad.AnnData:
    """Normalisation and feature selection (scanpy).

    Besides the scanpy steps, ``options`` may ask for ``tfidf`` (Signac
    method 1, with ``scale_factor``) and ``min_detect_frac`` (keep features
    detected in at least that fraction of cells). A sparse ``X`` is returned
    as CSR float32, the layout ``Multimodal_Corpus`` reads row by row.
    """
    options = options or {}
    if options.get('layer'):
        adata.X = adata.layers[options['layer']].copy()
    if options.get('tfidf'):
        adata.X = _tfidf(adata.X, options.get('scale_factor', 1e4))
    n_hvg = options.get('n_hvg')
    flavor = options.get('hvg_flavor', 'seurat_v3')
    if n_hvg or options.get('normalize_total') or options.get('log1p'):
        import scanpy as sc
    if n_hvg and flavor == 'seurat_v3':
        # seurat_v3 expects counts, so select features before normalising
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=flavor, subset=True)
    if options.get('normalize_total'):
        sc.pp.normalize_total(adata, target_sum=options.get('target_sum', 1e4))
    if options.get('log1p'):
        sc.pp.log1p(adata)
    if n_hvg and flavor != 'seurat_v3':
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=flavor, subset=True)
    if options.get('min_detect_frac'):
        detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        adata = adata[:, detected >= options['min_detect_frac'] * adata.n_obs].copy()
    if sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    return adata


def cell_states(adata: ad.AnnData, spec: dict) -> pd.Series:
    """Cell states from ``label_key``, merged through the optional ``label_map``.

    ``label_map`` renames or pools states, e.g. ``{"de": "dead-end"}`` or
    several clusters into one lineage; unmapped states keep their name.
    """
    states = adata.obs[spec['label_key']].astype(str)
    mapping = spec.get('label_map')
    return states.map(lambda s: mapping.get(s, s)) if mapping else states


def _ground_truth(adata: ad.AnnData, query_mask, spec: dict) -> pd.DataFrame:
    gt = spec.get('ground_truth')
    if not gt:
        return None
    if gt['type'] == 'clone_matrix':
        return clonal_fate_table(
            adata,
            fates=tuple(gt.get('fates', spec['terminal_classes'])),
            progenitor_mask=query_mask,
            clone_obsm=gt.get('clone_obsm', 'X_clone'),
            state_key=gt.get('state_key', spec['label_key']),
            time_key=gt.get('time_key'),
            later_only=gt.get('later_only', True),
            pure_threshold=gt.get('pure_threshold', 1.0),
        )
    if gt['type'] == 'obs_label':
        return labels_from_obs(adata, gt['key'], gt.get('mapping'), mask=query_mask)
    if gt['type'] == 'csv':
        return pd.read_csv(gt['path'], index_col=0)
    raise ValueError(f"Unknown ground_truth type {gt['type']!r}.")


def prepare_dataset(name: str, registry: dict = None, subset_features: list = None) -> Prepared:
    """Load, filter and preprocess one registry dataset.

    Ground truth is computed on the raw object before any cell filtering so
    that clonal relatives outside the terminal classes are still counted.
    Preprocessing runs on the kept cells, or on every cell of the object when
    ``preprocess.scope`` is ``"object"`` (e.g. a detection filter defined on
    the whole benchmark object).

    Args:
        name: Registry key.
        registry: Parsed registry; defaults to ``revision/datasets.json``.
        subset_features: Optional feature list to keep (after preprocessing).

    Returns:
        :class:`Prepared` dataset.
    """
    registry = registry or load_registry()
    spec = registry[name]
    adata = ad.read_h5ad(resolve_path(spec))
    adata.obs_names_make_unique()

    adata.obs[STATE_COL] = cell_states(adata, spec).values
    states = adata.obs[STATE_COL]
    query_mask = states.isin(spec['query_values']).values
    ground_truth = _ground_truth(adata, query_mask, spec)

    keep = states.isin(list(spec['terminal_classes']) + list(spec['query_values'])).values
    if spec.get('extra_keep_values'):
        keep |= states.isin(spec['extra_keep_values']).values
    options = spec.get('preprocess') or {}
    if options.get('scope') == 'object':
        adata = _preprocess(adata, options)[keep].copy()
    else:
        adata = _preprocess(adata[keep].copy(), options)
    if subset_features is not None:
        adata = adata[:, [f for f in subset_features if f in adata.var_names]].copy()
    if 'name' not in adata.var:
        adata.var['name'] = adata.var_names

    states = adata.obs[STATE_COL]
    classes = list(spec['terminal_classes'])
    train = adata[states.isin(classes).values].copy()
    train.obs[LABEL_COL] = pd.Categorical(train.obs[STATE_COL], categories=classes)
    query = adata[states.isin(spec['query_values']).values].copy()

    return Prepared(
        name=name, spec=spec, adata=adata, train=train, query=query,
        classes=classes, ground_truth=ground_truth,
    )


def dense(matrix) -> np.ndarray:
    """Dense float copy of a (possibly sparse) matrix."""
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix, dtype=float)
