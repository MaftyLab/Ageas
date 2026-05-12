Installation
============

Prerequisites
-------------
Before installing **Ageas**, please ensure your system meets the following
requirements:

* **Python:** Version 3.11 or newer is required.
* **Hardware:** A CUDA-capable GPU is recommended for the neural-network
  classifiers (:class:`~ageas.NN_Classifier`,
  :class:`~ageas.Mixer_Classifier`) and for SHAP-based explanation of
  XGBoost models. The classical classifiers run comfortably on CPU.


Install from Source
-------------------
The latest development version can be installed directly from GitHub:

.. code-block:: bash

    gh repo clone MaftyLab/Ageas
    cd Ageas
    pip install .

This command will automatically resolve and install all required core
dependencies (PyTorch, PyTorch Lightning, XGBoost, scikit-learn,
imbalanced-learn, AnnData, SHAP, Captum, and others).

Verifying the Installation
--------------------------
A quick sanity check uses the synthetic data helper bundled with the
:mod:`ageas.tool` module:

.. code-block:: python

    import ageas
    from ageas.tool import make_fake_adata, Multimodal_Corpus

    adata = make_fake_adata(n_cells=200, n_genes=50, n_class=3)
    corpus = Multimodal_Corpus(adata=adata, label_key='celltype')
    print(corpus.label_dict)

If the corpus prints a ``{0: 0, 1: 1, 2: 2}``-style label dictionary, the
package and its core dependencies are correctly installed.
