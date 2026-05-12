import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "ageas",
    version = "v0.0.1-alpha1",
    author = "nkmtmsys and JackSSK",
    author_email = "gyu@genzentrum.lmu.de, gyu17@alumni.jh.edu",
    description = "AutoML-based Genetic regulatory Element extrAction System",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/MaftyLab/Ageas",
    project_urls = {"Bug Tracker": "https://github.com/MaftyLab/Ageas/issues",},
    packages = setuptools.find_packages(),
    package_data = {'': [
        'data/*',
        'data/configs/*',
        'data/gene_dict/*',
    ]},
    include_package_data = True,
    classifiers = [
                    'Programming Language :: Python :: 3',
                    'License :: OSI Approved :: Apache License Version 2.0',
                    'Operating System :: OS Independent',
                    ],
    python_requires = ">=3.11",
    install_requires = [
        "shap >= 0.45.1",
        "captum >= 0.8.0",
        "torch >= 2.2.0",
        "pytorch-lightning >= 2.2.0",
        "torchmetrics >= 1.6.1",
        "xgboost == 2.0.3",
        "scikit-learn >= 1.5.1",
        "imbalanced-learn >= 0.13.0",
        "numpy >= 1.26.4",
        "pandas >= 2.2.2",
        "anndata >= 0.10.8",
        "hdf5plugin >= 4.3.0",
    ],
)
