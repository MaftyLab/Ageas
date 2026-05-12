#!/usr/bin/env python3
"""
random data generator for testing purposes
"""
import shutil
import logging
import warnings
from ageas import Hangar, n_iter_extraction
from ageas.tool import Multimodal_Corpus, make_fake_adata


# For filtering pytorch_lightning logs and slurm related warnings
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Set the parameters
n_extract_iter = 2
top_n = 5
seed = 42

# Set the seed for reproducibility
accelerator = 'cuda'
# accelerator = 'cpu'

# Make a fake adata corpus
fake_adata_path = '../data/fake_adata.h5ad'
fake_adata = make_fake_adata(n_class = 2, n_clusters_per_class = 1)
fake_adata.write_h5ad(
    fake_adata_path,
)
print(fake_adata)
del fake_adata

corpus = Multimodal_Corpus(
    fake_adata_path,
    label_key = 'celltype',
    backed=False
)

# # Load the foo mESC dataset
# corpus = Multimodal_Corpus(
#     "../data/foo.mESC.h5ad",
#     label_key = 'time',
#     backed=False
# )

# Test the n_iter_extraction function
config_folder = "../data/configs/config_larry_v5"
ans = n_iter_extraction(
    hangar = Hangar(config_folder = config_folder,),
    accelerator = accelerator,
    query_dataset = corpus,
    max_extraction_iter = n_extract_iter,
    extract_top_n = top_n,
    use_gene_names=False,
    verbose = True,
    selection_args = {
        'kfold_selection_list': [2],
    },
    explain_args = {
        'exp_step': 50,
        'exp_sample_limit': 10,
        'exp_batch_size': 5,
    },
)

# Remove the caches
shutil.rmtree('cache')

print("Extraction finished.")
print("Results:")
print(ans)