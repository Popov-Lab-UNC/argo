import os
import pandas as pd
from rdkit import Chem
from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder
from argo.frag_utils import SAFECodec
from argo.gen_models.api import GenerationModel, GenerationTask
from argo.vocab import FragmentVocabulary
from tqdm import tqdm
import torch

# 1. Load docking score CSV
df = pd.read_csv('CHD1_score0.csv')

# 2. Sort by score (ascending: best first)
df = df.sort_values('score', ascending=True)

# 3-5. Craft fragment vocabulary using the new class
vocab = FragmentVocabulary(
    data='CHD1_score0.csv',
    smiles_col='smiles',
    score_col='score',
    scoring_method='average',  # or 'enrichment'
    top_percent=10.0,  # only used for enrichment scoring
    min_frag_size=5,
    max_frag_size=30,
    min_count=1,
    max_fragments=None,
    verbose=True
)

# Save the vocabulary
vocab.save('fragment_scores.csv')
print('Fragment statistics written to fragment_scores.csv')

# 6. Instantiate all four generative models via the API
use_cuda = torch.cuda.is_available()
safegpt_model = GenerationModel('safegpt', use_cuda=use_cuda)
api_key = os.environ.get("MOLMIM_API_KEY")
molmim_model = GenerationModel('molmim', api_token=api_key)
gem_model = GenerationModel('gem', model_path='argo/gen_models/pretrained/gem_chembl.pt', use_cuda=use_cuda)

# Use the vocabulary directly with f-RAG
f_rag_model = GenerationModel('f-rag', 
                              vocab_path=vocab,  # Pass the class directly
                              injection_model_path="argo/gen_models/pretrained/model.safetensors",
                              frag_population_size=50,
                              mol_population_size=20,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=100,
                              mutation_rate=0.01
)

print('All four generative models instantiated.')
