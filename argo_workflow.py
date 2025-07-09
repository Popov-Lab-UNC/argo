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
lower_is_better = True

# 2. Sort by score (ascending: best first)
df = df.sort_values('score', ascending=lower_is_better)

# 3-5. Create fragment vocabulary using the new class with enrichment scoring
vocab = FragmentVocabulary(
    data='CHD1_score0.csv',
    smiles_col='smiles',
    score_col='score',
    scoring_method='enrichment',  # Changed from 'average' to 'enrichment'
    min_frag_size=5,
    max_frag_size=30,
    min_count=5,
    max_fragments=1000,
    lower_is_better=lower_is_better
)

# Save the vocabulary
vocab.save('fragment_scores_init.csv')
print('Fragment statistics written to fragment_scores.csv')

# Demonstrate loading from CSV
print('\nDemonstrating vocabulary loading functionality:')
print('1. Loading vocabulary from saved CSV file...')
vocab_from_csv = FragmentVocabulary.load_from_csv('fragment_scores_init.csv')
print(f'   Loaded vocabulary with {len(vocab_from_csv)} fragments')

# Show vocabulary info
info = vocab_from_csv.get_info()
print(f'   Vocabulary info: {info["vocab_size"]} fragments, loaded from CSV: {info["loaded_from_csv"]}')

# Save complete state for later use
vocab.save_state('complete_vocab_state.pkl')
print('2. Saved complete vocabulary state to complete_vocab_state.pkl')

# Load complete state
vocab_complete = FragmentVocabulary.load_state('complete_vocab_state.pkl')
print(f'3. Loaded complete vocabulary with {len(vocab_complete)} fragments')
print(f'   Has original data: {vocab_complete.get_info()["has_original_data"]}')

# TODO: A way to load the vocabulary from the csv file

# 6. Instantiate all four generative models via the API
use_cuda = torch.cuda.is_available()
safegpt_model = GenerationModel('safegpt', use_cuda=use_cuda)
api_key = os.environ.get("MOLMIM_API_KEY")
molmim_model = GenerationModel('molmim', api_token=api_key)
gem_model = GenerationModel('gem', model_path='argo/gen_models/pretrained/gem_chembl.pt', use_cuda=use_cuda)

# Use the vocabulary object directly with f-RAG
f_rag_model = GenerationModel('f-rag', 
                              vocab=vocab,
                              injection_model_path="argo/gen_models/pretrained/model.safetensors",
                              frag_population_size=50,
                              mol_population_size=20,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=100,
                              mutation_rate=0.01,
                              use_cuda=use_cuda
)

print('All four generative models instantiated.')
