import os
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import torch
import time
import random
from typing import List, Dict, Any

from argo.utils import clean_smiles, run_generation_task
from argo.gen_models import GenerationModel, GenerationTask
from argo.vocab import FragmentVocabulary
from argo.frag_utils import find_connected_rotatable_bond_ends
from argo.filter_models import ChemeleonFilterModel
from argo.gen_models.f_rag.ga.ga import reproduce

# --- 1. Data Loading and Preparation ---
print("="*60)
print("1. Loading and preparing initial data...")
print("="*60)
df = pd.read_csv('../benchmark/class1_compounds.csv')
actives_df = df[df['ic50'] < 10].copy()
hdg_smiles = clean_smiles(actives_df['SMILES'].tolist())

print(f"Loaded {len(df)} total compounds.")
print(f"Found {len(actives_df)} active compounds (IC50 < 10 uM).")

# --- 2. Vocabulary for f-RAG Model ---
print("\n" + "="*60)
print("2. Creating vocabulary for f-RAG model...")
print("="*60)
frag_vocab_for_frag = FragmentVocabulary(
    slicer='f-rag',
    data=actives_df,
    smiles_col='SMILES',
    score_col='ic50',
    lower_is_better=True
)
print(f"f-RAG vocabulary created with {len(frag_vocab_for_frag.get_vocab())} fragments.")

# --- 3. Vocabulary for Top Fragments ---
print("\n" + "="*60)
print("3. Creating vocabulary for top fragments...")
print("="*60)
rotbonds_vocab = FragmentVocabulary(
    slicer=find_connected_rotatable_bond_ends,
    data=actives_df,
    smiles_col='SMILES',
    score_col='ic50',
    min_frag_size=5,
    max_frag_size=30,
    min_count=1,
    max_fragments=50,
    lower_is_better=True
)
rotbonds_vocab_df = rotbonds_vocab.get_vocab()
top_arms = rotbonds_vocab_df[rotbonds_vocab_df['type'] == 'arm'].head(10)['frag'].tolist()
top_linkers = rotbonds_vocab_df[rotbonds_vocab_df['type'] == 'linker'].head(10)['frag'].tolist()
print(f"Rotatable bonds vocabulary created. Top 10 arms and linkers extracted.")

# --- 4. Instantiate All Models ---
print("\n" + "="*60)
print("4. Instantiating all generative models...")
print("="*60)
use_cuda = torch.cuda.is_available()
safegpt_model = GenerationModel('safegpt', use_cuda=use_cuda)
molmim_model = GenerationModel('molmim', server_address="g0313:8000") # change
gem_model = GenerationModel('gem', model_path='/nas/longleaf/home/shuhang/argo/argo/gen_models/pretrained/gem_chembl.pt', use_cuda=use_cuda)
f_rag_model = GenerationModel('f-rag', 
                              vocab=frag_vocab_for_frag.get_vocab(),
                              injection_model_path="/nas/longleaf/home/shuhang/argo/argo/gen_models/pretrained/model.safetensors",
                              frag_population_size=100,
                              mol_population_size=500,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=200,
                              use_cuda=use_cuda)
print("All four generative models instantiated.")

# --- 5. Define All Generation Tasks (Tasks Never Get Updated) ---
print("\n" + "="*60)
print("5. Defining all generation tasks...")
print("="*60)
tasks = []
n_de_novo = 1000
batch_size = 100

# SAFE-GPT Tasks
tasks.append(('SAFE-GPT De Novo', safegpt_model, GenerationTask(mode='de_novo', config={'n_samples': n_de_novo, 'batch_size': batch_size, 'sanitize': True})))
tasks.append(('SAFE-GPT Scaffold Decoration', safegpt_model, GenerationTask(mode='scaffold_decoration', scaffold=top_linkers, config={'n_samples': n_de_novo, 'batch_size': batch_size, 'sanitize': True, 'processing_mode': 'sample'})))
tasks.append(('SAFE-GPT Linker Generation', safegpt_model, GenerationTask(mode='linker_generation', fragments=top_arms, config={'n_samples': n_de_novo, 'batch_size': batch_size, 'sanitize': True, 'processing_mode': 'sample'})))

# MolMIM Task
tasks.append(('MolMiM Biased Generation', molmim_model, GenerationTask(mode='property_optimization', seed_smiles=hdg_smiles, config={'n_samples': n_de_novo, 'batch_size': batch_size, 'processing_mode': 'sample'})))

# GEM Tasks
tasks.append(('GEM De Novo', gem_model, GenerationTask(mode='de_novo', config={'n_samples': n_de_novo, 'batch_size': batch_size})))
tasks.append(('GEM Fine-tuned', gem_model, GenerationTask(mode='biased_generation', seed_smiles=hdg_smiles, config={'n_samples': n_de_novo, 'batch_size': batch_size})))

# f-RAG Tasks
tasks.append(('f-RAG Linker Generation', f_rag_model, GenerationTask(mode='linker_generation', config={'n_samples': n_de_novo})))
tasks.append(('f-RAG Scaffold Decoration', f_rag_model, GenerationTask(mode='scaffold_decoration', config={'n_samples': n_de_novo})))
print(f"{len(tasks)} generation tasks created.")

# --- 6. Start Generation Loop ---
print("\n" + "="*60)
print("6. Starting generation loop...")
print("="*60)

mol_population = []
chemeleon_filter = ChemeleonFilterModel(positive_controls=hdg_smiles)
collect_n = 3000
iter = 0

while len(mol_population) < collect_n:
    iter_start_time = time.time()
    iter += 1
    print(f"\n--- Iteration {iter} (Population: {len(mol_population)}/{collect_n}) ---")

    # 7. Generate All Tasks
    all_results = []
    for task_name, model, task in tasks:
        result = run_generation_task(model, task, task_name)
        all_results.append(result)

    # 8. Drop Duplicate SMILES
    generated_smiles = []
    for result in all_results:
        if result['success']:
            generated_smiles.extend(result['results'])
    
    initial_count = len(generated_smiles)
    unique_generated_smiles = list(set(generated_smiles))
    unique_count = len(unique_generated_smiles)
    print(f"Generated {initial_count} molecules, dropped {initial_count - unique_count} duplicates.")

    # 9. Filter using ChemeleonFilterModel
    filtered_smiles_scores = chemeleon_filter.filter(unique_generated_smiles)
    print(f"Filtered {unique_count} unique molecules, {len(filtered_smiles_scores)} passed Chemeleon filter.")

    # 10. Add new SMILES to mol_population
    new_mols_for_population = [(s, score) for s, score in filtered_smiles_scores if s not in [mol[0] for mol in mol_population]]
    mol_population.extend(new_mols_for_population)
    print(f"Added {len(new_mols_for_population)} new molecules to the population.")

    # 11. Use reproduce to get new SMILES
    if mol_population:
        reproduce_start_time = time.time()
        mol_population_smiles = [mol[0] for mol in mol_population]
        reproduced_mols = reproduce(mol_population_smiles, mutation_rate=0.1)
        print(f"Generated {len(reproduced_mols)} molecules via reproduce in {time.time() - reproduce_start_time:.2f}s.")

        # 12. Filter again using ChemeleonFilterModel
        reproduced_filtered = chemeleon_filter.filter(reproduced_mols)
        print(f"Filtered {len(reproduced_mols)} reproduced molecules, {len(reproduced_filtered)} passed.")

        new_reproduced_for_population = [(s, score) for s, score in reproduced_filtered if s not in [mol[0] for mol in mol_population]]
        mol_population.extend(new_reproduced_for_population)
        print(f"Added {len(new_reproduced_for_population)} new reproduced molecules to the population.")

    # Write mol population to .smi file for this iteration
    with open(f'generated_mols_{iter}.smi', 'w') as f:
        for smiles, score in mol_population:
            f.write(f"{smiles}\t{score}\n")
    print(f"Wrote current population to generated_mols_{iter}.smi")

    iter_duration = time.time() - iter_start_time
    print(f"--- Iteration {iter} finished in {iter_duration:.2f} seconds. Population size: {len(mol_population)} ---")

# --- 13. Exit while loop when enough is generated ---
print("\n" + "="*60)
print(f"Generation complete. Final population size: {len(mol_population)}")
print("="*60)
