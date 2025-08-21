import os
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import torch, joblib
import time, random
from typing import List, Dict, Any
from argo.utils import clean_smiles, run_generation_task

from argo.gen_models import GenerationModel, GenerationTask
from argo.vocab import FragmentVocabulary

from argo.frag_utils import find_connected_rotatable_bond_ends

from argo.filter_models import ChemeleonFilterModel
import argo.gen_models.f_rag.ga.crossover as co
from argo.gen_models.f_rag.ga.ga import reproduce

# Load HGD compounds
df = pd.read_csv('../benchmark/class1_compounds.csv')
df = df[df['Compound Class'] == 'HGD']
lower_is_better = True

df = df.sort_values('ic50', ascending=lower_is_better)
print(f"Loaded {len(df)} HGD compounds")
print(f"IC50 range: {df['ic50'].min():.3f} - {df['ic50'].max():.3f} μM")

print("\nCreating fragment vocabulary...")
vocab = FragmentVocabulary(
    slicer=find_connected_rotatable_bond_ends, # New slicer
    data=df,
    smiles_col='SMILES',
    score_col='ic50',
    scoring_method='average',
    min_frag_size=5,
    max_frag_size=30,
    min_count=1,
    max_fragments=50,
    lower_is_better=lower_is_better
)

# TODO: have a f-rag vocab

vocab.save('pptt_hgd_fragment_scores.csv')
vocab.save_state('pptt_hgd_fragment_scores.pt')

# Instantiate generative models
use_cuda = torch.cuda.is_available()
safegpt_model = GenerationModel('safegpt', use_cuda=use_cuda)
molmim_model = GenerationModel('molmim', server_address="g0313:8000") # change
gem_model = GenerationModel('gem', model_path='/nas/longleaf/home/shuhang/argo/argo/gen_models/pretrained/gem_chembl.pt', use_cuda=use_cuda)

f_rag_model = GenerationModel('f-rag', 
                              vocab=vocab.get_vocab(),
                              injection_model_path="/nas/longleaf/home/shuhang/argo/argo/gen_models/pretrained/model.safetensors",
                              frag_population_size=100,
                              mol_population_size=500,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=200,
                              use_cuda=use_cuda
)

print('All four generative models instantiated.')

# Get top fragments from vocabulary
vocab_df = vocab.get_vocab()
top_arms = vocab_df[vocab_df['type'] == 'arm'].head(10)['frag'].tolist()
top_linkers = vocab_df[vocab_df['type'] == 'linker'].head(10)['frag'].tolist()

# Define generation tasks
tasks = []
n_de_novo = 1000
batch_size = 100

# SAFE-GPT: De novo generation
safegpt_task = GenerationTask(
    mode='de_novo',
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size,
        'sanitize': True
    }
)
tasks.append(('SAFE-GPT De Novo', safegpt_model, safegpt_task))

# SAFE-GPT: Scaffold decoration using top linkers (batch processing)
safegpt_scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    scaffold=top_linkers,  # Pass list of scaffolds
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size,
        'sanitize': True,
        'processing_mode': 'sample'
    }
)
tasks.append(('SAFE-GPT Scaffold Decoration', safegpt_model, safegpt_scaffold_task))

# SAFE-GPT: Linker generation using all top arms (sample mode)
safegpt_linker_task = GenerationTask(
    mode='linker_generation',
    fragments=top_arms,  # Pass the whole list of arms
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size,
        'sanitize': True,
        'processing_mode': 'sample'
    }
)
tasks.append(('SAFE-GPT Linker Generation (Sample)', safegpt_model, safegpt_linker_task))

hdg_smiles = clean_smiles(df['SMILES'].tolist())

molmim_task = GenerationTask(
    mode='property_optimization',
    seed_smiles=hdg_smiles,
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size,
        'processing_mode': 'sample'
    }
)
tasks.append(('MolMiM Biased Generation', molmim_model, molmim_task))

gem_de_novo_task = GenerationTask(
    mode='de_novo',
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size,
    }
)
tasks.append(('GEM De Novo', gem_model, gem_de_novo_task))

gem_finetune_task = GenerationTask(
    mode='biased_generation',
    seed_smiles=hdg_smiles,
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size
    }
)
tasks.append(('GEM Fine-tuned', gem_model, gem_finetune_task))

frag_linker_task = GenerationTask(
    mode='linker_generation',
    config={
        'n_samples': n_de_novo
    }
)
tasks.append(('f-RAG Linker Generation', f_rag_model, frag_linker_task))

frag_scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    config={
        'n_samples': n_de_novo
    }
)
tasks.append(('f-RAG Scaffold Decoration', f_rag_model, frag_scaffold_task))

'''
frag_optimize_task = GenerationTask(
    mode='property_optimization',
    objective='qed',
    config={
        'n_samples': n_de_novo,
        'batch_size': batch_size * 2,
        'max_iter': 20
    }
)
tasks.append(('f-RAG Property Optimization (QED)', f_rag_model, frag_optimize_task))
'''

# Run all tasks and collect results
print(f"\n{'='*60}")
print("STARTING GENERATION TASKS")
print(f"{'='*60}")

# Set mol population
mol_population = []

# Create an empty vocabulary library outside the while loop
print("\nCreating empty vocabulary library for cumulative fragment collection...")
cumulative_vocab = FragmentVocabulary(
    slicer=find_connected_rotatable_bond_ends,
    data=pd.DataFrame(columns=['smiles', 'score']),  # Empty DataFrame
    smiles_col='smiles',
    score_col='score',
    scoring_method='average',
    min_frag_size=5,
    max_frag_size=30,
    min_count=1,
    max_fragments=50,
    lower_is_better=True
)

collect_n = 3000
iter = 0
while len(mol_population) < collect_n:
    all_results = []

    # Run all tasks in a single loop
    for task_name, model, task in tasks:
        result = run_generation_task(model, task, task_name)
        all_results.append(result)

    # Filter results using Chemeleon filter model
    chemeleon_filter = ChemeleonFilterModel(positive_controls=hdg_smiles)

    # Extract SMILES from successful results with tracking
    all_smiles = []
    smiles_to_task = {}  # Track which task each SMILES came from
    
    for result in all_results:
        if result['success']:
            task_name = result['task_name']
            model_type = result['model_type']
            task_key = f"{model_type}_{task_name}"
            
            for smiles in result['results']:
                all_smiles.append(smiles)
                smiles_to_task[smiles] = task_key

    # Drop duplicate smiles
    all_smiles_unique = list(set(all_smiles))
    print(f"Dropped {len(all_smiles) - len(all_smiles_unique)} duplicate smiles")

    # Filter the SMILES using Chemeleon
    filtered_results = chemeleon_filter.filter(all_smiles_unique)
    filtered_smiles = [smiles for smiles, score in filtered_results]

    # Analyze filtering performance by task
    task_stats = {}
    for smiles in all_smiles:
        task_key = smiles_to_task[smiles]
        if task_key not in task_stats:
            task_stats[task_key] = {'total': 0, 'passed': 0}
        task_stats[task_key]['total'] += 1
        if smiles in filtered_smiles:
            task_stats[task_key]['passed'] += 1

    # Print overall statistics
    print(f"\n=== Overall Filtering Results ===")
    print(f"Total molecules: {len(all_smiles)}")
    print(f"Filtered molecules: {len(filtered_results)}")
    print(f"Overall filtering rate: {len(filtered_results) / len(all_smiles) * 100:.2f}%")

    # Print detailed statistics by task
    print(f"\n=== Filtering Performance by Task ===")
    print(f"{'Task':<40} {'Total':<8} {'Passed':<8} {'Rate (%)':<10}")
    print("-" * 70)
    
    # Sort by filtering rate (highest first)
    sorted_tasks = sorted(task_stats.items(), 
                         key=lambda x: x[1]['passed']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                         reverse=True)
    
    # Add new molecules to population (with scores)
    new_mols = [(smiles, score) for smiles, score in filtered_results if smiles not in [mol[0] for mol in mol_population]]

    print(f"Adding {len(new_mols)} new molecules to population")
    mol_population.extend(new_mols)

    reproduce_results = []
    
    # Perform reproduce on filtered population (f-RAG style)
    if mol_population:
        mutation_rate = 0.1
        try:
            # Extract just SMILES for reproduce function
            mol_population_smiles = [mol[0] for mol in mol_population]
            reproduced_mols = reproduce(mol_population_smiles, mutation_rate)
            reproduce_results.extend(reproduced_mols)
        except Exception as e:
            print(f"Reproduce failed. Error: {e}")
            pass
    
    # Filter and add reproduce results
    if reproduce_results:
        new_filtered_results = chemeleon_filter.filter(reproduce_results)
        new_filtered_smiles_scores = [(smiles, score) for smiles, score in new_filtered_results if smiles not in [mol[0] for mol in mol_population]]
        mol_population.extend(new_filtered_smiles_scores)
        print(f"Reproduce: {len(reproduce_results)} generated, {len(new_filtered_smiles_scores)} passed filter")

    # Update tasks that have inputs
    for task_name, model, task in tasks:
        if task.config.get('seed_smiles'):
            # Extract SMILES from population (which contains (smiles, score) tuples)
            mol_population_smiles = [mol[0] for mol in mol_population]
            task.config['seed_smiles'] = random.sample(mol_population_smiles, min(10, len(mol_population_smiles)))

    cumulative_vocab.add(pd.DataFrame(filtered_results, columns=['smiles', 'score']), use_tqdm=False)
    # vocab.rescore(lower_is_better=True) # redundant
    cumulative_vocab_df = cumulative_vocab.get_vocab()
    top_arms = cumulative_vocab_df[cumulative_vocab_df['type'] == 'arm'].head(10)['frag'].tolist()
    top_linkers = cumulative_vocab_df[cumulative_vocab_df['type'] == 'linker'].head(10)['frag'].tolist()

    for task_name, model, task in tasks:
        if task.config.get('scaffold'):
            task.config['scaffold'] = top_linkers
        if task.config.get('fragments'):
            task.config['fragments'] = top_arms
    
    for task_key, stats in sorted_tasks:
        rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{task_key:<40} {stats['total']:<8} {stats['passed']:<8} {rate:<10.2f}")

    # Find best performing task
    if sorted_tasks:
        best_task, best_stats = sorted_tasks[0]
        best_rate = best_stats['passed'] / best_stats['total'] * 100
        print(f"\n=== Best Performing Task ===")
        print(f"Task: {best_task}")
        print(f"Filtering rate: {best_rate:.2f}%")
        print(f"Total molecules: {best_stats['total']}")
        print(f"Passed molecules: {best_stats['passed']}")

    iter += 1
    print(f"Population size: {len(mol_population)} after {iter} iterations.")

    # Write mol population to .smi file
    with open(f'generated_mols_{iter}.smi', 'w') as f:
        for smiles, score in mol_population:
            f.write(f"{smiles}\t{score}\n")
    print(f"Wrote mol population to generated_mols_{iter}.smi")

print(f"\n{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")
