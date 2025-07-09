import os
import pandas as pd
from rdkit import Chem
from argo.gen_models.api import GenerationModel, GenerationTask
from argo.vocab import FragmentVocabulary
from tqdm import tqdm
import torch
import time
from typing import List, Dict, Any

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
    max_fragments=2000,
    lower_is_better=lower_is_better
)

# Save the vocabulary
vocab.save('fragment_scores_init.csv')
print('Fragment statistics written to fragment_scores.csv')

vocab.save_state('fragment_scores_init.pt')
print('Fragment state written to fragment_scores_init.pt')

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
                              frag_population_size=500,
                              mol_population_size=200,
                              min_frag_size=5,
                              max_frag_size=30,
                              min_mol_size=10,
                              max_mol_size=150,
                              mutation_rate=0.01,
                              use_cuda=use_cuda
)

print('All four generative models instantiated.')

# 7. Generation Tasks with Timing
def run_generation_task(model: GenerationModel, task: GenerationTask, task_name: str) -> Dict[str, Any]:
    """
    Run a generation task and track timing.
    
    Args:
        model: The generation model to use
        task: The generation task configuration
        task_name: Name for the task (for logging)
        
    Returns:
        Dictionary with results and timing information
    """
    print(f"\n=== Running {task_name} ===")
    start_time = time.time()
    
    try:
        results = model.generate(task)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {task_name} completed in {duration:.2f} seconds")
        print(f"  Generated {len(results)} molecules")
        
        return {
            'task_name': task_name,
            'model_type': type(model).__name__,
            'results': results,
            'duration': duration,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"✗ {task_name} failed after {duration:.2f} seconds: {e}")
        
        return {
            'task_name': task_name,
            'model_type': type(model).__name__,
            'results': [],
            'duration': duration,
            'success': False,
            'error': str(e)
        }

# Get top compounds
top_100_compounds = df.head(100)['smiles'].tolist()
top_1_percent = df.head(int(len(df) * 0.01))['smiles'].tolist()

# Define generation tasks
tasks = []

# 1. SAFE-GPT: De novo generation
n_de_novo = 1000  # You can adjust this value
safegpt_task = GenerationTask(
    mode='de_novo',
    config={
        'n_samples': n_de_novo,
        'n_trials': 1,
        'sanitize': True
    }
)
tasks.append(('SAFE-GPT De Novo', safegpt_model, safegpt_task))

# 2. MolMiM: Biased generation based on top 100 compounds
# We'll run 100 batches of 10 samples each to get 1000 total
molmim_tasks = []
for i in range(100):  # 100 batches
    seed_smiles = top_100_compounds[i % len(top_100_compounds)]  # Cycle through top 100
    molmim_task = GenerationTask(
        mode='biased_generation',
        seed_smiles=seed_smiles,
        config={
            'n_samples': 10,
            'min_similarity': 0.7,
            'scaled_radius': 1.0
        }
    )
    molmim_tasks.append((f'MolMiM Biased Generation Batch {i+1}', molmim_model, molmim_task))

# 3. GEM: De novo generation (1000), then fine-tune and generate 1000 more
gem_de_novo_task = GenerationTask(
    mode='de_novo',
    config={
        'n_samples': 1000,
        'max_length': 100
    }
)
tasks.append(('GEM De Novo', gem_model, gem_de_novo_task))

# GEM fine-tune task (will be run after initial generation)
gem_finetune_task = GenerationTask(
    mode='biased_generation',
    seed_smiles=top_1_percent,
    config={
        'n_samples': 1000,
        'max_length': 100
    }
)
tasks.append(('GEM Fine-tuned', gem_model, gem_finetune_task))

# 4. f-RAG: Linker generation and scaffold decoration
frag_linker_task = GenerationTask(
    mode='linker_generation',
    config={
        'n_samples': 1000,
        'random_seed': 42
    }
)
tasks.append(('f-RAG Linker Generation', f_rag_model, frag_linker_task))

frag_scaffold_task = GenerationTask(
    mode='scaffold_decoration',
    config={
        'n_samples': 1000,
        'random_seed': 42
    }
)
tasks.append(('f-RAG Scaffold Decoration', f_rag_model, frag_scaffold_task))

# Run all tasks and collect results
print(f"\n{'='*60}")
print("STARTING GENERATION TASKS")
print(f"{'='*60}")

all_results = []

# Run SAFE-GPT task
result = run_generation_task(safegpt_model, safegpt_task, 'SAFE-GPT De Novo')
all_results.append(result)

# Run MolMiM tasks
print(f"\n=== Running MolMiM Biased Generation (100 batches of 10 samples each) ===")
molmim_start = time.time()
molmim_all_results = []
for i, (task_name, model, task) in enumerate(molmim_tasks):
    result = run_generation_task(model, task, f'MolMiM Batch {i+1}/100')
    molmim_all_results.append(result)
    if result['success']:
        print(f"  Batch {i+1}: {len(result['results'])} molecules in {result['duration']:.2f}s")
    else:
        print(f"  Batch {i+1}: FAILED - {result.get('error', 'Unknown error')}")

molmim_total_time = time.time() - molmim_start
molmim_total_molecules = sum(len(r['results']) for r in molmim_all_results if r['success'])
print(f"✓ MolMiM completed: {molmim_total_molecules} molecules in {molmim_total_time:.2f} seconds")

all_results.append({
    'task_name': 'MolMiM Biased Generation (All Batches)',
    'model_type': 'MolMIMGenerator',
    'results': [mol for r in molmim_all_results if r['success'] for mol in r['results']],
    'duration': molmim_total_time,
    'success': True,
    'batch_results': molmim_all_results
})

# Run GEM de novo task
result = run_generation_task(gem_model, gem_de_novo_task, 'GEM De Novo')
all_results.append(result)

# Run GEM fine-tune task
result = run_generation_task(gem_model, gem_finetune_task, 'GEM Fine-tuned')
all_results.append(result)

# Run f-RAG tasks
result = run_generation_task(f_rag_model, frag_linker_task, 'f-RAG Linker Generation')
all_results.append(result)

result = run_generation_task(f_rag_model, frag_scaffold_task, 'f-RAG Scaffold Decoration')
all_results.append(result)

# Summary
print(f"\n{'='*60}")
print("GENERATION TASK SUMMARY")
print(f"{'='*60}")

total_time = sum(r['duration'] for r in all_results)
total_molecules = sum(len(r['results']) for r in all_results if r['success'])
successful_tasks = sum(1 for r in all_results if r['success'])
failed_tasks = len(all_results) - successful_tasks

print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Total molecules generated: {total_molecules}")
print(f"Successful tasks: {successful_tasks}")
print(f"Failed tasks: {failed_tasks}")

print(f"\nDetailed Results:")
for result in all_results:
    status = "✓" if result['success'] else "✗"
    molecules = len(result['results']) if result['success'] else 0
    print(f"  {status} {result['task_name']}: {molecules} molecules in {result['duration']:.2f}s")

# Save results to files
print(f"\nSaving results...")

# Save all generated molecules
all_molecules = []
for result in all_results:
    if result['success']:
        all_molecules.extend(result['results'])

if all_molecules:
    molecules_df = pd.DataFrame({'smiles': all_molecules})
    molecules_df.to_csv('generated_molecules.csv', index=False)
    print(f"✓ Saved {len(all_molecules)} generated molecules to 'generated_molecules.csv'")

# Save timing results
timing_results = []
for result in all_results:
    timing_results.append({
        'task_name': result['task_name'],
        'model_type': result['model_type'],
        'duration_seconds': result['duration'],
        'molecules_generated': len(result['results']) if result['success'] else 0,
        'success': result['success'],
        'error': result.get('error', '')
    })

timing_df = pd.DataFrame(timing_results)
timing_df.to_csv('generation_timing.csv', index=False)
print(f"✓ Saved timing results to 'generation_timing.csv'")

print(f"\n{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")

# 8. Build and Apply Filter Model
print(f"\n{'='*60}")
print("BUILDING AND APPLYING FILTER MODEL")
print(f"{'='*60}")

from argo.filter_models import SmilesFilterModel
import numpy as np

def build_filter_model(df: pd.DataFrame, smiles_col: str = 'smiles', score_col: str = 'score', 
                      threshold_percentile: float = 10.0) -> SmilesFilterModel:
    """
    Build a filter model using the CHD1 data.
    
    Args:
        df: DataFrame with SMILES and scores
        smiles_col: Column name for SMILES
        score_col: Column name for scores
        threshold_percentile: Percentile to use as threshold for good/bad classification
        
    Returns:
        Trained SmilesFilterModel
    """
    print(f"Building filter model using {len(df)} compounds...")
    
    # Get SMILES and scores
    smiles = df[smiles_col].tolist()
    scores = df[score_col].values
    
    # Create binary labels based on percentile threshold
    threshold = np.percentile(scores, threshold_percentile)
    labels = (scores <= threshold).astype(int)  # 1 for good compounds (lower scores), 0 for bad
    
    print(f"  Score threshold: {threshold:.4f} (top {threshold_percentile}% of compounds)")
    print(f"  Good compounds (label=1): {np.sum(labels)}")
    print(f"  Bad compounds (label=0): {len(labels) - np.sum(labels)}")
    
    # Train the filter model
    filter_model = SmilesFilterModel()
    filter_model.train(smiles, labels)
    
    print(f"✓ Filter model trained successfully")
    return filter_model

def apply_filter_to_results(filter_model: SmilesFilterModel, all_results: List[Dict[str, Any]], 
                          conf_thresh: float = 0.6) -> Dict[str, Dict[str, Any]]:
    """
    Apply the filter model to all generated compounds and report results.
    
    Args:
        filter_model: Trained SmilesFilterModel
        all_results: List of generation results
        conf_thresh: Confidence threshold for filtering
        
    Returns:
        Dictionary with filtering results for each task
    """
    print(f"\nApplying filter model to generated compounds (confidence threshold: {conf_thresh})...")
    
    filtering_results = {}
    
    for result in all_results:
        if not result['success'] or len(result['results']) == 0:
            continue
            
        task_name = result['task_name']
        molecules = result['results']
        
        print(f"\n  Filtering {task_name} ({len(molecules)} molecules)...")
        
        # Apply filter
        try:
            filtered_molecules = filter_model.filter(molecules, conf_thresh=conf_thresh)
            pass_rate = len(filtered_molecules) / len(molecules) * 100
            
            print(f"    ✓ {len(filtered_molecules)}/{len(molecules)} molecules passed filter ({pass_rate:.1f}%)")
            
            filtering_results[task_name] = {
                'total_molecules': len(molecules),
                'passed_filter': len(filtered_molecules),
                'pass_rate': pass_rate,
                'filtered_molecules': filtered_molecules,
                'duration': result['duration'],
                'model_type': result['model_type']
            }
            
        except Exception as e:
            print(f"    ✗ Error filtering {task_name}: {e}")
            filtering_results[task_name] = {
                'total_molecules': len(molecules),
                'passed_filter': 0,
                'pass_rate': 0.0,
                'filtered_molecules': [],
                'duration': result['duration'],
                'model_type': result['model_type'],
                'error': str(e)
            }
    
    return filtering_results

# Build the filter model
filter_model = build_filter_model(df, 'smiles', 'score', threshold_percentile=20.0)

# Save the filter model
filter_model.save('chd1_filter_model.joblib')
print(f"✓ Filter model saved to 'chd1_filter_model.joblib'")

# Apply filter to all results
filtering_results = apply_filter_to_results(filter_model, all_results, conf_thresh=0.6)

# Summary of filtering results
print(f"\n{'='*60}")
print("FILTERING RESULTS SUMMARY")
print(f"{'='*60}")

total_generated = sum(r['total_molecules'] for r in filtering_results.values())
total_passed = sum(r['passed_filter'] for r in filtering_results.values())
overall_pass_rate = total_passed / total_generated * 100 if total_generated > 0 else 0

print(f"Overall results:")
print(f"  Total molecules generated: {total_generated}")
print(f"  Total molecules passed filter: {total_passed}")
print(f"  Overall pass rate: {overall_pass_rate:.1f}%")

print(f"\nResults by task:")
for task_name, result in filtering_results.items():
    status = "✓" if result['passed_filter'] > 0 else "✗"
    print(f"  {status} {task_name}:")
    print(f"    Generated: {result['total_molecules']}")
    print(f"    Passed filter: {result['passed_filter']}")
    print(f"    Pass rate: {result['pass_rate']:.1f}%")
    print(f"    Duration: {result['duration']:.2f}s")
    if 'error' in result:
        print(f"    Error: {result['error']}")

# Save filtering results
print(f"\nSaving filtering results...")

# Save filtered molecules
all_filtered_molecules = []
for task_name, result in filtering_results.items():
    if result['passed_filter'] > 0:
        for mol in result['filtered_molecules']:
            all_filtered_molecules.append({
                'smiles': mol,
                'task': task_name,
                'model_type': result['model_type']
            })

if all_filtered_molecules:
    filtered_df = pd.DataFrame(all_filtered_molecules)
    filtered_df.to_csv('filtered_molecules.csv', index=False)
    print(f"✓ Saved {len(all_filtered_molecules)} filtered molecules to 'filtered_molecules.csv'")

# Save filtering statistics
filtering_stats = []
for task_name, result in filtering_results.items():
    filtering_stats.append({
        'task_name': task_name,
        'model_type': result['model_type'],
        'total_molecules': result['total_molecules'],
        'passed_filter': result['passed_filter'],
        'pass_rate': result['pass_rate'],
        'duration_seconds': result['duration'],
        'error': result.get('error', '')
    })

filtering_df = pd.DataFrame(filtering_stats)
filtering_df.to_csv('filtering_results.csv', index=False)
print(f"✓ Saved filtering statistics to 'filtering_results.csv'")

print(f"\n{'='*60}")
print("FILTERING COMPLETE")
print(f"{'='*60}")