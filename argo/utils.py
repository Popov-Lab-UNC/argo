import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from argo.filter_models import SmilesFilterModel

def clean_smiles(smiles_list):
    """Clean SMILES strings by removing spaces, stereochemical annotations, and stripping salts/solvents using RDKit SaltRemover (keep largest fragment)."""
    remover = SaltRemover()
    cleaned = []
    for i, smiles in enumerate(smiles_list):
        cleaned_smiles = smiles.strip().replace(' ', '')
        if '|' in cleaned_smiles:
            parts = cleaned_smiles.split('|')
            cleaned_smiles = parts[0]
        if cleaned_smiles:
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is not None:
                desalted = remover.StripMol(mol)
                if desalted is not None:
                    smiles_main = Chem.MolToSmiles(desalted)
                    cleaned.append(smiles_main)
    return cleaned


def build_filter_model(df: pd.DataFrame, smiles_col: str = 'smiles', score_col: str = 'score', threshold_percentile: float = 10.0):
    """
    Build a filter model using data.
    Args:
        df: DataFrame with SMILES and scores
        smiles_col: Column name for SMILES
        score_col: Column name for scores
        threshold_percentile: Percentile to use as threshold for good/bad classification
    Returns:
        Trained SmilesFilterModel
    """
    if SmilesFilterModel is None:
        raise ImportError("SmilesFilterModel could not be imported. Please check your argo installation.")
    smiles = df[smiles_col].tolist()
    scores = df[score_col].values
    threshold = np.percentile(scores, threshold_percentile)
    labels = (scores <= threshold).astype(int)
    filter_model = SmilesFilterModel()
    filter_model.train(smiles, labels)
    return filter_model


def apply_filter_to_results(filter_model, all_results: List[Dict[str, Any]], conf_thresh: float = 0.6) -> Dict[str, Dict[str, Any]]:
    """
    Apply the filter model to all generated compounds and report results.
    Args:
        filter_model: Trained SmilesFilterModel
        all_results: List of generation results
        conf_thresh: Confidence threshold for filtering
    Returns:
        Dictionary with filtering results for each task
    """
    filtering_results = {}
    for result in all_results:
        if not result['success'] or len(result['results']) == 0:
            continue
        task_name = result['task_name']
        molecules = result['results']
        try:
            filtered_molecules = filter_model.filter(molecules, conf_thresh=conf_thresh)
            pass_rate = len(filtered_molecules) / len(molecules) * 100
            filtering_results[task_name] = {
                'total_molecules': len(molecules),
                'passed_filter': len(filtered_molecules),
                'pass_rate': pass_rate,
                'filtered_molecules': filtered_molecules,
                'duration': result['duration'],
                'model_type': result['model_type']
            }
        except Exception as e:
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


def run_generation_task(model, task, task_name: str) -> Dict[str, Any]:
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