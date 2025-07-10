#!/usr/bin/env python3
"""
Comprehensive test cases for f-RAG generation model.
Tests scaffold decoration, linker generation, and property optimization modes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from tdc import Oracle

# Add the parent directory to the path to import argo modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from argo.gen_models.api import GenerationModel, GenerationTask
from argo.vocab import FragmentVocabulary

def test_f_rag_scaffold_decoration():
    """Test f-RAG scaffold decoration mode."""
    print("\n" + "="*60)
    print("TESTING F-RAG SCAFFOLD DECORATION")
    print("="*60)
    
    # Use the provided vocab file
    vocab_path = str(Path(__file__).parent / "f_rag" / "example_vocab.csv")
    
    f_rag = GenerationModel(
        model_type='f-rag',
        vocab=vocab_path,
        injection_model_path="pretrained/model.safetensors",
        frag_population_size=50,
        mol_population_size=30,
        min_frag_size=1,
        max_frag_size=20,
        min_mol_size=10,
        max_mol_size=150,
        mutation_rate=0.01
    )
    
    task = GenerationTask(
        mode='scaffold_decoration',
        config={
            "n_samples": 10,
            "random_seed": 42
        }
    )
    
    try:
        results = f_rag.generate(task)
        print(f"✓ Scaffold decoration successful: {len(results)} molecules generated")
        
        # Validate results
        valid_molecules = []
        for smiles in results:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(smiles)
        
        print(f"  Valid molecules: {len(valid_molecules)}/{len(results)}")
        if valid_molecules:
            print(f"  Sample results: {valid_molecules[:3]}")
        
        return valid_molecules
        
    except Exception as e:
        print(f"✗ Scaffold decoration failed: {e}")
        return []

def test_f_rag_linker_generation():
    """Test f-RAG linker generation mode."""
    print("\n" + "="*60)
    print("TESTING F-RAG LINKER GENERATION")
    print("="*60)
    
    # Use the provided vocab file
    vocab_path = str(Path(__file__).parent / "f_rag" / "example_vocab.csv")
    
    f_rag = GenerationModel(
        model_type='f-rag',
        vocab=vocab_path,
        injection_model_path="pretrained/model.safetensors",
        frag_population_size=50,
        mol_population_size=30,
        min_frag_size=1,
        max_frag_size=20,
        min_mol_size=10,
        max_mol_size=150,
        mutation_rate=0.01
    )
    
    task = GenerationTask(
        mode='linker_generation',
        config={
            "n_samples": 10,
            "random_seed": 42
        }
    )
    
    try:
        results = f_rag.generate(task)
        print(f"✓ Linker generation successful: {len(results)} molecules generated")
        
        # Validate results
        valid_molecules = []
        for smiles in results:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(smiles)
        
        print(f"  Valid molecules: {len(valid_molecules)}/{len(results)}")
        if valid_molecules:
            print(f"  Sample results: {valid_molecules[:3]}")
        
        return valid_molecules
        
    except Exception as e:
        print(f"✗ Linker generation failed: {e}")
        return []

def test_f_rag_property_optimization():
    """Test f-RAG property optimization mode with QED."""
    print("\n" + "="*60)
    print("TESTING F-RAG PROPERTY OPTIMIZATION (QED)")
    print("="*60)
    
    # Initialize QED oracle
    try:
        qed_oracle = Oracle(name='qed')
        print("✓ QED oracle initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize QED oracle: {e}")
        return []
    
    # Use the provided vocab file
    vocab_path = str(Path(__file__).parent / "f_rag" / "example_vocab.csv")
    
    f_rag = GenerationModel(
        model_type='f-rag',
        vocab=vocab_path,
        injection_model_path="pretrained/model.safetensors",
        frag_population_size=50,
        mol_population_size=30,
        min_frag_size=1,
        max_frag_size=20,
        min_mol_size=10,
        max_mol_size=150,
        mutation_rate=0.01
    )
    
    task = GenerationTask(
        mode='property_optimization',
        objective='qed',
        config={
            "n_samples": 10,
            "random_seed": 42
        }
    )
    
    try:
        results = f_rag.generate(task)
        print(f"✓ Property optimization successful: {len(results)} molecules generated")
        
        # Validate results and calculate QED scores
        valid_molecules = []
        qed_scores = []
        
        for score, smiles in results:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    qed_score = qed_oracle(smiles)
                    valid_molecules.append(smiles)
                    qed_scores.append(qed_score)
                except Exception as e:
                    print(f"  Warning: Could not calculate QED for {smiles}: {e}")
        
        print(f"  Valid molecules: {len(valid_molecules)}/{len(results)}")
        
        if valid_molecules and qed_scores:
            avg_qed = np.mean(qed_scores)
            max_qed = np.max(qed_scores)
            min_qed = np.min(qed_scores)
            
            print(f"  QED Statistics:")
            print(f"    Average QED: {avg_qed:.4f}")
            print(f"    Max QED: {max_qed:.4f}")
            print(f"    Min QED: {min_qed:.4f}")
            
            # Show top 3 molecules by QED
            if len(valid_molecules) >= 3:
                top_indices = np.argsort(qed_scores)[-3:][::-1]
                print(f"  Top 3 molecules by QED:")
                for i, idx in enumerate(top_indices):
                    print(f"    {i+1}. QED: {qed_scores[idx]:.4f} | SMILES: {valid_molecules[idx]}")
        
        return valid_molecules
        
    except Exception as e:
        print(f"✗ Property optimization failed: {e}")
        return []

def run_all_tests():
    """Run all f-RAG tests."""
    print("COMPREHENSIVE F-RAG TESTING")
    print("="*60)
    
    all_results = {}
    
    # Test 1: Scaffold decoration
    scaffold_results = test_f_rag_scaffold_decoration()
    all_results['scaffold_decoration'] = scaffold_results
    
    # Test 2: Linker generation
    linker_results = test_f_rag_linker_generation()
    all_results['linker_generation'] = linker_results
    
    # Test 3: Property optimization
    property_results = test_f_rag_property_optimization()
    all_results['property_optimization'] = property_results
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_molecules = sum(len(results) for results in all_results.values())
    successful_tests = sum(1 for results in all_results.values() if len(results) > 0)
    
    print(f"Total molecules generated: {total_molecules}")
    print(f"Successful tests: {successful_tests}/{len(all_results)}")
    
    for test_name, results in all_results.items():
        status = "✓" if len(results) > 0 else "✗"
        print(f"  {status} {test_name}: {len(results)} molecules")
    
    return all_results

if __name__ == "__main__":
    run_all_tests() 