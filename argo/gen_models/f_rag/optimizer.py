"""
This script defines the `f_RAG` class, a powerful tool for de novo molecule
generation using a hybrid approach. It combines a deep learning model for intelligent
fragment assembly with a genetic algorithm for evolutionary optimization.

Dependencies:
- argo (a specialized library for generative chemistry)
- rdkit-pypi
- pandas
- numpy
"""

import os
import re
import random
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional

from argo.gen_models.f_rag.fusion.sample import SAFEFusionDesign
from argo.gen_models.f_rag.fusion.slicer import MolSlicer
import argo.gen_models.f_rag.ga.crossover as co
from argo.gen_models.f_rag.ga.ga import reproduce


class f_RAG:
    """
    This class orchestrates a hybrid strategy for de novo molecule design. It combines:
    1. A deep learning model (SAFEFusion) for intelligent fragment assembly.
    2. A Genetic Algorithm (GA) for evolving high-scoring molecules.
    
    The core idea is to maintain populations of high-quality fragments ("arms" with one
    attachment point and "linkers" with two) and molecules. New molecules are generated
    by combining fragments, and successful molecules are broken down into their
    constituent fragments to enrich the fragment populations over time.
    """

    def __init__(
        self,
        vocab_path: str,
        results_path: str = 'output/frag_run_results.csv',
        injection_model_path: Optional[str] = None,
        frag_population_size: int = 50,
        mol_population_size: int = 100,
        num_safe_per_gen: int = 10,
        num_ga_per_gen: int = 10,
        min_frag_size: int = 1,
        max_frag_size: int = 15,
        min_mol_size: int = 10,
        max_mol_size: int = 50,
        mutation_rate: float = 0.01,
        seed: int = 42,
    ):
        """
        Initializes the f-RAG system with explicit parameters.
        """
        print("Initializing f-RAG model...")
        # --- Store configuration as instance attributes ---
        self.vocab_path = vocab_path
        self.results_path = results_path
        self.injection_model_path = injection_model_path
        self.frag_population_size = frag_population_size
        self.mol_population_size = mol_population_size
        self.num_safe_per_gen = num_safe_per_gen
        self.num_ga_per_gen = num_ga_per_gen
        self.min_frag_size = min_frag_size
        self.max_frag_size = max_frag_size
        self.min_mol_size = min_mol_size
        self.max_mol_size = max_mol_size
        self.mutation_rate = mutation_rate
        self.seed = seed
        
        # --- Model and Tool Initialization ---
        self.designer = SAFEFusionDesign.load_default()
        
        if self.injection_model_path:
            self.designer.load_fuser(self.injection_model_path)
            print(f"Loaded custom fuser model from {self.injection_model_path}.")

        self.slicer = MolSlicer(shortest_linker=True)

        # --- Population Initialization ---
        self.molecule_population = []
        self.arm_population = []
        self.linker_population = []
        self.set_initial_population(self.vocab_path)

        # --- Configuration Settings ---
        co.MIN_SIZE, co.MAX_SIZE = self.min_mol_size, self.max_mol_size
        
        self.output_filepath = self.results_path
        output_dir = os.path.dirname(self.output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_filepath}")

    def attach(self, fragment_smiles_1, fragment_smiles_2):
        """Chemically joins two fragments together at their attachment points."""
        reaction = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mol1 = Chem.MolFromSmiles(fragment_smiles_1)
        mol2 = Chem.MolFromSmiles(fragment_smiles_2)
        products = reaction.RunReactants((mol1, mol2))
        random_product_idx = np.random.randint(len(products))
        return Chem.MolToSmiles(products[random_product_idx][0])

    def fragmentize(self, molecule_smiles):
        """Breaks a molecule down into its constituent chemical fragments."""
        try:
            fragments = set()
            for safe_fragment_mol in self.slicer(molecule_smiles):
                if safe_fragment_mol is None:
                    continue
                fragment_smiles = Chem.MolToSmiles(safe_fragment_mol)
                fragment_smiles = re.sub(r'\[\d+\*\]', '[1*]', fragment_smiles)
                if fragment_smiles.count('*') in {1, 2}:
                    fragments.add(fragment_smiles)
            
            valid_fragments = [
                frag for frag in fragments
                if self.min_frag_size <= Chem.MolFromSmiles(frag).GetNumAtoms() <= self.max_frag_size
            ]
            return valid_fragments
        except Exception:
            return None

    def set_initial_population(self, vocabulary_path):
        """Loads the initial fragment populations from a CSV file."""
        print(f"Loading initial fragment vocabulary from {vocabulary_path}...")
        try:
            vocabulary_df = pd.read_csv(vocabulary_path)
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {vocabulary_path}. Cannot set initial population.")
            return

        vocabulary_df = vocabulary_df[vocabulary_df['size'] >= self.min_frag_size]
        vocabulary_df = vocabulary_df[vocabulary_df['size'] <= self.max_frag_size]
        scores = vocabulary_df.get('score', [0.0] * len(vocabulary_df))

        for score, fragment_smiles in zip(scores, vocabulary_df['frag']):
            if fragment_smiles.count('*') == 1:
                self.arm_population.append((score, fragment_smiles))
            else:
                self.linker_population.append((score, fragment_smiles))
            if (len(self.arm_population) >= self.frag_population_size and
                len(self.linker_population) >= self.frag_population_size):
                break
        
        self.arm_population = self.arm_population[:self.frag_population_size]
        self.linker_population = self.linker_population[:self.frag_population_size]
        print(f"Initialized with {len(self.arm_population)} arms and {len(self.linker_population)} linkers.")

    def update_population(self, scores, new_molecule_smiles_list):
        """Updates all populations with new, high-scoring individuals."""
        new_molecules = list(set(zip(scores, new_molecule_smiles_list)))
        self.molecule_population.extend(new_molecules)
        self.molecule_population.sort(reverse=True, key=lambda x: x[0])
        self.molecule_population = self.molecule_population[:self.mol_population_size]

        existing_arms = {frag for _, frag in self.arm_population}
        existing_linkers = {frag for _, frag in self.linker_population}
        for score, smiles in zip(scores, new_molecule_smiles_list):
            new_fragments = self.fragmentize(smiles)
            if new_fragments:
                for fragment_smiles in new_fragments:
                    num_attachments = fragment_smiles.count('*')
                    if num_attachments == 1 and fragment_smiles not in existing_arms:
                        self.arm_population.append((score, fragment_smiles))
                        existing_arms.add(fragment_smiles)
                    elif num_attachments == 2 and fragment_smiles not in existing_linkers:
                        self.linker_population.append((score, fragment_smiles))
                        existing_linkers.add(fragment_smiles)

        self.arm_population.sort(reverse=True, key=lambda x: x[0])
        self.linker_population.sort(reverse=True, key=lambda x: x[0])
        self.arm_population = self.arm_population[:self.frag_population_size]
        self.linker_population = self.linker_population[:self.frag_population_size]

    def generate(self, num_to_generate):
        """Generates new molecules using the deep learning model."""
        generated_molecules = []
        max_attempts, attempts = num_to_generate * 10, 0
        while len(generated_molecules) < num_to_generate and attempts < max_attempts:
            attempts += 1
            try:
                if random.random() < 0.5:
                    arm_frag_1, arm_frag_2 = random.sample([frag for _, frag in self.arm_population], 2)
                    self.designer.frags = [frag for _, frag in self.linker_population]
                    smiles = self.designer.linker_generation(arm_frag_1, arm_frag_2, n_samples_per_trial=1, random_seed=self.seed)[0]
                else:
                    arm_frag = random.choice([frag for _, frag in self.arm_population])
                    linker_frag = random.choice([frag for _, frag in self.linker_population])
                    motif = re.sub(r'\[1\*\]', '[*]', self.attach(arm_frag, linker_frag))
                    self.designer.frags = [frag for _, frag in self.arm_population]
                    smiles = self.designer.motif_extension(motif, n_samples_per_trial=1, random_seed=self.seed)[0]
                    smiles = sorted(smiles.split('.'), key=len)[-1]
                
                if hasattr(self.designer, 'decode'):
                    smiles = self.designer.decode(smiles)
                mol = Chem.MolFromSmiles(smiles)
                if mol and self.min_mol_size <= mol.GetNumAtoms() <= self.max_mol_size:
                    generated_molecules.append(smiles)
            except Exception:
                continue
        return generated_molecules

    def record(self, molecule_smiles_list, scores):
        """Appends molecules and their scores to the output CSV file."""
        with open(self.output_filepath, 'a', newline='') as f:
            for smiles, score in zip(molecule_smiles_list, scores):
                f.write(f'"{smiles}",{score}\n')


if __name__ == "__main__":
    def evaluate_qed(smiles_list):
        """A simple QED evaluation function for demonstration."""
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                scores.append(float(Chem.QED.default(mol)) if mol else -1.0)
            except:
                scores.append(-1.0)
        return scores

    # --- 1. Set up Configuration ---
    config = {
        'vocab_path': 'my_fragment_vocab.csv',
        'results_path': 'output/frag_run_results.csv',
        'mol_population_size': 100,
        'num_generations': 5,
        'seed': 42
    }
    
    if not os.path.exists(config['vocab_path']):
        print(f"Creating a dummy vocabulary file at '{config['vocab_path']}' for demonstration.")
        dummy_data = {'frag': ['c1ccccc1[*:1]', 'C[*:1]', '[*:1]CC[*:2]'], 'size': [7, 2, 4]}
        pd.DataFrame(dummy_data).to_csv(config['vocab_path'], index=False)

    random.seed(config['seed'])
    np.random.seed(config['seed'])

    # --- 2. Initialize the f-RAG Model with keyword arguments ---
    model = f_RAG(**config)

    # --- 3. Run the Generative Loop ---
    print("\n--- Starting Generative Process ---")
    for generation in range(model.num_generations):
        print(f"\n>> Generation {generation + 1}/{model.num_generations}")
        
        safe_smiles = model.generate(num_to_generate=model.num_safe_per_gen)
        ga_smiles = []
        if len(model.molecule_population) >= model.num_ga_per_gen:
            ga_smiles = [reproduce(model.molecule_population, model.mutation_rate) for _ in range(model.num_ga_per_gen)]
        
        all_new_smiles = list(filter(None, safe_smiles + ga_smiles))
        if not all_new_smiles:
            print("No valid molecules were generated. Skipping.")
            continue
            
        scores = evaluate_qed(all_new_smiles)
        model.update_population(scores, all_new_smiles)
        model.record(all_new_smiles, scores)

        if model.molecule_population:
            best_score, best_smiles = model.molecule_population[0]
            print(f"Current best molecule: {best_smiles} (Score: {best_score:.2f})")
        else:
            print("No valid molecules in population yet.")

    print("\n--- Generative Process Finished ---")