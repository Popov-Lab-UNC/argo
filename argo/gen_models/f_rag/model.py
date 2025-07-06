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
#from argo.gen_models.f_rag.fusion.slicer import MolSlicer
from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder
from argo.frag_utils import SAFECodec
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
        injection_model_path: str,
        vocab_path: str,
        frag_population_size: int = 50,
        mol_population_size: int = 100,
        min_frag_size: int = 1,
        max_frag_size: int = 15,
        min_mol_size: int = 10,
        max_mol_size: int = 100,
        mutation_rate: float = 0.01,
    ):
        """
        Initializes the f-RAG system with explicit parameters.
        """
        print("Initializing f-RAG model...")
        # --- Store configuration as instance attributes ---
        self.vocab_path = vocab_path
        self.injection_model_path = injection_model_path
        self.frag_population_size = frag_population_size
        self.mol_population_size = mol_population_size
        self.min_frag_size = min_frag_size
        self.max_frag_size = max_frag_size
        self.min_mol_size = min_mol_size
        self.max_mol_size = max_mol_size
        self.mutation_rate = mutation_rate
        
        # --- Model and Tool Initialization ---
        self.designer = SAFEFusionDesign.load_default()
        
        if self.injection_model_path:
            self.designer.load_fuser(self.injection_model_path)
            print(f"Loaded custom fuser model from {self.injection_model_path}.")

        slicer = MolSlicerForSAFEEncoder(shortest_linker=True)
        self.sfcodec = SAFECodec(slicer=slicer, ignore_stereo=True)

        # --- Population Initialization ---
        self.molecule_population = []
        self.arm_population = []
        self.linker_population = []
        self.set_initial_population(self.vocab_path)

        # !!! TODO: Make sure population is up to frag_population_size
        # Rewrite linker and arm population logic

        if len(self.arm_population) < 2 and len(self.linker_population) < 1:
            raise ValueError("Not enough fragments in the population to perform linker generation or motif extension. Please initialize the population with a valid vocabulary.")

        # --- Configuration Settings ---
        co.MIN_SIZE, co.MAX_SIZE = self.min_mol_size, self.max_mol_size

    '''
    def prepare_attach(self, smiles):
        smiles = re.sub(r'\[\*:\d+\]', '*', smiles)
        return re.sub(r'\*', '[1*]', smiles)

    def attach(self, fragment_smiles_1, fragment_smiles_2):
        """Chemically joins two fragments together at their attachment points."""
        reaction = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mol1 = Chem.MolFromSmiles(fragment_smiles_1)
        mol2 = Chem.MolFromSmiles(fragment_smiles_2)
        print(f'Attaching {Chem.MolToSmiles(mol1)} and {Chem.MolToSmiles(mol2)}')
        products = reaction.RunReactants((mol1, mol2))
        random_product_idx = np.random.randint(len(products))
        return Chem.MolToSmiles(products[random_product_idx][0])

    def fragmentize(self, molecule_smiles):
        """Breaks a molecule down into its constituent chemical fragments."""
        try:
            fragments = set()
            for safe_fragment_mol in self.sfcodec.encode_fragment(molecule_smiles):
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
    '''

    def fragmentize(self, molecule_smiles):
        """Breaks a molecule down into its constituent chemical fragments."""
        try:
            fragments = set()
            molecule_sf = self.sfcodec.encode(molecule_smiles)

            if molecule_sf is None:
                return None

            for fragment_sf in molecule_sf.split('.'):
                fragment_smiles = self.sfcodec.decode(fragment_sf)
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

    def generate(self, num_to_generate, random_seed=42):
        """Generates new molecules using the deep learning model."""
        generated_molecules = []
        max_attempts, attempts = num_to_generate * 10, 0

        can_gen_linker = len(self.arm_population) >= 2
        can_gen_motif = len(self.linker_population) >= 1

        if not can_gen_linker and not can_gen_motif:
            print("Warning: Not enough fragments to perform linker generation or motif extension.")
            return generated_molecules
        elif not can_gen_linker:
            print("Warning: Not enough arms in the population to perform linker generation. Defaulting to motif extension.")
        elif not can_gen_motif:
            print("Warning: Not enough linkers in the population to perform motif extension. Defaulting to linker generation.")

        print(f'Generating {num_to_generate} molecules...')
        while len(generated_molecules) < num_to_generate and attempts < max_attempts:
            attempts += 1
            try:
                if can_gen_linker and can_gen_motif:
                    gen_linker = random.random() < 0.5
                elif can_gen_linker:
                    gen_linker = True
                else:
                    gen_linker = False

                if gen_linker:
                    arm_frag_1, arm_frag_2 = random.sample([frag for _, frag in self.arm_population], 2)
                    # Any post processing of fragments?
                    self.designer.frags = [frag for _, frag in self.linker_population]
                    smiles = self.designer.linker_generation(arm_frag_1, arm_frag_2, n_samples_per_trial=1, random_seed=random_seed)[0]
                else:
                    arm_frag = random.choice([frag for _, frag in self.arm_population])
                    linker_frag = random.choice([frag for _, frag in self.linker_population])
                    motif = self.sfcodec.link_fragments(arm_frag, linker_frag) # SMILES
                    #motif = self.attach(arm_frag, linker_frag)
                    self.designer.frags = [frag for _, frag in self.arm_population]
                    smiles = self.designer.motif_extension(motif, n_samples_per_trial=1, random_seed=random_seed)[0]
                    smiles = sorted(smiles.split('.'), key=len)[-1]
                
                if hasattr(self.designer, 'decode'):
                    smiles = self.designer.decode(smiles)
                mol = Chem.MolFromSmiles(smiles)
                if mol and self.min_mol_size <= mol.GetNumAtoms() <= self.max_mol_size:
                    generated_molecules.append(smiles)
            except Exception as e:
                print(f'Error during generation: {e}')
                continue
        return generated_molecules

    '''
    def record(self, molecule_smiles_list, scores):
        """Appends molecules and their scores to the output CSV file."""
        with open(self.output_filepath, 'a', newline='') as f:
            for smiles, score in zip(molecule_smiles_list, scores):
                f.write(f'"{smiles}",{score}\n')
    '''