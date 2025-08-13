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

from tdc import Oracle

from argo.gen_models.f_rag.fusion.sample import SAFEFusionDesign
#from argo.gen_models.f_rag.fusion.slicer import MolSlicer
#from argo.gen_models.f_rag.fusion.slicer import MolSlicerForSAFEEncoder
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
        vocab: "str | pd.DataFrame",
        frag_population_size: int = 50,
        mol_population_size: int = 100,
        min_frag_size: int = 1,
        max_frag_size: int = 15,
        min_mol_size: int = 10,
        max_mol_size: int = 100,
        use_cuda: bool = True,
    ):
        """
        Initializes the f-RAG system with explicit parameters.
        vocab can be a path to a CSV file or a pandas DataFrame.
        
        Args:
            injection_model_path: Path to the injection model
            vocab: Vocabulary data (path or DataFrame)
            frag_population_size: Size of fragment populations
            mol_population_size: Size of molecule population
            min_frag_size: Minimum fragment size
            max_frag_size: Maximum fragment size
            min_mol_size: Minimum molecule size
            max_mol_size: Maximum molecule size
            mutation_rate: Mutation rate for genetic algorithm
            use_cuda: Whether to use CUDA if available (default: True)
        """
        print("Initializing f-RAG model...")
        # --- Store configuration as instance attributes ---
        if frag_population_size < 10:
            raise ValueError("frag_population_size must be at least 10.")
        self.vocab = vocab
        self.injection_model_path = injection_model_path
        self.frag_population_size = frag_population_size
        self.mol_population_size = mol_population_size
        self.min_frag_size = min_frag_size
        self.max_frag_size = max_frag_size
        self.min_mol_size = min_mol_size
        self.max_mol_size = max_mol_size
        self.use_cuda = use_cuda
        
        # --- Model and Tool Initialization ---
        self.designer = SAFEFusionDesign.load_default(use_cuda=self.use_cuda)
        self.designer.load_fuser(self.injection_model_path, use_cuda=self.use_cuda)
        print(f"Loaded custom fuser model from {self.injection_model_path}.")

        #slicer = MolSlicerForSAFEEncoder(shortest_linker=True)
        self.sfcodec = SAFECodec(slicer='f-rag', ignore_stereo=True)

        # --- Population Initialization ---
        self.mol_population = []
        self.arm_population = []
        self.linker_population = []
        self.set_initial_population(self.vocab)

        # Check for minimum arms and linkers
        if len(self.arm_population) < 10 or len(self.linker_population) < 10:
            raise ValueError(f"Initialization failed: Need at least 10 arms and 10 linkers, got {len(self.arm_population)} arms and {len(self.linker_population)} linkers.")

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
                #fragment_smiles = re.sub(r'\[\d+\*\]', '[1*]', fragment_smiles)
                if fragment_smiles.count('*') in {1, 2}:
                    fragments.add(fragment_smiles)
            
            valid_fragments = [
                frag for frag in fragments
                if self.min_frag_size <= Chem.MolFromSmiles(frag).GetNumAtoms() <= self.max_frag_size
            ]
            return valid_fragments
        except Exception:
            return None

    def set_initial_population(self, vocabulary: "str | pd.DataFrame"):
        """Loads the initial fragment populations from a CSV file or DataFrame."""
        if isinstance(vocabulary, str):
            print(f"Loading initial fragment vocabulary from {vocabulary}.")
            try:
                vocabulary_df = pd.read_csv(vocabulary)
            except FileNotFoundError:
                print(f"Error: Vocabulary file not found at {vocabulary}. Cannot set initial population.")
                return
        elif isinstance(vocabulary, pd.DataFrame):
            print("Loading initial fragment vocabulary from provided DataFrame.")
            vocabulary_df = vocabulary.copy()
        else:
            print("Error: vocabulary must be a file path or pandas DataFrame.")
            return

        # Ensure required columns exist
        required_columns = {'frag', 'size'}
        if not required_columns.issubset(vocabulary_df.columns):
            raise ValueError(f"Vocabulary DataFrame must contain columns: {required_columns}. Found: {set(vocabulary_df.columns)}")

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

    def update_population(self, scores, new_molecule_smiles_list, higher_is_better=True):
        """Updates all populations with new, high-scoring individuals."""
        new_molecules = list(set(zip(scores, new_molecule_smiles_list)))
        self.mol_population.extend(new_molecules)
        self.mol_population.sort(reverse=higher_is_better, key=lambda x: x[0])
        self.mol_population = self.mol_population[:self.mol_population_size]

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

        self.arm_population.sort(reverse=higher_is_better, key=lambda x: x[0])
        self.linker_population.sort(reverse=higher_is_better, key=lambda x: x[0])
        self.arm_population = self.arm_population[:self.frag_population_size]
        self.linker_population = self.linker_population[:self.frag_population_size]

    def linker_generation(self, n_samples, random_seed=42):
        """
        Generates molecules by connecting two randomly selected arms using a linker.
        """
        generated_molecules = []
        max_attempts, attempts = n_samples * 3, 0

        while len(generated_molecules) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                arm_frag_1, arm_frag_2 = random.sample([frag for _, frag in self.arm_population], 2)
                self.designer.frags = [frag for _, frag in self.linker_population]
                designs = self.designer.linker_generation(arm_frag_1, arm_frag_2, n_samples_per_trial=1, random_seed=random_seed)
                if len(designs) == 0:
                    continue
                smiles = self.sfcodec.decode(designs[0])
                mol = Chem.MolFromSmiles(smiles)
                if mol and self.min_mol_size <= mol.GetNumAtoms() <= self.max_mol_size:
                    generated_molecules.append(smiles)
            except Exception as e:
                print(f'Error during linker generation: {e}')
                continue
        return generated_molecules

    def scaffold_decoration(self, n_samples, scaffold=None, random_seed=42):
        """
        Generates molecules by extending a motif (arm + linker) with additional arms.
        """
        generated_molecules = []
        max_attempts, attempts = n_samples * 3, 0

        while len(generated_molecules) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                arm_frag = random.choice([frag for _, frag in self.arm_population])
                if scaffold:
                    linker_frag = scaffold
                else:
                    linker_frag = random.choice([frag for _, frag in self.linker_population])
                motif = self.sfcodec.link_fragments(arm_frag, linker_frag)
                self.designer.frags = [frag for _, frag in self.arm_population]
                designs = self.designer.motif_extension(motif, n_samples_per_trial=1, random_seed=random_seed)
                if len(designs) == 0 or designs[0] == None:
                    continue
                smiles = self.sfcodec.decode(sorted(designs[0].split('.'), key=len)[-1])
                mol = Chem.MolFromSmiles(smiles)
                if mol and self.min_mol_size <= mol.GetNumAtoms() <= self.max_mol_size:
                    generated_molecules.append(smiles)
            except Exception as e:
                print(f'Error during scaffold decoration: {e}')
                continue
        return generated_molecules

    '''
    def record(self, molecule_smiles_list, scores):
        """Appends molecules and their scores to the output CSV file."""
        with open(self.output_filepath, 'a', newline='') as f:
            for smiles, score in zip(molecule_smiles_list, scores):
                f.write(f'"{smiles}",{score}\n')
    '''

    def _reset_fragment_scores(self):
        self.arm_population = [(0.0, frag) for _, frag in self.arm_population]
        self.linker_population = [(0.0, frag) for _, frag in self.linker_population]
        return

    # Call reset_mol_population if you want to reset the mol population before optimizing
    def reset_mol_population(self):
        self.mol_population = []

    def reset_mol_population_size(self, new_mol_population_size: int):
        self.mol_population_size = new_mol_population_size

    def reset_frag_population(self, new_vocab: 'str | pd.DataFrame' = None, new_frag_population_size: int = None):
        self.arm_population = []
        self.linker_population = []

        if new_frag_population_size < 10:
            raise ValueError("frag_population_size must be at least 10.")
        if new_frag_population_size is not None:
            self.frag_population_size = new_frag_population_size

        if new_vocab is not None:
            self.vocab = new_vocab
            self.set_initial_population(self.vocab)
        else:
            self.set_initial_population(self.vocab)

    # Optimization must reset fragment scores to 0.0
    # TODO: Logic for threshold for higher_is_better=False
    def optimize(self,
                 n_samples,
                 oracle_name='QED',
                 random_seed=42,
                 threshold=0.8,
                 higher_is_better=True,
                 max_iter=10,              
                 batch_size=50,
                 mutation_rate=0.01,
                 init_lg_wt=0.5,
                 init_sd_wt=0.5,
                 init_ga_wt=0.0
    ):
        """
        Optimizes and collects n_samples of molecules that meet a threshold.

        n_samples: The target number of optimized molecules to collect and return.
        batch_size: The number of new molecules to generate in each iteration.
        """
        # 1. Initialization
        assert oracle_name.lower() in ['qed', 'sa', 'logp'], f"Oracle name must be one of ['QED', 'SA', 'LogP'], got {oracle_name}"
        
        tdc_oracle = Oracle(name=oracle_name)
        print(f'Optimizing with {oracle_name} to collect {n_samples} molecules with score > {threshold}...')

        self._reset_fragment_scores()

        # The final collection of successful molecules
        optimized_molecules = []
        seen_smiles = set()

        molecule_sources = {}
        sampler_weights = {'lg': init_lg_wt, 'sd': init_sd_wt, 'ga': init_ga_wt}
        ga_active = False
        
        for i in range(max_iter):
            # 2. Check if we have collected enough molecules
            if len(optimized_molecules) >= n_samples:
                print(f"\nCollection complete: Found {len(optimized_molecules)} molecules meeting the threshold.")
                break

            # 3. Determine sample counts for this batch
            normalized_weights = self._normalize_weights(sampler_weights, ga_active)
            n_samples_by_source = self._get_batch_counts(batch_size, normalized_weights)

            # 4. Generate and Score a new Batch
            new_mols_with_source = self._generate_molecules(n_samples_by_source, random_seed + i, mutation_rate, ga_active)
            if not new_mols_with_source:
                print("Warning: No molecules generated this iteration.")
                continue

            smiles_list = [smiles for smiles, _ in new_mols_with_source]
            scores = tdc_oracle(smiles_list)
            
            # 5. Process the new batch: Collect good molecules and update the pool
            # Collect new, valid molecules in a batch
            new_mols = [
                (score, smiles)
                for score, smiles in zip(scores, smiles_list)
                if score >= threshold and smiles not in seen_smiles
            ]
            optimized_molecules.extend(new_mols)
            seen_smiles.update(smiles for _, smiles in new_mols)

            # Report Progress
            self._print_iteration_stats(i, self.mol_population, threshold, optimized_molecules, sampler_weights)
            
            # Update the revolving gene pool
            self.update_population(scores, smiles_list, higher_is_better)

            # Update source tracking for weight adaptation
            for smiles, source in new_mols_with_source:
                molecule_sources[smiles] = source

            # Adapt weights based on the top performers in the current gene pool
            sampler_weights = self._adapt_weights(self.mol_population, molecule_sources)

            # 6. Activate GA and Adapt Weights
            if not ga_active and len(self.mol_population) >= self.mol_population_size:
                ga_active = True
                print("--- Genetic Algorithm Activated ---")
                if sampler_weights['ga'] == 0.0:
                    sampler_weights['ga'] = 0.2

        else: # This 'else' belongs to the 'for' loop, executes if loop finishes without `break`
            print(f"\nMax iterations ({max_iter}) reached. Collected {len(optimized_molecules)} out of {n_samples} desired molecules.")

        # Sort the final collection and return the top n_samples
        optimized_molecules.sort(key=lambda x: x[0], reverse=higher_is_better)
        return [smiles for score, smiles in optimized_molecules[:n_samples]]

    def _get_batch_counts(self, batch_size, normalized_weights):
        """Calculates how many molecules to generate from each source for a batch."""
        counts = {
            'lg': int(batch_size * normalized_weights['lg']),
            'sd': int(batch_size * normalized_weights['sd']),
            'ga': int(batch_size * normalized_weights['ga'])
        }
        # Ensure the total is exactly batch_size by assigning remainder to the largest contributor
        remainder = batch_size - sum(counts.values())
        if remainder > 0:
            largest_source = max(counts, key=counts.get)
            counts[largest_source] += remainder
        return counts
    
    def _print_iteration_stats(self, iter_num, population, threshold, collection, weights):
        """Prints a summary of the current iteration."""
        pool_scores = [score for score, _ in population]
        mean_score = np.mean(pool_scores) if pool_scores else 0
        max_score = np.max(pool_scores) if pool_scores else 0
        
        stats_str = f"Iter {iter_num}: Pool Mean Score={mean_score:.3f}, Pool Max Score={max_score:.3f}"
        collection_str = f"Collected={len(collection)}"
        weights_str = f"Weights: lg={weights.get('lg', 0):.2f}, sd={weights.get('sd', 0):.2f}, ga={weights.get('ga', 0):.2f}"
        
        print(f"{stats_str} | {collection_str} | {weights_str}")

    # The helper methods _normalize_weights, _generate_molecules, and _adapt_weights
    # can remain largely the same as in the previous refactoring.
    # Make sure _adapt_weights uses the `self.mol_population` for its analysis.
    def _normalize_weights(self, weights, ga_active):
        """Normalizes the weights of the active samplers."""
        active_weights = weights.copy()
        if not ga_active:
            active_weights['ga'] = 0.0
        
        total_weight = sum(active_weights.values())
        if total_weight == 0: # Fallback if all weights are zero
            return {'lg': 0.5, 'sd': 0.5, 'ga': 0.0}
            
        return {key: val / total_weight for key, val in active_weights.items()}

    def _generate_molecules(self, n_samples_by_source, random_seed, mutation_rate, ga_active):
        """Generates molecules from different sources."""
        all_mols = []
        if n_samples_by_source['lg'] > 0:
            linker_mols = self.linker_generation(n_samples=n_samples_by_source['lg'], random_seed=random_seed)
            all_mols.extend([(smiles, 'lg') for smiles in linker_mols])
        
        if n_samples_by_source['sd'] > 0:
            scaffold_mols = self.scaffold_decoration(n_samples=n_samples_by_source['sd'], random_seed=random_seed)
            all_mols.extend([(smiles, 'sd') for smiles in scaffold_mols])

        if ga_active and n_samples_by_source['ga'] > 0 and self.mol_population:
            ga_mols = [reproduce(self.mol_population, mutation_rate) for _ in range(n_samples_by_source['ga'])]
            all_mols.extend([(smiles, 'ga') for smiles in ga_mols])
            
        return all_mols

    def _adapt_weights(self, top_mols, molecule_sources):
        """Adaptively updates sampler weights based on their contribution to the top molecules."""
        source_counts = {'lg': 0, 'sd': 0, 'ga': 0}
        for _, smiles in top_mols:
            source = molecule_sources.get(smiles)
            if source in source_counts:
                source_counts[source] += 1
        
        total = sum(source_counts.values())

        return {source: count / total for source, count in source_counts.items()}