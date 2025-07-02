"""
This script defines the `f_RAG_Verbose` class, a powerful tool for de novo molecule
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

    def __init__(self, args):
        """
        Initializes the f-RAG system.

        Args:
            args (Namespace or dict): A configuration object containing necessary
                                      parameters like file paths, population sizes,
                                      and molecule size constraints.
        """
        print("Initializing f-RAG model...")
        self.args = args

        # --- Model and Tool Initialization ---
        # Load the pre-trained deep learning model for fragment-based generation.
        self.designer = SAFEFusionDesign.load_default()
        
        # Optionally, load a user-specified, fine-tuned model for more specific tasks.
        if hasattr(args, 'injection_model_path') and args.injection_model_path:
            self.designer.load_fuser(args.injection_model_path)
            print(f"Loaded custom fuser model from {args.injection_model_path}.")

        # Initialize the tool used to break molecules down into fragments.
        self.slicer = MolSlicer(shortest_linker=True)

        # --- Population Initialization ---
        # These lists will store tuples of (score, smiles_string)
        self.molecule_population = []  # Population of full, high-scoring molecules
        self.arm_population = []       # Population of fragments with one attachment point
        self.linker_population = []    # Population of fragments with two attachment points

        # Load the initial set of fragments from a vocabulary file.
        self.set_initial_population(self.args.vocab_path)

        # --- Configuration Settings ---
        # Set molecule size constraints for the genetic algorithm crossover function.
        co.MIN_SIZE, co.MAX_SIZE = args.min_mol_size, args.max_mol_size
        
        # Set up the path for the results file and create the directory if it doesn't exist.
        self.output_filepath = getattr(args, 'results_path', 'frag_results.csv')
        output_dir = os.path.dirname(self.output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_filepath}")

    def attach(self, fragment_smiles_1, fragment_smiles_2):
        """
        Chemically joins two fragments together at their attachment points.

        Args:
            fragment_smiles_1 (str): SMILES string of the first fragment (e.g., 'c1ccccc1[*:1]').
            fragment_smiles_2 (str): SMILES string of the second fragment (e.g., 'C[*:1]').

        Returns:
            str: SMILES string of the newly formed, combined molecule.
        """
        # Defines a chemical reaction where two attachment points '[1*]' are merged into a single bond.
        # SMARTS reaction: [Fragment A with tag 1] + [Fragment B with tag 1] >> [A-B fused]
        reaction = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        
        # Convert SMILES to RDKit molecule objects.
        mol1 = Chem.MolFromSmiles(fragment_smiles_1)
        mol2 = Chem.MolFromSmiles(fragment_smiles_2)
        
        # Run the reaction. It can sometimes produce multiple outcomes.
        reactant_mols = (mol1, mol2)
        products = reaction.RunReactants(reactant_mols)
        
        # If multiple products are possible, pick one randomly.
        random_product_idx = np.random.randint(len(products))
        return Chem.MolToSmiles(products[random_product_idx][0])

    def fragmentize(self, molecule_smiles):
        """
        Breaks a molecule down into its constituent chemical fragments.

        Args:
            molecule_smiles (str): SMILES string of the molecule to be fragmented.

        Returns:
            list[str] or None: A list of valid fragment SMILES strings, or None if fragmentation fails.
        """
        try:
            fragments = set()
            # Use the MolSlicer to perform the fragmentation.
            for safe_fragment_mol in self.slicer(molecule_smiles):
                if safe_fragment_mol is None:
                    continue
                
                fragment_smiles = Chem.MolToSmiles(safe_fragment_mol)
                
                # Standardize attachment points. Slicers might output [2*], [3*], etc.
                # We simplify them all to '[1*]' for consistency.
                fragment_smiles = re.sub(r'\[\d+\*\]', '[1*]', fragment_smiles)
                
                # We are only interested in fragments that can be used as arms or linkers.
                num_attachment_points = fragment_smiles.count('*')
                if num_attachment_points in {1, 2}:
                    fragments.add(fragment_smiles)

            # Filter fragments based on size constraints defined in the arguments.
            valid_fragments = [frag for frag in fragments
                               if self.args.min_frag_size <= Chem.MolFromSmiles(frag).GetNumAtoms() <= self.args.max_frag_size]
            return valid_fragments
        except Exception:
            # RDKit can fail on malformed SMILES strings.
            return None

    def set_initial_population(self, vocabulary_path):
        """
        Loads the initial fragment populations from a CSV file.

        Args:
            vocabulary_path (str): The file path to the fragment vocabulary CSV.
                                   The CSV must contain 'frag', 'size', and optionally 'score' columns.
        """
        print(f"Loading initial fragment vocabulary from {vocabulary_path}...")
        try:
            vocabulary_df = pd.read_csv(vocabulary_path)
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {vocabulary_path}. Cannot set initial population.")
            return

        # Filter fragments to meet size constraints.
        vocabulary_df = vocabulary_df[vocabulary_df['size'] >= self.args.min_frag_size]
        vocabulary_df = vocabulary_df[vocabulary_df['size'] <= self.args.max_frag_size]

        # Use provided scores, or default to 0 if the 'score' column doesn't exist.
        scores = vocabulary_df.get('score', [0.0] * len(vocabulary_df))

        # Populate arm and linker lists based on the number of attachment points.
        for score, fragment_smiles in zip(scores, vocabulary_df['frag']):
            if fragment_smiles.count('*') == 1:
                self.arm_population.append((score, fragment_smiles))
            else:
                self.linker_population.append((score, fragment_smiles))
            
            # Optimization: Stop reading the file once we have enough fragments to fill our populations.
            if (len(self.arm_population) >= self.args.frag_population_size and
                len(self.linker_population) >= self.args.frag_population_size):
                break
        
        # Ensure populations are exactly the desired size by trimming any excess.
        self.arm_population = self.arm_population[:self.args.frag_population_size]
        self.linker_population = self.linker_population[:self.args.frag_population_size]
        print(f"Initialized with {len(self.arm_population)} arms and {len(self.linker_population)} linkers.")


    def update_population(self, scores, new_molecule_smiles_list):
        """
        Updates all populations (molecules, arms, linkers) with new, high-scoring individuals.
        This is the "learning" or "evolution" step of the algorithm.

        Args:
            scores (list[float]): A list of scores for the new molecules.
            new_molecule_smiles_list (list[str]): A list of SMILES for the new molecules.
        """
        # Add new, unique molecules to the main population.
        new_molecules = list(set(zip(scores, new_molecule_smiles_list)))
        self.molecule_population.extend(new_molecules)
        
        # Sort the population by score (descending) and keep only the top performers.
        self.molecule_population.sort(reverse=True, key=lambda x: x[0])
        self.molecule_population = self.molecule_population[:self.args.mol_population_size]

        # --- Enrich fragment populations from the new successful molecules ---
        # Create sets for fast checking of existing fragments.
        existing_arms = {frag for score, frag in self.arm_population}
        existing_linkers = {frag for score, frag in self.linker_population}

        for score, smiles in zip(scores, new_molecule_smiles_list):
            new_fragments = self.fragmentize(smiles)
            if new_fragments:
                for fragment_smiles in new_fragments:
                    num_attachments = fragment_smiles.count('*')
                    # If it's a new arm, add it to the arm population.
                    if num_attachments == 1 and fragment_smiles not in existing_arms:
                        self.arm_population.append((score, fragment_smiles))
                        existing_arms.add(fragment_smiles)
                    # If it's a new linker, add it to the linker population.
                    elif num_attachments == 2 and fragment_smiles not in existing_linkers:
                        self.linker_population.append((score, fragment_smiles))
                        existing_linkers.add(fragment_smiles)

        # Sort and trim the fragment populations, keeping the highest-scoring ones.
        self.arm_population.sort(reverse=True, key=lambda x: x[0])
        self.linker_population.sort(reverse=True, key=lambda x: x[0])
        self.arm_population = self.arm_population[:self.args.frag_population_size]
        self.linker_population = self.linker_population[:self.args.frag_population_size]

    def generate(self, num_to_generate):
        """
        Generates a specified number of new molecules using the deep learning model.

        Args:
            num_to_generate (int): The number of molecules to generate.

        Returns:
            list[str]: A list of SMILES strings for the newly generated molecules.
        """
        generated_molecules = []
        # Try up to 10 times per molecule to avoid infinite loops on generation failure.
        max_attempts = num_to_generate * 10
        attempts = 0

        while len(generated_molecules) < num_to_generate and attempts < max_attempts:
            attempts += 1
            try:
                # Randomly choose one of two generation strategies.
                if random.random() < 0.5:
                    # Strategy 1: Linker Generation. Create a bridge between two arms.
                    arm_frag_1, arm_frag_2 = random.sample([frag for score, frag in self.arm_population], 2)
                    self.designer.frags = [frag for score, frag in self.linker_population] # Provide inspiration
                    
                    smiles = self.designer.linker_generation(
                        arm_frag_1, arm_frag_2,
                        n_samples_per_trial=1,
                        random_seed=self.args.seed
                    )[0]
                else:
                    # Strategy 2: Motif Extension. Grow a new piece from an existing motif.
                    arm_frag = random.choice([frag for score, frag in self.arm_population])
                    linker_frag = random.choice([frag for score, frag in self.linker_population])
                    
                    # Create the initial motif by attaching the arm and linker.
                    motif = re.sub(r'\[1\*\]', '[*]', self.attach(arm_frag, linker_frag))
                    self.designer.frags = [frag for score, frag in self.arm_population] # Provide inspiration
                    
                    smiles = self.designer.motif_extension(
                        motif,
                        n_samples_per_trial=1,
                        random_seed=self.args.seed
                    )[0]
                    
                    # If generation results in disconnected parts ('.'), take the largest one.
                    smiles = sorted(smiles.split('.'), key=len)[-1]
                
                # Some models might output an intermediate representation that needs decoding.
                if hasattr(self.designer, 'decode'):
                    smiles = self.designer.decode(smiles)

                # Final validation: check if the generated molecule is valid and within size limits.
                mol = Chem.MolFromSmiles(smiles)
                if mol and self.args.min_mol_size <= mol.GetNumAtoms() <= self.args.max_mol_size:
                    generated_molecules.append(smiles)
            
            except Exception:
                # Generation can fail for many reasons (e.g., RDKit error, model failure).
                # We simply ignore the failure and try again.
                continue
        
        return generated_molecules

    def record(self, molecule_smiles_list, scores):
        """
        Appends a list of molecules and their scores to the output CSV file.

        Args:
            molecule_smiles_list (list[str]): The SMILES strings to record.
            scores (list[float]): The corresponding scores.
        """
        with open(self.output_filepath, 'a', newline='') as f:
            for smiles, score in zip(molecule_smiles_list, scores):
                f.write(f'"{smiles}",{score}\n')


if __name__ == "__main__":
    def evaluate_qed(smiles_list):
        """
        Evaluates molecules using QED (Quantitative Estimate of Drug-likeness).
        
        Args:
            smiles_list (list[str]): A list of molecules to score.
        
        Returns:
            list[float]: A list of corresponding QED scores between 0 and 1.
        """
        print(f"  (QED evaluation for {len(smiles_list)} molecules...)")
        scores = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    score = float(Chem.QED.default(mol))
                    scores.append(score)
                else:
                    scores.append(-1.0) # Penalize invalid SMILES
            except:
                scores.append(-1.0)
        return scores

    # --- 1. Set up Configuration Arguments ---
    # In a real script, this would be handled by `argparse.ArgumentParser`.
    parser = argparse.ArgumentParser(description="Run f-RAG molecule generation.")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to initial fragment vocabulary CSV.")
    parser.add_argument('--results_path', type=str, default='output/frag_run_results.csv', help="Path to save results.")
    parser.add_argument('--num_generations', type=int, default=5, help="Number of generations to run.")
    # Add other arguments from the class...
    # For this template, we create a simple namespace object manually.
    
    # NOTE: You must create a 'my_fragment_vocab.csv' file with 'frag', 'size' columns for this to work.
    # Example my_fragment_vocab.csv:
    # frag,size
    # "c1ccccc1[*:1]",7
    # "C[*:1]",2
    # "[*:1]CC[*:2]",4
    
    args = argparse.Namespace(
        # File paths
        vocab_path='my_fragment_vocab.csv',  # <--- USER MUST PROVIDE THIS FILE
        results_path='output/frag_run_results.csv',
        injection_model_path=None,
        # Population sizes
        frag_population_size=50,
        mol_population_size=100,
        # Generation batch sizes
        num_safe_per_gen=10,
        num_ga_per_gen=10,
        # Size constraints
        min_frag_size=1, max_frag_size=15,
        min_mol_size=10, max_mol_size=50,
        # Other parameters
        mutation_rate=0.01,
        seed=42,
        num_generations=5
    )
    
    # Ensure the dummy vocab file exists for the example to be understandable
    if not os.path.exists(args.vocab_path):
        print(f"Creating a dummy vocabulary file at '{args.vocab_path}' for demonstration.")
        dummy_data = {'frag': ['c1ccccc1[*:1]', 'C[*:1]', '[*:1]CC[*:2]'], 'size': [7, 2, 4]}
        pd.DataFrame(dummy_data).to_csv(args.vocab_path, index=False)


    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- 2. Initialize the f-RAG Model ---
    model = f_RAG(args)

    # --- 3. Run the Generative Loop ---
    print("\n--- Starting Generative Process ---")
    for generation in range(args.num_generations):
        print(f"\n>> Generation {generation + 1}/{args.num_generations}")

        # --- Step A: Generate new molecules ---
        print(f"Generating {args.num_safe_per_gen} molecules with SAFE model...")
        safe_smiles_list = model.generate(num_to_generate=args.num_safe_per_gen)
        all_new_smiles = safe_smiles_list

        # After the molecule population has been established, use the GA for diversity
        if len(model.molecule_population) >= args.num_ga_per_gen:
            print(f"Generating {args.num_ga_per_gen} molecules with Genetic Algorithm...")
            ga_smiles_list = [reproduce(model.molecule_population, args.mutation_rate)
                              for _ in range(args.num_ga_per_gen)]
            all_new_smiles.extend(ga_smiles_list)
        
        # --- Step B: Evaluate the new molecules using the custom function ---
        all_new_smiles = list(filter(None, all_new_smiles))
        if not all_new_smiles:
            print("No valid molecules were generated in this step. Skipping.")
            continue
            
        scores = evaluate_qed(all_new_smiles)

        # --- Step C: Update the populations and record results ---
        print("Updating populations with new results...")
        model.update_population(scores, all_new_smiles)
        model.record(all_new_smiles, scores)

        # --- Step D: Report status ---
        if model.molecule_population:
            best_score, best_smiles = model.molecule_population[0]
            print(f"Current best molecule: {best_smiles} (Score: {best_score:.2f})")
            print(f"Molecule population size: {len(model.molecule_population)}")
            print(f"Arm/Linker population sizes: {len(model.arm_population)} / {len(model.linker_population)}")
        else:
            print("No valid molecules in population yet.")

    print("\n--- Generative Process Finished ---")