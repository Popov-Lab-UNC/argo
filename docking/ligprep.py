from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from molvs import Standardizer
from molvs.charge import Uncharger
#from molscrub import Scrub
from scrubber import Scrub
from meeko import MoleculePreparation, PDBQTWriterLegacy
import multiprocessing
from multiprocessing import Pool
from time import sleep
import os
import argparse
import pandas as pd

def process_ligand(args):
   """
   Process a single ligand by scrubbing and preparing it for docking.

   Args:
      args (tuple): A tuple containing the SMILES string, ligand name, output directory,
                     scrub instance, and maximum attempts for scrubbing.

   Returns:
      int: Number of conformers generated for this ligand.
   """
   smi, name, outdir, scrub_instance, max_attempts = args
   conformer_count = 0

   # Convert SMILES string to molecule object
   mol = Chem.MolFromSmiles(smi)
   if mol is None:
      print(f"[SMILES] Failed to parse: {smi}")
      return 0

   # Safety check for multi-fragment molecules (should not happen after cleaning)
   num_fragments = len(Chem.GetMolFrags(mol))
   if num_fragments > 1:
      print(f"[Warning] Multi-fragment molecule found after cleaning: {name} ({num_fragments} fragments)")
      return 0

   # Apply scrub with retry in case of failure, primarily with conformer generation
   for attempt in range(1, max_attempts + 1):
      try:
            mol_states = list(scrub_instance(mol))
            break
      except RuntimeError as e:
            print(f"[Scrub Run {attempt}/{max_attempts}] Failed on {name}, molecule Smiles {smi}: {e}")
            sleep(0.1)
   else:
      print(f"[Scrub Run] Gave up on {name}")
      return 0

   # Initialize MoleculePreparation instance
   mk_prep = MoleculePreparation()
   counter = 0  # Counter for multiple outcomes from the same ligand SMILES

   # Prepare the molecules and generate the pdbqt files
   for mol_state in mol_states:
      molsetup_list = mk_prep.prepare(mol_state)
      for molsetup in molsetup_list:
            pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup)
            if success:
               path = os.path.join(outdir, f"{name}_{counter}.pdbqt")
               with open(path, "w") as f:
                  f.write(pdbqt_string)
               conformer_count += 1
               counter += 1
            else:
               print(f"[PDBQT] Write failed for {name}: {error_msg}")
               # Note: Failed writes are not counted in conformer_count

   return conformer_count

if __name__ == "__main__":
   # Parse command line arguments
   parser = argparse.ArgumentParser(description='Process ligands for docking preparation')
   parser.add_argument('--n_processes', type=int, default=min(multiprocessing.cpu_count(), 8),
                      help='Number of processes to use (default: min(CPU_count, 8))')
   parser.add_argument('--input-file', type=str, required=True,
                      help='Input file path (.smi or .csv)')
   parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for PDBQT files')
   parser.add_argument('--compound-id-col', type=str, default='compound_id',
                      help='Column name for compound ID in CSV (default: compound_id)')
   parser.add_argument('--smiles-col', type=str, default='smiles',
                      help='Column name for SMILES in CSV (default: smiles)')
   parser.add_argument('--verbose', type=bool, default=False,
                      help='Verbose output (default: False)')
   args = parser.parse_args()

   # Multiprocessing options
   n_processes = args.n_processes

   # Scrubbing and ligand preparation options
   max_attempts = 10  # Maximum attempts for scrubbing each ligand
   scrub = Scrub(ph_low=7.4, ph_high=7.4, skip_tautomers=False)  # Setup scrub instance with pH constraints

   # Input and output paths from arguments
   input_file = args.input_file
   output_dir = args.output_dir
   
   # Create output directory
   os.makedirs(output_dir, exist_ok=True)

   # Read ligands from input file and clean multi-fragment SMILES
   ligand_list = []
   multi_fragment_count = 0
   
   # Determine file type and read accordingly
   file_extension = os.path.splitext(input_file)[1].lower()
   
   if file_extension == '.csv':
      # Read CSV file
      try:
         df = pd.read_csv(input_file)
         if args.compound_id_col not in df.columns:
            raise ValueError(f"Column '{args.compound_id_col}' not found in CSV. Available columns: {list(df.columns)}")
         if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found in CSV. Available columns: {list(df.columns)}")
         
         s = Standardizer()
         remover = SaltRemover()
         uncharger = Uncharger()
         for idx, row in df.iterrows():
            # Extract and clean input data
            ligand_smi = str(row[args.smiles_col]).strip()
            ligand_name = str(row[args.compound_id_col]).strip()

            # Process SMILES through standardization pipeline
            try:
                mol = Chem.MolFromSmiles(ligand_smi)
                if mol is None:
                    print(f"[CSV] Failed to parse SMILES: {ligand_smi} for compound {ligand_name}")
                    continue
                    
                mol = s.standardize(mol)
                mol = remover.StripMol(mol)
                mol = uncharger.uncharge(mol)

                # Handle multi-fragment molecules
                fragments = Chem.GetMolFrags(mol, asMols=True)
                if len(fragments) > 1:
                    multi_fragment_count += 1
                    mol = max(fragments, key=lambda x: x.GetNumAtoms())
                    if args.verbose:
                        cleaned_smiles = Chem.MolToSmiles(mol)
                        print(f"{ligand_smi}\t{ligand_name}\t{len(fragments)} fragments -> {cleaned_smiles}\n")

                # Add processed molecule to list
                ligand_list.append((Chem.MolToSmiles(mol), ligand_name, output_dir, scrub, max_attempts))

            except Exception as e:
                print(f"[CSV] Error processing {ligand_name}: {str(e)}")
               
      except Exception as e:
         print(f"Error reading CSV file: {e}")
         exit(1)
         
   elif file_extension == '.smi':
      # Read SMILES file (original logic)
      with open(input_file, "r") as f:
         for line_num, line in enumerate(f, 1):
            if len(line.split()) >= 2:
               ligand_smi, ligand_name = line.split()[0], line.split()[-1]
            elif len(line.split()) == 1:
               ligand_smi = line.split()[0]
               ligand_name = 'ligand_' + str(line_num)
               
            # Check for multi-fragment SMILES
            mol = Chem.MolFromSmiles(ligand_smi)
            if mol is not None:
               num_fragments = len(Chem.GetMolFrags(mol))
               if num_fragments > 1:
                  multi_fragment_count += 1
                  
                  # Get the largest fragment
                  fragments = Chem.GetMolFrags(mol, asMols=True)
                  largest_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
                  cleaned_smiles = Chem.MolToSmiles(largest_fragment)
                  
                  # Record original and cleaned SMILES
                  if args.verbose:
                     print(f"{ligand_smi}\t{ligand_name}\t{num_fragments} fragments -> {cleaned_smiles}\n")
                  
                  # Use the cleaned SMILES for processing
                  ligand_list.append((cleaned_smiles, ligand_name, output_dir, scrub, max_attempts))
               else:
                  # Single fragment, use as is
                  ligand_list.append((ligand_smi, ligand_name, output_dir, scrub, max_attempts))
   else:
      print(f"Unsupported file format: {file_extension}. Please use .smi or .csv files.")
      exit(1)

   print(f"Found {len(ligand_list)} ligands from {input_file}")
   print(f"Processing all {len(ligand_list)} ligands with {n_processes} processes")

   # Process ligands
   total_conformers = 0
   failed_conformers = 0
   
   with Pool(processes=n_processes) as pool:
      for result in pool.imap_unordered(process_ligand, ligand_list):
         total_conformers += result
         if total_conformers % 1000 == 0:
            print(f"Generated {total_conformers} conformers so far")
   
   # Count actual files created to verify
   import glob
   actual_files = len(glob.glob(os.path.join(output_dir, "*.pdbqt")))
   
   print(f"Successfully generated {total_conformers} conformers in {output_dir}")
   print(f"Actual PDBQT files created: {actual_files}")
   if actual_files != total_conformers:
      print(f"Warning: Mismatch between reported ({total_conformers}) and actual files ({actual_files})")
      print("This may indicate PDBQT write failures during processing")