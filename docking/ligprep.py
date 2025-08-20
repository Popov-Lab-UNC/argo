from rdkit import Chem
#from molscrub import Scrub
from scrubber import Scrub
from meeko import MoleculePreparation, PDBQTWriterLegacy
import multiprocessing
from multiprocessing import Pool
from time import sleep
import os
import argparse

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

   return conformer_count

if __name__ == "__main__":
   # Parse command line arguments
   parser = argparse.ArgumentParser(description='Process ligands for docking preparation')
   parser.add_argument('--n_processes', type=int, default=min(multiprocessing.cpu_count(), 8),
                      help='Number of processes to use (default: min(CPU_count, 8))')
   args = parser.parse_args()

   # Multiprocessing options
   n_processes = args.n_processes

   # Scrubbing and ligand preparation options
   max_attempts = 5  # Maximum attempts for scrubbing each ligand
   scrub = Scrub(ph_low=7.4, ph_high=7.4, skip_tautomers=True)  # Setup scrub instance with pH constraints

   # Manual input and output paths
   input_file = "PptT/class1_compounds.smi"
   output_dir = "/work/users/s/h/shuhang/argo_docking/PptT_bm/class1_compounds"
   
   # Create output directory
   os.makedirs(output_dir, exist_ok=True)

   # Read ligands from input file and clean multi-fragment SMILES
   ligand_list = []
   multi_fragment_count = 0
   multi_fragment_file = "multi_fragment_smiles.txt"
   
   with open(input_file, "r") as f:
      for line_num, line in enumerate(f, 1):
         if len(line.split()) >= 2:
            ligand_smi, ligand_name = line.split()[0], line.split()[-1]
            
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
                  with open(multi_fragment_file, "a") as mf:
                     mf.write(f"{ligand_smi}\t{ligand_name}\t{num_fragments} fragments -> {cleaned_smiles}\n")
                  
                  # Use the cleaned SMILES for processing
                  ligand_list.append((cleaned_smiles, ligand_name, output_dir, scrub, max_attempts))
               else:
                  # Single fragment, use as is
                  ligand_list.append((ligand_smi, ligand_name, output_dir, scrub, max_attempts))

   print(f"Found {len(ligand_list)} ligands from {input_file}")
   print(f"Cleaned {multi_fragment_count} multi-fragment SMILES (saved to {multi_fragment_file})")
   print(f"Processing all {len(ligand_list)} ligands with {n_processes} processes")

   # Process ligands
   total_conformers = 0
   with Pool(processes=n_processes) as pool:
      for result in pool.imap_unordered(process_ligand, ligand_list):
         total_conformers += result
         if total_conformers % 1000 == 0:
            print(f"Generated {total_conformers} conformers so far")
   
   print(f"Successfully generated {total_conformers} conformers in {output_dir}")